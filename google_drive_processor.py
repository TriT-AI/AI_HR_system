# google_drive_processor.py
from __future__ import annotations

import asyncio
import io
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

import requests
from config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

class GoogleDriveProcessor:
    def __init__(self, cv_processor, database):
        self.cv_processor = cv_processor
        self.db = database
        self.api_key = settings.GOOGLE_DRIVE_API_KEY
        self.folder_id = settings.GOOGLE_DRIVE_FOLDER_ID
        self.processed_files: Set[str] = set()
        self._load_processed_files()
        
        if not self.api_key or not self.folder_id:
            logger.warning("Google Drive API key or folder ID not configured")

    def _load_processed_files(self):
        """Load list of already processed files from database."""
        try:
            # Check if we have a tracking table
            self.db.conn.execute("""
                CREATE TABLE IF NOT EXISTS processed_drive_files (
                    file_id VARCHAR PRIMARY KEY,
                    file_name VARCHAR,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    candidate_id INTEGER
                )
            """)
            
            # Load previously processed file IDs
            processed = self.db.conn.execute(
                "SELECT file_id FROM processed_drive_files"
            ).fetchall()
            
            self.processed_files = {row[0] for row in processed}
            logger.info(f"Loaded {len(self.processed_files)} previously processed files")
            
        except Exception as e:
            logger.error(f"Error loading processed files: {e}")
            self.processed_files = set()

    def list_pdf_files_in_folder(self) -> List[Dict]:
        """List all PDF files in the specified Google Drive folder."""
        if not self.api_key or not self.folder_id:
            return []

        try:
            # Google Drive API v3 endpoint to list files
            url = "https://www.googleapis.com/drive/v3/files"
            
            params = {
                'key': self.api_key,
                'q': f"'{self.folder_id}' in parents and mimeType='application/pdf' and trashed=false",
                'fields': 'files(id,name,modifiedTime,size)',
                'orderBy': 'modifiedTime desc',
                'pageSize': 100
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            files = data.get('files', [])
            
            logger.info(f"Found {len(files)} PDF files in Google Drive folder")
            return files
            
        except Exception as e:
            logger.error(f"Error listing Google Drive files: {e}")
            return []

    def download_file(self, file_id: str, file_name: str) -> Optional[Path]:
        """Download a file from Google Drive to temporary storage."""
        if not self.api_key:
            return None

        try:
            # Google Drive API v3 endpoint to download files
            url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
            
            params = {
                'key': self.api_key,
                'alt': 'media'
            }
            
            response = requests.get(url, params=params, stream=True)
            response.raise_for_status()
            
            # Create temporary file
            temp_dir = Path("temp_drive_downloads")
            temp_dir.mkdir(exist_ok=True)
            
            temp_file = temp_dir / f"{file_id}_{file_name}"
            
            # Write file content
            with open(temp_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"Downloaded file: {file_name} -> {temp_file}")
            return temp_file
            
        except Exception as e:
            logger.error(f"Error downloading file {file_name}: {e}")
            return None

    def mark_file_as_processed(self, file_id: str, file_name: str, candidate_id: int):
        """Mark a file as processed in the database."""
        try:
            self.db.conn.execute(
                """INSERT INTO processed_drive_files (file_id, file_name, candidate_id) 
                   VALUES (?, ?, ?)""",
                [file_id, file_name, candidate_id]
            )
            self.processed_files.add(file_id)
            logger.info(f"Marked file {file_name} as processed")
            
        except Exception as e:
            logger.error(f"Error marking file as processed: {e}")

    async def process_new_files(self) -> List[Dict]:
        """Check for new PDF files and process them."""
        logger.info("Checking Google Drive for new PDF files...")
        
        files = self.list_pdf_files_in_folder()
        new_files = [f for f in files if f['id'] not in self.processed_files]
        
        if not new_files:
            logger.info("No new files found in Google Drive")
            return []

        logger.info(f"Found {len(new_files)} new files to process")
        results = []

        for file_info in new_files:
            file_id = file_info['id']
            file_name = file_info['name']
            
            try:
                logger.info(f"Processing: {file_name}")
                
                # Download file
                temp_file = self.download_file(file_id, file_name)
                if not temp_file:
                    continue

                # Process the CV
                structured = await self.cv_processor.process_resume(str(temp_file))
                
                # Import to database
                candidate_id = self.db.import_resume_data(structured)
                
                # Mark as processed
                self.mark_file_as_processed(file_id, file_name, candidate_id)
                
                results.append({
                    "file_id": file_id,
                    "file_name": file_name,
                    "candidate_id": candidate_id,
                    "data": structured,
                    "temp_path": temp_file
                })
                
                logger.info(f"✅ Successfully processed {file_name} -> Candidate ID {candidate_id}")
                
            except Exception as e:
                logger.error(f"❌ Error processing {file_name}: {e}")
                continue
            
            finally:
                # Clean up temporary file
                if temp_file and temp_file.exists():
                    try:
                        temp_file.unlink()
                    except Exception:
                        pass

        return results

    def get_drive_stats(self) -> Dict:
        """Get statistics about Google Drive processing."""
        try:
            total_processed = self.db.conn.execute(
                "SELECT COUNT(*) FROM processed_drive_files"
            ).fetchone()[0]
            
            # Use DuckDB's date_diff function instead of datetime('now', '-7 days')
            recent_processed = self.db.conn.execute(
                """SELECT COUNT(*) FROM processed_drive_files 
                WHERE date_diff('day', processed_at, current_timestamp) <= 7"""
            ).fetchone()[0]
            
            return {
                "total_processed_files": total_processed,
                "files_processed_last_week": recent_processed,
                "configured": bool(self.api_key and self.folder_id)
            }
            
        except Exception as e:
            logger.error(f"Error getting drive stats: {e}")
            return {
                "total_processed_files": 0,
                "files_processed_last_week": 0,
                "configured": bool(self.api_key and self.folder_id)
            }

