import os
import re

def sanitize_filename(title):
    # Removes characters that are illegal in Windows file names: \ / * ? " < > | :
    return re.sub(r'[\\/*?:"<>|]', "", title).strip()

def setup_directories(base_path="d:/Notes/youtube_playlist_notes", playlist_name="Default_Playlist"):
    safe_playlist_name = sanitize_filename(playlist_name)
    
    # Define paths
    playlist_dir = os.path.join(base_path, safe_playlist_name)
    md_dir = os.path.join(playlist_dir, "NotebookLM_Sources")
    pdf_dir = os.path.join(playlist_dir, "PDF_Output")
    
    # Auto-create folders if they don't exist
    os.makedirs(md_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)
    
    return md_dir, pdf_dir
