import os
import re
import glob
import tempfile
import yt_dlp

def download_and_parse_subtitle(video_url):
    """
    Downloads the auto-generated subtitle using yt-dlp, parses the raw text out of the VTT file,
    and returns (raw_text, vtt_file_path, error_message).
    """
    temp_dir = tempfile.gettempdir()
    video_id = video_url.split("v=")[-1].split("&")[0]
    
    # Clean up any leftover vtt files for this video ID
    for f in glob.glob(os.path.join(temp_dir, f"{video_id}_sub*")):
        try: os.remove(f)
        except: pass
        
    outtmpl = os.path.join(temp_dir, f"{video_id}_sub.%(ext)s")
    
    ydl_opts = {
        'skip_download': True,
        'writeautomaticsub': True,
        'writesubtitles': True,
        'subtitleslangs': ['en', 'hi', 'en-IN'],
        'subtitlesformat': 'vtt',
        'outtmpl': outtmpl,
        'quiet': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
    except Exception as e:
        print(f"yt-dlp subtitle download failed: {e}")
        # We continue to check if a file was downloaded anyway
        
    # Find the downloaded vtt file
    sub_files = glob.glob(os.path.join(temp_dir, f"{video_id}_sub*"))
    if not sub_files:
        return None, None, "No auto-generated subtitles could be downloaded for this video."
        
    vtt_file = sub_files[0]
    
    try:
        with open(vtt_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Parse the VTT file
        # 1. Remove WEBVTT header and metadata
        content = re.sub(r'WEBVTT[\s\S]*?\n\n', '', content)
        # 2. Remove timestamps
        content = re.sub(r'\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}.*?\n', '', content)
        # 3. Remove HTML/XML tags (like <c>)
        content = re.sub(r'<[^>]+>', '', content)
        # 4. Extract non-empty lines, ignoring styling meta
        lines = [line.strip() for line in content.split('\n') if line.strip() and "align:" not in line and "position:" not in line]
        
        # 5. Deduplicate rolling captions
        clean_lines = []
        for line in lines:
            if not clean_lines or clean_lines[-1] != line:
                clean_lines.append(line)
                
        raw_text = " ".join(clean_lines)
        
        if not raw_text.strip():
            return None, vtt_file, "Subtitles were downloaded but contained no readable text."
            
        return raw_text, vtt_file, None
        
    except Exception as e:
        return None, vtt_file, f"Failed to parse subtitle file: {str(e)}"
