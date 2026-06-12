import yt_dlp
from utils import sanitize_filename

def get_playlist_metadata(playlist_url):
    """Fetches ONLY the playlist metadata (video IDs and titles). Extremely fast."""
    ydl_opts = {
        'extract_flat': True,
        'quiet': True
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
        playlist_title = info.get('title', 'Unknown_Playlist')
        
        videos = []
        for entry in info['entries']:
            if not entry:
                continue
                
            video_url = f"https://www.youtube.com/watch?v={entry['id']}"
            original_title = entry['title']
            safe_title = sanitize_filename(original_title)
            
            videos.append({
                'video_url': video_url,
                'original_title': original_title,
                'safe_title': safe_title,
                'video_id': entry['id']
            })
            
        return playlist_title, videos
