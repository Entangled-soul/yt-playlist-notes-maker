from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp
from utils import sanitize_filename

def get_playlist_metadata(playlist_url, cookies_file=None):
    """Fetches ONLY the playlist metadata (video IDs and titles). Extremely fast."""
    ydl_opts = {
        'extract_flat': True, # Use True instead of 'in_playlist' to bypass 100 limit sometimes
        'quiet': True
    }
    if cookies_file:
        ydl_opts['cookiefile'] = cookies_file
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
        playlist_title = info.get('title', 'Unknown_Playlist')
        
        videos = []
        for entry in info['entries']:
            videos.append({
                'video_id': entry['id'],
                'original_title': entry['title'],
                'safe_title': sanitize_filename(entry['title'])
            })
            
        return playlist_title, videos

def fetch_transcript(video_id, cookies_file=None):
    """Fetches the transcript for a single video. Returns (raw_text, error_message)."""
    try:
        if cookies_file:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id, cookies=cookies_file)
        else:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        transcript = None
        try:
            transcript = transcript_list.find_transcript(['en', 'en-IN', 'en-US', 'en-GB', 'hi'])
        except:
            for t in transcript_list:
                transcript = t
                break
        
        if not transcript:
            return None, "No transcripts found for this video."
            
        # Ensure transcript is a valid object before accessing attributes
        if hasattr(transcript, 'language_code') and not transcript.language_code.startswith('en') and getattr(transcript, 'is_translatable', False):
            try:
                transcript = transcript.translate('en')
            except:
                pass
                
        transcript_data = transcript.fetch()
        raw_text = " ".join([t['text'] for t in transcript_data])
        return raw_text, None
        
    except Exception as e:
        if "Too Many Requests" in str(e):
            return None, "Rate limited by YouTube."
        else:
            return None, f"{type(e).__name__}"
