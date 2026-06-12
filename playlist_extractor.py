from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp
from utils import sanitize_filename
import time
import random

def get_playlist_data(playlist_url):
    ydl_opts = {
        'extract_flat': 'in_playlist',
        'quiet': True
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
        playlist_title = info.get('title', 'Unknown_Playlist')
        
        videos = []
        for entry in info['entries']:
            video_id = entry['id']
            original_title = entry['title']
            safe_title = sanitize_filename(original_title)
            
            try:
                # Fetch robust transcript
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                
                transcript = None
                try:
                    transcript = transcript_list.find_transcript(['en', 'en-IN', 'en-US', 'en-GB', 'hi'])
                except:
                    for t in transcript_list:
                        transcript = t
                        break
                
                if not transcript:
                    raise Exception("No transcripts found")
                    
                if not transcript.language_code.startswith('en') and transcript.is_translatable:
                    try:
                        transcript = transcript.translate('en')
                    except:
                        pass
                        
                transcript_data = transcript.fetch()
                raw_text = " ".join([t['text'] for t in transcript_data])
                
                videos.append({
                    'video_id': video_id,
                    'original_title': original_title,
                    'safe_title': safe_title,
                    'raw_text': raw_text
                })
            except Exception as e:
                if "Too Many Requests" in str(e):
                    print(f"Skipping '{original_title}': Rate limited by YouTube. You may need to wait before trying again.")
                    time.sleep(10)
                else:
                    print(f"Skipping '{original_title}': No transcript available ({type(e).__name__}).")
            
            # Add a random delay to prevent rate limiting
            time.sleep(random.uniform(2.0, 4.0))
                
        return playlist_title, videos
