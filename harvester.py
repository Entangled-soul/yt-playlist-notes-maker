"""
Stage 1: Harvester
==================
harvester.py

This script extracts all metadata from the provided YouTube playlist URL. 
It intentionally bypasses the YouTube 100-video cap by manually chunking the 
API request and appending the results into a unified manifest file.
"""
import sys
import yt_dlp

def harvest_playlist(playlist_url):
    print("🔍 Harvesting video links from playlist via chunking...")
    all_entries = []
    current_start = 1
    
    # We will also grab the playlist title and save it so processor.py can still use it!
    # Writing to the root directory temporarily so the downloader's auto-wipe doesn't kill it.
    playlist_title = "Generated_Textbook"
    
    # Phase 1: Bypassing YouTube 100-video pagination
    while True:
        opts = {
            'extract_flat': True,
            'playliststart': current_start,
            'playlistend': current_start + 99,
            'cookiefile': 'cookies.txt',
            'quiet': True,
        }
        
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(playlist_url, download=False)
            
            if not info:
                break
                
            if current_start == 1:
                playlist_title = info.get('title', 'Generated_Textbook')
                with open('playlist_name.txt', 'w', encoding='utf-8') as f:
                    f.write(playlist_title)
                    
            entries = list(info.get('entries', []))
            
        if not entries:
            break
            
        valid_entries = [e for e in entries if e is not None and e.get('id')]
        all_entries.extend(valid_entries)
        current_start += 100

    # Phase 2: Manifest Generation
    manifest_path = "manifest_urls.txt"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in all_entries:
            # Comma-separated format: URL,Title
            url = f"https://www.youtube.com/watch?v={entry['id']}"
            # Replace commas in title to prevent breaking the CSV format
            title = str(entry.get('title', 'Unknown_Video')).replace(',', '')
            f.write(f"{url},{title}\n")
            
    print(f"✅ Harvested {len(all_entries)} video URLs to {manifest_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️ ERROR: No playlist URL provided. Usage: python main.py <PLAYLIST_URL>")
        sys.exit(1)
        
    playlist_url = sys.argv[1]
    harvest_playlist(playlist_url)
