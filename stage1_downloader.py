import os
import argparse
import yt_dlp
from utils import sanitize_filename

def download_playlist_subtitles(playlist_url, output_dir="Raw_Transcripts"):
    """
    Downloads auto-generated subtitles for an entire YouTube playlist using yt-dlp.
    Forces pagination up to 200 videos to bypass the 100-video cap.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Use playlist_index to keep the files ordered
    outtmpl = os.path.join(output_dir, "%(playlist_index)s_%(title)s.%(ext)s")

    ydl_opts = {
        'skip_download': True,           # Don't download the video
        'writeautomaticsub': True,       # Download auto-generated subtitles
        'writesubtitles': True,          # Download manual subtitles if available
        'subtitleslangs': ['en', 'hi', 'en-IN'], # Prefer English, then Hindi
        'subtitlesformat': 'vtt',        # Save as VTT
        'outtmpl': outtmpl,              # Output format
        'playlistend': 200,              # Bypass 100 video limit
        'ignoreerrors': True,            # Continue on error
    }

    print(f"🚀 Starting mass subtitle download for playlist...")
    print(f"URL: {playlist_url}")
    print(f"Output Directory: {output_dir}\n")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([playlist_url])

    print(f"\n✅ Stage 1 Complete! All available subtitles downloaded to '{output_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Mass Subtitle Downloader")
    parser.add_argument("playlist_url", help="The YouTube playlist URL to download.")
    parser.add_argument("--out", default="Raw_Transcripts", help="Output directory (default: Raw_Transcripts)")
    args = parser.parse_args()

    download_playlist_subtitles(args.playlist_url, args.out)
