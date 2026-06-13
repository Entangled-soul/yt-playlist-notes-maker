import os
import argparse
import time
from dotenv import load_dotenv
from google import genai
from playlist_extractor import get_playlist_metadata
from subtitle_parser import download_and_parse_subtitle
from pdf_generator import generate_pdf
from utils import setup_directories

load_dotenv()

def process_playlist(playlist_url):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        print("❌ Error: GEMINI_API_KEY is missing. Please update your .env file.")
        return

    client = genai.Client(api_key=api_key)

    print(f"🚀 Fetching metadata for playlist: {playlist_url}")
    try:
        playlist_title, videos = get_playlist_metadata(playlist_url)
    except Exception as e:
        print(f"❌ Failed to fetch playlist metadata: {e}")
        return

    print(f"✅ Found {len(videos)} videos in playlist '{playlist_title}'.")
    md_dir, pdf_dir = setup_directories(playlist_name=playlist_title)
    # Actually, let's use global folders instead of playlist-specific folders
    # so the app.py dashboard can easily find them without knowing the playlist name.
    # The user asked to just put them in "Final_PDFs".
    pdf_dir = "Final_PDFs"
    md_dir = "Markdown_Notes"
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(md_dir, exist_ok=True)

    for i, video in enumerate(videos):
        video_title = video['original_title']
        video_url = video['video_url']
        safe_title = video['safe_title']
        
        print(f"\n[{i+1}/{len(videos)}] Processing: {video_title}")
        
        raw_text, vtt_file, err = download_and_parse_subtitle(video_url)
        if not raw_text:
            print(f"  ⏭️ Skipping: {err or 'Failed to extract subtitle text'}")
            continue

        prompt = f"""
        You are an expert Data Science and Machine Learning tutor. Read this video transcript and convert it into comprehensive, structured Markdown notes. 
        
        Transcript:
        {raw_text}
        
        Requirements:
        Create comprehensive, well-structured Markdown textbook notes. Focus heavily on providing plain-English intuition and clean Python code implementations. Do not output raw math derivations or complex calculus.
        """

        try:
            print(f"  🧠 Asking Gemini to write notes (gemini-3.1-flash-lite)...")
            response = client.models.generate_content(
                model='gemini-3.1-flash-lite',
                contents=[prompt]
            )
            
            md_text = response.text
            
            md_path = os.path.join(md_dir, f"{safe_title}.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_text)

            print(f"  📄 Converting Markdown to PDF...")
            pdf_path = os.path.join(pdf_dir, f"{safe_title}.pdf")
            css_path = os.path.join(os.path.dirname(__file__), "style.css")
            
            generate_pdf(md_text, pdf_path, css_path=css_path)
            print(f"  ✅ Saved: {pdf_path}")

        except Exception as e:
            print(f"  ❌ Failed to process {video_title}: {e}")

        # Cleanup the VTT file
        if vtt_file and os.path.exists(vtt_file):
            try:
                os.remove(vtt_file)
            except:
                pass

        # Rate Limit Control
        if i < len(videos) - 1:
            print(f"  ⏳ Waiting 5 seconds to prevent Google API rate limits...")
            time.sleep(5.0)

    print(f"\n🎉 Batch Processing Complete! All PDFs saved in '{pdf_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone CLI Batch Processor")
    parser.add_argument("playlist_url", help="The YouTube playlist URL to process.")
    args = parser.parse_args()

    process_playlist(args.playlist_url)
