import os
import re
import glob
import time
from google import genai
from dotenv import load_dotenv
from pdf_generator import generate_pdf
from utils import sanitize_filename

load_dotenv()

def clean_vtt_text(vtt_file):
    """
    Reads a VTT/SRT file and strips timestamps, HTML tags, and deduplicates rolling captions
    to return a clean spoken text string.
    """
    try:
        with open(vtt_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 1. Remove WEBVTT header and metadata
        content = re.sub(r'WEBVTT[\s\S]*?\n\n', '', content)
        # 2. Remove timestamps (e.g. 00:00:00.000 --> 00:00:02.000)
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
        return raw_text
    except Exception as e:
        print(f"Error reading {vtt_file}: {e}")
        return None

def process_transcripts(input_dir="Raw_Transcripts", output_dir="Final_PDFs"):
    """
    Loops through the Raw_Transcripts folder, cleans text, calls Gemini API, and saves PDFs.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        print("❌ Error: GEMINI_API_KEY is missing. Please update your .env file.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    client = genai.Client(api_key=api_key)
    
    # Locate all subtitle files
    vtt_files = sorted(glob.glob(os.path.join(input_dir, "*.vtt")) + glob.glob(os.path.join(input_dir, "*.srt")))
    
    if not vtt_files:
        print(f"⚠️ No subtitle files found in '{input_dir}'. Did you run Stage 1?")
        return

    print(f"🚀 Found {len(vtt_files)} transcript files. Beginning processing...")

    for i, vtt_file in enumerate(vtt_files):
        # Extract title from filename (e.g., "001_Video_Title.en.vtt" -> "001_Video_Title")
        base_name = os.path.basename(vtt_file)
        # Strip extension and language codes
        video_title = base_name.replace(".en.vtt", "").replace(".hi.vtt", "").replace(".en-IN.vtt", "").replace(".vtt", "").replace(".srt", "")
        safe_title = sanitize_filename(video_title)
        
        print(f"\n[{i+1}/{len(vtt_files)}] Processing: {video_title}")
        
        raw_text = clean_vtt_text(vtt_file)
        if not raw_text or len(raw_text.strip()) < 10:
            print(f"  ⏭️ Skipping: No readable text extracted.")
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
            
            # Save Markdown locally temporarily or permanently (optional)
            md_path = os.path.join(output_dir, f"{safe_title}.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_text)

            print(f"  📄 Converting Markdown to PDF...")
            pdf_path = os.path.join(output_dir, f"{safe_title}.pdf")
            css_path = os.path.join(os.path.dirname(__file__), "style.css")
            
            # Using WeasyPrint from pdf_generator
            generate_pdf(md_text, pdf_path, css_path=css_path)
            
            print(f"  ✅ Saved: {pdf_path}")

        except Exception as e:
            print(f"  ❌ Failed to process {video_title}: {e}")

        # Rate Limit Control
        if i < len(vtt_files) - 1:
            print(f"  ⏳ Waiting 5 seconds to prevent Google API rate limits...")
            time.sleep(5.0)

    print(f"\n🎉 Stage 2 Complete! All PDFs saved in '{output_dir}'.")

if __name__ == "__main__":
    process_transcripts()
