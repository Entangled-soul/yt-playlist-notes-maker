"""
Stage 3: Processor
==================
processor.py

This script cleans raw transcript files, dynamically adjusts its AI persona based on 
the playlist domain, generates exhaustive textbook chapters via the Gemini API, 
and ultimately compiles them into a beautiful, production-ready PDF using WeasyPrint.
"""
import os
import re
import glob
import time
from google import genai
from dotenv import load_dotenv
import weasyprint
import markdown

def sanitize_filename(name):
    # Replace spaces with underscores and strip invalid Windows characters
    name = str(name).replace(' ', '_')
    return re.sub(r'[<>:"/\\|?*]', '', name).strip()

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
        # 2. Remove timestamps
        content = re.sub(r'\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}.*?\n', '', content)
        # 3. Remove HTML/XML tags
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

def stitch_and_compile(chapters_dir="Raw_Chapters", output_pdf="Master_ML_Textbook.pdf"):
    """
    Reads all Markdown chapters in numerical order, concatenates them with page breaks,
    and uses WeasyPrint to output the Master PDF Textbook.
    """
    print("\n📚 Stitching chapters into Master Textbook...")
    md_files = sorted(glob.glob(os.path.join(chapters_dir, "*.md")))
    
    if not md_files:
        print("❌ Error: No markdown chapters found to stitch.")
        return
        
    master_markdown = f"# Master Machine Learning Textbook\n\n<div class='page-break'></div>\n\n"
    
    for i, md_file in enumerate(md_files):
        with open(md_file, "r", encoding="utf-8") as f:
            master_markdown += f.read()
            
        # Inject HTML page break after every chapter except the very last one
        if i < len(md_files) - 1:
            master_markdown += "\n\n<div class='page-break'></div>\n\n"
            
    print(f"📄 Compiling Master PDF (this may take a minute for {len(md_files)} chapters)...")
    
    # Convert combined markdown to HTML
    html_content = markdown.markdown(master_markdown, extensions=['fenced_code', 'tables', 'extra'])

    # Clean multi-line string (No backslashes!)
    styled_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; }}
            h1, h2, h3 {{ color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 5px; }}
            code {{ background-color: #f4f4f4; padding: 2px 4px; border-radius: 4px; font-family: monospace; }}
            pre {{ background-color: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
            pre code {{ background-color: transparent; padding: 0; }}
            blockquote {{ border-left: 4px solid #ccc; margin-left: 0; padding-left: 15px; color: #666; }}
            .page-break {{ page-break-after: always; }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    # Generate the PDF
    weasyprint.HTML(string=styled_html).write_pdf(output_pdf)
    print(f"🎉 SUCCESS: '{output_pdf}' is ready!")

def process_transcripts(input_dir="Raw_Transcripts", chapters_dir="Raw_Chapters"):
    """
    Loops through the Raw_Transcripts folder, calls Gemini API to extract exhaustive textbook chapters,
    saves them to Raw_Chapters/, and then compiles the final textbook.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        print("❌ Error: GEMINI_API_KEY is missing. Please create a .env file and add your key.")
        return

    print("\n🧹 Initializing Workspace (Auto-Wipe)...")
    if os.path.exists(chapters_dir):
        import shutil
        shutil.rmtree(chapters_dir, ignore_errors=True)
        print(f"   Deleted old '{chapters_dir}' directory to prevent ghost chapters.")
    os.makedirs(chapters_dir, exist_ok=True)
    
    # Phase 1: Reading domain context for AI persona
    playlist_name_file = os.path.join(input_dir, 'playlist_name.txt')
    if os.path.exists(playlist_name_file):
        with open(playlist_name_file, 'r', encoding='utf-8') as f:
            raw_title = f.read().strip()
            course_domain = raw_title
            sanitized_title = re.sub(r'[<>:"/\\|?*]', '', raw_title).strip()
            if not sanitized_title:
                sanitized_title = "Master_Textbook"
    else:
        course_domain = "Advanced University-Level Subject"
        sanitized_title = "Master_Textbook"

    output_pdf = f"{sanitized_title}.pdf"
    if os.path.exists(output_pdf):
        os.remove(output_pdf)
        print(f"   Deleted old '{output_pdf}'.")

    client = genai.Client(api_key=api_key)
    
    # Locate all subtitle files
    vtt_files = sorted(glob.glob(os.path.join(input_dir, "*.vtt")) + glob.glob(os.path.join(input_dir, "*.srt")))
    
    if not vtt_files:
        print(f"⚠️ No subtitle files found in '{input_dir}'. Did you run Stage 1 (downloader.py)?")
        return

    print(f"🚀 Found {len(vtt_files)} transcript files. Beginning processing (Stage 2)...")

    for i, vtt_file in enumerate(vtt_files):
        # Extract title from filename (e.g., "001_Video_Title.en.vtt")
        base_name = os.path.basename(vtt_file)
        video_title = base_name.replace(".en.vtt", "").replace(".hi.vtt", "").replace(".hi-orig.vtt", "").replace(".en-IN.vtt", "").replace(".vtt", "").replace(".srt", "")
        safe_title = sanitize_filename(video_title)
        
        md_path = os.path.join(chapters_dir, f"{safe_title}.md")
        
        # Skip if already processed
        if os.path.exists(md_path):
            print(f"[{i+1}/{len(vtt_files)}] ⏭️ Skipping '{video_title}' (Already generated)")
            continue
            
        print(f"\n[{i+1}/{len(vtt_files)}] Processing: {video_title}")
        
        raw_text = clean_vtt_text(vtt_file)
        if not raw_text or len(raw_text.strip()) < 10:
            print(f"  ⏭️ Skipping: No readable text extracted.")
            continue

        # Phase 2: Dynamic System Prompt Injection
        system_instruction = f"""
        You are a world-class professor, researcher, and textbook author specializing in: {course_domain}. 
        I will provide you with a raw video transcript. Do NOT just summarize the transcript. You must use the transcript solely as a topical syllabus, and write an exhaustive, university-level textbook chapter that expands on these topics using your own deep external knowledge.

        Expand the length, depth, and density of the content by at least 3x. For every concept mentioned in the transcript, you MUST inject the following external knowledge based on the domain:
        1. Core Theory & Mechanics: Explain the fundamental underlying principles, formulas, or historical context that the video skipped.
        2. Domain-Specific Deep Dives: 
           - If this is a programming/tech topic, provide production-grade code implementations (e.g., Python, Rust, C++) with exhaustive comments and architecture logic.
           - If this is a math/science topic, provide step-by-step derivations and physical world examples.
           - If this is humanities/business, provide real-world case studies, financial impacts, and historical consequences.
        3. Edge Cases & Limitations: Detail the exact scenarios where these methods or concepts fail in the real world and how experts solve them.

        Output strictly in beautiful Markdown. Do not hold back on length. Be incredibly dense, detailed, and comprehensive.
        """
        
        prompt = f"{system_instruction}\n\nTranscript:\n{raw_text}"

        try:
            print(f"  🧠 Generating exhaustive chapter (gemini-3.1-flash-lite)...")
            response = client.models.generate_content(
                model='gemini-3.1-flash-lite',
                contents=[prompt]
            )
            
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(response.text)

            print(f"  ✅ Saved Chapter: {md_path}")

        except Exception as e:
            print(f"  ❌ Failed to process {video_title}: {e}")

        # Rate Limit Control
        if i < len(vtt_files) - 1:
            print(f"  ⏳ Waiting 5 seconds to prevent Google API rate limits...")
            time.sleep(5.0)

    print(f"\n✅ All chapters generated successfully in '{chapters_dir}'.")
    
    # Finally, trigger the Master PDF Compiler
    stitch_and_compile(chapters_dir, output_pdf=output_pdf)

if __name__ == "__main__":
    process_transcripts()
