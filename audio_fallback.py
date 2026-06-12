import os
import tempfile
import yt_dlp
from google import genai
import time

def download_audio(video_id, cookies_file=None):
    """Downloads the smallest audio stream to a temporary file."""
    temp_dir = tempfile.gettempdir()
    output_template = os.path.join(temp_dir, f"{video_id}_audio.%(ext)s")
    
    ydl_opts = {
        'format': 'worstaudio/bestaudio', # Smallest file size
        'outtmpl': output_template,
        'quiet': True,
        'extract_audio': True
    }
    
    if cookies_file:
        ydl_opts['cookiefile'] = cookies_file
        
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f'https://www.youtube.com/watch?v={video_id}'])
            
        # The exact filename might end in .m4a, .webm, etc.
        # Find the downloaded file
        for file in os.listdir(temp_dir):
            if file.startswith(f"{video_id}_audio."):
                return os.path.join(temp_dir, file)
                
        return None
    except Exception as e:
        print(f"Audio download failed: {e}")
        return None

def enrich_from_audio(audio_path, safe_title, md_output_dir):
    """Uploads the audio to Gemini and generates structured Markdown notes."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        raise ValueError("GEMINI_API_KEY is not set or is invalid.")
        
    client = genai.Client(api_key=api_key)
    
    # 1. Upload audio to Gemini
    uploaded_file = client.files.upload(file=audio_path)
    
    # 2. Wait for processing (if necessary)
    # google-genai SDK handles states
    while True:
        file_info = client.files.get(name=uploaded_file.name)
        state_str = str(file_info.state).upper()
        if "PROCESSING" in state_str:
            time.sleep(2)
        elif "FAILED" in state_str:
            raise Exception("Audio file processing failed on Gemini servers.")
        else:
            break
    
    prompt = f"""
    You are an expert Data Science and Machine Learning tutor. Listen to this audio lecture and convert it into comprehensive, structured Markdown notes. 
    Keep the summary highly detailed, around 100 to 200 words per minute of the video, or whatever is standard for excellent technical notes.
    
    Requirements:
    1. Organize the content with clear headings (H1, H2, H3).
    2. Summarize the core concepts deeply.
    3. Provide 'Extra Build-Up': Expand on the ML/AI concepts, mathematical intuition, and practical applications mentioned in the audio. Add relevant Python code snippets (e.g., NumPy, Pandas, Scikit-learn, TensorFlow) where they illustrate the concept.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[uploaded_file, prompt]
        )
        enriched_markdown = response.text
        
        # Save the markdown file
        file_path = os.path.join(md_output_dir, f"{safe_title}.md")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(enriched_markdown)
            
        return enriched_markdown
        
    finally:
        # ALWAYS clean up the file from Gemini to save user quota
        try:
            client.files.delete(name=uploaded_file.name)
        except:
            pass
