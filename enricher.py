import os
import time
from google import genai
from google.genai import types

def enrich_notes(raw_text, safe_title, md_output_dir):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        raise ValueError("GEMINI_API_KEY is not set or is invalid.")
        
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    You are an expert Data Science and Machine Learning tutor. Read this video transcript and convert it into comprehensive, structured Markdown notes. 
    
    Transcript:
    {raw_text}
    
    Requirements:
    1. Organize the content with clear headings (H1, H2, H3).
    2. Summarize the core concepts covered deeply.
    3. Provide 'Extra Build-Up': Expand on the ML/AI concepts and practical applications. Add relevant Python code snippets where they illustrate the concept.
    4. NO RAW MATH OR COMPLEX DERIVATIONS: Keep explanations practical and intuitive. DO NOT output complex mathematical formulas or raw LaTeX (like $\\frac{{\\partial}}{{\\partial W}}$). Focus on the high-level intuition and code instead.
    """
    
    max_retries = 3
    last_error = None
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=[
                    prompt
                ]
            )
            
            enriched_markdown = response.text
            
            file_path = os.path.join(md_output_dir, f"{safe_title}.md")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(enriched_markdown)
                
            return enriched_markdown
            
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            if "429" in error_str or "quota" in error_str or "exhausted" in error_str:
                if attempt < max_retries - 1:
                    time.sleep(65) # Wait 65 seconds if rate limited
                    continue
            raise e
            
    raise last_error
