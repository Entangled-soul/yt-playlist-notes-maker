from google import genai
import os

def enrich_notes(raw_text, safe_title, md_output_dir):
    # Initialize API inside the function to ensure the env var is loaded first by app.py
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        raise ValueError("GEMINI_API_KEY is not set or is invalid.")
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    You are an expert Data Science and Machine Learning tutor. Take the following raw, unformatted transcript from a video lecture and convert it into comprehensive, structured Markdown notes. 
    
    Requirements:
    1. Clean up spoken errors and conversational filler.
    2. Organize the content with clear headings (H1, H2, H3).
    3. Summarize the core concepts.
    4. Provide 'Extra Build-Up': Expand on the ML/AI concepts, mathematical intuition, and practical applications mentioned in the text. Add relevant Python code snippets (e.g., NumPy, Pandas, Scikit-learn, TensorFlow) where they illustrate the concept.
    
    Raw Transcript:
    {raw_text}
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt
    )
    enriched_markdown = response.text
    
    # Save the markdown file using the sanitized original title
    file_path = os.path.join(md_output_dir, f"{safe_title}.md")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(enriched_markdown)
        
    return enriched_markdown
