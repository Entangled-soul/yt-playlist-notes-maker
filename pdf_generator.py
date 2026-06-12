import markdown
import os
from weasyprint import HTML

def generate_pdf(markdown_content, output_pdf_path, css_path="style.css"):
    # Convert markdown to HTML using extensions for tables and code highlighting
    html_content = markdown.markdown(
        markdown_content, 
        extensions=['fenced_code', 'tables', 'nl2br']
    )
    
    # Wrap in a full HTML document with a reference to the stylesheet or embedded CSS
    css_content = ""
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
            
    full_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <style>
            {css_content}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Generate PDF from the HTML string using WeasyPrint
    HTML(string=full_html).write_pdf(output_pdf_path)
    return output_pdf_path
