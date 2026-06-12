import pdfkit
import markdown
import os

def generate_pdf(markdown_content, output_pdf_path, css_path="style.css"):
    # Convert markdown to HTML using extensions for tables and code highlighting
    html_content = markdown.markdown(
        markdown_content, 
        extensions=['fenced_code', 'tables', 'nl2br']
    )
    
    # Wrap in a full HTML document with a reference to the stylesheet or embedded CSS
    # For local CSS, we can read the file and embed it so pdfkit picks it up easily
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
    
    # Generate PDF from the HTML string
    # Ensure wkhtmltopdf is in your system PATH
    options = {
        'page-size': 'A4',
        'margin-top': '1in',
        'margin-right': '1in',
        'margin-bottom': '1in',
        'margin-left': '1in',
        'encoding': "UTF-8",
        'no-outline': None,
        'enable-local-file-access': None
    }
    
    pdfkit.from_string(full_html, output_pdf_path, options=options)
    return output_pdf_path
