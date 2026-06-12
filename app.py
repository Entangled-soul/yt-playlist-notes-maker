import streamlit as st
import os
import shutil
from playlist_extractor import get_playlist_data
from enricher import enrich_notes
from utils import setup_directories
from pdf_generator import generate_pdf
from dotenv import load_dotenv

# Load env variables if they exist in a .env file
load_dotenv()

st.set_page_config(page_title="YouTube Notes App", layout="wide", page_icon="📝")

st.title("🎬 YouTube Playlist to Premium Notes")
st.markdown("Automate the extraction and enrichment of video transcripts into beautiful PDFs and Markdown using the Gemini API.")

# Sidebar - Settings or status
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Gemini API Key", value=os.environ.get("GEMINI_API_KEY", ""), type="password")
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
        
    st.markdown("---")
    st.markdown("""
    **Pipeline Steps:**
    1. Extract transcripts
    2. Enrich via Gemini 2.5 Flash
    3. Save to Markdown & PDF
    """)

playlist_url = st.text_input("Enter YouTube Playlist URL:", placeholder="https://www.youtube.com/playlist?list=...")

if st.button("🚀 Process Playlist", type="primary"):
    if not playlist_url:
        st.warning("Please enter a playlist URL.")
    elif not os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY") == "your_gemini_api_key_here":
        st.error("Please provide a valid Gemini API Key in the sidebar or .env file.")
    else:
        with st.spinner("Extracting metadata and transcripts (this may take a moment)..."):
            try:
                playlist_title, videos = get_playlist_data(playlist_url)
                
                # Automatically create the required folders in D:/Notes/...
                # Note: utils expects "Default_Playlist" if title is missing.
                md_dir, pdf_dir = setup_directories(playlist_name=playlist_title)
                st.success(f"📂 Directories created for: **{playlist_title}**")
                
                # Create a placeholder for live updates
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, video in enumerate(videos):
                    status_text.text(f"Processing ({i+1}/{len(videos)}): {video['original_title']}")
                    
                    if not video.get('raw_text'):
                        st.warning(f"⚠️ Skipped '{video['original_title']}': No transcript found.")
                        continue
                        
                    # 1. Enrich Text
                    enriched_md = enrich_notes(video['raw_text'], video['safe_title'], md_dir)
                    
                    # 2. Convert to PDF (using original title)
                    pdf_path = os.path.join(pdf_dir, f"{video['safe_title']}.pdf")
                    # Make sure the current working directory contains style.css or provide absolute path
                    css_path = os.path.join(os.path.dirname(__file__), "style.css")
                    generate_pdf(enriched_md, pdf_path, css_path=css_path)
                    
                    # Display the successful parsing for the user
                    with st.expander(f"✅ {video['original_title']} - Processed"):
                        st.markdown(enriched_md)
                        with open(pdf_path, "rb") as pdf_file:
                            st.download_button(
                                label="📥 Download PDF",
                                data=pdf_file,
                                file_name=f"{video['safe_title']}.pdf",
                                mime="application/pdf",
                                key=f"dl_{video['video_id']}"
                            )
                            
                    progress_bar.progress((i + 1) / len(videos))
                    
                status_text.text("All videos processed!")
                st.balloons()
                
                # Zip the PDFs directory
                st.info("📦 Zipping your notes...")
                zip_path = shutil.make_archive(
                    base_name=pdf_dir,
                    format='zip',
                    root_dir=pdf_dir
                )
                
                st.success("All done! You can now download your complete notes.")
                with open(zip_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download All Notes (ZIP)",
                        data=f,
                        file_name=f"{playlist_title}_notes.zip",
                        mime="application/zip",
                        type="primary",
                        use_container_width=True
                    )
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
