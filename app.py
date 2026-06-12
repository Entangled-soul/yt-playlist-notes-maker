import streamlit as st
import os
import shutil
import time
from playlist_extractor import get_playlist_metadata
from enricher import enrich_notes
from utils import setup_directories
from pdf_generator import generate_pdf
from dotenv import load_dotenv

# Load env variables if they exist in a .env file
load_dotenv()

st.set_page_config(page_title="YouTube Notes App", layout="wide", page_icon="📝")

st.title("🎬 YouTube Playlist to Premium Notes")
st.markdown("Automate the extraction and enrichment of video lectures into beautiful PDFs and Markdown using the Gemini API's native Video Understanding capabilities.")

# Check Streamlit Secrets and os.environ
gemini_key = os.environ.get("GEMINI_API_KEY")
if not gemini_key and hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
    gemini_key = st.secrets["GEMINI_API_KEY"]
    os.environ["GEMINI_API_KEY"] = gemini_key

st.markdown("#### Step 1: Enter Playlist URL")
playlist_url = st.text_input("YouTube Playlist URL:", placeholder="https://www.youtube.com/playlist?list=...")

st.markdown("---")

if st.button("🚀 Process Playlist", type="primary", use_container_width=True):
    if not playlist_url:
        st.warning("Please enter a playlist URL.")
        st.stop()
    elif not os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY") == "your_gemini_api_key_here":
        st.error("Please add your GEMINI_API_KEY to your Streamlit Secrets.")
        st.stop()
    else:
        with st.spinner("Fetching playlist metadata (this may take a moment)..."):
            try:
                playlist_title, videos = get_playlist_metadata(playlist_url)
            except Exception as exc:
                st.error(f"❌ Could not fetch playlist: {exc}")
                st.stop()
                
            md_dir, pdf_dir = setup_directories(playlist_name=playlist_title)
            st.success(f"📂 Directories created for: **{playlist_title}**")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            processed, failed_videos = [], []
            
            for i, video in enumerate(videos):
                video_title = video['original_title']
                video_url = video['video_url']
                
                try:
                    status_text.info(f"🧠 **[Video {i+1}/{len(videos)}]** Gemini is natively watching: *{video_title}* (this may take 15-30 seconds)...")
                    enriched_md = enrich_notes(video_url, video['safe_title'], md_dir)
                    
                    # Convert to PDF
                    status_text.info(f"📄 **[Video {i+1}/{len(videos)}]** Converting notes to PDF...")
                    pdf_path = os.path.join(pdf_dir, f"{video['safe_title']}.pdf")
                    css_path = os.path.join(os.path.dirname(__file__), "style.css")
                    generate_pdf(enriched_md, pdf_path, css_path=css_path)
                    
                    processed.append(video)
                    
                    # Display the successful parsing for the user
                    with st.expander(f"✅ {video_title} - Processed natively via Gemini"):
                        st.markdown(enriched_md)
                        with open(pdf_path, "rb") as pdf_file:
                            st.download_button(
                                label="📥 Download PDF",
                                data=pdf_file,
                                file_name=f"{video['safe_title']}.pdf",
                                mime="application/pdf",
                                key=f"dl_{video['video_id']}"
                            )
                except Exception as exc:
                    failed_videos.append({**video, "error": f"Failed to process video natively: {str(exc)}", "error_type": "gemini_error"})

                progress_bar.progress((i + 1) / len(videos))
                
                # Rate limit delay for gemini-2.0-flash free tier
                if i < len(videos) - 1:
                    status_text.info(f"⏳ **[Video {i+1}/{len(videos)}]** Waiting 7 seconds to prevent Google API rate limits...")
                    time.sleep(7.0)
                

            status_text.success(f"Done! — ✅ {len(processed)} processed, ⚠️ {len(failed_videos)} skipped.")
            if len(processed) > 0:
                st.balloons()
            
            # Error summary
            if failed_videos:
                with st.expander(f"⚠️ {len(failed_videos)} video(s) could not be processed"):
                    for v in failed_videos:
                        st.markdown(f"**Skipped — {v.get('original_title', 'Unknown')}**")
                        st.caption(f"Reason: {v.get('error', 'Unknown Error')}")
                        st.divider()

            if len(processed) > 0:
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
