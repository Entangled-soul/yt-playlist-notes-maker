import streamlit as st
import os
import shutil
import tempfile
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

# API Key Setup
st.markdown("### Step 1: Set your Gemini API Key")
st.markdown("Don't have one? [Click here to get your API key from Google AI Studio](https://aistudio.google.com/app/apikey)")
api_key = st.text_input("Gemini API Key:", value=os.environ.get("GEMINI_API_KEY", ""), type="password")
if api_key:
    os.environ["GEMINI_API_KEY"] = api_key

st.markdown("### Step 2: Enter Playlist URL")
playlist_url = st.text_input("YouTube Playlist URL:", placeholder="https://www.youtube.com/playlist?list=...")

st.markdown("### Step 3: Rate-limit Bypass (Optional)")
st.markdown("YouTube blocks rapid scraping. If you are getting rate-limited, upload your browser's **cookies.txt** file to completely bypass the block. You can get this using the [Get cookies.txt LOCALLY](https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc) Chrome extension.")
cookie_file = st.file_uploader("Upload cookies.txt", type=["txt"])

delay = st.slider("Delay between videos (seconds):", min_value=1.0, max_value=15.0, value=3.0, step=0.5, help="Slows down extraction to prevent YouTube from blocking your IP.")

st.markdown("---")

if st.button("🚀 Process Playlist", type="primary", use_container_width=True):
    if not playlist_url:
        st.warning("Please enter a playlist URL.")
        st.stop()
    elif not os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY") == "your_gemini_api_key_here":
        st.error("Please provide a valid Gemini API Key above.")
        st.stop()
    else:
        # Save cookies to temp file
        cookies_path = None
        if cookie_file:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="wb")
            tmp.write(cookie_file.read())
            tmp.close()
            cookies_path = tmp.name
            st.success("✅ Cookies loaded securely.")

        with st.spinner("Fetching playlist metadata (this may take a moment)..."):
            try:
                playlist_title, videos = get_playlist_data(playlist_url, delay=delay, cookies_file=cookies_path)
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
                status_text.info(f"⏳ **[Video {i+1}/{len(videos)}]** Extracting transcript for: *{video_title}*...")
                
                if not video.get('raw_text'):
                    failed_videos.append(video)
                    progress_bar.progress((i + 1) / len(videos))
                    continue
                    
                try:
                    # 1. Enrich Text
                    status_text.info(f"🧠 **[Video {i+1}/{len(videos)}]** AI is generating notes for: *{video_title}* (this will take 15-30 seconds)...")
                    enriched_md = enrich_notes(video['raw_text'], video['safe_title'], md_dir)
                    
                    # 2. Convert to PDF
                    status_text.info(f"📄 **[Video {i+1}/{len(videos)}]** Converting notes to PDF...")
                    pdf_path = os.path.join(pdf_dir, f"{video['safe_title']}.pdf")
                    css_path = os.path.join(os.path.dirname(__file__), "style.css")
                    generate_pdf(enriched_md, pdf_path, css_path=css_path)
                    
                    processed.append(video)
                    
                    # Display the successful parsing for the user
                    with st.expander(f"✅ {video_title} - Processed"):
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
                    failed_videos.append({**video, "error": str(exc), "error_type": "enrichment_error"})

                progress_bar.progress((i + 1) / len(videos))
                
            # Cleanup temp cookie file
            if cookies_path and os.path.exists(cookies_path):
                os.unlink(cookies_path)

            status_text.success(f"Done! — ✅ {len(processed)} processed, ⚠️ {len(failed_videos)} skipped.")
            if len(processed) > 0:
                st.balloons()
            
            # Error summary
            if failed_videos:
                with st.expander(f"⚠️ {len(failed_videos)} video(s) could not be processed"):
                    for v in failed_videos:
                        st.markdown(f"**Skipped — {v.get('original_title', 'Unknown')}**")
                        st.caption(f"Reason: {v.get('error', 'No transcript found or rate limited.')}")
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
