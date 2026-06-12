import streamlit as st
import os
import shutil
import tempfile
import time
from playlist_extractor import get_playlist_metadata, fetch_transcript
from audio_fallback import download_audio, enrich_from_audio
from enricher import enrich_notes
from utils import setup_directories
from pdf_generator import generate_pdf
from dotenv import load_dotenv

# Load env variables if they exist in a .env file
load_dotenv()

st.set_page_config(page_title="YouTube Notes App", layout="wide", page_icon="📝")

st.title("🎬 YouTube Playlist to Premium Notes")
st.markdown("Automate the extraction and enrichment of video transcripts into beautiful PDFs and Markdown using the Gemini API.")

# Check Streamlit Secrets and os.environ
gemini_key = os.environ.get("GEMINI_API_KEY")
if not gemini_key and hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
    gemini_key = st.secrets["GEMINI_API_KEY"]
    os.environ["GEMINI_API_KEY"] = gemini_key

st.markdown("#### Step 1: Enter Playlist URL")
playlist_url = st.text_input("YouTube Playlist URL:", placeholder="https://www.youtube.com/playlist?list=...")
st.markdown("#### Step 2: Upload Cookies (Mandatory for Streamlit Cloud!)")
st.error("🚨 **CRITICAL FIX:** YouTube has blocked Streamlit Cloud's IP! You **MUST** upload your `cookies.txt` file here so YouTube thinks you are a real person, otherwise every video will fail and say '0 processed, 100 skipped'.")
cookie_file = st.file_uploader("Upload your browser cookies.txt file", type=["txt"])

st.caption("⏳ *Note: We automatically wait 5 seconds between successful videos to prevent YouTube from banning your cookies.*")

st.markdown("---")

if st.button("🚀 Process Playlist", type="primary", use_container_width=True):
    if not playlist_url:
        st.warning("Please enter a playlist URL.")
        st.stop()
    elif not os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY") == "your_gemini_api_key_here":
        st.error("Please add your GEMINI_API_KEY to your Streamlit Secrets.")
        st.stop()
    else:
        # Save cookies to temp file
        cookies_path = None
        if cookie_file:
            content = cookie_file.read().decode('utf-8', errors='ignore')
            
            if "youtube.com" not in content:
                st.error("❌ Invalid cookies.txt! You must go to **youtube.com** in your browser before clicking Export.")
                st.stop()
                
            # Fix Python's MozillaCookieJar bug with HttpOnly cookies
            content = content.replace("#HttpOnly_", "")
            
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8")
            tmp.write(content)
            tmp.close()
            cookies_path = tmp.name
            st.success("✅ Cookies parsed and loaded securely.")

        with st.spinner("Fetching playlist metadata (this may take a moment)..."):
            try:
                playlist_title, videos = get_playlist_metadata(playlist_url, cookies_file=cookies_path)
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
                
                # Fetch transcript
                raw_text, error = fetch_transcript(video['video_id'], cookies_file=cookies_path)
                
                if not raw_text:
                    status_text.warning(f"🎧 **[Video {i+1}/{len(videos)}]** Transcript missing! Falling back to Native Audio AI for: *{video_title}*...")
                    
                    audio_path = download_audio(video['video_id'], cookies_file=cookies_path)
                    if not audio_path:
                        video['error'] = error + " (And audio fallback also failed)."
                        failed_videos.append(video)
                        progress_bar.progress((i + 1) / len(videos))
                        continue
                        
                    try:
                        status_text.info(f"🧠 **[Video {i+1}/{len(videos)}]** Gemini AI is listening to the audio for: *{video_title}* (this will take a minute)...")
                        enriched_md = enrich_from_audio(audio_path, video['safe_title'], md_dir)
                        
                        # Cleanup local audio file immediately
                        try:
                            os.remove(audio_path)
                        except:
                            pass
                            
                        # Convert to PDF
                        status_text.info(f"📄 **[Video {i+1}/{len(videos)}]** Converting audio notes to PDF...")
                        pdf_path = os.path.join(pdf_dir, f"{video['safe_title']}.pdf")
                        css_path = os.path.join(os.path.dirname(__file__), "style.css")
                        generate_pdf(enriched_md, pdf_path, css_path=css_path)
                        
                        processed.append(video)
                        
                        with st.expander(f"✅ {video_title} - Processed via Audio AI"):
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
                        # Cleanup local audio file
                        try:
                            os.remove(audio_path)
                        except:
                            pass
                        failed_videos.append({**video, "error": f"Audio fallback failed: {str(exc)}", "error_type": "audio_fallback_error"})
                        
                    progress_bar.progress((i + 1) / len(videos))
                    continue
                    
                time.sleep(5.0)
                    
                try:
                    # 1. Enrich Text
                    status_text.info(f"🧠 **[Video {i+1}/{len(videos)}]** AI is generating notes for: *{video_title}* (this will take 15-30 seconds)...")
                    enriched_md = enrich_notes(raw_text, video['safe_title'], md_dir)
                    
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
