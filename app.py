import streamlit as st
import os
import glob
import shutil
import base64

st.set_page_config(page_title="YouTube Notes Library", layout="wide", page_icon="📚")

st.title("📚 Premium Notes Library")
st.markdown("Welcome to your beautifully formatted, read-only PDF library. All heavy processing is now safely managed by the background `batch_processor.py` script.")

pdf_dir = "Final_PDFs"
os.makedirs(pdf_dir, exist_ok=True)

# Scan for PDFs
pdf_files = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))

if not pdf_files:
    st.info("No PDFs found yet. Open your terminal and run `python batch_processor.py <PLAYLIST_URL>` to start generating notes!")
    st.stop()

# Master Download Button
st.markdown("### 📥 Download All")
if st.button("Download All Notes (ZIP)", type="primary"):
    with st.spinner("Zipping files..."):
        zip_path = shutil.make_archive(
            base_name="All_Notes",
            format='zip',
            root_dir=pdf_dir
        )
        with open(zip_path, "rb") as f:
            st.download_button(
                label="Click here to download ZIP",
                data=f,
                file_name="All_Notes.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True
            )

st.divider()

# Search UI
st.markdown(f"### 🔍 Browse {len(pdf_files)} Generated Notes")
search_query = st.text_input("Search by video title:", placeholder="e.g. Linear Regression...")

filtered_pdfs = []
for pdf in pdf_files:
    title = os.path.basename(pdf).replace(".pdf", "")
    if search_query.lower() in title.lower():
        filtered_pdfs.append((title, pdf))

if not filtered_pdfs:
    st.warning("No notes match your search.")
else:
    # Display in a clean grid layout
    cols = st.columns(3)
    for i, (title, pdf_path) in enumerate(filtered_pdfs):
        with cols[i % 3]:
            st.markdown(f"**{title}**")
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="📥 Download PDF",
                    data=f,
                    file_name=f"{title}.pdf",
                    mime="application/pdf",
                    key=f"dl_pdf_{i}"
                )
            st.caption("---")
