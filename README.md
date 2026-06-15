# YouTube to AI Textbook Compiler

A battle-tested Python pipeline that converts any YouTube playlist into a 1000+ page, deeply technical, domain-agnostic PDF textbook.

## 🏗 Architecture
This project utilizes a highly modular, decoupled 3-stage process to bypass YouTube's internal caps, extract raw knowledge, and use Google's Gemini AI to expand the transcripts into exhaustive textbooks.

* **Stage 1: Harvester (`harvester.py`)**  
  Uses a custom manual chunking loop to bypass YouTube's 100-video cap and extract a master manifest of all video IDs and titles in the playlist.

* **Stage 2: Downloader (`downloader.py`)**  
  Consumes the manifest and individually pulls the raw native `.vtt` / `.srt` subtitle tracks for every video, avoiding the IP-blocking auto-translation API.

* **Stage 3: AI Processor (`processor.py`)**  
  Cleans the raw transcripts, automatically detects the playlist's core subject, dynamically modifies its persona to match the domain expertise, expands the content 3x via the Gemini API, and stitches everything into a beautiful PDF using WeasyPrint.

## 🔐 Prerequisites & Authentication (CRITICAL)

Because YouTube aggressively throttles and blocks automated data scraping, this pipeline uses a secure local session cookie override. **You MUST do the following before running the script:**

1. Install a "Get cookies.txt LOCALLY" extension in Google Chrome.
2. Go to YouTube, ensure you are logged in, and use the extension to export your current session cookies.
3. Save the exported file exactly as `cookies.txt` into the root directory of this project.

*(Note: `cookies.txt` and `.env` are explicitly git-ignored to ensure your personal credentials are never exposed.)*

You will also need a `.env` file containing your Gemini API key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

## 🚀 Usage

Once your `cookies.txt` and `.env` are in place, the entire architecture is managed by a single master orchestrator.

Run the following command in your terminal:
```bash
python main.py "https://www.youtube.com/playlist?list=YOUR_PLAYLIST_ID"
```

The pipeline will safely initialize workspace wipes, output colorful terminal statuses, catch non-zero exit codes securely, and output a pristine PDF titled after the playlist itself!
