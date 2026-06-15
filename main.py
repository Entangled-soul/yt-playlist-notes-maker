"""
YouTube to AI Textbook Compiler
===============================
main.py (The Orchestrator)

This script acts as the master entry point for the pipeline. It orchestrates
the sequential execution of the 3-stage decoupled architecture:
1. Harvester (Metadata extraction)
2. Downloader (Transcript retrieval)
3. Processor (AI generation and PDF compilation)

It is secured with a try/except block to halt execution instantly if any stage fails.
"""
import sys
import subprocess

def run_pipeline():
    # Phase 1: Terminal Validation & Initialization
    if len(sys.argv) < 2:
        print("⚠️ ERROR: No playlist URL provided. Usage: python main.py <PLAYLIST_URL>")
        sys.exit(1)
        
    playlist_url = sys.argv[1]
    
    print("\n" + "="*50)
    print("🚀 THE MASTER EXECUTION PIPELINE INITIATED")
    print("="*50 + "\n")
    
    try:
        # Stage 1: Harvester
        print("\033[96m🚀 STAGE 1: Harvesting URLs...\033[0m")
        subprocess.run(["python", "harvester.py", playlist_url], check=True)
        print("\n" + "-"*50 + "\n")
        
        # Stage 2: Downloader
        print("\033[96m🚀 STAGE 2: Downloading Transcripts...\033[0m")
        subprocess.run(["python", "downloader.py"], check=True)
        print("\n" + "-"*50 + "\n")
        
        # Stage 3: Processor (Gemini Generation & PDF Compilation)
        print("\033[96m🚀 STAGE 3: Processing & Compiling Textbook...\033[0m")
        subprocess.run(["python", "processor.py"], check=True)
        print("\n" + "="*50)
        print("🎉 PIPELINE COMPLETE!")
        print("="*50 + "\n")
        
    except subprocess.CalledProcessError as e:
        # Phase 3: Critical Failure Interception
        failed_script = e.cmd[1] if len(e.cmd) > 1 else e.cmd[0]
        print(f"\n\033[91m🚨 CRITICAL ERROR: The pipeline halted because '{failed_script}' crashed with exit code {e.returncode}.\033[0m")
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()
