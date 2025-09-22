"""
Main application entry point for the Deep Research Agent
Simplified for easy deployment and usage
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Simple environment setup
os.environ.setdefault("STREAMLIT_THEME_PRIMARY_COLOR", "#1f77b4")
os.environ.setdefault("STREAMLIT_THEME_BACKGROUND_COLOR", "#FFFFFF")

def main():
    """Main entry point with error handling."""
    try:
        from src.chat.chat_interface import main as chat_main
        chat_main()
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Please install required dependencies: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Application Error: {e}")
        print("Please check Ollama is running: ollama serve")

if __name__ == "__main__":
    main()