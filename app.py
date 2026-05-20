# Hugging Face Spaces entry point
# HF Spaces looks for 'app' in app.py by default
from server import app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
