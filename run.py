# run.py
import uvicorn
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",  # Changed to localhost for local testing
        port=8000,
        reload=True,
        log_level="info"
    )
