"""ULCA_agent application entry point."""
from backend.app import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8001)
