from fastapi import FastAPI
import youtube_transcript_api, yt_dlp, webvtt, requests
from openai import OpenAI

app = FastAPI()

@app.get("/")
def root():
    return {"ok": True}

@app.get("/version")
def version():
    import youtube_transcript_api as yta
    return {"yta_version": getattr(yta, "__version__", "unknown")}
