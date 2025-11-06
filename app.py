from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable

# === CONFIG ===
API_KEY = None  # set from env if you like; we'll pass it via header check below

app = FastAPI(title="YouTube Transcript Service")

# Allow the GPT Actions runtime to call you
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

YID_RE = re.compile(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})")

def extract_video_id(url_or_id: str):
    if not url_or_id:
        return None
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id):
        return url_or_id
    m = YID_RE.search(url_or_id)
    return m.group(1) if m else None

def ensure_key(header_key: Optional[str]):
    # Replace this with an env check if you prefer:
    #   import os; expected = os.getenv("API_KEY")
    # For simplicity, paste your key string here OR set expected in your host env.
    expected = "CHANGE_ME_SUPER_SECRET"
    if header_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")

class TranscriptReq(BaseModel):
    url: Optional[str] = None
    video_id: Optional[str] = None
    lang: Optional[str] = None
    translate_to: Optional[str] = None
    allow_auto_captions: bool = True

class MetaReq(BaseModel):
    url: Optional[str] = None
    video_id: Optional[str] = None

class ChaptersReq(BaseModel):
    url: Optional[str] = None
    video_id: Optional[str] = None

@app.post("/api/transcript")
def get_transcript(req: TranscriptReq, x_api_key: Optional[str] = Header(None)):
    ensure_key(x_api_key)
    vid = req.video_id or extract_video_id(req.url or "")
    if not vid:
        raise HTTPException(status_code=400, detail="Missing or invalid video id/url")

    langs = []
    if req.lang: langs.append(req.lang)
    langs.extend(["en", "en-US", "en-GB"])

    try:
        tlist = YouTubeTranscriptApi.list_transcripts(vid)
        transcript = None
        source = "human_captions"

        # Try human captions first
        for code in langs:
            try:
                transcript = tlist.find_manually_created_transcript([code])
                break
            except:
                pass

        # Fall back to auto captions
        if transcript is None and req.allow_auto_captions:
            source = "auto_captions"
            for code in langs:
                try:
                    transcript = tlist.find_generated_transcript([code])
                    break
                except:
                    pass

        # Try any available if still none
        if transcript is None:
            try:
                transcript = tlist.find_manually_created_transcript(tlist._manually_created_transcripts.keys())
                source = "human_captions"
            except:
                if req.allow_auto_captions:
                    try:
                        transcript = tlist.find_generated_transcript(tlist._generated_transcripts.keys())
                        source = "auto_captions"
                    except:
                        pass

        if transcript is None:
            raise NoTranscriptFound(vid)

        translated = False
        if req.translate_to and req.translate_to != transcript.language_code:
            try:
                transcript = transcript.translate(req.translate_to)
                translated = True
            except:
                translated = False

        items = transcript.fetch()
        segments = [
            {"start": float(it["start"]), "duration": float(it.get("duration", 0.0)), "text": it.get("text", "")}
            for it in items
        ]
        word_count = sum(len(s["text"].split()) for s in segments)
        approx_runtime = max((s["start"] + s["duration"]) for s in segments) if segments else 0

        return {
            "video_id": vid,
            "source": source,
            "language": transcript.language_code,
            "translated": translated,
            "segments": segments,
            "word_count": word_count,
            "approx_runtime_seconds": approx_runtime
        }

    except (TranscriptsDisabled, NoTranscriptFound):
        raise HTTPException(status_code=404, detail="Transcript unavailable")
    except VideoUnavailable:
        raise HTTPException(status_code=404, detail="Video unavailable")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/metadata")
def get_metadata(req: MetaReq, x_api_key: Optional[str] = Header(None)):
    ensure_key(x_api_key)
    vid = req.video_id or extract_video_id(req.url or "")
    if not vid:
        raise HTTPException(status_code=400, detail="Missing or invalid video id/url")
    return {
        "video_id": vid,
        "title": None, "channel": None, "publish_date": None,
        "duration_seconds": None, "description": None,
        "url": f"https://www.youtube.com/watch?v={vid}"
    }

@app.post("/api/chapters")
def get_chapters(req: ChaptersReq, x_api_key: Optional[str] = Header(None)):
    ensure_key(x_api_key)
    vid = req.video_id or extract_video_id(req.url or "")
    if not vid:
        raise HTTPException(status_code=400, detail="Missing or invalid video id/url")
    return {"video_id": vid, "chapters": []}
