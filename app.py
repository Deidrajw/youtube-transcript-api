import os, re
from typing import Optional
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# youtube-transcript-api
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

app = FastAPI(title="YouTube Transcript Service")

# CORS so GPT Actions can call you
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- config / helpers ---
EXPECTED_API_KEY = os.getenv("API_KEY", "CHANGE_ME_SUPER_SECRET")
YID_RE = re.compile(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})")

def ensure_key(header_key: Optional[str]):
    if header_key != EXPECTED_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

def extract_video_id(url_or_id: str):
    if not url_or_id:
        return None
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id):
        return url_or_id
    m = YID_RE.search(url_or_id)
    return m.group(1) if m else None

# --- models ---
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

# --- health/version ---
@app.get("/")
def root():
    return {"ok": True, "service": "YouTube Transcript Service"}

@app.get("/version")
def version():
    import youtube_transcript_api as yta
    return {"yta_version": getattr(yta, "__version__", "unknown")}

# --- core endpoints ---
@app.post("/api/transcript")
def get_transcript(req: TranscriptReq, x_api_key: Optional[str] = Header(None)):
    ensure_key(x_api_key)
    vid = req.video_id or extract_video_id(req.url or "")
    if not vid:
        raise HTTPException(status_code=400, detail="Missing or invalid video id/url")

    # language preferences
    langs = []
    if req.lang:
        langs.append(req.lang)
    langs.extend(["en", "en-US", "en-GB"])

    try:
        # Path A: modern API (list_transcripts exists)
        if hasattr(YouTubeTranscriptApi, "list_transcripts"):
            tlist = YouTubeTranscriptApi.list_transcripts(vid)
            transcript = None
            source = "human_captions"

            # try human captions by preferred langs
            for code in langs:
                try:
                    transcript = tlist.find_manually_created_transcript([code])
                    break
                except Exception:
                    pass

            # fall back to auto captions
            if transcript is None and req.allow_auto_captions:
                source = "auto_captions"
                for code in langs:
                    try:
                        transcript = tlist.find_generated_transcript([code])
                        break
                    except Exception:
                        pass

            # last-ditch: any human, then any generated
            if transcript is None:
                try:
                    transcript = tlist.find_manually_created_transcript(
                        list(getattr(tlist, "_manually_created_transcripts", {}).keys())
                    )
                    source = "human_captions"
                except Exception:
                    if req.allow_auto_captions:
                        try:
                            transcript = tlist.find_generated_transcript(
                                list(getattr(tlist, "_generated_transcripts", {}).keys())
                            )
                            source = "auto_captions"
                        except Exception:
                            pass

            if transcript is None:
                raise NoTranscriptFound(vid)

            translated = False
            if req.translate_to and req.translate_to != getattr(transcript, "language_code", None):
                try:
                    transcript = transcript.translate(req.translate_to)
                    translated = True
                except Exception:
                    translated = False

            items = transcript.fetch()
            language = getattr(transcript, "language_code", None)

        # Path B: older API (no list_transcripts)
        else:
            source = "unknown"
            items = None
            language = req.lang or "unknown"
            for code in langs:
                try:
                    items = YouTubeTranscriptApi.get_transcript(vid, languages=[code])
                    language = code
                    break
                except NoTranscriptFound:
                    continue
                except TranscriptsDisabled:
                    raise HTTPException(status_code=404, detail="Transcript unavailable")

            if items is None and req.allow_auto_captions:
                try:
                    items = YouTubeTranscriptApi.get_transcript(vid)
                except Exception:
                    pass

            if items is None:
                raise NoTranscriptFound(vid)

            translated = False

        segments = [
            {"start": float(it.get("start", 0.0)),
             "duration": float(it.get("duration", 0.0)),
             "text": it.get("text", "")}
            for it in items
        ]
        word_count = sum(len(s["text"].split()) for s in segments)
        approx_runtime = max((s["start"] + s["duration"]) for s in segments) if segments else 0

        return {
            "video_id": vid,
            "source": source,
            "language": language or "unknown",
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
    return {"video_id": vid, "url": f"https://www.youtube.com/watch?v={vid}"}

@app.post("/api/chapters")
def get_chapters(req: ChaptersReq, x_api_key: Optional[str] = Header(None)):
    ensure_key(x_api_key)
    vid = req.video_id or extract_video_id(req.url or "")
    if not vid:
        raise HTTPException(status_code=400, detail="Missing or invalid video id/url")
    return {"video_id": vid, "chapters": []}  # stub; you can enrich later
