import os, re, io, requests
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Subtitles & transcript backends
import yt_dlp
import webvtt
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

# ------------------------------------------------------------------------------
# App + CORS
# ------------------------------------------------------------------------------
app = FastAPI(title="YouTube Transcript Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # lock this down later if you like
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------
EXPECTED_API_KEY = os.getenv("API_KEY", "CHANGE_ME_SUPER_SECRET")
YID_RE = re.compile(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})")

# ------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
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

def to_seconds(ts: str) -> float:
    # HH:MM:SS.mmm
    h, m, s = ts.split(":")
    return float(h) * 3600 + float(m) * 60 + float(s)

def vtt_to_segments(vtt_text: str) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    buf = io.StringIO(vtt_text)
    for caption in webvtt.read_buffer(buf):
        start = to_seconds(caption.start)
        end = to_seconds(caption.end)
        segments.append({
            "start": start,
            "duration": max(0.0, end - start),
            "text": (caption.text or "").strip()
        })
    return segments

def fetch_subs_with_ytdlp(url_or_id: str):
    """
    Try to fetch subtitles via yt-dlp first.
    Returns (segments, source, language) or (None, None, None) if not found.
    """
    ydl_opts = {"quiet": True, "nocheckcertificate": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url_or_id, download=False)

    # Prefer human subtitles, then auto
    for bucket in ("subtitles", "automatic_captions"):
        submap = info.get(bucket) or {}

        # Try English variants first
        for lang_try in ("en", "en-US", "en-GB"):
            tracks = submap.get(lang_try)
            if not tracks:
                continue
            for t in tracks:
                url = t.get("url")
                ext = (t.get("ext") or "").lower()
                if not url:
                    continue
                r = requests.get(url, timeout=20)
                r.raise_for_status()
                if ext == "vtt":
                    segs = vtt_to_segments(r.text)
                else:
                    # If non-VTT (e.g., srv3/ttml/json3), return as 1 chunk (still usable)
                    segs = [{"start": 0.0, "duration": 0.0, "text": r.text}]
                return segs, ("human_captions" if bucket == "subtitles" else "auto_captions"), lang_try

        # If no English, take the first available track
        for lang, tracks in submap.items():
            for t in tracks or []:
                url = t.get("url")
                ext = (t.get("ext") or "").lower()
                if not url:
                    continue
                r = requests.get(url, timeout=20)
                r.raise_for_status()
                if ext == "vtt":
                    segs = vtt_to_segments(r.text)
                else:
                    segs = [{"start": 0.0, "duration": 0.0, "text": r.text}]
                return segs, ("human_captions" if bucket == "subtitles" else "auto_captions"), lang

    return None, None, None

# ------------------------------------------------------------------------------
# Health / Version / Debug
# ------------------------------------------------------------------------------
@app.get("/")
def root():
    return {"ok": True, "service": "YouTube Transcript Service"}

@app.get("/version")
def version():
    import youtube_transcript_api as yta
    return {"yta_version": getattr(yta, "__version__", "unknown")}

@app.get("/debug/transcripts")
def debug_transcripts(
    url: Optional[str] = Query(None),
    video_id: Optional[str] = Query(None),
    x_api_key: Optional[str] = Header(None)
):
    ensure_key(x_api_key)
    vid = video_id or extract_video_id(url or "")
    if not vid:
        raise HTTPException(status_code=400, detail="Missing or invalid video id/url")

    try:
        info = {"video_id": vid, "tracks": []}
        # Show what yt-dlp sees
        try:
            ydl_opts = {"quiet": True, "nocheckcertificate": True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                meta = ydl.extract_info(vid, download=False)
            for bucket in ("subtitles", "automatic_captions"):
                submap = meta.get(bucket) or {}
                for lang, tracks in submap.items():
                    info["tracks"].append({
                        "backend": "yt-dlp",
                        "bucket": bucket,
                        "language": lang,
                        "count": len(tracks or []),
                    })
        except Exception as e:
            info["yt_dlp_probe_error"] = str(e)

        # Show what youtube-transcript-api sees
        try:
            if hasattr(YouTubeTranscriptApi, "list_transcripts"):
                tl = YouTubeTranscriptApi.list_transcripts(vid)
                for t in getattr(tl, "_manually_created_transcripts", {}).values():
                    info["tracks"].append({
                        "backend": "youtube-transcript-api",
                        "bucket": "human",
                        "language": t.language_code,
                        "translatable_to": [x["language_code"] for x in (t.translation_languages or [])]
                    })
                for t in getattr(tl, "_generated_transcripts", {}).values():
                    info["tracks"].append({
                        "backend": "youtube-transcript-api",
                        "bucket": "auto",
                        "language": t.language_code,
                        "translatable_to": [x["language_code"] for x in (t.translation_languages or [])]
                    })
            else:
                # older API path – not much to list
                info["yt_api_mode"] = "legacy_get_transcript_only"
        except Exception as e:
            info["yta_probe_error"] = str(e)

        return info
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"probe_failed: {type(e).__name__}: {str(e)}")

# ------------------------------------------------------------------------------
# Core endpoints
# ------------------------------------------------------------------------------
@app.post("/api/transcript")
def get_transcript(req: TranscriptReq, x_api_key: Optional[str] = Header(None)):
    ensure_key(x_api_key)

    vid = req.video_id or extract_video_id(req.url or "")
    if not vid:
        raise HTTPException(status_code=400, detail="Missing or invalid video id/url")

    # 0) Try yt-dlp subtitles first (most reliable, no API cost)
    try:
        segs, src, lang_found = fetch_subs_with_ytdlp(req.url or vid)
        if segs:
            word_count = sum(len(s["text"].split()) for s in segs)
            approx_runtime = max((s["start"] + s["duration"]) for s in segs if s.get("duration") is not None) if segs else 0
            return {
                "video_id": vid,
                "source": src,
                "language": lang_found or req.lang or "unknown",
                "translated": False,
                "segments": segs,
                "word_count": word_count,
                "approx_runtime_seconds": approx_runtime
            }
    except Exception:
        # swallow and fall through to youtube-transcript-api path
        pass

    # 1) Fall back to youtube-transcript-api (human → auto → any)
    langs: List[str] = []
    if req.lang: langs.append(req.lang)
    langs.extend(["en", "en-US", "en-GB"])

    try:
        # Modern path: list_transcripts exists
        if hasattr(YouTubeTranscriptApi, "list_transcripts"):
            tlist = YouTubeTranscriptApi.list_transcripts(vid)
            transcript = None
            source = "human_captions"

            # try human in preferred languages
            for code in langs:
                try:
                    transcript = tlist.find_manually_created_transcript([code])
                    break
                except Exception:
                    pass

            # try auto in preferred languages
            if transcript is None and req.allow_auto_captions:
                source = "auto_captions"
                for code in langs:
                    try:
                        transcript = tlist.find_generated_transcript([code])
                        break
                    except Exception:
                        pass

            # last-ditch: any human, then any auto
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

        else:
            # Legacy path: get_transcript only
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

        # Normalize output
        segments = [
            {
                "start": float(it.get("start", 0.0)),
                "duration": float(it.get("duration", 0.0)),
                "text": it.get("text", "")
            } for it in items
        ]
        word_count = sum(len(s["text"].split()) for s in segments)
        approx_runtime = max((s["start"] + s["duration"]) for s in segments) if segments else 0

        return {
            "video_id": vid,
            "source": source,
            "language": language or req.lang or "unknown",
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
        "url": f"https://www.youtube.com/watch?v={vid}",
        # expand later with title/channel/date if you like
    }

@app.post("/api/chapters")
def get_chapters(req: ChaptersReq, x_api_key: Optional[str] = Header(None)):
    ensure_key(x_api_key)
    vid = req.video_id or extract_video_id(req.url or "")
    if not vid:
        raise HTTPException(status_code=400, detail="Missing or invalid video id/url")
    # TODO: you can enrich via yt-dlp "chapters" if present in metadata
    return {"video_id": vid, "chapters": []}
