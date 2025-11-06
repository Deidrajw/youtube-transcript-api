import os, re, io, base64, tempfile, requests
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

# OpenAI (Whisper fallback)
from openai import OpenAI

# ------------------------------------------------------------------------------
# App + CORS
# ------------------------------------------------------------------------------
app = FastAPI(title="YouTube Transcript Service", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later if you want
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------
EXPECTED_API_KEY = os.getenv("API_KEY", "CHANGE_ME_SUPER_SECRET")
OPENAI_MODEL = os.getenv("WHISPER_MODEL", "whisper-1")  # or "gpt-4o-mini-transcribe"
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

# ----- optional cookie support (base64-encoded cookies.txt in env) -----
def get_cookiefile_path_from_env() -> Optional[str]:
    b64 = os.getenv("YT_COOKIES_B64")
    if not b64:
        return None
    try:
        raw = base64.b64decode(b64)
        fp = tempfile.NamedTemporaryFile(delete=False, suffix=".cookies.txt")
        fp.write(raw); fp.flush(); fp.close()
        return fp.name
    except Exception:
        return None

def ydl_with_optional_cookies_opts() -> dict:
    opts = {"quiet": True, "nocheckcertificate": True}
    cookiefile = get_cookiefile_path_from_env()
    if cookiefile:
        opts["cookiefile"] = cookiefile
    return opts

def fetch_subs_with_ytdlp(url_or_id: str):
    """
    Try subtitles via yt-dlp first.
    Returns (segments, source, language) or (None, None, None) if not found.
    """
    with yt_dlp.YoutubeDL(ydl_with_optional_cookies_opts()) as ydl:
        info = ydl.extract_info(url_or_id, download=False)

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
                    segs = [{"start": 0.0, "duration": 0.0, "text": r.text}]
                return segs, ("human_captions" if bucket == "subtitles" else "auto_captions"), lang_try

        # If no English, take first available
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

def select_best_audio_format(formats: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    audio_only = [f for f in formats or [] if f.get("vcodec") in (None, "none") and f.get("acodec") not in (None, "none")]
    if not audio_only:
        return None
    audio_only.sort(key=lambda f: float(f.get("abr", 0.0)), reverse=True)
    return audio_only[0]

def get_audio_url_with_ytdlp(url_or_id: str) -> Optional[str]:
    with yt_dlp.YoutubeDL(ydl_with_optional_cookies_opts()) as ydl:
        info = ydl.extract_info(url_or_id, download=False)
    fmt = select_best_audio_format(info.get("formats", []))
    if fmt and fmt.get("url"):
        return fmt["url"]
    return info.get("url")

def download_to_temp(url: str) -> str:
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    fd, path = tempfile.mkstemp(suffix=".audio")
    with os.fdopen(fd, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 256):
            if chunk:
                f.write(chunk)
    return path

def transcribe_with_whisper(path: str) -> str:
    client = OpenAI()
    with open(path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model=OPENAI_MODEL,  # "whisper-1" or "gpt-4o-mini-transcribe"
            file=f,
            response_format="text",
        )
    return resp  # string when response_format="text"

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

    info = {"video_id": vid, "tracks": []}
    try:
        with yt_dlp.YoutubeDL(ydl_with_optional_cookies_opts()) as ydl:
            meta = ydl.extract_info(vid, download=False)
        for bucket in ("subtitles", "automatic_captions"):
            submap = meta.get(bucket) or {}
            for lang, tracks in submap.items():
                info["tracks"].append({"backend": "yt-dlp", "bucket": bucket, "language": lang, "count": len(tracks or [])})
    except Exception as e:
        info["yt_dlp_probe_error"] = str(e)

    try:
        if hasattr(YouTubeTranscriptApi, "list_transcripts"):
            tl = YouTubeTranscriptApi.list_transcripts(vid)
            for t in getattr(tl, "_manually_created_transcripts", {}).values():
                info["tracks"].append({"backend": "youtube-transcript-api", "bucket": "human", "language": t.language_code})
            for t in getattr(tl, "_generated_transcripts", {}).values():
                info["tracks"].append({"backend": "youtube-transcript-api", "bucket": "auto", "language": t.language_code})
        else:
            info["yt_api_mode"] = "legacy_get_transcript_only"
    except Exception as e:
        info["yta_probe_error"] = str(e)

    return info

@app.get("/debug/env")
def debug_env(x_api_key: Optional[str] = Header(None)):
    ensure_key(x_api_key)
    return {
        "has_openai_key": bool(os.getenv("OPENAI_API_KEY")),
        "whisper_model": os.getenv("WHISPER_MODEL", "whisper-1")
    }

@app.get("/debug/audio")
def debug_audio(
    url: Optional[str] = Query(None),
    video_id: Optional[str] = Query(None),
    x_api_key: Optional[str] = Header(None)
):
    ensure_key(x_api_key)
    vid = video_id or extract_video_id(url or "")
    if not vid:
        raise HTTPException(status_code=400, detail="Missing or invalid video id/url")
    try:
        aurl = get_audio_url_with_ytdlp(url or vid)
        if not aurl:
            return {"video_id": vid, "audio_url": None, "note": "yt-dlp returned no audio url"}
        r = requests.head(aurl, timeout=15, allow_redirects=True)
        return {"video_id": vid, "audio_url": aurl, "status": r.status_code}
    except Exception as e:
        return {"video_id": vid, "audio_probe_error": f"{type(e).__name__}: {e}"}

# ------------------------------------------------------------------------------
# Core endpoint
# ------------------------------------------------------------------------------
@app.post("/api/transcript")
def get_transcript(req: TranscriptReq, x_api_key: Optional[str] = Header(None)):
    ensure_key(x_api_key)

    vid = req.video_id or extract_video_id(req.url or "")
    if not vid:
        raise HTTPException(status_code=400, detail="Missing or invalid video id/url")

    last_error = "none"

    # 0) Try yt-dlp subtitles first
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
    except Exception as e:
        last_error = f"yt-dlp subtitles: {type(e).__name__}: {e}"

    # 1) Fall back to youtube-transcript-api (human -> auto -> any)
    langs: List[str] = []
    if req.lang: langs.append(req.lang)
    langs.extend(["en", "en-US", "en-GB"])

    try:
        if hasattr(YouTubeTranscriptApi, "list_transcripts"):
            tlist = YouTubeTranscriptApi.list_transcripts(vid)
            transcript = None
            source = "human_captions"
            for code in langs:
                try:
                    transcript = tlist.find_manually_created_transcript([code]); break
                except Exception:
                    pass
            if transcript is None and req.allow_auto_captions:
                source = "auto_captions"
                for code in langs:
                    try:
                        transcript = tlist.find_generated_transcript([code]); break
                    except Exception:
                        pass
            if transcript is None:
                try:
                    transcript = tlist.find_manually_created_transcript(list(getattr(tlist, "_manually_created_transcripts", {}).keys()))
                    source = "human_captions"
                except Exception:
                    if req.allow_auto_captions:
                        try:
                            transcript = tlist.find_generated_transcript(list(getattr(tlist, "_generated_transcripts", {}).keys()))
                            source = "auto_captions"
                        except Exception:
                            pass
            if transcript is None:
                raise NoTranscriptFound(vid)

            translated = False
            if req.translate_to and req.translate_to != getattr(transcript, "language_code", None):
                try:
                    transcript = transcript.translate(req.translate_to); translated = True
                except Exception:
                    translated = False

            items = transcript.fetch()
            language = getattr(transcript, "language_code", None)
        else:
            source = "unknown"
            items = None
            language = req.lang or "unknown"
            for code in langs:
                try:
                    items = YouTubeTranscriptApi.get_transcript(vid, languages=[code]); language = code; break
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

        segments = [{"start": float(it.get("start", 0.0)), "duration": float(it.get("duration", 0.0)), "text": it.get("text", "")} for it in items]
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

    except Exception as e:
        last_error = f"yta fallback: {type(e).__name__}: {e}"

    # ===== FINAL FALLBACK: Whisper transcription from audio =====
    try:
        audio_url = get_audio_url_with_ytdlp(req.url or vid)
        if not audio_url:
            raise RuntimeError("yt-dlp could not get an audio URL (likely cookie/challenge)")

        temp_path = download_to_temp(audio_url)
        try:
            text = transcribe_with_whisper(temp_path)
        finally:
            try: os.remove(temp_path)
            except Exception: pass

        words = text.split()
        return {
            "video_id": vid,
            "source": "whisper_transcription",
            "language": req.lang or "unknown",
            "translated": False,
            "segments": [{"start": 0.0, "duration": 0.0, "text": text}],
            "word_count": len(words),
            "approx_runtime_seconds": 0.0
        }
    except Exception as e2:
        raise HTTPException(
            status_code=404,
            detail=f"Transcript unavailable; last_error='{last_error}' whisper_error='{type(e2).__name__}: {e2}'"
        )

# Minimal stubs (kept for schema stability)
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
    return {"video_id": vid, "chapters": []}
