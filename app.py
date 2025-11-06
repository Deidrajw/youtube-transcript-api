@app.post("/api/transcript")
def get_transcript(req: TranscriptReq, x_api_key: Optional[str] = Header(None)):
    ensure_key(x_api_key)
    vid = req.video_id or extract_video_id(req.url or "")
    if not vid:
        raise HTTPException(status_code=400, detail="Missing or invalid video id/url")

    # Preferred language order
    langs = []
    if req.lang:
        langs.append(req.lang)
    langs.extend(["en", "en-US", "en-GB"])

    try:
        # ---------- Path A: modern API with list_transcripts ----------
        if hasattr(YouTubeTranscriptApi, "list_transcripts"):
            tlist = YouTubeTranscriptApi.list_transcripts(vid)
            transcript = None
            source = "human_captions"

            # Try human captions first (preferred langs)
            for code in langs:
                try:
                    transcript = tlist.find_manually_created_transcript([code])
                    break
                except Exception:
                    pass

            # Fall back to auto captions
            if transcript is None and req.allow_auto_captions:
                source = "auto_captions"
                for code in langs:
                    try:
                        transcript = tlist.find_generated_transcript([code])
                        break
                    except Exception:
                        pass

            # Last-ditch: any manually created, then any generated
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

        # ---------- Path B: older API (no list_transcripts) ----------
        else:
            source = "unknown"
            items = None
            language = req.lang or "unknown"

            # Try preferred languages
            for code in langs:
                try:
                    items = YouTubeTranscriptApi.get_transcript(vid, languages=[code])
                    language = code
                    break
                except NoTranscriptFound:
                    continue
                except TranscriptsDisabled:
                    raise HTTPException(status_code=404, detail="Transcript unavailable")

            # Try any transcript if allowed
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
            }
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
@app.get("/")
def root():
    return {"ok": True, "service": "YouTube Transcript Service"}

@app.get("/version")
def version():
    import youtube_transcript_api as yta
    return {"yta_version": getattr(yta, "__version__", "unknown")}

