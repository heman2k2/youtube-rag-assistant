from urllib.parse import parse_qs, urlparse

from langchain_core.documents import Document
from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)
from youtube_transcript_api.formatters import TextFormatter

yt = YouTubeTranscriptApi()
formatter = TextFormatter()


def extract_video_id(url_or_id: str) -> str:
    """
    Extracts the 11-character YouTube video ID from a YouTube URL or raw ID.

    Raises:
        ValueError: If the input is empty, not a string, or does not yield a
            valid 11-character YouTube video ID.
    """
    if not isinstance(url_or_id, str):
        raise ValueError(f"Expected a string URL or video ID, got {type(url_or_id).__name__}")

    value = url_or_id.strip()
    if not value:
        raise ValueError("YouTube URL or video ID cannot be empty")

    parsed = urlparse(value)
    host = parsed.netloc.lower()
    video_id = ""

    if host == "youtu.be":
        parts = [p for p in parsed.path.split("/") if p]
        video_id = parts[0] if parts else ""
    elif host == "youtube.com" or host.endswith(".youtube.com"):
        path_parts = [p for p in parsed.path.split("/") if p]
        if path_parts and path_parts[0] in {"shorts", "embed", "v", "live"} and len(path_parts) > 1:
            video_id = path_parts[1]
        else:
            video_id = parse_qs(parsed.query).get("v", [""])[0]
    elif not parsed.scheme and not parsed.netloc:
        # Allow passing a raw 11-character video ID directly
        video_id = value

    if len(video_id) != 11:
        raise ValueError(f"Invalid YouTube URL or video ID: {url_or_id!r}")

    return video_id


def get_transcript(
    url_or_id: str,
    languages: list[str] | None = None,
    translate_to: str | None = None,
) -> Document:
    """
    Fetches a YouTube transcript as a LangChain Document from a YouTube URL
    or video ID, with optional automatic fallback to a translated caption track
    when a requested language is unavailable.

    Args:
        url_or_id: Full YouTube URL or 11-character video ID.
        languages: Ordered list of preferred language codes (defaults to ["en"]).
        translate_to: Target language code to translate into if no direct match
            is found. Defaults to the first language in `languages` (or "en").

    Raises:
        ValueError: If the URL/ID is invalid or no transcript/translation can be retrieved.
    """
    clean_id = extract_video_id(url_or_id)
    preferred_langs = languages or ["en"]
    target_lang = translate_to or preferred_langs[0]

    try:
        transcript_list = yt.list(clean_id)

        is_translated = False
        source_language_code: str | None = None

        try:
            transcript = transcript_list.find_transcript(preferred_langs)
            fetched = transcript.fetch()
            actual_lang = transcript.language_code
            is_generated = transcript.is_generated
        except NoTranscriptFound:
            source_transcript = next(
                (t for t in transcript_list if t.is_translatable),
                None,
            )
            if source_transcript is None:
                raise NoTranscriptFound(
                    clean_id,
                    preferred_langs,
                    transcript_list,
                )

            translated_transcript = source_transcript.translate(target_lang)
            fetched = translated_transcript.fetch()
            actual_lang = target_lang
            is_generated = source_transcript.is_generated
            is_translated = True
            source_language_code = source_transcript.language_code

        transcript_text = formatter.format_transcript(fetched)
        if not transcript_text.strip():
            raise ValueError(f"Retrieved an empty transcript for video ID {clean_id}")

        return Document(
            page_content=transcript_text,
            metadata={
                "video_id": clean_id,
                "source_url": url_or_id.strip(),
                "language_code": actual_lang,
                "is_generated": is_generated,
                "is_translated": is_translated,
                "source_language_code": source_language_code,
            },
        )
    except (
        TranscriptsDisabled,
        NoTranscriptFound,
        VideoUnavailable,
        CouldNotRetrieveTranscript,
    ) as e:
        raise ValueError(
            f"Could not retrieve or translate transcript for video ID {clean_id}: {e}"
        ) from e


if __name__ == "__main__":
    test_url = "https://youtu.be/3lPnN8omdPA?si=g5RlUsTGWiu0l4J5"
    doc = get_transcript(test_url, languages=["en", "hi"], translate_to="en")
    print("Video ID:", doc.metadata["video_id"])
    print("Language:", doc.metadata["language_code"])
    print("Length:", len(doc.page_content))
    print("Preview:", doc.page_content[:300])