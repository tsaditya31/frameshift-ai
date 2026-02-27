"""YouTube upload with OAuth2 + multilingual dubbed audio tracks."""
import json
import os
from pathlib import Path
from typing import Dict, Optional

from config import get_settings
from database import async_session, Episode, Storyboard
from sqlalchemy import select

settings = get_settings()

TOKEN_FILE = "youtube_token.json"
SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
]

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "bn": "Bengali",
}

BCP47_CODES = {
    "en": "en",
    "hi": "hi",
    "ta": "ta",
    "te": "te",
    "kn": "kn",
    "ml": "ml",
    "bn": "bn",
}


def _get_authenticated_service():
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        from google.auth.transport.requests import Request
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            client_config = {
                "installed": {
                    "client_id": settings.youtube_client_id,
                    "client_secret": settings.youtube_client_secret,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                }
            }
            flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())

    return build("youtube", "v3", credentials=creds)


async def upload_episode(ep_id: int):
    """Upload an episode to YouTube with all language tracks."""
    async with async_session() as db:
        ep = await db.get(Episode, ep_id)
        if not ep:
            return
        sb = await db.get(Storyboard, ep.storyboard_id)

    out_dir = Path(settings.output_dir) / str(ep.storyboard_id) / f"ep_{ep.episode_num:02d}"
    base_video = out_dir / "base.mp4"
    if not base_video.exists():
        base_video = out_dir / "base_raw.mp4"
    if not base_video.exists():
        print(f"[youtube] No base video found for ep {ep_id}")
        return

    try:
        youtube = _get_authenticated_service()

        # Ensure playlist exists
        playlist_id = await _ensure_playlist(youtube, sb)

        # Build description
        description = (
            f"{ep.title}\n\n"
            f"Episode {ep.episode_num} of {sb.title}.\n\n"
            f"{ep.storyline}\n\n"
            f"Available languages: English, Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali"
        )

        # Upload base video
        video_id = _upload_video(
            youtube,
            video_path=base_video,
            title=f"{sb.title} | Ep {ep.episode_num}: {ep.title}",
            description=description,
            tags=[sb.title, "mythology", "epic", "story"],
            playlist_id=playlist_id,
        )

        youtube_urls = {"en": f"https://youtu.be/{video_id}"}

        # Try to add dubbed audio tracks
        dubbed_ok = await _try_add_dubbed_tracks(youtube, video_id, out_dir)

        if not dubbed_ok:
            # Fallback: upload separate language videos
            for lang_code, lang_name in LANGUAGE_NAMES.items():
                if lang_code == "en":
                    continue
                lang_video = out_dir / f"{lang_code}.mp4"
                if lang_video.exists():
                    lang_vid_id = _upload_video(
                        youtube,
                        video_path=lang_video,
                        title=f"{sb.title} | Ep {ep.episode_num}: {ep.title} [{lang_name}]",
                        description=description,
                        tags=[sb.title, "mythology", lang_name],
                        playlist_id=playlist_id,
                    )
                    youtube_urls[lang_code] = f"https://youtu.be/{lang_vid_id}"

        async with async_session() as db:
            ep = await db.get(Episode, ep_id)
            ep.youtube_urls_json = json.dumps(youtube_urls)
            ep.production_status = "published"
            await db.commit()

    except Exception as e:
        print(f"[youtube] Upload failed for ep {ep_id}: {e}")


def _upload_video(youtube, video_path: Path, title: str, description: str, tags: list, playlist_id: Optional[str]) -> str:
    from googleapiclient.http import MediaFileUpload

    body = {
        "snippet": {
            "title": title[:100],
            "description": description[:5000],
            "tags": tags,
            "categoryId": "22",
            "defaultLanguage": "en",
        },
        "status": {"privacyStatus": "public"},
    }

    media = MediaFileUpload(str(video_path), chunksize=-1, resumable=True, mimetype="video/mp4")
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    while response is None:
        _, response = request.next_chunk()

    video_id = response["id"]

    if playlist_id:
        youtube.playlistItems().insert(
            part="snippet",
            body={
                "snippet": {
                    "playlistId": playlist_id,
                    "resourceId": {"kind": "youtube#video", "videoId": video_id},
                }
            },
        ).execute()

    return video_id


async def _ensure_playlist(youtube, sb: Storyboard) -> Optional[str]:
    """Create a playlist for this storyboard if it doesn't exist."""
    try:
        result = youtube.playlists().insert(
            part="snippet,status",
            body={
                "snippet": {
                    "title": sb.title,
                    "description": sb.description,
                },
                "status": {"privacyStatus": "public"},
            },
        ).execute()
        return result["id"]
    except Exception as e:
        print(f"[youtube] Playlist creation failed: {e}")
        return None


async def _try_add_dubbed_tracks(youtube, video_id: str, out_dir: Path) -> bool:
    """Try YouTube dubbing API for multilingual audio tracks. Returns True if successful."""
    try:
        for lang_code, lang_name in LANGUAGE_NAMES.items():
            if lang_code == "en":
                continue
            audio_path = out_dir / "audio" / f"{lang_code}_full.mp3"
            if not audio_path.exists():
                continue

            from googleapiclient.http import MediaFileUpload
            media = MediaFileUpload(str(audio_path), mimetype="audio/mpeg")

            youtube.videos().insertDubbing(
                part="snippet",
                videoId=video_id,
                body={
                    "snippet": {
                        "language": BCP47_CODES[lang_code],
                        "name": lang_name,
                    }
                },
                media_body=media,
            ).execute()

        return True
    except Exception as e:
        print(f"[youtube] Dubbed tracks API unavailable ({e}), falling back to separate videos")
        return False
