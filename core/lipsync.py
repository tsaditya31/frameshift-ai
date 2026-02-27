"""Sync.so lip sync API integration."""
import asyncio
import httpx
from pathlib import Path
from typing import Optional

from config import get_settings

settings = get_settings()

SYNC_SO_BASE = "https://api.sync.so/v2"


async def submit_lipsync_job(video_path: Path, audio_path: Path) -> Optional[str]:
    """Submit a lip sync job to Sync.so. Returns job ID."""
    if not settings.sync_so_api_key:
        print("[lipsync] No SYNC_SO_API_KEY configured, skipping lip sync")
        return None

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            with open(video_path, "rb") as vf, open(audio_path, "rb") as af:
                resp = await client.post(
                    f"{SYNC_SO_BASE}/generate",
                    headers={"x-api-key": settings.sync_so_api_key},
                    files={
                        "video": (video_path.name, vf, "video/mp4"),
                        "audio": (audio_path.name, af, "audio/mpeg"),
                    },
                    data={"model": "sync-1.9.0-beta"},
                )
            resp.raise_for_status()
            data = resp.json()
            return data.get("id")
    except Exception as e:
        print(f"[lipsync] Submit failed: {e}")
        return None


async def poll_lipsync_job(job_id: str, timeout_seconds: int = 600) -> Optional[str]:
    """Poll Sync.so until the job is done. Returns output video URL."""
    if not job_id:
        return None

    elapsed = 0
    interval = 10
    while elapsed < timeout_seconds:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    f"{SYNC_SO_BASE}/generate/{job_id}",
                    headers={"x-api-key": settings.sync_so_api_key},
                )
                resp.raise_for_status()
                data = resp.json()
                status = data.get("status", "")
                if status == "completed":
                    return data.get("outputUrl") or data.get("output_url")
                if status in ("failed", "error"):
                    print(f"[lipsync] Job {job_id} failed: {data}")
                    return None
        except Exception as e:
            print(f"[lipsync] Poll error: {e}")

        await asyncio.sleep(interval)
        elapsed += interval

    print(f"[lipsync] Job {job_id} timed out after {timeout_seconds}s")
    return None


async def download_video(url: str, output_path: Path) -> bool:
    """Download a video from a URL to a local path."""
    try:
        async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(resp.content)
            return True
    except Exception as e:
        print(f"[lipsync] Download failed: {e}")
        return False


async def lipsync_video(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
) -> bool:
    """Full lip sync pipeline: submit → poll → download. Returns True on success."""
    job_id = await submit_lipsync_job(video_path, audio_path)
    if not job_id:
        return False

    video_url = await poll_lipsync_job(job_id)
    if not video_url:
        return False

    return await download_video(video_url, output_path)
