"""End-to-end video production pipeline for an episode."""
import json
import asyncio
import subprocess
from pathlib import Path
from typing import List, Optional

from config import get_settings
from database import async_session, Episode, Job, Character, KnowledgeChunk, Storyboard
from sqlalchemy import select
from core.story_generator import generate_scene_script
from core.translator import translate_narrations, SUPPORTED_LANGUAGES
from core.audio_generator import generate_episode_audio
from core.lipsync import lipsync_video

settings = get_settings()


async def produce_episode(ep_id: int):
    """Main pipeline: script → images → animate → audio → lipsync → stitch → per-language outputs."""
    async with async_session() as db:
        ep = await db.get(Episode, ep_id)
        if not ep:
            return

        ep.production_status = "producing"
        await db.commit()

        sb = await db.get(Storyboard, ep.storyboard_id)
        kb_result = await db.execute(
            select(KnowledgeChunk).where(KnowledgeChunk.storyboard_id == ep.storyboard_id)
        )
        kb_chunks = kb_result.scalars().all()

        char_result = await db.execute(
            select(Character).where(
                Character.storyboard_id == ep.storyboard_id,
                Character.training_status == "trained",
            )
        )
        characters = char_result.scalars().all()
        soul_map = {c.name: c.higgsfield_soul_id for c in characters}

        out_dir = Path(settings.output_dir) / str(ep.storyboard_id) / f"ep_{ep.episode_num:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Generate script
        async with async_session() as db:
            ep = await db.get(Episode, ep_id)
            if ep.script_json:
                scenes = json.loads(ep.script_json)
            else:
                sb = await db.get(Storyboard, ep.storyboard_id)
                kb_result = await db.execute(
                    select(KnowledgeChunk).where(KnowledgeChunk.storyboard_id == ep.storyboard_id)
                )
                kb_chunks = kb_result.scalars().all()
                scenes = await generate_scene_script(sb, kb_chunks, ep.episode_num, ep.title, ep.storyline)
                ep.script_json = json.dumps(scenes)
                await db.commit()

        # Step 2: Generate scene images + animate
        base_video_path = await _produce_base_video(ep_id, scenes, soul_map, out_dir)

        # Step 3: English narration audio
        en_audio_paths = await generate_episode_audio(scenes, "en", out_dir / "audio")

        # Step 4: English lip sync
        en_lipsync_path = out_dir / "base.mp4"
        if base_video_path and any(p for p in en_audio_paths if p):
            combined_audio = await _concat_audio(en_audio_paths, out_dir / "audio" / "en_full.mp3")
            if combined_audio:
                await lipsync_video(base_video_path, combined_audio, en_lipsync_path)
            else:
                en_lipsync_path = base_video_path

        # Step 5: Translate + TTS + lip sync for each Indic language
        translated = await translate_narrations(scenes)

        async with async_session() as db:
            ep = await db.get(Episode, ep_id)
            ep.narration_translations = json.dumps({
                lang: [s.get("narration", "") for s in sc]
                for lang, sc in translated.items()
            })
            await db.commit()

        for lang_code in SUPPORTED_LANGUAGES:
            lang_scenes = translated.get(lang_code, scenes)
            lang_audio_paths = await generate_episode_audio(lang_scenes, lang_code, out_dir / "audio")
            combined_audio = await _concat_audio(lang_audio_paths, out_dir / "audio" / f"{lang_code}_full.mp3")
            lang_out = out_dir / f"{lang_code}.mp4"
            base = en_lipsync_path if en_lipsync_path.exists() else base_video_path
            if base and combined_audio:
                await lipsync_video(base, combined_audio, lang_out)

        # Mark done
        async with async_session() as db:
            ep = await db.get(Episode, ep_id)
            ep.production_status = "done"
            await db.commit()

    except Exception as e:
        async with async_session() as db:
            ep = await db.get(Episode, ep_id)
            ep.production_status = "draft"
            await db.commit()
        print(f"[video_producer] Production failed for ep {ep_id}: {e}")


async def _produce_base_video(
    ep_id: int,
    scenes: List[dict],
    soul_map: dict,
    out_dir: Path,
) -> Optional[Path]:
    """Generate still images then animate each scene; stitch into base video."""
    clip_paths = []
    for scene in scenes:
        scene_num = scene.get("scene", 1)
        img_path = out_dir / f"scene_{scene_num:02d}.jpg"
        clip_path = out_dir / f"scene_{scene_num:02d}.mp4"

        # Generate still image
        img_ok = await _generate_scene_image(scene, soul_map, img_path, ep_id)
        if not img_ok:
            continue

        # Animate still → clip
        clip_ok = await _animate_scene(img_path, scene.get("duration_seconds", 5), clip_path, ep_id)
        if clip_ok:
            clip_paths.append(clip_path)

    if not clip_paths:
        return None

    base_path = out_dir / "base_raw.mp4"
    await _stitch_clips(clip_paths, base_path)
    return base_path if base_path.exists() else None


async def _generate_scene_image(scene: dict, soul_map: dict, out_path: Path, ep_id: int) -> bool:
    description = scene.get("description", "")
    characters = scene.get("characters", [])
    soul_ids = [soul_map[c] for c in characters if c in soul_map]

    # Create job record
    async with async_session() as db:
        job = Job(
            episode_id=ep_id,
            job_type="scene_image",
            language="en",
            status="running",
        )
        db.add(job)
        await db.commit()
        job_id = job.id

    try:
        from higgsfield.client import HiggsFieldClient
        hf = HiggsFieldClient(api_key=settings.higgsfield_api_key)

        kwargs = {"prompt": f"Epic cinematic scene: {description}", "width": 1280, "height": 720}
        if soul_ids:
            kwargs["soul_ids"] = soul_ids

        result = hf.image.generate(**kwargs)
        img_url = result.image_url if hasattr(result, "image_url") else result.get("image_url", "")

        if img_url:
            import httpx
            async with httpx.AsyncClient(timeout=60) as c:
                resp = await c.get(img_url)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(resp.content)

            async with async_session() as db:
                j = await db.get(Job, job_id)
                j.status = "done"
                j.output_path = str(out_path)
                await db.commit()
            return True

    except Exception as e:
        print(f"[video_producer] Scene image failed: {e}")

    async with async_session() as db:
        j = await db.get(Job, job_id)
        j.status = "failed"
        j.error_msg = str(e) if 'e' in dir() else "unknown"
        await db.commit()
    return False


async def _animate_scene(img_path: Path, duration: int, out_path: Path, ep_id: int) -> bool:
    """Higgsfield image-to-video animation."""
    async with async_session() as db:
        job = Job(episode_id=ep_id, job_type="animate", language="en", status="running")
        db.add(job)
        await db.commit()
        job_id = job.id

    try:
        from higgsfield.client import HiggsFieldClient
        hf = HiggsFieldClient(api_key=settings.higgsfield_api_key)

        result = hf.video.animate(
            image_path=str(img_path),
            duration=duration,
            motion_strength=0.6,
        )

        video_url = result.video_url if hasattr(result, "video_url") else result.get("video_url", "")
        if video_url:
            import httpx
            async with httpx.AsyncClient(timeout=120) as c:
                resp = await c.get(video_url)
                out_path.write_bytes(resp.content)

            async with async_session() as db:
                j = await db.get(Job, job_id)
                j.status = "done"
                j.output_path = str(out_path)
                await db.commit()
            return True

    except Exception as e:
        print(f"[video_producer] Animate failed: {e}")

    async with async_session() as db:
        j = await db.get(Job, job_id)
        j.status = "failed"
        await db.commit()
    return False


async def _stitch_clips(clip_paths: List[Path], out_path: Path):
    """Use ffmpeg to concatenate video clips."""
    concat_file = out_path.parent / "concat.txt"
    concat_file.write_text(
        "\n".join(f"file '{p.resolve()}'" for p in clip_paths)
    )
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",
        str(out_path),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()
    concat_file.unlink(missing_ok=True)


async def _concat_audio(audio_paths: List[Optional[Path]], out_path: Path) -> Optional[Path]:
    """Concatenate MP3 audio files using ffmpeg."""
    valid = [p for p in audio_paths if p and p.exists()]
    if not valid:
        return None

    concat_file = out_path.parent / "audio_concat.txt"
    concat_file.write_text(
        "\n".join(f"file '{p.resolve()}'" for p in valid)
    )
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",
        str(out_path),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()
    concat_file.unlink(missing_ok=True)
    return out_path if out_path.exists() else None
