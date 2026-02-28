"""End-to-end video production pipeline for an episode."""
import json
import asyncio
from pathlib import Path
from typing import List, Optional

from config import get_settings
from database import async_session, Episode, Job, Character, KnowledgeChunk, Storyboard
from sqlalchemy import select
from core.story_generator import generate_scene_script
from core.translator import translate_narrations, SUPPORTED_LANGUAGES
from core.audio_generator import generate_episode_audio

settings = get_settings()

# Ken Burns effect types — cycled across scenes for visual variety
KEN_BURNS_EFFECTS = [
    "zoom_in",
    "zoom_out",
    "pan_left",
    "pan_right",
    "pan_up",
    "zoom_in_left",
    "zoom_in_right",
]


async def produce_episode(ep_id: int):
    """Main pipeline: script → images → Ken Burns → audio → composite → per-language outputs."""
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

        # Step 2 & 3: Generate scene images + Ken Burns animate
        scene_clips = await _produce_base_video(ep_id, scenes, soul_map, out_dir)

        # Step 4: English narration audio
        en_audio_paths = await generate_episode_audio(scenes, "en", out_dir / "audio")

        # Step 5: Composite English video (per-scene video+audio → concat)
        en_out = out_dir / "en.mp4"
        await _produce_language_video(scene_clips, en_audio_paths, en_out)

        # Step 6: Translate narrations to Indic languages
        translated = await translate_narrations(scenes)

        async with async_session() as db:
            ep = await db.get(Episode, ep_id)
            ep.narration_translations = json.dumps({
                lang: [s.get("narration", "") for s in sc]
                for lang, sc in translated.items()
            })
            await db.commit()

        # Step 7: Per-language: ElevenLabs audio → composite → concat
        for lang_code in SUPPORTED_LANGUAGES:
            lang_scenes = translated.get(lang_code, scenes)
            lang_audio_paths = await generate_episode_audio(lang_scenes, lang_code, out_dir / "audio")
            lang_out = out_dir / f"{lang_code}.mp4"
            await _produce_language_video(scene_clips, lang_audio_paths, lang_out)

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
) -> List[dict]:
    """Generate still images then Ken Burns animate each scene. Returns list of scene clip metadata."""
    scene_clips = []
    for i, scene in enumerate(scenes):
        scene_num = scene.get("scene", i + 1)
        img_path = out_dir / f"scene_{scene_num:02d}.jpg"
        clip_path = out_dir / f"scene_{scene_num:02d}.mp4"
        duration = scene.get("duration_seconds", 5)
        effect = KEN_BURNS_EFFECTS[i % len(KEN_BURNS_EFFECTS)]

        # Generate still image
        img_ok = await _generate_scene_image(scene, soul_map, img_path, ep_id)
        if not img_ok:
            continue

        # Ken Burns animate still → clip
        clip_ok = await _ken_burns_animate(img_path, duration, clip_path, ep_id, effect)
        if clip_ok:
            scene_clips.append({
                "scene_num": scene_num,
                "clip_path": clip_path,
                "duration": duration,
            })

    return scene_clips


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


async def _ken_burns_animate(
    img_path: Path,
    duration: int,
    out_path: Path,
    ep_id: int,
    effect: str = "zoom_in",
) -> bool:
    """Apply Ken Burns (pan/zoom) effect to a still image using FFmpeg."""
    async with async_session() as db:
        job = Job(episode_id=ep_id, job_type="ken_burns", language="en", status="running")
        db.add(job)
        await db.commit()
        job_id = job.id

    fps = 24
    frames = duration * fps

    # Build zoompan filter based on effect type
    effects = {
        "zoom_in": (
            f"z='1+0.3*on/{frames}':"
            f"x='iw/2-(iw/zoom/2)':"
            f"y='ih/2-(ih/zoom/2)'"
        ),
        "zoom_out": (
            f"z='1.3-0.3*on/{frames}':"
            f"x='iw/2-(iw/zoom/2)':"
            f"y='ih/2-(ih/zoom/2)'"
        ),
        "pan_left": (
            f"z='1.1':"
            f"x='iw*0.1*(1-on/{frames})':"
            f"y='ih/2-(ih/zoom/2)'"
        ),
        "pan_right": (
            f"z='1.1':"
            f"x='iw*0.1*on/{frames}':"
            f"y='ih/2-(ih/zoom/2)'"
        ),
        "pan_up": (
            f"z='1.1':"
            f"x='iw/2-(iw/zoom/2)':"
            f"y='ih*0.1*(1-on/{frames})'"
        ),
        "zoom_in_left": (
            f"z='1+0.3*on/{frames}':"
            f"x='iw*0.3*(1-on/{frames})':"
            f"y='ih/2-(ih/zoom/2)'"
        ),
        "zoom_in_right": (
            f"z='1+0.3*on/{frames}':"
            f"x='iw*0.3*on/{frames}':"
            f"y='ih/2-(ih/zoom/2)'"
        ),
    }

    zp = effects.get(effect, effects["zoom_in"])
    vf = f"zoompan={zp}:d={frames}:s=1280x720:fps={fps}"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(img_path),
        "-vf", vf,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()

        if proc.returncode == 0 and out_path.exists():
            async with async_session() as db:
                j = await db.get(Job, job_id)
                j.status = "done"
                j.output_path = str(out_path)
                await db.commit()
            return True
        else:
            print(f"[video_producer] Ken Burns FFmpeg failed: {stderr.decode()}")

    except Exception as e:
        print(f"[video_producer] Ken Burns failed: {e}")

    async with async_session() as db:
        j = await db.get(Job, job_id)
        j.status = "failed"
        await db.commit()
    return False


async def _composite_scene_with_audio(
    video_path: Path,
    audio_path: Path,
    out_path: Path,
) -> bool:
    """Mux a video clip with an audio file. Extends video if audio is longer."""
    cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-shortest",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()

    if proc.returncode == 0 and out_path.exists():
        return True

    print(f"[video_producer] Composite failed: {stderr.decode()}")
    return False


async def _produce_language_video(
    scene_clips: List[dict],
    audio_paths: List[Optional[Path]],
    out_path: Path,
) -> bool:
    """Composite each scene clip with its audio, then concatenate into a final video."""
    composited = []
    tmp_dir = out_path.parent / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for i, clip_info in enumerate(scene_clips):
        clip_path = clip_info["clip_path"]
        audio = audio_paths[i] if i < len(audio_paths) else None

        if audio and audio.exists():
            comp_path = tmp_dir / f"comp_{clip_info['scene_num']:02d}_{out_path.stem}.mp4"
            ok = await _composite_scene_with_audio(clip_path, audio, comp_path)
            if ok:
                composited.append(comp_path)
                continue

        # No audio or composite failed — use silent clip as-is
        composited.append(clip_path)

    if not composited:
        return False

    if len(composited) == 1:
        # Single scene — just copy
        import shutil
        shutil.copy2(composited[0], out_path)
        _cleanup_tmp(tmp_dir)
        return True

    # Concatenate all composited clips
    await _stitch_clips(composited, out_path)
    _cleanup_tmp(tmp_dir)
    return out_path.exists()


def _cleanup_tmp(tmp_dir: Path):
    """Remove temporary composited clips."""
    if tmp_dir.exists():
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


async def _stitch_clips(clip_paths: List[Path], out_path: Path):
    """Use ffmpeg to concatenate video clips."""
    concat_file = out_path.parent / f"concat_{out_path.stem}.txt"
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
