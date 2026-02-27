import os
import json
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, Depends, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sse_starlette.sse import EventSourceResponse

from config import get_settings
from database import init_db, get_db, Storyboard, Episode, Character, KnowledgeChunk, ChatMessage, Job, UploadSchedule
from core.knowledge_base import ingest_file, ingest_url
from core.story_generator import stream_chat, generate_episode_outlines
from core.character_manager import extract_characters, start_soul_training, generate_hero_image
from core.video_producer import produce_episode
from core.scheduler import start_scheduler, stop_scheduler

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    Path(settings.knowledge_dir).mkdir(exist_ok=True)
    Path(settings.output_dir).mkdir(exist_ok=True)
    start_scheduler()
    yield
    stop_scheduler()


app = FastAPI(title="Clawd Bot", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="web/static"), name="static")
app.mount("/output", StaticFiles(directory="output"), name="output")
templates = Jinja2Templates(directory="web/templates")


# ─────────────────────────────── HOME ────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Storyboard).order_by(Storyboard.created_at.desc()))
    storyboards = result.scalars().all()

    # Attach counts
    boards_data = []
    for sb in storyboards:
        ep_result = await db.execute(select(Episode).where(Episode.storyboard_id == sb.id))
        episodes = ep_result.scalars().all()
        published = sum(1 for e in episodes if e.production_status == "published")
        producing = sum(1 for e in episodes if e.production_status in ("queued", "producing"))
        schedule = (await db.execute(
            select(UploadSchedule).where(UploadSchedule.storyboard_id == sb.id)
        )).scalar_one_or_none()
        boards_data.append({
            "sb": sb,
            "published": published,
            "producing": producing,
            "total": len(episodes),
            "next_upload": schedule.first_upload_at if schedule else None,
        })

    return templates.TemplateResponse("home.html", {"request": request, "boards": boards_data})


@app.post("/storyboards/new")
async def new_storyboard(
    title: str = Form(...),
    description: str = Form(""),
    db: AsyncSession = Depends(get_db),
):
    sb = Storyboard(title=title, description=description)
    db.add(sb)
    await db.commit()
    await db.refresh(sb)
    kb_dir = Path(settings.knowledge_dir) / str(sb.id)
    (kb_dir / "stories").mkdir(parents=True, exist_ok=True)
    (kb_dir / "character_refs").mkdir(parents=True, exist_ok=True)
    return RedirectResponse(f"/storyboards/{sb.id}", status_code=303)


# ─────────────────────────── STORYBOARD ──────────────────────────────

@app.get("/storyboards/{sb_id}", response_class=HTMLResponse)
async def storyboard_page(sb_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    sb = await db.get(Storyboard, sb_id)
    if not sb:
        raise HTTPException(404)
    ep_result = await db.execute(
        select(Episode).where(Episode.storyboard_id == sb_id).order_by(Episode.episode_num)
    )
    episodes = ep_result.scalars().all()
    msg_result = await db.execute(
        select(ChatMessage).where(ChatMessage.storyboard_id == sb_id).order_by(ChatMessage.created_at)
    )
    messages = msg_result.scalars().all()
    schedule = (await db.execute(
        select(UploadSchedule).where(UploadSchedule.storyboard_id == sb_id)
    )).scalar_one_or_none()
    return templates.TemplateResponse("storyboard.html", {
        "request": request, "sb": sb, "episodes": episodes,
        "messages": messages, "schedule": schedule,
    })


@app.post("/storyboards/{sb_id}/upload-knowledge")
async def upload_knowledge(
    sb_id: int,
    files: list[UploadFile] = File(default=[]),
    urls: str = Form(""),
    db: AsyncSession = Depends(get_db),
):
    sb = await db.get(Storyboard, sb_id)
    if not sb:
        raise HTTPException(404)

    dest_dir = Path(settings.knowledge_dir) / str(sb_id) / "stories"
    dest_dir.mkdir(parents=True, exist_ok=True)

    added = 0
    for f in files:
        if not f.filename:
            continue
        dest = dest_dir / f.filename
        content = await f.read()
        dest.write_bytes(content)
        chunks = await ingest_file(dest, sb_id)
        for chunk in chunks:
            db.add(chunk)
        added += len(chunks)

    for url in [u.strip() for u in urls.splitlines() if u.strip()]:
        chunks = await ingest_url(url, sb_id)
        for chunk in chunks:
            db.add(chunk)
        added += len(chunks)

    await db.commit()
    return JSONResponse({"added_chunks": added})


@app.get("/storyboards/{sb_id}/chat/stream")
async def chat_stream(sb_id: int, message: str, request: Request, db: AsyncSession = Depends(get_db)):
    sb = await db.get(Storyboard, sb_id)
    if not sb:
        raise HTTPException(404)

    # Save user message
    user_msg = ChatMessage(storyboard_id=sb_id, role="user", content=message)
    db.add(user_msg)
    await db.commit()

    # Load history + knowledge
    msg_result = await db.execute(
        select(ChatMessage).where(ChatMessage.storyboard_id == sb_id).order_by(ChatMessage.created_at)
    )
    history = msg_result.scalars().all()

    kb_result = await db.execute(
        select(KnowledgeChunk).where(KnowledgeChunk.storyboard_id == sb_id)
    )
    kb_chunks = kb_result.scalars().all()

    async def event_generator():
        full_response = ""
        async for token in stream_chat(sb, history, kb_chunks, message):
            full_response += token
            yield {"data": json.dumps({"token": token})}
            if await request.is_disconnected():
                break

        # Persist assistant message
        async with db.__class__(bind=db.get_bind()) as new_session:
            pass
        # Use a new session to avoid closed session issues
        from database import async_session
        async with async_session() as sess:
            asst_msg = ChatMessage(storyboard_id=sb_id, role="assistant", content=full_response)
            sess.add(asst_msg)
            await sess.commit()

        yield {"data": json.dumps({"done": True})}

    return EventSourceResponse(event_generator())


@app.post("/storyboards/{sb_id}/generate-outlines")
async def generate_outlines(
    sb_id: int,
    num_episodes: int = Form(24),
    db: AsyncSession = Depends(get_db),
):
    sb = await db.get(Storyboard, sb_id)
    if not sb:
        raise HTTPException(404)

    kb_result = await db.execute(
        select(KnowledgeChunk).where(KnowledgeChunk.storyboard_id == sb_id)
    )
    kb_chunks = kb_result.scalars().all()

    outlines = await generate_episode_outlines(sb, kb_chunks, num_episodes)

    # Clear existing draft episodes
    existing = await db.execute(
        select(Episode).where(Episode.storyboard_id == sb_id, Episode.production_status == "draft")
    )
    for ep in existing.scalars().all():
        await db.delete(ep)

    for i, outline in enumerate(outlines, 1):
        ep = Episode(
            storyboard_id=sb_id,
            episode_num=i,
            title=outline.get("title", f"Episode {i}"),
            storyline=outline.get("storyline", ""),
        )
        db.add(ep)

    sb.total_episodes = len(outlines)
    sb.status = "ready"
    await db.commit()
    return JSONResponse({"episodes": outlines})


@app.post("/storyboards/{sb_id}/extract-characters")
async def extract_characters_route(sb_id: int, db: AsyncSession = Depends(get_db)):
    sb = await db.get(Storyboard, sb_id)
    if not sb:
        raise HTTPException(404)
    kb_result = await db.execute(
        select(KnowledgeChunk).where(KnowledgeChunk.storyboard_id == sb_id)
    )
    kb_chunks = kb_result.scalars().all()
    characters = await extract_characters(sb, kb_chunks)
    added = []
    for char in characters:
        existing = (await db.execute(
            select(Character).where(
                Character.storyboard_id == sb_id,
                Character.name == char["name"],
            )
        )).scalar_one_or_none()
        if not existing:
            c = Character(
                storyboard_id=sb_id,
                name=char["name"],
                description=char.get("description", ""),
            )
            db.add(c)
            added.append(char["name"])
    await db.commit()
    return JSONResponse({"added": added})


@app.post("/storyboards/{sb_id}/schedule")
async def set_schedule(
    sb_id: int,
    first_upload_at: str = Form(...),
    interval_days: int = Form(7),
    timezone: str = Form("UTC"),
    db: AsyncSession = Depends(get_db),
):
    from datetime import datetime
    schedule = (await db.execute(
        select(UploadSchedule).where(UploadSchedule.storyboard_id == sb_id)
    )).scalar_one_or_none()

    dt = datetime.fromisoformat(first_upload_at)
    if schedule:
        schedule.first_upload_at = dt
        schedule.interval_days = interval_days
        schedule.timezone = timezone
    else:
        schedule = UploadSchedule(
            storyboard_id=sb_id,
            first_upload_at=dt,
            interval_days=interval_days,
            timezone=timezone,
        )
        db.add(schedule)
    await db.commit()
    return JSONResponse({"ok": True})


# ─────────────────────────── EPISODES ────────────────────────────────

@app.get("/episodes/{ep_id}", response_class=HTMLResponse)
async def episode_page(ep_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    ep = await db.get(Episode, ep_id)
    if not ep:
        raise HTTPException(404)
    sb = await db.get(Storyboard, ep.storyboard_id)
    jobs_result = await db.execute(select(Job).where(Job.episode_id == ep_id))
    jobs = jobs_result.scalars().all()
    return templates.TemplateResponse("episode.html", {
        "request": request, "ep": ep, "sb": sb, "jobs": jobs,
        "script": json.loads(ep.script_json) if ep.script_json else [],
        "translations": json.loads(ep.narration_translations) if ep.narration_translations else {},
        "youtube_urls": json.loads(ep.youtube_urls_json) if ep.youtube_urls_json else {},
    })


@app.post("/episodes/{ep_id}/update-storyline")
async def update_storyline(
    ep_id: int,
    storyline: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    ep = await db.get(Episode, ep_id)
    if not ep:
        raise HTTPException(404)
    ep.storyline = storyline
    await db.commit()
    return JSONResponse({"ok": True})


@app.post("/episodes/{ep_id}/approve")
async def approve_episode(ep_id: int, db: AsyncSession = Depends(get_db)):
    ep = await db.get(Episode, ep_id)
    if not ep:
        raise HTTPException(404)
    ep.production_status = "approved"
    await db.commit()
    return JSONResponse({"ok": True})


@app.post("/episodes/{ep_id}/produce")
async def produce(ep_id: int, db: AsyncSession = Depends(get_db)):
    ep = await db.get(Episode, ep_id)
    if not ep:
        raise HTTPException(404)
    ep.production_status = "queued"
    await db.commit()
    asyncio.create_task(produce_episode(ep_id))
    return JSONResponse({"ok": True, "status": "queued"})


# ─────────────────────────── CHARACTERS ──────────────────────────────

@app.get("/storyboards/{sb_id}/characters", response_class=HTMLResponse)
async def characters_page(sb_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    sb = await db.get(Storyboard, sb_id)
    if not sb:
        raise HTTPException(404)
    result = await db.execute(
        select(Character).where(Character.storyboard_id == sb_id).order_by(Character.name)
    )
    characters = result.scalars().all()
    return templates.TemplateResponse("characters.html", {
        "request": request, "sb": sb, "characters": characters,
    })


@app.post("/characters/{char_id}/upload-refs")
async def upload_ref_images(
    char_id: int,
    images: list[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
):
    char = await db.get(Character, char_id)
    if not char:
        raise HTTPException(404)
    dest_dir = Path(settings.knowledge_dir) / str(char.storyboard_id) / "character_refs" / char.name.lower().replace(" ", "_")
    dest_dir.mkdir(parents=True, exist_ok=True)
    for img in images:
        if img.filename:
            data = await img.read()
            (dest_dir / img.filename).write_bytes(data)
    char.ref_images_dir = str(dest_dir)
    char.training_status = "ready_to_train"
    await db.commit()
    return JSONResponse({"ok": True, "dir": str(dest_dir)})


@app.post("/characters/{char_id}/train")
async def train_character(char_id: int, db: AsyncSession = Depends(get_db)):
    char = await db.get(Character, char_id)
    if not char:
        raise HTTPException(404)
    char.training_status = "training"
    await db.commit()
    asyncio.create_task(start_soul_training(char_id))
    return JSONResponse({"ok": True})


@app.post("/characters/{char_id}/hero-image")
async def hero_image(char_id: int, db: AsyncSession = Depends(get_db)):
    char = await db.get(Character, char_id)
    if not char:
        raise HTTPException(404)
    asyncio.create_task(generate_hero_image(char_id))
    return JSONResponse({"ok": True})


@app.get("/characters/{char_id}/status")
async def character_status(char_id: int, db: AsyncSession = Depends(get_db)):
    char = await db.get(Character, char_id)
    if not char:
        raise HTTPException(404)
    return JSONResponse({
        "training_status": char.training_status,
        "soul_id": char.higgsfield_soul_id,
        "hero_image": char.hero_image_path,
    })


# ─────────────────────────── JOBS API ────────────────────────────────

@app.get("/episodes/{ep_id}/jobs")
async def episode_jobs(ep_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Job).where(Job.episode_id == ep_id).order_by(Job.created_at))
    jobs = result.scalars().all()
    return JSONResponse([{
        "id": j.id, "job_type": j.job_type, "language": j.language,
        "status": j.status, "output_path": j.output_path, "error_msg": j.error_msg,
    } for j in jobs])
