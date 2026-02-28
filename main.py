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
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
import re
from sqlalchemy import select, func, delete
from sse_starlette.sse import EventSourceResponse

from config import get_settings
from database import init_db, get_db, Storyboard, Episode, Character, KnowledgeChunk, ChatMessage, Job, UploadSchedule, User
from auth import router as auth_router
from core.knowledge_base import ingest_file, ingest_url
from core.story_generator import stream_chat, stream_episode_chat, generate_episode_outlines
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


app = FastAPI(title="Frameshift AI", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="web/static"), name="static")
app.mount("/output", StaticFiles(directory="output"), name="output")
app.mount("/knowledge", StaticFiles(directory="knowledge"), name="knowledge")
templates = Jinja2Templates(directory="web/templates")

app.include_router(auth_router)

# ─── Auth middleware ──────────────────────────────────────────────────
# NOTE: require_auth is registered first so that SessionMiddleware (added
# below via add_middleware) ends up outermost and runs first, populating
# request.session before require_auth inspects it.

PUBLIC_PATHS = {
    "/login", "/register", "/forgot-password", "/reset-password",
    "/auth/google", "/auth/google/callback", "/static", "/favicon.ico",
}


@app.middleware("http")
async def require_auth(request: Request, call_next):
    path = request.url.path
    if any(path == p or path.startswith(p) for p in PUBLIC_PATHS):
        return await call_next(request)
    if not request.session.get("user_id"):
        return RedirectResponse("/login", status_code=303)
    return await call_next(request)


# SessionMiddleware must be added AFTER require_auth so it wraps it (runs first).
app.add_middleware(SessionMiddleware, secret_key=settings.secret_key)


# ─── Current user dependency ─────────────────────────────────────────

async def get_current_user(request: Request, db: AsyncSession = Depends(get_db)) -> User:
    user_id = request.session.get("user_id")
    if not user_id:
        raise HTTPException(401)
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(401)
    return user


# ─── Ownership helper ────────────────────────────────────────────────

async def get_owned_storyboard(sb_id: int, user: User, db: AsyncSession) -> Storyboard:
    sb = await db.get(Storyboard, sb_id)
    if not sb or sb.user_id != user.id:
        raise HTTPException(404)
    return sb


# ─────────────────────────────── HOME ────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Storyboard)
        .where(Storyboard.user_id == user.id)
        .order_by(Storyboard.created_at.desc())
    )
    storyboards = result.scalars().all()

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

    return templates.TemplateResponse("home.html", {"request": request, "boards": boards_data, "user": user})


@app.post("/storyboards/new")
async def new_storyboard(
    title: str = Form(...),
    description: str = Form(""),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    sb = Storyboard(title=title, description=description, user_id=user.id)
    db.add(sb)
    await db.commit()
    await db.refresh(sb)
    kb_dir = Path(settings.knowledge_dir) / str(sb.id)
    (kb_dir / "stories").mkdir(parents=True, exist_ok=True)
    (kb_dir / "character_refs").mkdir(parents=True, exist_ok=True)
    return RedirectResponse(f"/storyboards/{sb.id}", status_code=303)


# ─────────────────────────── STORYBOARD ──────────────────────────────

@app.get("/storyboards/{sb_id}", response_class=HTMLResponse)
async def storyboard_page(
    sb_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    sb = await get_owned_storyboard(sb_id, user, db)
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
        "messages": messages, "schedule": schedule, "user": user,
    })


@app.post("/storyboards/{sb_id}/upload-knowledge")
async def upload_knowledge(
    sb_id: int,
    files: list[UploadFile] = File(default=[]),
    urls: str = Form(""),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    sb = await get_owned_storyboard(sb_id, user, db)

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


@app.get("/storyboards/{sb_id}/knowledge-files")
async def knowledge_files(
    sb_id: int,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    await get_owned_storyboard(sb_id, user, db)
    result = await db.execute(
        select(
            KnowledgeChunk.source_file,
            KnowledgeChunk.chunk_type,
            func.count(KnowledgeChunk.id).label("count"),
        )
        .where(KnowledgeChunk.storyboard_id == sb_id)
        .group_by(KnowledgeChunk.source_file, KnowledgeChunk.chunk_type)
    )
    rows = result.all()
    files = {}
    for source_file, chunk_type, count in rows:
        if source_file not in files:
            files[source_file] = {"source_file": source_file, "chunk_type": chunk_type, "chunks": 0}
        files[source_file]["chunks"] += count
        files[source_file]["chunk_type"] = chunk_type
    return JSONResponse(list(files.values()))


@app.delete("/storyboards/{sb_id}/knowledge-files")
async def delete_knowledge_file(
    sb_id: int,
    source_file: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    await get_owned_storyboard(sb_id, user, db)
    result = await db.execute(
        select(KnowledgeChunk).where(
            KnowledgeChunk.storyboard_id == sb_id,
            KnowledgeChunk.source_file == source_file,
        )
    )
    chunks = result.scalars().all()
    for chunk in chunks:
        await db.delete(chunk)
    await db.commit()
    return JSONResponse({"ok": True, "deleted": len(chunks)})


@app.get("/storyboards/{sb_id}/chat/stream")
async def chat_stream(
    sb_id: int,
    message: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    sb = await get_owned_storyboard(sb_id, user, db)

    user_msg = ChatMessage(storyboard_id=sb_id, role="user", content=message)
    db.add(user_msg)
    await db.commit()

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
    user: User = Depends(get_current_user),
):
    sb = await get_owned_storyboard(sb_id, user, db)

    kb_result = await db.execute(
        select(KnowledgeChunk).where(KnowledgeChunk.storyboard_id == sb_id)
    )
    kb_chunks = kb_result.scalars().all()

    outlines = await generate_episode_outlines(sb, kb_chunks, num_episodes)

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


@app.post("/storyboards/{sb_id}/parse-episodes-from-chat")
async def parse_episodes_from_chat(
    sb_id: int,
    content: str = Form(...),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    sb = await get_owned_storyboard(sb_id, user, db)

    episodes = []

    # Try combined JSON object format first: {"episodes": [...], "characters": [...]}
    obj_match = re.search(r"\{.*\}", content, re.DOTALL)
    if obj_match:
        try:
            parsed_obj = json.loads(obj_match.group())
            if isinstance(parsed_obj, dict) and "episodes" in parsed_obj:
                for item in parsed_obj["episodes"]:
                    episodes.append({
                        "title": item.get("title", ""),
                        "storyline": item.get("storyline", item.get("summary", item.get("description", ""))),
                    })
        except json.JSONDecodeError:
            pass

    # Fall back to JSON array format: [{"title": ..., "storyline": ...}]
    if not episodes:
        json_match = re.search(r"\[.*\]", content, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
                    for item in parsed:
                        episodes.append({
                            "title": item.get("title", ""),
                            "storyline": item.get("storyline", item.get("summary", item.get("description", ""))),
                        })
            except json.JSONDecodeError:
                pass

    if not episodes:
        pattern = r"(?:^|\n)\s*(\d+)\.\s*\*{0,2}(.+?)\*{0,2}\s*[-:–]\s*(.+?)(?=\n\s*\d+\.|\n*$)"
        matches = re.findall(pattern, content, re.DOTALL)
        for num, title, storyline in matches:
            episodes.append({"title": title.strip(), "storyline": storyline.strip()})

    if not episodes:
        return JSONResponse({"error": "Could not parse episodes from content"}, status_code=400)

    existing = await db.execute(
        select(Episode).where(Episode.storyboard_id == sb_id, Episode.production_status == "draft")
    )
    for ep in existing.scalars().all():
        await db.delete(ep)

    created = []
    for i, ep_data in enumerate(episodes, 1):
        ep = Episode(
            storyboard_id=sb_id,
            episode_num=i,
            title=ep_data["title"],
            storyline=ep_data["storyline"],
        )
        db.add(ep)
        created.append({"episode": i, "title": ep_data["title"], "storyline": ep_data["storyline"]})

    sb.total_episodes = len(episodes)
    await db.commit()
    return JSONResponse({"episodes": created})


@app.post("/storyboards/{sb_id}/extract-characters")
async def extract_characters_route(
    sb_id: int,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    sb = await get_owned_storyboard(sb_id, user, db)
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


@app.post("/storyboards/{sb_id}/parse-characters-from-chat")
async def parse_characters_from_chat(
    sb_id: int,
    content: str = Form(...),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    sb = await get_owned_storyboard(sb_id, user, db)

    characters = []

    # Try combined JSON object format first: {"episodes": [...], "characters": [...]}
    obj_match = re.search(r"\{.*\}", content, re.DOTALL)
    if obj_match:
        try:
            parsed_obj = json.loads(obj_match.group())
            if isinstance(parsed_obj, dict) and "characters" in parsed_obj:
                for item in parsed_obj["characters"]:
                    if isinstance(item, dict) and "name" in item:
                        characters.append({
                            "name": item["name"],
                            "description": item.get("description", ""),
                        })
        except json.JSONDecodeError:
            pass

    # Fall back to JSON array format: [{"name": ..., "description": ...}]
    if not characters:
        json_match = re.search(r"\[.*\]", content, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and "name" in item:
                            characters.append({
                                "name": item["name"],
                                "description": item.get("description", ""),
                            })
            except json.JSONDecodeError:
                pass

    if not characters:
        return JSONResponse({"error": "Could not parse characters from content"}, status_code=400)

    added = []
    for char_data in characters:
        existing = (await db.execute(
            select(Character).where(
                Character.storyboard_id == sb_id,
                Character.name == char_data["name"],
            )
        )).scalar_one_or_none()
        if not existing:
            c = Character(
                storyboard_id=sb_id,
                name=char_data["name"],
                description=char_data["description"],
            )
            db.add(c)
            added.append(char_data["name"])
    await db.commit()
    return JSONResponse({"added": added})


@app.get("/storyboards/{sb_id}/characters-list")
async def characters_list(
    sb_id: int,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    await get_owned_storyboard(sb_id, user, db)
    result = await db.execute(
        select(Character).where(Character.storyboard_id == sb_id).order_by(Character.name)
    )
    characters = result.scalars().all()
    return JSONResponse([{
        "id": c.id,
        "name": c.name,
        "description": c.description,
        "training_status": c.training_status,
        "locked": c.locked,
    } for c in characters])


@app.get("/storyboards/{sb_id}/editor", response_class=HTMLResponse)
async def storyboard_editor(
    sb_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    sb = await get_owned_storyboard(sb_id, user, db)
    ep_result = await db.execute(
        select(Episode).where(Episode.storyboard_id == sb_id).order_by(Episode.episode_num)
    )
    episodes = ep_result.scalars().all()
    return templates.TemplateResponse("storyboard_editor.html", {
        "request": request, "sb": sb, "episodes": episodes, "user": user,
    })


@app.post("/storyboards/{sb_id}/characters/new")
async def new_character(
    sb_id: int,
    name: str = Form(...),
    description: str = Form(""),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    await get_owned_storyboard(sb_id, user, db)
    c = Character(storyboard_id=sb_id, name=name, description=description)
    db.add(c)
    await db.commit()
    return RedirectResponse(f"/storyboards/{sb_id}/characters", status_code=303)


@app.post("/characters/{char_id}/update")
async def update_character(
    char_id: int,
    name: str = Form(...),
    description: str = Form(""),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    char = await db.get(Character, char_id)
    if not char:
        raise HTTPException(404)
    await get_owned_storyboard(char.storyboard_id, user, db)
    if char.locked:
        return JSONResponse({"error": "Character is locked after Soul ID training"}, status_code=400)
    char.name = name
    char.description = description
    await db.commit()
    return JSONResponse({"ok": True})


@app.post("/storyboards/{sb_id}/schedule")
async def set_schedule(
    sb_id: int,
    first_upload_at: str = Form(...),
    interval_days: int = Form(7),
    timezone: str = Form("UTC"),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    from datetime import datetime
    await get_owned_storyboard(sb_id, user, db)
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

async def get_owned_episode(ep_id: int, user: User, db: AsyncSession) -> Episode:
    ep = await db.get(Episode, ep_id)
    if not ep:
        raise HTTPException(404)
    await get_owned_storyboard(ep.storyboard_id, user, db)
    return ep


@app.get("/episodes/{ep_id}", response_class=HTMLResponse)
async def episode_page(
    ep_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    ep = await get_owned_episode(ep_id, user, db)
    sb = await db.get(Storyboard, ep.storyboard_id)
    jobs_result = await db.execute(select(Job).where(Job.episode_id == ep_id))
    jobs = jobs_result.scalars().all()
    msg_result = await db.execute(
        select(ChatMessage).where(ChatMessage.episode_id == ep_id).order_by(ChatMessage.created_at)
    )
    messages = msg_result.scalars().all()
    return templates.TemplateResponse("episode.html", {
        "request": request, "ep": ep, "sb": sb, "jobs": jobs, "messages": messages,
        "script": json.loads(ep.script_json) if ep.script_json else [],
        "translations": json.loads(ep.narration_translations) if ep.narration_translations else {},
        "youtube_urls": json.loads(ep.youtube_urls_json) if ep.youtube_urls_json else {},
        "user": user,
    })


@app.get("/episodes/{ep_id}/chat/stream")
async def episode_chat_stream(
    ep_id: int,
    message: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    ep = await get_owned_episode(ep_id, user, db)
    sb = await db.get(Storyboard, ep.storyboard_id)

    user_msg = ChatMessage(storyboard_id=ep.storyboard_id, episode_id=ep_id, role="user", content=message)
    db.add(user_msg)
    await db.commit()

    msg_result = await db.execute(
        select(ChatMessage).where(ChatMessage.episode_id == ep_id).order_by(ChatMessage.created_at)
    )
    history = msg_result.scalars().all()

    kb_result = await db.execute(
        select(KnowledgeChunk).where(KnowledgeChunk.storyboard_id == ep.storyboard_id)
    )
    kb_chunks = kb_result.scalars().all()

    async def event_generator():
        full_response = ""
        async for token in stream_episode_chat(sb, ep, history, kb_chunks, message):
            full_response += token
            yield {"data": json.dumps({"token": token})}
            if await request.is_disconnected():
                break

        from database import async_session
        async with async_session() as sess:
            asst_msg = ChatMessage(storyboard_id=ep.storyboard_id, episode_id=ep_id, role="assistant", content=full_response)
            sess.add(asst_msg)
            await sess.commit()

        yield {"data": json.dumps({"done": True})}

    return EventSourceResponse(event_generator())


@app.post("/episodes/{ep_id}/update-storyline")
async def update_storyline(
    ep_id: int,
    storyline: str = Form(...),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    ep = await get_owned_episode(ep_id, user, db)
    ep.storyline = storyline
    await db.commit()
    return JSONResponse({"ok": True})


@app.post("/episodes/{ep_id}/approve")
async def approve_episode(
    ep_id: int,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    ep = await get_owned_episode(ep_id, user, db)
    ep.production_status = "approved"
    await db.commit()
    return JSONResponse({"ok": True})


@app.post("/episodes/{ep_id}/produce")
async def produce(
    ep_id: int,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    ep = await get_owned_episode(ep_id, user, db)
    ep.production_status = "queued"
    await db.commit()
    asyncio.create_task(produce_episode(ep_id))
    return JSONResponse({"ok": True, "status": "queued"})


# ─────────────────────────── CHARACTERS ──────────────────────────────

async def get_owned_character(char_id: int, user: User, db: AsyncSession) -> Character:
    char = await db.get(Character, char_id)
    if not char:
        raise HTTPException(404)
    await get_owned_storyboard(char.storyboard_id, user, db)
    return char


@app.get("/storyboards/{sb_id}/characters", response_class=HTMLResponse)
async def characters_page(
    sb_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    sb = await get_owned_storyboard(sb_id, user, db)
    result = await db.execute(
        select(Character).where(Character.storyboard_id == sb_id).order_by(Character.name)
    )
    characters = result.scalars().all()
    return templates.TemplateResponse("characters.html", {
        "request": request, "sb": sb, "characters": characters, "user": user,
    })


@app.get("/characters/{char_id}", response_class=HTMLResponse)
async def character_detail_page(
    char_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    char = await get_owned_character(char_id, user, db)
    sb = await db.get(Storyboard, char.storyboard_id)

    all_chars_result = await db.execute(
        select(Character).where(Character.storyboard_id == char.storyboard_id).order_by(Character.name)
    )
    all_chars = all_chars_result.scalars().all()
    char_ids = [c.id for c in all_chars]
    current_idx = char_ids.index(char_id) if char_id in char_ids else 0
    prev_id = char_ids[current_idx - 1] if current_idx > 0 else None
    next_id = char_ids[current_idx + 1] if current_idx < len(char_ids) - 1 else None

    ref_images = []
    if char.ref_images_dir:
        ref_dir = Path(char.ref_images_dir)
        if ref_dir.exists():
            ref_images = [f.name for f in ref_dir.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp')]

    return templates.TemplateResponse("character_detail.html", {
        "request": request, "char": char, "sb": sb,
        "ref_images": ref_images,
        "current_num": current_idx + 1, "total_chars": len(char_ids),
        "prev_id": prev_id, "next_id": next_id,
        "user": user,
    })


@app.post("/characters/{char_id}/update-profile")
async def update_character_profile(
    char_id: int,
    name: str = Form(...),
    description: str = Form(""),
    profile_prompt: str = Form(""),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    char = await get_owned_character(char_id, user, db)
    if char.locked:
        return JSONResponse({"error": "Character is locked after Soul ID training"}, status_code=400)
    char.name = name
    char.description = description
    char.profile_prompt = profile_prompt
    await db.commit()
    return JSONResponse({"ok": True})


@app.post("/characters/{char_id}/upload-refs")
async def upload_ref_images(
    char_id: int,
    images: list[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    char = await get_owned_character(char_id, user, db)
    if char.locked:
        return JSONResponse({"error": "Character is locked after Soul ID training"}, status_code=400)
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
async def train_character(
    char_id: int,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    char = await get_owned_character(char_id, user, db)
    char.training_status = "training"
    await db.commit()
    asyncio.create_task(start_soul_training(char_id))
    return JSONResponse({"ok": True})


@app.post("/characters/{char_id}/hero-image")
async def hero_image(
    char_id: int,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    await get_owned_character(char_id, user, db)
    asyncio.create_task(generate_hero_image(char_id))
    return JSONResponse({"ok": True})


@app.post("/characters/{char_id}/unlock")
async def unlock_character(
    char_id: int,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    char = await get_owned_character(char_id, user, db)
    char.locked = 0
    char.training_status = "needs_images"
    char.higgsfield_soul_id = ""
    await db.commit()
    return JSONResponse({"ok": True})


@app.get("/characters/{char_id}/status")
async def character_status(
    char_id: int,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    char = await get_owned_character(char_id, user, db)
    return JSONResponse({
        "training_status": char.training_status,
        "soul_id": char.higgsfield_soul_id,
        "hero_image": char.hero_image_path,
        "locked": char.locked,
    })


# ─────────────────────────── JOBS API ────────────────────────────────

@app.get("/episodes/{ep_id}/jobs")
async def episode_jobs(
    ep_id: int,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    await get_owned_episode(ep_id, user, db)
    result = await db.execute(select(Job).where(Job.episode_id == ep_id).order_by(Job.created_at))
    jobs = result.scalars().all()
    return JSONResponse([{
        "id": j.id, "job_type": j.job_type, "language": j.language,
        "status": j.status, "output_path": j.output_path, "error_msg": j.error_msg,
    } for j in jobs])
