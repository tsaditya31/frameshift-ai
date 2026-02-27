"""APScheduler-based job scheduling for episode production and uploads."""
import asyncio
from datetime import datetime, timedelta

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from database import async_session, Episode, UploadSchedule, Storyboard
from sqlalchemy import select

_scheduler = AsyncIOScheduler()


def start_scheduler():
    _scheduler.add_job(
        _check_scheduled_uploads,
        trigger=IntervalTrigger(minutes=5),
        id="check_uploads",
        replace_existing=True,
    )
    _scheduler.add_job(
        _advance_production_queue,
        trigger=IntervalTrigger(minutes=10),
        id="production_queue",
        replace_existing=True,
    )
    _scheduler.start()


def stop_scheduler():
    if _scheduler.running:
        _scheduler.shutdown(wait=False)


async def _check_scheduled_uploads():
    """Check for episodes scheduled to be uploaded now."""
    from core.youtube_uploader import upload_episode

    now = datetime.utcnow()
    async with async_session() as db:
        result = await db.execute(
            select(Episode).where(
                Episode.production_status == "done",
                Episode.scheduled_upload_at <= now,
            )
        )
        episodes = result.scalars().all()

    for ep in episodes:
        print(f"[scheduler] Uploading episode {ep.id} (scheduled at {ep.scheduled_upload_at})")
        await upload_episode(ep.id)


async def _advance_production_queue():
    """Find approved episodes and kick off production if not already running."""
    from core.video_producer import produce_episode

    async with async_session() as db:
        result = await db.execute(
            select(Episode).where(Episode.production_status == "approved")
        )
        episodes = result.scalars().all()

    for ep in episodes:
        print(f"[scheduler] Queuing production for episode {ep.id}")
        asyncio.create_task(produce_episode(ep.id))


async def schedule_storyboard_uploads(storyboard_id: int):
    """Set upload times for all done episodes based on the storyboard schedule."""
    async with async_session() as db:
        schedule = (await db.execute(
            select(UploadSchedule).where(UploadSchedule.storyboard_id == storyboard_id)
        )).scalar_one_or_none()

        if not schedule or not schedule.first_upload_at:
            return

        done_episodes = (await db.execute(
            select(Episode).where(
                Episode.storyboard_id == storyboard_id,
                Episode.production_status == "done",
            ).order_by(Episode.episode_num)
        )).scalars().all()

        upload_time = schedule.first_upload_at
        for ep in done_episodes:
            if not ep.scheduled_upload_at:
                ep.scheduled_upload_at = upload_time
                upload_time += timedelta(days=schedule.interval_days)

        await db.commit()
