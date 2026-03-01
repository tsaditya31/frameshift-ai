"""Character extraction and Higgsfield Soul ID management."""
import json
import re
import asyncio
from pathlib import Path
from typing import List, Optional

import anthropic
import httpx

from config import get_settings
from database import async_session, Character, KnowledgeChunk, Storyboard
from sqlalchemy import select

settings = get_settings()
client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

HIGGSFIELD_BASE = "https://platform.higgsfield.ai"


def _hf_headers() -> dict:
    return {
        "hf-api-key": settings.higgsfield_api_key,
        "hf-secret": settings.higgsfield_api_secret,
        "Content-Type": "application/json",
    }


async def extract_characters(sb: Storyboard, kb_chunks: List[KnowledgeChunk]) -> List[dict]:
    """Use Claude to identify key characters from the knowledge base."""
    kb_text = "\n\n".join(c.content_text for c in kb_chunks[:20])[:8000]

    msg = await client.messages.create(
        model=settings.model,
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": (
                f"From this knowledge base about '{sb.title}', identify the main characters.\n\n"
                f"{kb_text}\n\n"
                "Return ONLY a JSON array:\n"
                '[{"name": "CharacterName", "description": "Brief physical + personality description"}]'
            ),
        }],
    )
    text = msg.content[0].text.strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return []


def _get_public_base_url() -> str:
    """Get the app's public base URL (Railway or localhost)."""
    import os
    domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN", "")
    if domain:
        return f"https://{domain}"
    port = settings.port
    return f"http://localhost:{port}"


def _local_path_to_public_url(image_path: Path) -> Optional[str]:
    """Convert a local image path to a public URL served by the app."""
    # Images are under knowledge/ which is mounted at /knowledge
    path_str = str(image_path)
    knowledge_dir = settings.knowledge_dir
    if knowledge_dir in path_str:
        relative = path_str.split(knowledge_dir, 1)[1].lstrip("/")
        base = _get_public_base_url()
        return f"{base}/knowledge/{relative}"
    return None


async def start_soul_training(char_id: int):
    """Create a Higgsfield character reference (Soul ID) from uploaded images."""
    async with async_session() as db:
        char = await db.get(Character, char_id)
        if not char or not char.ref_images_dir:
            return

        ref_dir = Path(char.ref_images_dir)
        image_paths = list(ref_dir.glob("*.*"))[:5]  # Max 5 images
        if not image_paths:
            async with async_session() as sess:
                c = await sess.get(Character, char_id)
                c.training_status = "failed"
                await sess.commit()
            return

        char.training_status = "training"
        await db.commit()

    try:
        # Convert local paths to public URLs (served by the app)
        image_urls = []
        for img_path in image_paths:
            url = _local_path_to_public_url(img_path)
            if url:
                image_urls.append(url)

        if not image_urls:
            raise ValueError("No images have public URLs — is RAILWAY_PUBLIC_DOMAIN set?")

        # Create character reference via Higgsfield API
        print(f"[character_manager] Creating character '{char.name}' with {len(image_urls)} images: {image_urls}")
        async with httpx.AsyncClient(timeout=30) as http:
            resp = await http.post(
                f"{HIGGSFIELD_BASE}/v1/custom-references",
                headers=_hf_headers(),
                json={"name": char.name, "image_urls": image_urls},
            )
            print(f"[character_manager] Higgsfield response {resp.status_code}: {resp.text}")
            resp.raise_for_status()
            data = resp.json()

        character_id = data.get("id") or data.get("character_id", "")
        if not character_id:
            raise ValueError(f"No character ID in response: {data}")

        # Poll until character is ready
        soul_id = await _poll_character_status(character_id)

        async with async_session() as sess:
            c = await sess.get(Character, char_id)
            c.higgsfield_soul_id = str(soul_id or character_id)
            c.training_status = "trained" if soul_id else "failed"
            c.locked = 1 if soul_id else 0
            await sess.commit()

    except Exception as e:
        async with async_session() as sess:
            c = await sess.get(Character, char_id)
            c.training_status = "failed"
            await sess.commit()
        print(f"[character_manager] Soul training failed for {char_id}: {e}")


async def _poll_character_status(character_id: str, timeout: int = 300) -> Optional[str]:
    """Poll Higgsfield until character reference is ready. Returns character ID on success."""
    elapsed = 0
    interval = 5
    while elapsed < timeout:
        try:
            async with httpx.AsyncClient(timeout=30) as http:
                resp = await http.get(
                    f"{HIGGSFIELD_BASE}/v1/custom-references/{character_id}",
                    headers=_hf_headers(),
                )
                resp.raise_for_status()
                data = resp.json()
                status = data.get("status", "")
                if status == "completed":
                    return character_id
                if status in ("failed", "error"):
                    print(f"[character_manager] Character {character_id} training failed: {data}")
                    return None
        except Exception as e:
            print(f"[character_manager] Poll error: {e}")

        await asyncio.sleep(interval)
        elapsed += interval

    print(f"[character_manager] Character {character_id} timed out after {timeout}s")
    return None


async def _poll_job_set(job_set_id: str, timeout: int = 120) -> Optional[str]:
    """Poll a Higgsfield job set until completion. Returns image URL on success."""
    elapsed = 0
    interval = 5
    while elapsed < timeout:
        try:
            async with httpx.AsyncClient(timeout=30) as http:
                resp = await http.get(
                    f"{HIGGSFIELD_BASE}/v1/job-sets/{job_set_id}",
                    headers=_hf_headers(),
                )
                resp.raise_for_status()
                data = resp.json()
                status = data.get("status", "")
                if status == "completed":
                    # Extract image URL from results
                    jobs = data.get("jobs", [])
                    if jobs:
                        outputs = jobs[0].get("outputs", [])
                        if outputs:
                            return outputs[0].get("url", "")
                    return None
                if status in ("failed", "error"):
                    print(f"[character_manager] Job {job_set_id} failed: {data}")
                    return None
        except Exception as e:
            print(f"[character_manager] Poll error: {e}")

        await asyncio.sleep(interval)
        elapsed += interval

    print(f"[character_manager] Job {job_set_id} timed out")
    return None


async def generate_hero_image(char_id: int):
    """Generate a canonical portrait for a character using their Soul ID."""
    async with async_session() as db:
        char = await db.get(Character, char_id)
        if not char or not char.higgsfield_soul_id:
            return

        # Generate prompt via Claude
        msg = await client.messages.create(
            model=settings.model,
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": (
                    f"Write a detailed image generation prompt for a portrait of '{char.name}'.\n"
                    f"Character description: {char.description}\n"
                    "The prompt should describe: pose, lighting, background, style (cinematic, epic fantasy). "
                    "Return only the prompt text, no explanation."
                ),
            }],
        )
        prompt_text = msg.content[0].text.strip()

        try:
            # Generate image via Higgsfield Soul with character reference
            async with httpx.AsyncClient(timeout=30) as http:
                payload = {
                    "prompt": prompt_text,
                    "width_and_height": "1280x720",
                    "quality": "1080p",
                    "batch_size": 1,
                    "custom_reference_id": char.higgsfield_soul_id,
                }
                resp = await http.post(
                    f"{HIGGSFIELD_BASE}/v1/text2image/soul",
                    headers=_hf_headers(),
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()

            job_set_id = data.get("job_set_id", "")
            if not job_set_id:
                print(f"[character_manager] No job_set_id in response: {data}")
                return

            image_url = await _poll_job_set(job_set_id)

            if image_url:
                out_dir = Path(settings.output_dir) / str(char.storyboard_id) / "characters"
                out_dir.mkdir(parents=True, exist_ok=True)
                hero_path = out_dir / f"{char.name.lower().replace(' ', '_')}_hero.jpg"
                async with httpx.AsyncClient(timeout=60) as client_http:
                    resp = await client_http.get(image_url)
                    hero_path.write_bytes(resp.content)

                async with async_session() as sess:
                    c = await sess.get(Character, char_id)
                    c.hero_image_path = str(hero_path)
                    await sess.commit()

        except Exception as e:
            print(f"[character_manager] Hero image generation failed for {char_id}: {e}")
