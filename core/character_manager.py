"""Character extraction and Higgsfield Soul ID management."""
import json
import re
import asyncio
from pathlib import Path
from typing import List

import anthropic

from config import get_settings
from database import async_session, Character, KnowledgeChunk, Storyboard
from sqlalchemy import select

settings = get_settings()
client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)


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


async def start_soul_training(char_id: int):
    """Submit Soul ID training job to Higgsfield and poll until complete."""
    async with async_session() as db:
        char = await db.get(Character, char_id)
        if not char or not char.ref_images_dir:
            return

        ref_dir = Path(char.ref_images_dir)
        image_paths = list(ref_dir.glob("*.*"))
        if not image_paths:
            async with async_session() as sess:
                c = await sess.get(Character, char_id)
                c.training_status = "failed"
                await sess.commit()
            return

        try:
            from higgsfield.client import HiggsFieldClient
            hf = HiggsFieldClient(api_key=settings.higgsfield_api_key)

            # Submit training job
            job = hf.soul.train(
                name=char.name,
                images=[str(p) for p in image_paths],
            )
            soul_id = job.soul_id if hasattr(job, "soul_id") else job.get("soul_id", "")

            async with async_session() as sess:
                c = await sess.get(Character, char_id)
                c.higgsfield_soul_id = soul_id
                c.training_status = "trained"
                c.locked = 1
                await sess.commit()

        except Exception as e:
            async with async_session() as sess:
                c = await sess.get(Character, char_id)
                c.training_status = "failed"
                await sess.commit()
            print(f"[character_manager] Soul training failed for {char_id}: {e}")


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
            from higgsfield.client import HiggsFieldClient
            hf = HiggsFieldClient(api_key=settings.higgsfield_api_key)
            result = hf.image.generate(
                prompt=prompt_text,
                soul_id=char.higgsfield_soul_id,
            )
            image_url = result.image_url if hasattr(result, "image_url") else result.get("image_url", "")

            if image_url:
                # Download and save
                import httpx
                out_dir = Path(settings.output_dir) / str(char.storyboard_id) / "characters"
                out_dir.mkdir(parents=True, exist_ok=True)
                hero_path = out_dir / f"{char.name.lower().replace(' ', '_')}_hero.jpg"
                async with httpx.AsyncClient() as client_http:
                    resp = await client_http.get(image_url)
                    hero_path.write_bytes(resp.content)

                async with async_session() as sess:
                    c = await sess.get(Character, char_id)
                    c.hero_image_path = str(hero_path)
                    await sess.commit()

        except Exception as e:
            print(f"[character_manager] Hero image generation failed for {char_id}: {e}")
