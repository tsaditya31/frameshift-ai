"""Claude-powered story generation and chat."""
import json
import re
from typing import AsyncIterator, List

import anthropic

from config import get_settings
from database import Storyboard, Episode, ChatMessage, KnowledgeChunk

settings = get_settings()
client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)


def _kb_context(chunks: List[KnowledgeChunk]) -> str:
    if not chunks:
        return "No knowledge base loaded yet."
    texts = [f"[{c.source_file}]\n{c.content_text}" for c in chunks[:40]]
    combined = "\n\n---\n\n".join(texts)
    return combined[:12000]  # Limit context size


def _build_system(sb: Storyboard, kb_chunks: List[KnowledgeChunk]) -> str:
    return f"""You are a creative story planning assistant for the project: "{sb.title}".
Description: {sb.description}

You help plan multi-episode video series based on the provided knowledge base.
When asked to generate episode outlines, respond with a JSON object in this exact format:
{{
  "episodes": [
    {{"episode": 1, "title": "Episode Title", "storyline": "One paragraph summary..."}},
    ...
  ],
  "characters": [
    {{"name": "Character Name", "description": "Brief character description..."}},
    ...
  ]
}}

Include all main characters identified from the storylines. For regular conversation, respond naturally.

Knowledge Base:
{_kb_context(kb_chunks)}
"""


async def stream_chat(
    sb: Storyboard,
    history: List[ChatMessage],
    kb_chunks: List[KnowledgeChunk],
    user_message: str,
) -> AsyncIterator[str]:
    system = _build_system(sb, kb_chunks)

    messages = []
    for msg in history[:-1]:  # Exclude the last user message (already in history)
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": user_message})

    async with client.messages.stream(
        model=settings.model,
        max_tokens=4096,
        system=system,
        messages=messages,
    ) as stream:
        async for text in stream.text_stream:
            yield text


async def generate_episode_outlines(
    sb: Storyboard,
    kb_chunks: List[KnowledgeChunk],
    num_episodes: int,
) -> List[dict]:
    system = _build_system(sb, kb_chunks)
    prompt = (
        f"Generate exactly {num_episodes} episode outlines for '{sb.title}'. "
        "Return ONLY a JSON array with no extra text. Each item: "
        '{"episode": N, "title": "...", "storyline": "one paragraph..."}'
    )

    msg = await client.messages.create(
        model=settings.model,
        max_tokens=8192,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    text = msg.content[0].text.strip()

    # Extract JSON array from response
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: return placeholder outlines
    return [
        {"episode": i, "title": f"Episode {i}", "storyline": ""}
        for i in range(1, num_episodes + 1)
    ]


async def stream_episode_chat(
    sb: Storyboard,
    episode: Episode,
    history: List[ChatMessage],
    kb_chunks: List[KnowledgeChunk],
    user_message: str,
) -> AsyncIterator[str]:
    """Stream chat for a specific episode, with episode context in the system prompt."""
    kb_text = _kb_context(kb_chunks)
    system = f"""You are a creative story writing assistant for the project: "{sb.title}".
Description: {sb.description}

You are working on Episode {episode.episode_num}: "{episode.title}".
Current storyline: {episode.storyline or '(not yet defined)'}
Current script: {episode.script_json or '(not yet generated)'}

Help refine the storyline, generate or improve the scene-by-scene script, and answer questions about this episode.
When generating a script, return a JSON array of scenes with keys: scene, description, narration, characters, duration_seconds.

Knowledge Base:
{kb_text}
"""

    messages = []
    for msg in history[:-1]:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": user_message})

    async with client.messages.stream(
        model=settings.model,
        max_tokens=4096,
        system=system,
        messages=messages,
    ) as stream:
        async for text in stream.text_stream:
            yield text


async def generate_scene_script(
    sb: Storyboard,
    kb_chunks: List[KnowledgeChunk],
    episode_num: int,
    episode_title: str,
    storyline: str,
    num_scenes: int = 6,
) -> List[dict]:
    """Generate detailed scene-by-scene script for an episode."""
    system = _build_system(sb, kb_chunks)
    prompt = (
        f"Write a detailed {num_scenes}-scene script for Episode {episode_num}: '{episode_title}'.\n"
        f"Storyline: {storyline}\n\n"
        "Return ONLY a JSON array. Each scene:\n"
        '{"scene": N, "description": "visual description for image generation", '
        '"narration": "narration text to be read aloud", '
        '"characters": ["Char1", "Char2"], '
        '"duration_seconds": 5}'
    )

    msg = await client.messages.create(
        model=settings.model,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    text = msg.content[0].text.strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return [{"scene": i, "description": "", "narration": "", "characters": [], "duration_seconds": 5}
            for i in range(1, num_scenes + 1)]
