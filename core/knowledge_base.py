"""Ingest txt/md/pdf/URLs into KnowledgeChunk rows."""
import re
from pathlib import Path
from typing import List

import httpx
from bs4 import BeautifulSoup

from database import KnowledgeChunk

CHUNK_SIZE = 2000  # characters per chunk


def _split_text(text: str, source: str, sb_id: int, chunk_type: str = "text") -> List[KnowledgeChunk]:
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        # Try to break at paragraph boundary
        if end < len(text):
            para_break = text.rfind("\n\n", start, end)
            if para_break > start:
                end = para_break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(KnowledgeChunk(
                storyboard_id=sb_id,
                source_file=source,
                content_text=chunk,
                chunk_type=chunk_type,
            ))
        start = end
    return chunks


async def ingest_file(path: Path, sb_id: int) -> List[KnowledgeChunk]:
    suffix = path.suffix.lower()
    source = path.name

    if suffix in (".txt", ".md"):
        text = path.read_text(encoding="utf-8", errors="replace")
        return _split_text(text, source, sb_id)

    if suffix == ".pdf":
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(path))
            pages = [page.get_text() for page in doc]
            text = "\n\n".join(pages)
            return _split_text(text, source, sb_id)
        except ImportError:
            return []

    if suffix in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
        # Store reference — actual image paths used by character_manager
        return [KnowledgeChunk(
            storyboard_id=sb_id,
            source_file=source,
            content_text=f"[Image reference: {path}]",
            chunk_type="image",
        )]

    return []


async def ingest_url(url: str, sb_id: int) -> List[KnowledgeChunk]:
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "ClaiwdBot/1.0"})
            resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove scripts/styles
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        return _split_text(text, url, sb_id, chunk_type="url")
    except Exception as e:
        return [KnowledgeChunk(
            storyboard_id=sb_id,
            source_file=url,
            content_text=f"[Failed to fetch: {e}]",
            chunk_type="url",
        )]
