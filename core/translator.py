"""Claude-powered English → Indic language translation."""
import json
from typing import Dict, List

import anthropic

from config import get_settings

settings = get_settings()
client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

SUPPORTED_LANGUAGES = {
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "bn": "Bengali",
}


async def translate_narrations(scenes: List[dict]) -> Dict[str, List[dict]]:
    """Translate all scene narrations from English to all 6 Indic languages.

    Returns: {lang_code: [translated_scenes, ...]}
    """
    narrations = [s.get("narration", "") for s in scenes]
    narrations_text = json.dumps(narrations, ensure_ascii=False)

    results: Dict[str, List[dict]] = {}

    for lang_code, lang_name in SUPPORTED_LANGUAGES.items():
        translated = await _translate_to_language(narrations_text, lang_name, lang_code)
        translated_scenes = []
        for i, scene in enumerate(scenes):
            scene_copy = dict(scene)
            scene_copy["narration"] = translated[i] if i < len(translated) else scene.get("narration", "")
            translated_scenes.append(scene_copy)
        results[lang_code] = translated_scenes

    return results


async def _translate_to_language(narrations_json: str, lang_name: str, lang_code: str) -> List[str]:
    msg = await client.messages.create(
        model=settings.model,
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": (
                f"Translate the following English narration texts to {lang_name}.\n"
                "Maintain the storytelling tone and epic narrative style.\n"
                f"Input JSON array of strings:\n{narrations_json}\n\n"
                f"Return ONLY a JSON array of translated strings in {lang_name}. "
                "No explanation, no extra text."
            ),
        }],
    )
    text = msg.content[0].text.strip()
    import re
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    # Fallback: return originals
    return json.loads(narrations_json)
