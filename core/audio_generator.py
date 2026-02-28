"""ElevenLabs multilingual narration audio generation."""
from pathlib import Path
from typing import List, Optional

from config import get_settings

settings = get_settings()

# Language code → ElevenLabs language name (eleven_multilingual_v2 supports all)
LANG_MAP = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "bn": "Bengali",
}


async def generate_narration_audio(
    text: str,
    lang_code: str,
    output_path: Path,
) -> bool:
    """Generate TTS audio for a narration text using ElevenLabs. Returns True on success."""
    if not text.strip():
        return False

    if not settings.elevenlabs_api_key:
        print("[audio_generator] No ELEVENLABS_API_KEY configured, skipping TTS")
        return False

    try:
        from elevenlabs.client import ElevenLabs

        client = ElevenLabs(api_key=settings.elevenlabs_api_key)

        audio = client.text_to_speech.convert(
            voice_id=settings.elevenlabs_voice_id,
            text=text,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # audio is a generator of bytes chunks
        with open(output_path, "wb") as f:
            for chunk in audio:
                f.write(chunk)

        return True

    except Exception as e:
        print(f"[audio_generator] ElevenLabs TTS failed for lang={lang_code}: {e}")
        return False


async def generate_episode_audio(
    scenes: List[dict],
    lang_code: str,
    output_dir: Path,
) -> List[Optional[Path]]:
    """Generate per-scene audio files. Returns list of paths (None if failed)."""
    paths = []
    for scene in scenes:
        narration = scene.get("narration", "")
        scene_num = scene.get("scene", len(paths) + 1)
        audio_path = output_dir / f"scene_{scene_num:02d}_{lang_code}.mp3"
        success = await generate_narration_audio(narration, lang_code, audio_path)
        paths.append(audio_path if success else None)
    return paths
