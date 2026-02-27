"""Google Cloud Text-to-Speech narration generation."""
import os
from pathlib import Path
from typing import List, Optional

from config import get_settings

settings = get_settings()

# Language code → Google TTS voice config
VOICE_CONFIG = {
    "en": {"language_code": "en-US", "name": "en-US-Neural2-D", "gender": "MALE"},
    "hi": {"language_code": "hi-IN", "name": "hi-IN-Neural2-B", "gender": "MALE"},
    "ta": {"language_code": "ta-IN", "name": "ta-IN-Neural2-D", "gender": "MALE"},
    "te": {"language_code": "te-IN", "name": "te-IN-Standard-B", "gender": "MALE"},
    "kn": {"language_code": "kn-IN", "name": "kn-IN-Standard-B", "gender": "MALE"},
    "ml": {"language_code": "ml-IN", "name": "ml-IN-Standard-B", "gender": "MALE"},
    "bn": {"language_code": "bn-IN", "name": "bn-IN-Neural2-D", "gender": "MALE"},
}


async def generate_narration_audio(
    text: str,
    lang_code: str,
    output_path: Path,
) -> bool:
    """Generate TTS audio for a narration text. Returns True on success."""
    if not text.strip():
        return False

    voice = VOICE_CONFIG.get(lang_code, VOICE_CONFIG["en"])

    try:
        from google.cloud import texttospeech

        client = texttospeech.TextToSpeechClient(
            client_options={"api_key": settings.google_cloud_tts_key}
            if settings.google_cloud_tts_key
            else {}
        )

        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice_params = texttospeech.VoiceSelectionParams(
            language_code=voice["language_code"],
            name=voice["name"],
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=0.95,
        )

        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice_params,
            audio_config=audio_config,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(response.audio_content)
        return True

    except Exception as e:
        print(f"[audio_generator] TTS failed for lang={lang_code}: {e}")
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
