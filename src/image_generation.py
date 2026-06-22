"""Image and audio helpers for notebook 3."""

from __future__ import annotations

import base64
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from src.utils import GENERATED_IMAGES_DIR, ensure_output_dirs, get_env


def _ensure_parent(path: Path) -> Path:
    ensure_output_dirs()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def build_travel_image_prompt(
    destination_city: str,
    destination_country: str,
    places: list[dict[str, str]],
) -> str:
    """Create a careful image-generation prompt."""
    place_lines = "\n".join(
        f"{i + 1}. {p['name']}: {p['short_note']}"
        for i, p in enumerate(places[:3])
    )
    return f"""Create a respectful illustrated travel poster for {destination_city}, {destination_country}.
Include visual references to these three places:
{place_lines}

Style:
- Clean educational travel-poster style
- Warm, professional, not stereotyped
- No fake logos
- No real identifiable people
- Add small caption areas for each place
"""


def generate_image_openai(prompt: str, output_path: str | Path) -> dict[str, Any]:
    """Generate an image with OpenAI."""
    from openai import OpenAI

    path = _ensure_parent(Path(output_path))
    client = OpenAI(api_key=get_env("OPENAI_API_KEY"))
    response = client.images.generate(
        model=get_env("OPENAI_IMAGE_MODEL", "gpt-image-1"),
        prompt=prompt,
        size="1024x1024",
    )
    image_data = response.data[0]
    if getattr(image_data, "b64_json", None):
        path.write_bytes(base64.b64decode(image_data.b64_json))
    elif getattr(image_data, "url", None):
        import requests

        img_response = requests.get(image_data.url, timeout=60)
        img_response.raise_for_status()
        path.write_bytes(img_response.content)
    else:
        raise RuntimeError("OpenAI image response did not include image data.")
    return {"status": "success", "provider": "openai", "path": str(path)}


def generate_image_gemini(prompt: str, output_path: str | Path) -> dict[str, Any]:
    """Generate an image with Gemini / Imagen where available."""
    from google import genai

    path = _ensure_parent(Path(output_path))
    client = genai.Client(api_key=get_env("GEMINI_API_KEY"))
    model = get_env("GEMINI_IMAGE_MODEL", "imagen-3.0-generate-002")
    response = client.models.generate_images(model=model, prompt=prompt)
    generated = response.generated_images[0]
    path.write_bytes(generated.image.image_bytes)
    return {"status": "success", "provider": "gemini", "path": str(path)}


def generate_image_stability(prompt: str, output_path: str | Path) -> dict[str, Any]:
    """Generate an image with Stability AI."""
    import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
    from stability_sdk import client

    path = _ensure_parent(Path(output_path))
    stability_api = client.StabilityInference(
        key=get_env("STABILITY_API_KEY"),
        verbose=False,
    )
    answers = stability_api.generate(
        prompt=prompt,
        steps=30,
        width=1024,
        height=1024,
        samples=1,
    )
    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.type == generation.ARTIFACT_IMAGE:
                path.write_bytes(artifact.binary)
                return {"status": "success", "provider": "stability", "path": str(path)}
    raise RuntimeError("Stability AI did not return an image.")


def generate_travel_poster_with_pil(
    destination_city: str,
    destination_country: str,
    places: list[dict[str, str]],
    output_path: str | Path = "data/outputs/generated_images/travel_poster.png",
) -> dict[str, Any]:
    """Create a simple educational poster that always works offline."""
    path = _ensure_parent(Path(output_path))
    width, height = 1024, 1280
    image = Image.new("RGB", (width, height), color=(245, 240, 230))
    draw = ImageDraw.Draw(image)

    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 42)
        body_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 24)
        small_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
    except OSError:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    draw.rectangle([(30, 30), (width - 30, height - 30)], outline=(60, 90, 120), width=4)
    title = f"Data Scientist Travel Guide: {destination_city}, {destination_country}"
    draw.text((60, 60), title, fill=(30, 60, 90), font=title_font)

    y = 170
    for idx, place in enumerate(places[:3], start=1):
        draw.rectangle([(60, y), (width - 60, y + 180)], fill=(255, 255, 255), outline=(180, 180, 180))
        draw.text((80, y + 20), f"{idx}. {place['name']}", fill=(20, 20, 20), font=body_font)
        note = place.get("short_note", "")
        draw.text((80, y + 70), note[:90], fill=(60, 60, 60), font=small_font)
        angle = place.get("data_scientist_angle", "")
        draw.text((80, y + 110), f"Data scientist angle: {angle[:85]}", fill=(80, 80, 120), font=small_font)
        y += 210

    footer = "AI-generated learning demo. Verify travel details before use."
    draw.text((60, height - 70), footer, fill=(100, 100, 100), font=small_font)
    image.save(path)
    return {"status": "success", "provider": "pil_fallback", "path": str(path)}


def generate_travel_poster(
    destination_city: str,
    destination_country: str,
    places: list[dict[str, str]],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Try cloud image APIs, then fall back to PIL poster."""
    if output_path is None:
        safe_name = destination_city.lower().replace(" ", "_")
        output_path = GENERATED_IMAGES_DIR / f"travel_poster_{safe_name}.png"

    prompt = build_travel_image_prompt(destination_city, destination_country, places)

    if get_env("OPENAI_API_KEY"):
        try:
            return generate_image_openai(prompt, output_path)
        except Exception:
            pass

    if get_env("GEMINI_API_KEY"):
        try:
            return generate_image_gemini(prompt, output_path)
        except Exception:
            pass

    if get_env("STABILITY_API_KEY"):
        try:
            return generate_image_stability(prompt, output_path)
        except Exception:
            pass

    return generate_travel_poster_with_pil(
        destination_city, destination_country, places, output_path
    )


async def text_to_speech_edge(
    text: str,
    output_path: str | Path = "data/outputs/travel_brief.mp3",
) -> str:
    """Generate speech audio with edge-tts."""
    import edge_tts

    path = _ensure_parent(Path(output_path))
    communicate = edge_tts.Communicate(text[:3000], voice="en-GB-SoniaNeural")
    await communicate.save(str(path))
    return str(path)


def _run_async(coro):
    """Run async code from sync code, including inside Jupyter notebooks."""
    import asyncio
    import concurrent.futures

    try:
        asyncio.get_running_loop()
        in_notebook = True
    except RuntimeError:
        in_notebook = False

    if not in_notebook:
        return asyncio.run(coro)

    # Jupyter already has a running event loop; run in a worker thread.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()


def text_to_speech_gtts(
    text: str,
    output_path: str | Path = "data/outputs/travel_brief.mp3",
) -> str:
    """Generate speech audio with gTTS."""
    try:
        from gtts import gTTS
    except ImportError as exc:
        raise RuntimeError(
            "gTTS is not installed. Run: pip install gTTS"
        ) from exc

    path = _ensure_parent(Path(output_path))
    tts = gTTS(text=text[:3000], lang="en")
    tts.save(str(path))
    return str(path)


def text_to_speech_macos(
    text: str,
    output_path: str | Path = "data/outputs/travel_brief.m4a",
) -> str:
    """Generate speech audio with the built-in macOS `say` command."""
    if platform.system() != "Darwin" or not shutil.which("say"):
        raise RuntimeError("macOS say command is not available.")

    path = _ensure_parent(Path(output_path))
    aiff_path = path.with_suffix(".aiff")
    spoken_text = text[:3000].replace("\n", " ").strip()
    if not spoken_text:
        raise RuntimeError("No text provided for speech synthesis.")

    subprocess.run(
        ["say", "-o", str(aiff_path), spoken_text],
        check=True,
        capture_output=True,
        text=True,
    )

    if shutil.which("afconvert"):
        audio_path = path.with_suffix(".m4a")
        try:
            subprocess.run(
                ["afconvert", "-f", "m4af", "-d", "aac", str(aiff_path), str(audio_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            aiff_path.unlink(missing_ok=True)
            return str(audio_path)
        except subprocess.CalledProcessError:
            pass

    return str(aiff_path)


def text_to_speech(text: str, output_path: str | Path | None = None) -> str:
    """
    Generate audio for the travel guide demo.

    Fallback order (first success wins):
    1. edge-tts
    2. gTTS
    3. macOS built-in `say` command
    """
    if output_path is None:
        output_path = GENERATED_IMAGES_DIR.parent / "travel_brief.mp3"

    errors: list[str] = []

    try:
        return _run_async(text_to_speech_edge(text, output_path))
    except Exception as exc:
        errors.append(f"edge-tts failed: {exc}")

    try:
        return text_to_speech_gtts(text, output_path)
    except Exception as exc:
        errors.append(f"gTTS failed: {exc}")

    try:
        return text_to_speech_macos(text, output_path)
    except Exception as exc:
        errors.append(f"macOS say failed: {exc}")

    raise RuntimeError(
        "Text-to-speech failed. "
        + " ".join(errors)
        + " On macOS, the built-in say command should work without extra packages."
    )
