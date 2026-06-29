"""Format LLM and tool outputs as readable markdown for notebooks and Gradio."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import PROJECT_ROOT


def media_path_for_gradio(path: str | Path | None) -> str | None:
    """Return an absolute path Gradio can load, even when Jupyter cwd is notebooks/."""
    if not path:
        return None
    candidates = [Path(path), PROJECT_ROOT / path]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return str(resolved)
    return str(Path(path).resolve())


def strip_markdown_for_speech(text: str, max_chars: int = 1200) -> str:
    """Remove markdown noise before text-to-speech."""
    cleaned = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"^#+\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\*+", "", cleaned)
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:max_chars]


def format_llm_response_markdown(text: str, title: str | None = None) -> str:
    """Turn numbered LLM answers into clearer markdown sections."""
    if not text or text.strip().startswith("Error"):
        return f"**{text}**" if text else "_No output yet._"

    lines = text.strip().splitlines()
    formatted: list[str] = []
    if title:
        formatted.append(f"## {title}\n")

    for line in lines:
        stripped = line.strip()
        if not stripped:
            formatted.append("")
            continue
        section = re.match(r"^(\d+)\.\s*([^:]+):\s*(.*)$", stripped)
        if section:
            _, heading, body = section.groups()
            formatted.append(f"### {heading.strip()}")
            if body:
                formatted.append(body.strip())
            formatted.append("")
            continue
        if stripped.startswith(("-", "*", "•")):
            formatted.append(stripped)
            continue
        formatted.append(stripped)

    return "\n".join(formatted).strip()


def show_llm_output(text: str, title: str | None = None) -> None:
    """Display an LLM answer as rendered markdown inside a notebook."""
    from IPython.display import Markdown, display

    display(Markdown(format_llm_response_markdown(text, title=title)))


def format_travel_brief_markdown(
    brief: str,
    base_city: str,
    base_country: str,
    destination_city: str,
    destination_country: str,
) -> str:
    """Wrap a travel brief with a clear route header."""
    header = (
        f"## {base_city}, {base_country} → {destination_city}, {destination_country}\n\n"
        "_Facts below come from live tools. Verify before real travel._\n"
    )
    return header + "\n" + format_llm_response_markdown(brief)


def format_tool_evidence_dataframe(evidence: dict[str, Any]) -> pd.DataFrame:
    """Readable tool-evidence table for notebooks and Gradio."""
    weather = evidence.get("weather", {})
    distance = evidence.get("distance", {})
    exchange = evidence.get("exchange", {})

    rows: list[dict[str, str]] = []

    if weather.get("error"):
        rows.append({"Source": "Weather", "Fact": "Status", "Value": weather["error"]})
    else:
        temp = weather.get("temperature")
        temp_unit = weather.get("units", {}).get("temperature", "°C")
        wind = weather.get("wind_speed")
        wind_unit = weather.get("units", {}).get("wind_speed", "km/h")
        rows.extend(
            [
                {"Source": "Weather", "Fact": "Summary", "Value": weather.get("weather_summary", "—")},
                {
                    "Source": "Weather",
                    "Fact": "Temperature",
                    "Value": f"{temp}{temp_unit}" if temp is not None else "—",
                },
                {
                    "Source": "Weather",
                    "Fact": "Rain chance",
                    "Value": (
                        f"{weather.get('precipitation_probability')}%"
                        if weather.get("precipitation_probability") is not None
                        else "—"
                    ),
                },
                {
                    "Source": "Weather",
                    "Fact": "Wind",
                    "Value": f"{wind} {wind_unit}" if wind is not None else "—",
                },
            ]
        )

    if distance.get("error"):
        rows.append({"Source": "Distance", "Fact": "Status", "Value": distance["error"]})
    else:
        rows.append(
            {
                "Source": "Distance",
                "Fact": "Straight-line km",
                "Value": str(distance.get("distance_km", "—")),
            }
        )
        rows.append(
            {
                "Source": "Distance",
                "Fact": "Note",
                "Value": distance.get("note", "Approximate only."),
            }
        )

    if exchange.get("error"):
        rows.append({"Source": "Exchange", "Fact": "Status", "Value": exchange["error"]})
    else:
        rows.extend(
            [
                {
                    "Source": "Exchange",
                    "Fact": "Rate",
                    "Value": (
                        f"1 {exchange.get('base_currency')} = "
                        f"{exchange.get('rate')} {exchange.get('destination_currency')}"
                    ),
                },
                {
                    "Source": "Exchange",
                    "Fact": "Example",
                    "Value": (
                        f"{exchange.get('amount')} {exchange.get('base_currency')} → "
                        f"{exchange.get('converted_amount')} {exchange.get('destination_currency')}"
                    ),
                },
                {"Source": "Exchange", "Fact": "Date", "Value": str(exchange.get("date", "—"))},
            ]
        )

    return pd.DataFrame(rows)


def format_places_dataframe(places: list[dict[str, str]]) -> pd.DataFrame:
    """Compact places table with the most useful columns first."""
    if not places:
        return pd.DataFrame(
            [{"Place": "—", "Why visit": "No places found.", "Data angle": "—"}]
        )
    return pd.DataFrame(
        [
            {
                "Place": place.get("name", "—"),
                "Why visit": place.get("why_visit", "—"),
                "Data angle": place.get("data_scientist_angle", "—"),
            }
            for place in places[:3]
        ]
    )


def format_error_markdown(step: str, exc: Exception) -> str:
    """User-friendly markdown error for Gradio."""
    return (
        f"## Could not complete: {step}\n\n"
        f"**What happened:** `{exc}`\n\n"
        "**Try:** check internet access, currency codes (e.g. RWF, GHS), "
        "that Ollama or your API key is ready, then click Generate again."
    )


def show_multimodal_travel_guide(
    brief_md: str,
    poster_path: str | Path,
    route_label: str | None = None,
) -> None:
    """Show text + image side by side — the core multimodal idea in the notebook."""
    from IPython.display import HTML, Image, Markdown, display

    title = route_label or "Your multimodal travel guide"
    style = """
            <style>
              .mm-grid { display: flex; gap: 1.25rem; flex-wrap: wrap; margin: 0.5rem 0 1rem; }
              .mm-card {
                flex: 1 1 280px; border: 1px solid #d0d7de; border-radius: 10px;
                padding: 0.75rem 1rem; background: #f6f8fa;
              }
              .mm-card h4 { margin: 0 0 0.35rem; font-size: 0.95rem; }
              .mm-card p { margin: 0; color: #57606a; font-size: 0.9rem; }
            </style>
    """
    body = f"""
            <h3 style="margin-bottom:0.25rem;">{title}</h3>
            <div class="mm-grid">
              <div class="mm-card">
                <h4>Modality 1 — Text</h4>
                <p>Grounded travel brief from tools + LLM</p>
              </div>
              <div class="mm-card">
                <h4>Modality 2 — Image</h4>
                <p>Generated travel poster for the destination</p>
              </div>
            </div>
    """
    display(HTML(style + body))
    display(Markdown(brief_md))
    display(Image(filename=str(poster_path), width=420))
