"""
Gradio apps used in notebooks 2 and 3.

You can read these functions in the notebook first, then open this file
if you want to see how the interface is built.
"""

from __future__ import annotations

from typing import Generator

import gradio as gr
import pandas as pd

from src.llm_gateway import check_available_providers, run_llm, stream_llm
from src.output_formatting import (
    format_error_markdown,
    format_llm_response_markdown,
    format_places_dataframe,
    format_tool_evidence_dataframe,
    format_travel_brief_markdown,
    media_path_for_gradio,
    strip_markdown_for_speech,
)
from src.prompt_templates import (
    PROMPT_PRESETS,
    build_learning_support_prompt,
    build_travel_brief_prompt,
)
from src.utils import APP_TRANSCRIPTS_DIR, save_text_output

# ---------------------------------------------------------------------------
# App-wide safety settings
# ---------------------------------------------------------------------------
MAX_INPUT_CHARS = 8000
PRIVATE_DATA_WARNING = (
    "Do not paste passwords, exam scripts, student records, or private documents."
)


def _provider_choices() -> list[str]:
    available = check_available_providers()
    choices = ["auto"]
    choices.extend([name for name, ok in available.items() if ok])
    return choices or ["auto"]


def _normalize_history(history: list | None) -> list:
    """Gradio may pass None for an empty chatbot; normalise before appending."""
    return history or []


def generate_learning_support(
    user_text: str,
    audience: str,
    task_type: str,
    provider: str = "auto",
    temperature: float = 0.3,
    max_tokens: int = 1000,
    system_prompt: str | None = None,
) -> str:
    """Core learning support function used by notebook 2."""
    if not user_text.strip():
        return "Please enter some source text first."
    if len(user_text) > MAX_INPUT_CHARS:
        return f"Input is too long. Limit is {MAX_INPUT_CHARS} characters."

    prompt = build_learning_support_prompt(user_text, audience, task_type)
    raw = run_llm(
        prompt,
        provider=provider,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return format_llm_response_markdown(raw, title=task_type)


def build_learning_support_app() -> gr.Blocks:
    """Build the structured learning support Gradio app."""
    provider_choices = _provider_choices()
    task_types = [
        "Explain simply",
        "Summarise",
        "Generate quiz questions",
        "Identify key concepts",
        "Rewrite for beginners",
        "Create study notes",
    ]
    audiences = [
        "undergraduate students",
        "postgraduate students",
        "first-year students",
        "workshop participants",
    ]

    with gr.Blocks(title="Learning Support Assistant") as demo:
        gr.Markdown("# Learning Support Assistant")
        gr.Markdown(PRIVATE_DATA_WARNING)

        with gr.Row():
            with gr.Column():
                user_text = gr.Textbox(label="Source text", lines=8)
                preset = gr.Dropdown(
                    label="Prompt preset library",
                    choices=list(PROMPT_PRESETS.keys()),
                    value=None,
                )
                task_type = gr.Dropdown(label="Task type", choices=task_types, value=task_types[0])
                audience = gr.Dropdown(label="Audience", choices=audiences, value=audiences[0])
                provider = gr.Dropdown(label="Provider", choices=provider_choices, value="auto")
                temperature = gr.Slider(0, 1, value=0.3, label="Temperature")
                max_tokens = gr.Slider(200, 2000, value=1000, step=50, label="Max tokens")
                system_prompt = gr.Textbox(
                    label="System prompt",
                    value="You are a helpful and responsible university learning assistant.",
                )
                generate_btn = gr.Button("Generate", variant="primary")
                clear_btn = gr.Button("Clear")
                export_btn = gr.Button("Export output")
            with gr.Column():
                provider_status = gr.JSON(label="Provider status", value=check_available_providers())
                prompt_preview = gr.Textbox(label="Prompt preview", lines=10)
                output = gr.Markdown(label="Output")
                export_status = gr.Textbox(label="Export status", interactive=False)

        def preview_prompt(text, audience_value, task_value):
            if not text.strip():
                return "Enter source text to preview the prompt."
            return build_learning_support_prompt(text, audience_value, task_value)

        def apply_preset(preset_name, text):
            if not preset_name:
                return gr.update(), gr.update(), gr.update(), text
            preset = PROMPT_PRESETS[preset_name]
            return (
                preset["task_type"],
                preset["audience"],
                preset["system_prompt"],
                text,
            )

        def run_app(text, audience_value, task_value, provider_value, temp, tokens, system):
            try:
                result = generate_learning_support(
                    text, audience_value, task_value, provider_value, temp, tokens, system
                )
                return result, "Generation complete."
            except Exception as exc:
                return format_error_markdown("learning support generation", exc), "Generation failed."

        def export_output(content):
            if not content or content.startswith("Error:"):
                return "Nothing to export."
            path = save_text_output(content, APP_TRANSCRIPTS_DIR, "learning_support", "md")
            return f"Saved to {path}"

        preset.change(
            apply_preset,
            inputs=[preset, user_text],
            outputs=[task_type, audience, system_prompt, user_text],
        )
        user_text.change(
            preview_prompt,
            inputs=[user_text, audience, task_type],
            outputs=prompt_preview,
        )
        generate_btn.click(
            run_app,
            inputs=[user_text, audience, task_type, provider, temperature, max_tokens, system_prompt],
            outputs=[output, export_status],
        )
        clear_btn.click(
            lambda: ("", "", "Cleared."),
            outputs=[user_text, output, export_status],
        )
        export_btn.click(export_output, inputs=output, outputs=export_status)

        gr.Examples(
            examples=[
                ["Transformers use attention mechanisms to weigh relationships between tokens in a sequence.", "Explain like a lecturer"],
                ["Overfitting happens when a model memorises training data and performs poorly on new data.", "Create study notes"],
            ],
            inputs=[user_text, preset],
        )

    return demo


def respond_chat(message: str, history: list | None, provider: str, system_prompt: str, temperature: float, max_tokens: int):
    """Non-streaming chat response."""
    history = _normalize_history(history)
    if not message.strip():
        return history, ""
    try:
        reply = run_llm(
            message,
            provider=provider,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return history + [[message, format_llm_response_markdown(reply)]], ""
    except Exception as exc:
        return history + [[message, format_error_markdown("chat response", exc)]], ""


def respond_streaming(
    message: str,
    history: list | None,
    provider: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
) -> Generator[tuple[list, str], None, None]:
    """Streaming chat response for Gradio."""
    history = _normalize_history(history)
    if not message.strip():
        yield history, ""
        return

    history = history + [[message, ""]]
    try:
        for chunk in stream_llm(
            message,
            provider=provider,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            history[-1][1] += chunk
            yield history, ""
    except Exception as exc:
        history[-1][1] = format_error_markdown("streaming response", exc)
        yield history, ""


def build_chatbot_app(streaming: bool = False) -> gr.Blocks:
    """Build a chatbot-style app with optional streaming."""
    provider_choices = _provider_choices()
    title = "Streaming Learning Chatbot" if streaming else "Learning Chatbot"

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}")
        gr.Markdown(PRIVATE_DATA_WARNING)

        chatbot = gr.Chatbot(label="Conversation")
        msg = gr.Textbox(label="Your message")
        provider = gr.Dropdown(label="Provider", choices=provider_choices, value="auto")
        system_prompt = gr.Textbox(
            label="System prompt",
            value="You are a helpful university learning assistant.",
        )
        temperature = gr.Slider(0, 1, value=0.3, label="Temperature")
        max_tokens = gr.Slider(200, 2000, value=1000, step=50, label="Max tokens")

        with gr.Row():
            send = gr.Button("Send", variant="primary")
            clear = gr.Button("Clear")
            export = gr.Button("Export chat")

        export_status = gr.Textbox(label="Export status", interactive=False)

        chat_inputs = [msg, chatbot, provider, system_prompt, temperature, max_tokens]
        chat_outputs = [chatbot, msg]

        def export_chat(history: list | None) -> str:
            history = _normalize_history(history)
            if not history:
                return "Nothing to export."
            transcript = "\n\n".join(f"User: {u}\nAssistant: {a}" for u, a in history)
            path = save_text_output(transcript, APP_TRANSCRIPTS_DIR, "chat_transcript", "txt")
            return f"Saved to {path}"

        respond_fn = respond_streaming if streaming else respond_chat
        send.click(respond_fn, inputs=chat_inputs, outputs=chat_outputs)
        msg.submit(respond_fn, inputs=chat_inputs, outputs=chat_outputs)

        clear.click(lambda: ([], ""), outputs=[chatbot, msg])
        export.click(export_chat, inputs=chatbot, outputs=export_status)

    return demo


def run_tour_guide_pipeline(
    base_city: str,
    base_country: str,
    destination_city: str,
    destination_country: str,
    base_currency: str,
    destination_currency: str,
    traveller_profile: str,
    provider_value: str = "auto",
    temperature: float = 0.3,
    max_tokens: int = 1200,
    amount: float = 100.0,
):
    """
    Run the notebook 3 travel workflow.

    Yields (brief_md, tool_df, places_df, audio_path, poster_path, status)
    so Gradio can show progress between slow steps.
    """
    from src.image_generation import generate_travel_poster, text_to_speech
    from src.travel_tools import gather_travel_tool_results

    empty_places = format_places_dataframe([])

    try:
        yield (
            "_Fetching live travel data…_",
            pd.DataFrame(),
            empty_places,
            None,
            None,
            "Step 1/4 — calling weather, distance, exchange, and places tools…",
        )
        evidence = gather_travel_tool_results(
            base_city,
            base_country,
            destination_city,
            destination_country,
            base_currency,
            destination_currency,
            amount=amount,
        )
        tool_df = format_tool_evidence_dataframe(evidence)
        places_df = format_places_dataframe(evidence["places"])

        yield (
            "_Generating grounded travel brief…_",
            tool_df,
            places_df,
            None,
            None,
            "Step 2/4 — writing brief with the LLM…",
        )
        prompt = build_travel_brief_prompt(
            base_city,
            base_country,
            destination_city,
            destination_country,
            traveller_profile,
            evidence["weather"],
            evidence["distance"],
            evidence["exchange"],
            evidence["places"],
        )
        brief = run_llm(
            prompt,
            provider=provider_value,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        brief_md = format_travel_brief_markdown(
            brief,
            base_city,
            base_country,
            destination_city,
            destination_country,
        )

        yield (
            brief_md,
            tool_df,
            places_df,
            None,
            None,
            "Step 3/4 — creating audio reading…",
        )
        speech_text = strip_markdown_for_speech(brief)
        audio_path = text_to_speech(speech_text)

        yield (
            brief_md,
            tool_df,
            places_df,
            media_path_for_gradio(audio_path),
            None,
            "Step 4/4 — creating travel poster…",
        )
        poster = generate_travel_poster(
            destination_city,
            destination_country,
            evidence["places"],
        )

        yield (
            brief_md,
            tool_df,
            places_df,
            media_path_for_gradio(audio_path),
            media_path_for_gradio(poster["path"]),
            "Done — brief, audio, and poster are ready.",
        )
    except Exception as exc:
        yield (
            format_error_markdown("travel guide", exc),
            pd.DataFrame(),
            empty_places,
            None,
            None,
            f"Failed: {exc}",
        )


def build_tour_guide_app() -> gr.Blocks:
    """Build the multimodal African travel guide app."""
    provider_choices = _provider_choices()

    with gr.Blocks(title="African Travel Guide for Data Scientists") as demo:
        gr.Markdown("# African Travel Guide for Data Scientists")
        gr.Markdown(
            "**Verify Before Travel:** Check official sources for visas, health guidance, "
            "and safety updates before making real travel plans."
        )

        with gr.Row():
            with gr.Column():
                base_city = gr.Textbox(value="Kigali", label="Base city")
                base_country = gr.Textbox(value="Rwanda", label="Base country")
                destination_city = gr.Textbox(value="Accra", label="Destination city")
                destination_country = gr.Textbox(value="Ghana", label="Destination country")
                base_currency = gr.Textbox(value="RWF", label="Base currency code")
                destination_currency = gr.Textbox(value="GHS", label="Destination currency code")
                traveller_profile = gr.Textbox(
                    value="data scientist attending an AI workshop",
                    label="Traveller profile",
                )
                provider = gr.Dropdown(label="Provider", choices=provider_choices, value="auto")
                temperature = gr.Slider(0, 1, value=0.3, label="Temperature")
                generate_btn = gr.Button("Generate travel brief", variant="primary")

            with gr.Column():
                travel_brief = gr.Markdown(value="_Your travel brief will appear here._")
                status = gr.Textbox(label="Status", interactive=False, value="Ready.")
                tool_table = gr.Dataframe(label="Tool evidence")
                places_table = gr.Dataframe(label="Top 3 places")

        with gr.Row():
            audio_output = gr.Audio(label="Audio reading", type="filepath")
            poster_output = gr.Image(label="Travel poster", type="filepath")

        generate_btn.click(
            run_tour_guide_pipeline,
            inputs=[
                base_city,
                base_country,
                destination_city,
                destination_country,
                base_currency,
                destination_currency,
                traveller_profile,
                provider,
                temperature,
            ],
            outputs=[travel_brief, tool_table, places_table, audio_output, poster_output, status],
        )

    return demo
