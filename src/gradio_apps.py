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
)
from src.prompt_templates import (
    PROMPT_PRESETS,
    build_learning_support_prompt,
    build_travel_brief_prompt,
)
from src.tour_session import TourSession
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
    """Build a simple learning support Gradio app for Notebook 2."""
    task_types = [
        "Explain simply",
        "Summarise",
        "Generate quiz questions",
        "Identify key concepts",
        "Rewrite for beginners",
        "Create study notes",
    ]
    default_audience = "undergraduate students"
    default_system = "You are a helpful and responsible university learning assistant."
    default_temperature = 0.3
    default_max_tokens = 1000

    def _system_prompt_for_task(task_value: str) -> str:
        for preset in PROMPT_PRESETS.values():
            if preset["task_type"] == task_value:
                return preset["system_prompt"]
        return default_system

    with gr.Blocks(title="Learning Support Assistant") as demo:
        gr.Markdown(
            "# Learning Support Assistant\n\n"
            "Paste your source text, choose what you need, and click **Generate**."
        )
        gr.Markdown(PRIVATE_DATA_WARNING)

        with gr.Row():
            with gr.Column():
                user_text = gr.Textbox(
                    label="Source text",
                    lines=10,
                    placeholder="Paste a paragraph from a lecture, article, or your notes…",
                )
                task_type = gr.Dropdown(
                    label="What do you need?",
                    choices=task_types,
                    value=task_types[0],
                )
                with gr.Row():
                    generate_btn = gr.Button("Generate", variant="primary")
                    clear_btn = gr.Button("Clear")

                gr.Examples(
                    examples=[
                        [
                            "Transformers use attention mechanisms to weigh relationships between tokens in a sequence.",
                            "Explain simply",
                        ],
                        [
                            "Overfitting happens when a model memorises training data and performs poorly on new data.",
                            "Create study notes",
                        ],
                    ],
                    inputs=[user_text, task_type],
                    label="Try an example",
                )
            with gr.Column():
                output = gr.Markdown(value="_Your answer will appear here._")
                status = gr.Textbox(label="Status", interactive=False, value="Ready.")

        def run_app(text, task_value):
            text = text or ""
            task_value = task_value or task_types[0]
            if not text.strip():
                return "_Enter some source text first._", "Nothing to generate."
            try:
                result = generate_learning_support(
                    text,
                    default_audience,
                    task_value,
                    provider="auto",
                    temperature=default_temperature,
                    max_tokens=default_max_tokens,
                    system_prompt=_system_prompt_for_task(task_value),
                )
                return str(result), "Done."
            except Exception as exc:
                message = f"**Generation failed:** {exc}"
                return message, f"Failed: {exc}"

        generate_btn.click(
            run_app,
            inputs=[user_text, task_type],
            outputs=[output, status],
            queue=True,
        )
        clear_btn.click(
            lambda: ("", task_types[0], "_Your answer will appear here._", "Cleared."),
            outputs=[user_text, task_type, output, status],
            queue=False,
        )

    demo.queue(concurrency_count=1)
    return demo


def respond_chat(
    message: str,
    history: list | None,
    provider: str = "auto",
    system_prompt: str = "You are a helpful university learning assistant.",
    temperature: float = 0.3,
    max_tokens: int = 1000,
):
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
        return history + [[message, f"**Error:** {exc}"]], ""


def respond_streaming(
    message: str,
    history: list | None,
    provider: str = "auto",
    system_prompt: str = "You are a helpful university learning assistant.",
    temperature: float = 0.3,
    max_tokens: int = 1000,
) -> Generator[tuple[list, str], None, None]:
    """Streaming chat response for Gradio."""
    history = _normalize_history(history)
    if not message.strip():
        yield history, ""
        return

    reply = ""
    try:
        for chunk in stream_llm(
            message,
            provider=provider,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            reply += chunk
            yield history + [[message, reply]], ""
    except Exception as exc:
        yield history + [[message, f"**Error:** {exc}"]], ""


def build_chatbot_app(streaming: bool = False) -> gr.Blocks:
    """Build a simple chatbot app with optional streaming."""
    title = "Streaming Learning Chatbot" if streaming else "Learning Chatbot"
    default_system = "You are a helpful university learning assistant."
    default_temperature = 0.3
    default_max_tokens = 1000

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}")
        if streaming:
            gr.Markdown(
                "Ask a study question below. Replies stream in token by token. "
                "Uses provider `auto` (Ollama or your API key)."
            )
        else:
            gr.Markdown("Ask a study question below. Uses provider `auto`.")
        gr.Markdown(PRIVATE_DATA_WARNING)

        chatbot = gr.Chatbot(label="Conversation")
        msg = gr.Textbox(
            label="Your message",
            placeholder="e.g. Explain overfitting in simple terms…",
        )

        with gr.Row():
            send = gr.Button("Send", variant="primary")
            clear = gr.Button("Clear")

        def send_chat(message, history):
            return respond_chat(
                message,
                history,
                provider="auto",
                system_prompt=default_system,
                temperature=default_temperature,
                max_tokens=default_max_tokens,
            )

        def send_stream(message, history):
            yield from respond_streaming(
                message,
                history,
                provider="auto",
                system_prompt=default_system,
                temperature=default_temperature,
                max_tokens=default_max_tokens,
            )

        respond_fn = send_stream if streaming else send_chat
        send.click(
            respond_fn,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg],
            queue=True,
        )
        msg.submit(
            respond_fn,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg],
            queue=True,
        )
        clear.click(lambda: ([], ""), outputs=[chatbot, msg], queue=False)

    demo.queue(concurrency_count=1)
    return demo


def _empty_tour_outputs() -> tuple:
    empty_places = format_places_dataframe([])
    empty_tools = pd.DataFrame(columns=["Source", "Fact", "Value"])
    return (
        "_Your travel brief will appear here._",
        empty_tools,
        empty_places,
        None,
        "Ready.",
    )


def run_tour_guide_progressive(
    base_city: str,
    base_country: str,
    destination_city: str,
    destination_country: str,
    base_currency: str,
    destination_currency: str,
    traveller_profile: str,
    provider_value: str = "auto",
    temperature: float = 0.3,
    max_tokens: int = 800,
    amount: float = 100.0,
) -> Generator[tuple, None, None]:
    """Run the travel workflow, yielding partial results for Gradio status updates."""
    from src.image_generation import generate_travel_poster
    from src.travel_tools import gather_travel_tool_results

    brief_md, tool_df, places_df, poster_out, _ = _empty_tour_outputs()

    try:
        yield (
            brief_md,
            tool_df,
            places_df,
            poster_out,
            "Fetching weather, distance, exchange rate, and places…",
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
            "_Writing your travel brief — this usually takes 20–40 seconds…_",
            tool_df,
            places_df,
            None,
            "Tool data ready. Calling the LLM…",
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
            "Brief ready. Creating travel poster…",
        )

        poster = generate_travel_poster(
            destination_city,
            destination_country,
            evidence["places"],
            fast=True,
        )

        yield (
            brief_md,
            tool_df,
            places_df,
            media_path_for_gradio(poster["path"]),
            "Done — brief and poster are ready.",
        )
    except Exception as exc:
        _, empty_tools, empty_places, _, _ = _empty_tour_outputs()
        yield (
            f"**Generation failed:** {exc}",
            empty_tools,
            empty_places,
            None,
            f"Failed: {exc}",
        )


def run_tour_guide_once(
    base_city: str,
    base_country: str,
    destination_city: str,
    destination_country: str,
    base_currency: str,
    destination_currency: str,
    traveller_profile: str,
    provider_value: str = "auto",
    temperature: float = 0.3,
    max_tokens: int = 800,
    amount: float = 100.0,
) -> tuple:
    """Run the full notebook 3 travel workflow and return all outputs at once."""
    result = None
    for result in run_tour_guide_progressive(
        base_city,
        base_country,
        destination_city,
        destination_country,
        base_currency,
        destination_currency,
        traveller_profile,
        provider_value=provider_value,
        temperature=temperature,
        max_tokens=max_tokens,
        amount=amount,
    ):
        pass
    return result if result is not None else _empty_tour_outputs()


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
    max_tokens: int = 800,
    amount: float = 100.0,
):
    """Backward-compatible wrapper used in tests."""
    yield run_tour_guide_once(
        base_city,
        base_country,
        destination_city,
        destination_country,
        base_currency,
        destination_currency,
        traveller_profile,
        provider_value,
        temperature,
        max_tokens,
        amount,
    )


def build_tour_guide_app() -> gr.Blocks:
    """Build a simple multimodal African travel guide app."""
    default_profile = "data scientist attending an AI workshop"

    with gr.Blocks(title="African Travel Guide for Data Scientists") as demo:
        gr.Markdown(
            "# African Travel Guide for Data Scientists\n\n"
            "Enter your trip details and click **Generate travel brief**. "
            "Tool data appears in a few seconds; the LLM brief takes 20–40 seconds; "
            "then a travel poster is generated. Watch **Status** for each step."
        )
        gr.Markdown(
            "**Verify before travel:** check official sources for visas, health, "
            "and safety updates. This is a teaching demo, not live travel advice."
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Your trip")
                with gr.Row():
                    base_city = gr.Textbox(value="Kigali", label="From (city)")
                    base_country = gr.Textbox(value="Rwanda", label="Country")
                with gr.Row():
                    destination_city = gr.Textbox(value="Accra", label="To (city)")
                    destination_country = gr.Textbox(value="Ghana", label="Country")
                with gr.Row():
                    base_currency = gr.Textbox(value="RWF", label="Home currency")
                    destination_currency = gr.Textbox(value="GHS", label="Dest. currency")
                traveller_profile = gr.Textbox(
                    value=default_profile,
                    label="Traveller profile",
                )
                generate_btn = gr.Button("Generate travel brief", variant="primary")

            with gr.Column():
                status = gr.Textbox(label="Status", interactive=False, value="Ready.")
                travel_brief = gr.Markdown(value="_Your travel brief will appear here._")
                tool_table = gr.Dataframe(label="Tool evidence")
                places_table = gr.Dataframe(label="Top 3 places")

        poster_output = gr.Image(label="Travel poster", type="filepath")

        def generate_travel_guide(
            b_city,
            b_country,
            d_city,
            d_country,
            b_curr,
            d_curr,
            profile,
        ):
            yield from run_tour_guide_progressive(
                b_city,
                b_country,
                d_city,
                d_country,
                b_curr,
                d_curr,
                profile,
                provider_value="auto",
                temperature=0.3,
            )

        generate_btn.click(
            generate_travel_guide,
            inputs=[
                base_city,
                base_country,
                destination_city,
                destination_country,
                base_currency,
                destination_currency,
                traveller_profile,
            ],
            outputs=[travel_brief, tool_table, places_table, poster_output, status],
            queue=True,
        )

    demo.queue(concurrency_count=1)
    return demo


def build_tour_guide_preview_app(session: TourSession) -> gr.Blocks:
    """Instant multimodal UI — shows what the student already built in the notebook."""
    poster_path = session.poster_path or None

    with gr.Blocks(title="Multimodal Travel Guide — Preview") as demo:
        gr.Markdown(
            "# Multimodal Travel Guide\n\n"
            f"**Route:** {session.route_label}\n\n"
            "This app opens **instantly** with the text brief and poster you already "
            "created in the notebook. Same agent workflow — two output types "
            "(**text** + **image**)."
        )
        gr.Markdown(
            "_To change the trip, edit Step 2 in the notebook and re-run Steps 3–7._"
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Pipeline recap")
                gr.Markdown(
                    "1. **Tools** → weather, distance, exchange, places\n"
                    "2. **LLM** → grounded text brief\n"
                    "3. **Code** → travel poster image\n"
                    "4. **Gradio** → one UI for both modalities"
                )
                refresh_btn = gr.Button("Refresh preview", variant="primary")
            with gr.Column(scale=2):
                status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    value="Loaded from your notebook — no extra wait.",
                )
                travel_brief = gr.Markdown(value=session.brief_md)
                tool_table = gr.Dataframe(value=session.tool_df, label="Tool evidence")
                places_table = gr.Dataframe(value=session.places_df, label="Top 3 places")
                poster_output = gr.Image(
                    value=poster_path,
                    label="Travel poster (image modality)",
                    type="filepath",
                )

        with gr.Accordion("Optional: generate a different trip (slow — 20–40s)", open=False):
            gr.Markdown(
                "Only use this if you want the app to re-run tools + LLM + poster. "
                "For class, it is faster to edit the notebook and re-launch the preview."
            )
            with gr.Row():
                base_city = gr.Textbox(value=session.base_city, label="From (city)")
                base_country = gr.Textbox(value=session.base_country, label="Country")
            with gr.Row():
                destination_city = gr.Textbox(value=session.destination_city, label="To (city)")
                destination_country = gr.Textbox(
                    value=session.destination_country, label="Country"
                )
            with gr.Row():
                base_currency = gr.Textbox(value=session.base_currency, label="Home currency")
                destination_currency = gr.Textbox(
                    value=session.destination_currency, label="Dest. currency"
                )
            traveller_profile = gr.Textbox(
                value=session.traveller_profile, label="Traveller profile"
            )
            generate_btn = gr.Button("Regenerate full trip")

            def regenerate_trip(
                b_city,
                b_country,
                d_city,
                d_country,
                b_curr,
                d_curr,
                profile,
            ):
                yield from run_tour_guide_progressive(
                    b_city,
                    b_country,
                    d_city,
                    d_country,
                    b_curr,
                    d_curr,
                    profile,
                    provider_value="auto",
                    temperature=0.3,
                    max_tokens=500,
                )

            generate_btn.click(
                regenerate_trip,
                inputs=[
                    base_city,
                    base_country,
                    destination_city,
                    destination_country,
                    base_currency,
                    destination_currency,
                    traveller_profile,
                ],
                outputs=[travel_brief, tool_table, places_table, poster_output, status],
                queue=True,
            )

        def show_preview():
            return (
                session.brief_md,
                session.tool_df,
                session.places_df,
                poster_path,
                "Showing your notebook travel guide (instant).",
            )

        refresh_btn.click(
            show_preview,
            outputs=[travel_brief, tool_table, places_table, poster_output, status],
            queue=False,
        )

    demo.queue(concurrency_count=1)
    return demo
