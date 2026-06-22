# Generative AI Practical Workshop

Materials for a hands-on session on prompting, simple LLM apps, and multimodal AI.

You do **not** need every API key in the list below. One cloud key **or** a local Ollama install is enough.

---

## Before the workshop (do this at home)

1. Install Python 3.10+ and create a folder for the project.
2. Open a terminal in the project folder and run:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

3. Open `.env` and paste **one** API key you have (or skip this if you will use Ollama only).

   **Environment files (important):**
   - `.env.example` — template in the repo (no secrets); do not put real keys here
   - `.env` — your private copy with real keys; created by `cp .env.example .env`; never commit or push this file

4. If using Ollama: install it from [ollama.com](https://ollama.com), then run `ollama serve` and `ollama pull llama3.2` (any installed model works).
5. Start Jupyter from the project folder:

```bash
jupyter notebook
```

6. Open the notebooks **in order** (see below). Run cells from top to bottom.

If something fails in Step 1 of a notebook, check you started Jupyter from the project root (the folder that contains `src/` and `notebooks/`).

---

## Notebooks (main path)

| Order | File | What you will do |
|------:|------|------------------|
| 1 | `notebooks/01_prompt_engineering_lab.ipynb` | Write and test prompts |
| 2 | `notebooks/02_build_simple_llm_app_gradio.ipynb` | Turn a prompt into a small Gradio app |
| 3 | `notebooks/03_multimodal_tour_guide_agent.ipynb` | Tools + text + audio + image |

The notebooks contain the explanations. The `src/` folder holds short helper code so cells stay readable.

---

## Project layout

```text
generative-ai-workshop/
├── notebooks/              ← start here
├── src/                    ← helpers (LLM calls, prompts, tools, Gradio)
├── data/sample_texts/      ← example inputs
├── data/outputs/           ← saved notebook outputs
├── .env.example            ← template (committed); copy to .env and add keys
├── .env                    ← your secrets (local only — not in git)
└── requirements.txt
```

---

## Privacy

Do not paste passwords, student records, exam scripts, or personal documents into cloud models. For sensitive work, use Ollama locally or anonymised data.

**Do not commit `.env`.** Only `.env.example` belongs in the repository. If you accidentally commit a key, rotate it immediately on the provider’s website and remove the file from git history.

---

## Supported models

OpenAI, Anthropic, Gemini, Mistral, Cohere, DeepSeek, Groq, OpenRouter, and **Ollama** (local).

Set `SELECTED_PROVIDER = "auto"` in a notebook to use the first available option, or set e.g. `"ollama"` or `"openai"` to choose one.

---

## The point of the workshop

Chatting with AI is easy. The skill is making outputs **useful, testable, and responsible** — especially in university settings.
