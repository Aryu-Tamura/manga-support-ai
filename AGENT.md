# AGENT NOTES — NarrAIve (Manga Support AI)

This document captures the essentials an agent needs to quickly understand and work on the project.

## Project Snapshot
- **Goal**: Streamlit app that helps editors convert long-form prose into comic panels with metadata (speaker, tone, etc.) using OpenAI Chat Completions.
- **Main entry point**: `app.py`
- **Primary dependencies**: `streamlit`, `openai>=1.51`, `python-dotenv`, `pandas`, `tqdm`, `regex`, `beautifulsoup4`, `ebooklib` (future work), `loguru`.
- **Runtime expectations**: OpenAI API access (GPT-5 family by default), text length around 10k chars per run, Streamlit UI served locally.

## Environment & Secrets
- API key is loaded from `.env` via `python-dotenv`.
- Variable name: `OPENAI_API_KEY`; stored in memory as `api_key`.
- Streamlit refuses to start if the key is missing or not prefixed with `sk-`.

## High-Level Flow (`app.py`)
1. Load env vars, initialize OpenAI client.
2. User pastes prose or loads a saved project JSON.
3. Text is chunked (`chunk_for_llm`) with sliding window to stay within LLM context.
4. Each chunk is sent to the LLM (`call_llm_chunk`), expecting a JSON array of panel metadata.
5. Panels normalized into `Panel` dataclass instances (`llm_cut_and_label`), with fallback to a single narration panel if parsing fails.
6. UI displays filters, panel table, download button, and two auxiliary tools:
   - **Plot variants**: `llm_plot_variants` generates three rewrite options for a selected panel range.
   - **Character sheet**: `build_character_brief` filters panels by name; `llm_character_sheet` produces Markdown guidance.

## Data Model
- `Panel` fields: `id`, `text`, `type`, `speaker`, `time`, `location/scene`, `tone`, `emotion`, `action`, `entities`, `source_span`, `checksum`.
- Export schema saved as JSON: top-level `schema` (`comicizer/v1`), `model`, `cut_policy`, `windowing`, `doc_meta`, `full_text`, `panels[]`.

## Streamlit UI Landmarks
- Sidebar: style preset, chunk size info, download/upload controls.
- Main area:
  - Text input (cleared via reset button).
  - Results table with filters (`type`, `time`, `tone`, keyword).
  - Plot support section (select range, generate variants).
  - Character support section (name input → filtered panels → Markdown sheet).

## Error Handling & Logs
- Missing/invalid API key halts early.
- JSON parsing errors per chunk trigger fallback panel creation and log warnings.
- Authentication errors bubble up with Streamlit error message.
- Progress tracked with `st.progress` and `st.empty` status text.
- Logging via standard `logging` (not `loguru` yet).

## Common Tasks
- **Run locally**: `pip install -r requirements.txt`, ensure `.env` in place, then `streamlit run app.py`.
- **Adjust panel granularity**: tweak `CUT_MIN`, `CUT_MAX`, `TARGET_*`.
- **Adapt for longer texts**: adjust `WINDOW`/`OVERLAP`.
- **Switch models**: change `MODEL` constant (current default `gpt-5-mini`).
- **Troubleshoot**: check API key validity, inspect console logs for chunk failures, verify JSON format returned by the model.

## Future Work Hooks
- README mentions planned features: plot idea suggestions, character summaries via RAG, EPUB ingestion.
- `ebooklib`, `beautifulsoup4`, `regex` included for upcoming parsing features but not heavily used yet.
- Consider unit-style tests for `chunk_for_llm`, `safe_json_loads`, and fallback logic.

## Quick Verification Checklist
1. `.env` present with valid `OPENAI_API_KEY`.
2. Streamlit launches without `NameError` (ensure variable names match).
3. Upload/download flows maintain `panels` and `source_text`.
4. Plot variants and character sheet buttons produce output (with API enabled).
5. No stray prints; rely on logging and Streamlit messages.

Happy hacking! Reach out to the README for user-facing guidance, while this file keeps the implementation picture handy.

