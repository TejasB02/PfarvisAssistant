# PfarvisAssistant
Building a prototype for Hackathon organized by Pfizer- Newspage
A 100% local, privacy-first AI knowledge assistant that advises on deadlines and commitments

# Local Notes Assistant

*A 100% local, privacy-first AI knowledge assistant that advises on deadlines and commitments.*

A manager's notes are scattered across cluttered folders: text files, Markdown, PDFs, Word documents and photos of whiteboards. This assistant runs entirely on the manager's own machine and does four things:

- It **ingests** all of those notes.
- It **links** related notes (same client, shared `#tags`).
- It **answers questions** in a Streamlit chat, including follow-ups like *"Which client?"* and *"Is it virtual or in person?"*.
- It **raises reminders** such as *"You have a call with the clients at 4pm today"*.

The local LLM uses conversation history to decide when to search notes, ask for clarification, or explain that a request is outside its scope. Answers based on notes include expandable source references.

**Nothing leaves the machine.** The local services it uses:

| What | How |
|---|---|
| LLM | Ollama at `127.0.0.1:11434` |
| Embeddings | `all-MiniLM-L6-v2`, loaded from `models/` |
| Vector DB | ChromaDB on disk in `data/`, telemetry off |
| OCR | Local Tesseract binary |
| Voice | Whisper Tiny English, loaded from `models/whisper-tiny.en/` |
| UI | Streamlit bound to `127.0.0.1`, usage stats off |

Install the dependencies and download the models once before going offline.

---

## Data flow

```
  sample_notes/   (config.NOTE_DIR: .txt .md .pdf .docx .png .jpg .jpeg)
       |
       v
  +-----------------------------+
  | module_ingest    (Person 1) |   scan_and_parse() -> list[dict]  (== asdict(NoteItem))
  +-----------------------------+
       |  store.add(notes)                notes also feed reminders (Person 4)
       v
  +-----------------------------+  all_notes()  +-------------------------------+
  | module_storage   (Person 2) | ------------> | module_ui/notifier (Person 4) |
  | NotesStore: Chroma + graph  |               | every 30 s: any event due?    |
  +-----------------------------+               | -> desktop popup + queue      |
       |                                        +-------------------------------+
       |  hybrid_search() -> list[ContextItem]          |  pop_pending_reminders()
       v                                                v
  +-----------------------------+  AnswerDict   +-------------------------------+
  | module_rag       (Person 3) | ------------> | module_ui/ui_app   (Person 4) |
  | pipeline.ask -> guardrails  |               | Streamlit chat + sidebar      |
  | -> Ollama at 127.0.0.1      | <------------ | reminder card "Add to chat"   |
  +-----------------------------+   question,   +-------------------------------+
                                    history,
                                    pinned ids
```

`app.py` wires it together:

1. It runs `scan_and_parse` and `NotesStore.add` once, inside `st.cache_resource`.
2. It starts the reminder scheduler once, also inside `st.cache_resource`.
3. It renders Pfarvis Assistant with the cached store and notes on every rerun.

## Owners

| Person | Folder | Deliverable | Handoff |
|---|---|---|---|
| 1 - Ingestion & OCR | `module_ingest/` | `parser.scan_and_parse(dir_path) -> list[dict]` | note dicts -> Person 2 (`store.add`) and Person 4 (sidebar, reminders) |
| 2 - Vector DB & Knowledge Graph | `module_storage/` | `NotesStore` (`add`, `hybrid_search`, `get_note`, `all_notes`, `remove`) | `list[ContextItem]` -> Person 3 (`pipeline.ask`) |
| 3 - LLM RAG & Guardrails | `module_rag/` | `engine.answer_user_query(...) -> AnswerDict` and `pipeline.ask(...)` | `AnswerDict` -> Person 4 (UI) |
| 4 - Frontend, State & Notifications | `module_ui/`, `app.py` | `ui_app.main(store, notes)`, `notifier.start_scheduler(get_notes)` | the app the manager uses |

### Shared contracts (`contracts.py`)

All four people share `contracts.py`. **Any change needs agreement from all four.**

**`NoteItem`.** The note dict is exactly `asdict(NoteItem)`. Fields, in this order:

| Field | Type | Meaning |
|---|---|---|
| `id` | `str` | sha256 of the absolute filepath, from `make_note_id` |
| `title` | `str` | |
| `content` | `str` | |
| `filepath` | `str` | absolute path |
| `timestamp` | `datetime \| None` | file modified time |
| `client` | `str \| None` | |
| `tags` | `list[str]` | lowercase, no `#` |
| `event_time` | `datetime \| None` | when the thing in the note happens |

All datetimes are naive local time, and they stay `datetime` objects inside dicts.

**Other shared shapes:**

- `ContextItem`: `filepath`, `title`, `content`, `score` (0 to 1), `via` (`"semantic"` or `"graph"`).
- `AnswerDict`: `answer`, `sources` (cited filepaths), `grounded`.
- `FALLBACK_MESSAGE = "I couldn't find that in your notes."`, plus `fallback_answer()`.

`tests/test_contracts.py` and `tests/test_smoke.py` fail if a contract or a public signature drifts.

---

## Person 1: Ingestion & OCR (`module_ingest/`)

**Files**

| File | Contents |
|---|---|
| `parser.py` | `scan_and_parse`, `parse_file` |
| `ocr.py` | `image_to_text` |
| `extract.py` | `guess_client`, `extract_tags`, `extract_event_time` |
| `watcher.py` | `watch` (stretch goal) |

**Done when:** `scan_and_parse("sample_notes")` returns a real dict for every sample file, including OCR text from the PNG. In the dict for `call_prep.txt`, `event_time` is 16:00 on the date of the file's own modified time.

**Prompt to your AI tool** (it must keep every signature and the dict schema in `contracts.py` unchanged):

> I'm implementing `module_ingest/` in a Python 3.11 repo, "Local Notes Assistant". It must run 100% offline: no network calls, no cloud APIs, no hosted models. Do not change `contracts.py` or any public function signature. Each note dict must be exactly `asdict(NoteItem)` in field order, with `id = make_note_id(filepath)`, absolute `filepath`, naive local datetimes kept as `datetime` objects, and lowercase tags without `#`.
>
> 1. **`parser.scan_and_parse(dir_path)`**: scan the folder recursively, in sorted order, skipping hidden files and unsupported extensions. Return one dict per readable file and replace the mock data.
> 2. **`parser.parse_file(path)`**: dispatch on extension.
>    - `.txt` and `.md`: read as utf-8, falling back to `errors="replace"`.
>    - `.pdf`: pypdf.
>    - `.docx`: python-docx.
>    - `.png`, `.jpg`, `.jpeg`: `ocr.image_to_text`.
>
>    Title comes from the first heading or first non-empty line. Timestamp is the file's mtime. Unreadable or empty files return `None` and are logged. The scan must never crash.
> 3. **`ocr.image_to_text(path)`**: use pytesseract with the local Tesseract binary, honouring `config.TESSERACT_CMD`. Return `""` on any failure.
> 4. **`extract.guess_client`**: use an explicit `Client: X` line, or a known-client list built from `.../Clients/<Name>/` folders. Otherwise return `None`.
> 5. **`extract.extract_tags`**: return `#hashtags` plus a small documented keyword map.
> 6. **`extract.extract_event_time(text, reference)`**: use `dateparser.search.search_dates` with `RELATIVE_BASE=reference`. "today", "tomorrow" and "4pm" resolve against the note's own timestamp, never the current date. "Q3" or version numbers must not become dates.
>
> Import pypdf, docx, pytesseract, PIL and dateparser inside the functions that use them. Add `tests/test_ingest.py` covering every sample file in `sample_notes/`, and keep `pytest` green.

## Person 2: Vector DB & Knowledge Graph (`module_storage/`)

**Files**

| File | Contents |
|---|---|
| `store.py` | the public `NotesStore` facade |
| `vector_db.py` | ChromaDB plus local embeddings |
| `knowledge_graph.py` | NetworkX graph |

**Done when:** "Which client is the 4pm call with?" returns `call_prep.txt` as a semantic hit and the Quillmont note as a graph neighbour, linked by the shared `#q3-rollout` tag.

**Prompt to your AI tool** (it must keep every signature in `contracts.py` and `NotesStore` unchanged):

> I'm implementing `module_storage/` in a Python 3.11 repo, "Local Notes Assistant". It must run 100% offline. Do not change `contracts.py` or any public signature on `NotesStore`: `add`, `query`, `hybrid_search`, `get_note`, `all_notes`, `remove`.
>
> **Vectors (`vector_db.py`).**
> - Open Chroma with `chromadb.PersistentClient(path=config.CHROMA_DIR, settings=Settings(anonymized_telemetry=False))`.
> - Create the collection with `get_or_create_collection(name=config.CHROMA_COLLECTION, embedding_function=None, metadata={"hnsw:space": "cosine"})`.
> - Always pass our own embeddings: `embeddings=` on upsert and `query_embeddings=` on query. Never use `query_texts` or Chroma's default embedding function, which downloads a model.
> - Embed with `SentenceTransformer(config.EMBED_MODEL_PATH)`, loaded from the local folder and never by hub name, using `normalize_embeddings=True`.
> - Chroma metadata must be scalar. Store datetimes as ISO strings and tags as a comma-joined string, using `note_to_metadata` and `metadata_to_note`.
> - Similarity = 1 - cosine distance, clamped to 0..1.
>
> **Graph (`knowledge_graph.py`).** Use a networkx `Graph` with one node per note id. Add an edge when two notes share the same client or any tag, with a `reason` attribute such as `"tag:q3-rollout"`.
>
> **Search (`NotesStore.hybrid_search(query, k, hops)`).** Return the semantic hits first, then graph neighbours of those hits within `hops`. De-duplicate, and score each neighbour as its parent's score times `GRAPH_SCORE_DECAY` with `via="graph"`. Results are a `list[ContextItem]`.
>
> Import chromadb, sentence_transformers and networkx inside functions or `__init__`. Add `tests/test_storage.py` with a fake embedder so the tests don't need the model, and keep `pytest` green.

## Person 3: LLM RAG & Guardrails (`module_rag/`)

**Files**

| File | Contents |
|---|---|
| `llm_client.py` | Ollama wrapper |
| `guardrails.py` | `SYSTEM_PROMPT` and the checks |
| `engine.py` | `answer_user_query` |
| `pipeline.py` | `rewrite_query`, `ask` (the only function the UI calls) |

**Done when:** the demo conversation below works, every answer cites `[filename]`, and the parking question returns the fallback.

**Prompt to your AI tool** (it must keep every signature in `contracts.py` and `module_rag` unchanged):

> I'm implementing `module_rag/` in a Python 3.11 repo, "Local Notes Assistant". It must run 100% offline. The only LLM is Ollama through `ollama.Client(host=config.OLLAMA_HOST)` (127.0.0.1) with `model=config.OLLAMA_MODEL`. No other hosts and no hosted APIs. Do not change `contracts.py` or any public signature.
>
> 1. **`llm_client.is_available()`**: a fast check that never raises and verifies the model is pulled.
> 2. **`llm_client.chat(messages, temperature=0.0)`**: return the reply text, and raise on errors.
> 3. **`guardrails.SYSTEM_PROMPT`**: refine it, keeping these rules:
>    - Answer only from the CONTEXT block, and cite filenames in square brackets.
>    - If the answer isn't in the CONTEXT, reply exactly with `FALLBACK_MESSAGE`.
>    - Never use outside knowledge.
>    - If a question is ambiguous (two clients match), ask which one instead of guessing.
> 4. **Engine rules (keep these in `engine.answer_user_query`):**
>    - If `has_enough_context` is False, return `fallback_answer()` **without calling the LLM**.
>    - If `validate_citations` fails, return the fallback.
>    - LLM errors return the fallback.
> 5. **`pipeline.rewrite_query`**: when the history is non-empty, turn a follow-up like "Which client?" into a standalone retrieval question using a temperature-0 LLM call. If that call fails, return the query unchanged.
> 6. **`pipeline.ask`**: keep the flow rewrite -> `store.hybrid_search` -> add pinned notes -> `answer_user_query`.
>
> Import ollama inside functions. Tune `config.MIN_SIMILARITY` on `sample_notes/`. Extend `tests/test_guardrails.py`, mocking `llm_client.chat`, and keep `pytest` green.

## Person 4: Frontend, State & Notifications (`module_ui/`, `app.py`)

**Files**

| File | Contents |
|---|---|
| `app.py` | Pfarvis chat layout and history |
| `ui_app.py` | notes sidebar and reminder fragment |
| `notifier.py` | scheduler, desktop popup, queue |
| `app.py` | entry point |

**Done when:**

- A reminder toast and card appear within 30 s of start-up.
- "Add to chat" injects `Reminder: <title> [<filename>]` and pins the note.
- The follow-up "Is it virtual or in person?" is answered from the pinned note.

**Prompt to your AI tool** (it must keep every signature in `contracts.py`, `ui_app.main` and `notifier` unchanged):

> I'm implementing `module_ui/` and `app.py` in a Python 3.11 Streamlit (>= 1.37) repo, "Local Notes Assistant". It must run 100% offline: no CDNs, web fonts, analytics or remote assets. Do not change `contracts.py` or any public signature.
>
> **Chat.** Use `st.chat_message` and `st.chat_input`, with history in `st.session_state["messages"]`. Each assistant reply shows its source filepaths in a small expander. Get answers only from `module_rag.pipeline.ask(query, history, store, pinned_note_ids)`.
>
> **Sidebar.** Show notes grouped by client, with a text filter. Each note gets an expander with title, folder, event_time and tags, plus a pin toggle. Pinned ids live in `st.session_state["pinned_note_ids"]`.
>
> **Scheduler (`notifier.py`).**
> - Use an APScheduler `BackgroundScheduler` that runs every `config.SCHEDULER_INTERVAL_SECONDS`.
> - Fire once per note id when `event_time` falls within `config.REMINDER_LEAD_MINUTES`.
> - Show a desktop popup with `plyer`, wrapped in try/except.
> - Put reminder dicts on a thread-safe queue that `pop_pending_reminders()` drains.
> - The scheduler thread must never touch `st.session_state` or import streamlit.
>
> **Reminder UI.** An `st.fragment(run_every=...)` polls the queue. It shows `st.toast` plus a card with an "Add to chat" button. The button appends an assistant message `Reminder: <title> [<filename>]` followed by the note content, then pins that note id.
>
> **`app.py`.** `import config` stays the first import. Keep `scan_and_parse` + `NotesStore` and the scheduler inside `st.cache_resource`.
>
> Add `streamlit.testing.v1.AppTest` tests in `tests/test_ui.py`, and keep `pytest` green.

---

## One-time setup (online)

Clone the submission branch and run the setup from the repository root. These commands use Python 3.12, the version tested on macOS.

**macOS or Linux**

```sh
python3.12 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install Ollama for your operating system and start it. Run the following in the activated environment:

```sh
python -m pip install -r requirements.txt
ollama pull llama3.1:8b
python scripts/prefetch_models.py
```

The prefetch script installs MiniLM into `models/all-MiniLM-L6-v2/`. The demo uses `llama3.1:8b`, which you select explicitly at launch. The shared application default is `llama3`.

**Optional voice input.** Download the English Whisper model to its expected local folder:

```sh
python -c "from huggingface_hub import snapshot_download; snapshot_download('openai/whisper-tiny.en', local_dir='models/whisper-tiny.en', allow_patterns=['*.json', '*.safetensors', 'merges.txt', 'vocab.txt'])"
```

**Optional image OCR.** Install the local Tesseract binary to search for text in images.

- Windows uses `winget install UB-Mannheim.TesseractOCR`. Add the installation to PATH or set `TESSERACT_CMD` to the full executable path.
- macOS uses `brew install tesseract`.
- Ubuntu uses `sudo apt install tesseract-ocr`.

## Run offline

Start Ollama if it is stopped. If your installation does not launch it automatically, run `ollama serve` in a separate terminal. Launch Streamlit from the repo root with the environment activated.

**macOS or Linux**

```sh
OLLAMA_MODEL=llama3.1:8b python -m streamlit run app.py
```

**Windows PowerShell**

```powershell
$env:OLLAMA_MODEL = "llama3.1:8b"
python -m streamlit run app.py
```

Open [Pfarvis Assistant](http://127.0.0.1:8501). The app indexes `sample_notes/` by default. Set `NOTE_DIR` before launch to use another folder.

## Demo conversation

Type a question or click **Voice** beside the text field and grant browser microphone access. Record a short question, review or edit the transcript in the text field, and press **Send**. Expand the references beneath an answer to read the supporting notes.

1. "When is my call with the clients?" -> 4pm today `[call_prep.txt]`
2. "Which client?" -> Quillmont Systems `[account_overview.md]`. The link comes from the shared `#q3-rollout` tag.
3. "Is it virtual or in person?" -> virtual `[untitled 3.txt]`
4. "What is our parking reimbursement policy?" -> *I couldn't find that in your notes.* No note mentions it.

**Reminders.** "4pm today" resolves against the file's modified time. On demo day, re-date `call_prep.txt` and run the demo after 15:45, or raise `REMINDER_LEAD_MINUTES`. To re-date the file:

- Windows: `(Get-Item "sample_notes\meetings\2026-11\call_prep.txt").LastWriteTime = Get-Date`
- macOS/Linux: `touch "sample_notes/meetings/2026-11/call_prep.txt"`

### Sample notes

`sample_notes/` holds about 10 fictional notes spread across messy folders: `Work/Clients/…`, `Work/team/`, `meetings/`, `Desktop dump/` and `old_notes/2026/…`. One of them is a PNG of a whiteboard.

All names and companies are invented. They set up the demo scenarios:

- **One client in three folders.** Quillmont Systems is mentioned in `Work/Clients`, `old_notes` and `Desktop dump`.
- **A 4pm call split across files.** The call itself is in `meetings/`, the client name is in `Work/Clients/`, and the meeting format is in `Desktop dump/`.
- **Shared tags across folders.** `#q3-rollout`, `#budget`, `#offsite`, `#hiring`.
- **One topic that appears nowhere.** Parking reimbursement, used to demo the fallback.

## Tests

```
python -m pytest -q
```

| File | What it checks |
|---|---|
| `test_contracts.py` | `NoteItem` field names, order and defaults; deterministic ids; dict round-trip |
| `test_guardrails.py` | empty or weak context -> fallback without calling the LLM; bad or missing citations -> fallback |
| `test_smoke.py` | see below |

`test_smoke.py` covers the scaffold as a whole:

- Every module imports, without heavy dependencies loaded at import time.
- Every public signature matches this README.
- The sample scan returns the full note schema.
- A reminder fires exactly once.
- The Streamlit app runs end to end, via `AppTest`.
- No project file except `scripts/prefetch_models.py` contains a URL other than `127.0.0.1` or `localhost`.

## Going offline checklist

- [ ] `pip install -r requirements.txt` finished inside `.venv`
- [ ] `python scripts/prefetch_models.py` ran, and `models/all-MiniLM-L6-v2/config.json` exists
- [ ] `ollama list` shows `llama3.1:8b`, and the launch command selects it
- [ ] If demoing voice, `models/whisper-tiny.en/` is present and browser microphone permission works
- [ ] If demoing image OCR, `tesseract --version` works or `TESSERACT_CMD` is set
- [ ] `OLLAMA_HOST` is not set to `0.0.0.0` in your shell, so Ollama stays on 127.0.0.1
- [ ] `python -m pytest -q` passes
- [ ] Turn Wi-Fi off and unplug Ethernet
- [ ] Restart `streamlit run app.py`, ask "When is my call with the clients?", and confirm it still answers with a `[filename]` citation

## Team rules

- `contracts.py` changes need agreement from all four.
- `config.py` changes must be announced to the team.
- No network code anywhere except `scripts/prefetch_models.py`. No hosted APIs, CDNs, web fonts or analytics, not even in comments.
- Heavy libraries are imported inside functions, so the app imports with only streamlit installed.
- Every entry point imports `config` first.
- Each person adds their tests as `tests/test_<module>.py`. Keep `pytest` green before pushing.
