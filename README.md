# Pfarvis Assistant

*A 100% local, privacy-first AI assistant for a manager's scattered notes: it answers questions about deadlines and commitments, and reminds you before they happen.*

Prototype built for the Pfizer Newspage hackathon.

A manager's notes live everywhere: text files, Markdown, PDFs, Word documents and photos of whiteboards, spread across cluttered folders. Pfarvis Assistant runs entirely on the manager's own machine and:

- **Ingests** every note in a folder, including text read from images (OCR).
- **Keeps itself up to date.** Add, edit, rename or delete a file and the index updates within about a second.
- **Links related notes** that share a client or a `#tag`.
- **Answers questions** in a chat, with follow-ups like *"Which client?"* and *"Is it virtual or in person?"*, and cites the `[filename]` behind every answer.
- **Raises reminders** such as *"You have a call with the clients at 4pm today"*.

It never invents an answer. If the notes don't contain it, it replies **"I couldn't find that in your notes."**

### Nothing leaves the machine

| Part | Runs on |
|---|---|
| LLM | Ollama at `127.0.0.1:11434` (default model `llama3`) |
| Embeddings | `all-MiniLM-L6-v2`, loaded from `models/` |
| Vector DB | ChromaDB on disk in `data/chroma/`, telemetry off |
| Knowledge graph | NetworkX, in memory |
| OCR | Local Tesseract binary |
| UI | Streamlit bound to `127.0.0.1:8501`, usage stats off |

The only script that ever touches the internet is `scripts/prefetch_models.py`, and only if you need to re-download the embedding model.

---

## Quick start

### 1. One-time setup

```powershell
py -3.11 -m venv .venv            # macOS/Linux: python3.11 -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Then install the two local programs:

| Program | Windows | macOS | Ubuntu |
|---|---|---|---|
| Ollama | `winget install Ollama.Ollama` | `brew install ollama` | official install script |
| Tesseract (OCR) | `winget install UB-Mannheim.TesseractOCR` | `brew install tesseract` | `sudo apt install tesseract-ocr` |

Pull the local LLM once:

```
ollama pull llama3
```

**Embedding model:** it ships in the repo at `models/all-MiniLM-L6-v2/`, so there is nothing to download. If that folder is missing or damaged, re-fetch it with `python scripts/prefetch_models.py --force`.

**Tesseract** is found automatically in its standard install folders, even when it isn't on PATH. Set `TESSERACT_CMD` only if you installed it somewhere else. Without Tesseract, image notes are skipped.

### 2. Run

```
ollama serve              # skip if the Ollama app is already running
streamlit run app.py      # from the project root
```

Open **http://localhost:8501**.

That single `streamlit run app.py` runs the whole project:

1. **Indexes your notes.** Every file in `NOTE_DIR` goes into the vector DB and the knowledge graph.
2. **Starts live sync,** which watches the folder for changes.
3. **Starts the reminder scheduler.**
4. **Serves the chat app.**

You don't need to run `pipeline.py` separately. Running it alongside the app makes two processes write to the same database. Stop everything with Ctrl+C.

To use your own notes, point `NOTE_DIR` at them before starting:

```powershell
$env:NOTE_DIR = "C:\Users\me\Documents\notes"   # macOS/Linux: export NOTE_DIR=~/notes
streamlit run app.py
```

---

## How it works

```
  NOTE_DIR  (.txt .md .pdf .docx .png .jpg .jpeg, any folder depth)
     |                                    ^
     | startup: scan_and_parse()          | file created / edited / moved / deleted
     v                                    |
  +---------------------------+     +---------------------------+
  | module_ingest             |     | module_ingest/watcher.py  |
  | parse, OCR, client, tags, |     | watchdog, 1 s debounce    |
  | event_time -> note dicts  |     +---------------------------+
  +---------------------------+                 |
     |                                          | add([note]) / remove(id)
     v                                          v
  +-----------------------------------------------------+
  | pipeline.py  run_ingestion_pipeline + start_live_sync|
  +-----------------------------------------------------+
     |
     v
  +---------------------------+  all_notes()  +---------------------------+
  | module_storage            | ------------> | module_ui/notifier.py     |
  | NotesStore:               |               | every 30 s: event due?    |
  | ChromaDB + NetworkX graph |               | -> desktop popup + queue  |
  +---------------------------+               +---------------------------+
     |  hybrid_search()                                 |
     v                                                  v
  +---------------------------+  AnswerDict   +---------------------------+
  | module_rag                | ------------> | module_ui (Streamlit)     |
  | rewrite -> search ->      |               | chat, notes sidebar,      |
  | date filter -> guardrails |  <----------  | reminder cards            |
  | -> Ollama (127.0.0.1)     |   question,   +---------------------------+
  +---------------------------+   history, pinned notes
```

### Ingestion (`module_ingest/`)

`parser.scan_and_parse(dir)` walks the folder recursively and builds one note dict per readable file, skipping hidden, temp and unsupported files. Each file type has its own reader:

| File type | Read with |
|---|---|
| `.txt`, `.md` | read directly as text |
| `.pdf` | pypdf |
| `.docx` | python-docx |
| `.png`, `.jpg`, `.jpeg` | Tesseract OCR |

`extract.py` then adds metadata to each note:
- **Client:** from a `Client: X` line, a known client name, or a `.../Clients/<Name>/` folder.
- **Tags:** `#hashtags` in the text.
- **`event_time`:** when the thing in the note happens. Phrases like "4pm today" resolve against the file's own modified time, not the current date.

### Live sync (`pipeline.py` + `module_ingest/watcher.py`)

After the first scan, a background watcher mirrors every change in `NOTE_DIR` into the store:

| On disk | In the vector DB and graph |
|---|---|
| New file, or file moved in | added |
| File edited | re-embedded under the same id |
| File renamed or moved | old entry removed, new one added |
| File deleted, or emptied | removed |
| Folder deleted or moved out | every note inside it removed |

Editors save in bursts, so changes are debounced (about 1 s per file). Changes made while the app is closed are picked up by the full scan at the next start, which also prunes deleted notes.

### Storage (`module_storage/`)

`NotesStore` is the single storage API:
- **Vector search:** ChromaDB with local `all-MiniLM-L6-v2` embeddings and cosine similarity from 0 to 1.
- **Knowledge graph:** NetworkX links notes that share a client or a tag.

`hybrid_search()` returns the semantic hits first, then their graph neighbours, scored 0.7 times their parent's score. It is thread-safe, because the watcher thread writes while the UI reads.

### Answering (`module_rag/`)

`pipeline.ask()` is the only function the UI calls. It answers in five steps:
1. Rewrites a follow-up ("Which client?") into a standalone question, using the chat history.
2. Runs a hybrid search over the notes.
3. Adds notes whose `event_time` falls in a time span named in the question ("in December", "this week", "the next 90 days"), using `dates.py`.
4. Adds any notes the user pinned.
5. Calls the local LLM with strict guardrails:
   - If no note scores at least `MIN_SIMILARITY`, it returns the fallback **without calling the LLM**.
   - Every answer must cite `[filename]`s that were actually in the context, or it is replaced by the fallback.
   - LLM errors also return the fallback.

### UI and reminders (`module_ui/`)

- **Chat:** answers come with expandable source citations.
- **Sidebar:** notes grouped by client, with a text filter and pin toggles.
- **Reminders:** an APScheduler job checks event times every `SCHEDULER_INTERVAL_SECONDS`. When an event is within `REMINDER_LEAD_MINUTES`, it shows a desktop popup and an in-app card. **Add to chat** posts the reminder and pins its note, so follow-up questions stay about it.

---

## Project layout

```
app.py                  Entry point: streamlit run app.py
pipeline.py             Initial indexing + live sync; python pipeline.py runs it without the UI
config.py               All settings and offline switches (imported first everywhere)
contracts.py            Shared data shapes: NoteItem, ContextItem, AnswerDict, fallback

module_ingest/          Files -> note dicts
  parser.py               scan_and_parse, parse_file
  ocr.py                  image_to_text (Tesseract)
  extract.py              client, tags, event_time
  watcher.py              live file watching (watchdog)
module_storage/         Note dicts -> searchable store
  store.py                NotesStore facade
  vector_db.py            ChromaDB + local embeddings
  knowledge_graph.py      NetworkX client/tag links
module_rag/             Question -> grounded answer
  pipeline.py             ask, rewrite_query
  engine.py               prompt -> LLM -> citation check
  guardrails.py           system prompt, context and citation checks
  dates.py                time-span detection in questions
  llm_client.py           Ollama wrapper
module_ui/              Streamlit app
  rag_app.py              wires the store, scheduler and chat together
  app.py                  chat layout and history
  chat_service.py         chat message and source types
  ui_app.py               notes sidebar, reminder panel
  notifier.py             reminder scheduler and desktop popups
  styles.css              app styling

models/all-MiniLM-L6-v2/  Bundled embedding model
sample_notes/           Fictional demo notes in deliberately messy folders
scripts/                Developer tools (see below)
tests/                  pytest suite
data/                   Created at runtime (ChromaDB); not committed
```

## Configuration

Everything is set in `config.py` and can be overridden with environment variables:

| Variable | Default | What it does |
|---|---|---|
| `NOTE_DIR` | `sample_notes` | Folder of notes to index and watch |
| `DATA_DIR` | `data` | Where ChromaDB stores its files |
| `OLLAMA_MODEL` | `llama3` | Local LLM to use (must be pulled) |
| `OLLAMA_TIMEOUT_SECONDS` | `120` | LLM timeout; the first call loads the model and is slow |
| `RETRIEVAL_TOP_K` | `5` | Semantic hits per question |
| `MIN_SIMILARITY` | `0.35` | Below this best score, answer with the fallback |
| `GRAPH_HOPS` | `1` | How far to follow client/tag links |
| `REMINDER_LEAD_MINUTES` | `15` | How early reminders fire |
| `SCHEDULER_INTERVAL_SECONDS` | `30` | How often reminders are checked |
| `TESSERACT_CMD` | auto-detected | Full path to `tesseract.exe` |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

`OLLAMA_HOST` is fixed to `127.0.0.1` on purpose and cannot be overridden.

## Developer tools

| Command | What it does |
|---|---|
| `python pipeline.py` | Index `NOTE_DIR`, then keep watching and log every change; no UI |
| `python scripts/search_notes.py "question"` | Show what the store returns for a question: score, semantic/graph, file |
| `python scripts/search_notes.py --list` | List every indexed note |
| `python scripts/try_rag.py` | Run the demo questions against the local LLM and print raw replies and citations |
| `python scripts/prefetch_models.py --force` | Re-download the embedding model (the only online step) |

## Demo

With the sample notes and Ollama running:

1. "When is my call with the clients?" → 4pm today `[call_prep.txt]`
2. "Which client?" → Quillmont Systems `[account_overview.md]`. The link comes from the shared `#q3-rollout` tag.
3. "Is it virtual or in person?" → virtual `[untitled 3.txt]`
4. "What is our parking reimbursement policy?" → *I couldn't find that in your notes.* No note mentions it.

**Live sync demo:** while the app runs, add a new `.md` file to `sample_notes/` and ask about it. Then delete the file and ask again.

**Reminders:** "4pm today" resolves against the file's modified time. On demo day, re-date `call_prep.txt` and run the demo after 15:45, or raise `REMINDER_LEAD_MINUTES`. To re-date the file:

- Windows: `(Get-Item "sample_notes\meetings\2026-11\call_prep.txt").LastWriteTime = Get-Date`
- macOS/Linux: `touch "sample_notes/meetings/2026-11/call_prep.txt"`

### Sample notes

`sample_notes/` holds 12 fictional notes in messy folders: `Work/Clients/…`, `Work/team/`, `meetings/`, `Desktop dump/` and `old_notes/2026/…`. Two of them are images (a whiteboard photo and a phone photo). All names and companies are invented. They set up the demo scenarios:

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
| `test_ingest.py` | parsing every sample file, OCR, client/tag/event extraction, watcher debounce and deletes |
| `test_dates.py` | time-span questions match the right event times |
| `test_guardrails.py` | weak context → fallback without calling the LLM; bad citations → fallback |
| `test_smoke.py` | every module imports, public signatures, the full note schema, reminders fire once, the app runs end to end, and no remote URLs anywhere |

## Troubleshooting

| Symptom | Fix |
|---|---|
| Sidebar says "Local model … is not running" | Start Ollama (`ollama serve` or the desktop app), check `ollama list` shows the model, then refresh |
| Error loading the embedding model | Check `models/all-MiniLM-L6-v2/config.json` exists; otherwise run `python scripts/prefetch_models.py --force` |
| Image notes never show up in answers | Install Tesseract, or set `TESSERACT_CMD`, then restart the app |
| "database is locked" | Only one process should use `data/chroma/`: don't run `python pipeline.py` while the app is running |
| A file change isn't picked up | Check it has a supported extension and isn't hidden (`.name`) or a temp file (`~$…`, `.swp`, `.tmp`) |
| Every answer is the fallback | Check `MIN_SIMILARITY`, and run `python scripts/search_notes.py "your question"` to see the scores |

## Going offline checklist

- [ ] `pip install -r requirements.txt` finished inside `.venv`
- [ ] `models/all-MiniLM-L6-v2/config.json` exists
- [ ] `ollama list` shows `llama3`
- [ ] Tesseract is installed (or `TESSERACT_CMD` is set)
- [ ] `OLLAMA_HOST` is not set to `0.0.0.0` in your shell
- [ ] `python -m pytest -q` passes
- [ ] Turn off Wi-Fi and unplug Ethernet
- [ ] Restart `streamlit run app.py`, ask "When is my call with the clients?", and confirm it answers with a `[filename]` citation

---

## Team

| Person | Area | Owns |
|---|---|---|
| 1 | Ingestion & OCR | `module_ingest/` |
| 2 | Vector DB & knowledge graph | `module_storage/` |
| 3 | LLM RAG & guardrails | `module_rag/` |
| 4 | Frontend, state & notifications | `module_ui/`, `app.py` |

### Shared contracts (`contracts.py`)

Every module passes data in these shapes:

- **`NoteItem`.** A note dict is exactly `asdict(NoteItem)`, with fields in this order: `id` (sha256 of the absolute path), `title`, `content`, `filepath` (absolute), `timestamp` (file modified time), `client`, `tags` (lowercase, no `#`), `event_time`. Datetimes are naive local time and stay `datetime` objects.
- **`ContextItem`.** `filepath`, `title`, `content`, `score` (0 to 1), `via` (`"semantic"` or `"graph"`).
- **`AnswerDict`.** `answer`, `sources` (cited filepaths), `grounded`.
- **`FALLBACK_MESSAGE`.** The fixed reply `"I couldn't find that in your notes."`, plus `fallback_answer()`.

`tests/test_contracts.py` and `tests/test_smoke.py` fail if a contract or a public signature drifts.

### Team rules

- `contracts.py` changes need agreement from all four.
- `config.py` changes must be announced to the team.
- No network code anywhere except `scripts/prefetch_models.py`. No hosted APIs, CDNs, web fonts or analytics, not even in comments.
- Heavy libraries are imported inside functions, so modules import with only streamlit installed.
- Every entry point imports `config` first.
- Keep `python -m pytest -q` green before pushing.
