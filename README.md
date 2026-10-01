# Cognitive STT: Multilingual Speech-to-Text Module (FastAPI & Deepgram)

A high-performance Speech-to-Text (STT) service designed specifically for **intra-sentential code-switching (Hinglish: English & Hindi)** and **2-person conversational diarization**, powered by FastAPI and Deepgram's `nova-3` model.

---

## Key Features

- **Seamless Code-Switching (English + Hindi):** Uses Deepgram's flagship `nova-3` multilingual model with `language="multi"`. Speakers can switch freely between English and Hindi in the same conversation without language token lock-in or phonetic breakdown.
- **2-Person Speaker Diarization:** Distinguishes between speakers (`Speaker 0`, `Speaker 1`) with millisecond timestamps, speech bubbles, and chronological turn segmentation.
- **Versatile API Ingestion:**
  - `POST /api/v1/transcribe`: Ingest audio files via `multipart/form-data` (supports `.mp3`, `.wav`, `.m4a`, `.flac`, `.webm`, `.ogg`, `.aac`).
  - `POST /api/v1/transcribe/raw`: Ingest direct binary audio streams via request body.
  - `POST /api/v1/transcribe/url`: Ingest audio from remote URLs.
- **Single-Page Interactive UI:** Clean, responsive web interface served directly at `http://localhost:8000/` featuring:
  - Drag-and-drop audio file upload.
  - Quick-load button for local sample audio (`cost_of_care.mp3`).
  - In-browser microphone recorder (HTML5 MediaRecorder) with live timer.
  - Model & language configuration controls.
  - Color-coded speaker diarization speech bubbles with clickable audio seeking.
  - One-click transcript copying and `.txt` download.
- **Bruno Collection Included:** Ready-to-import Bruno API collection (`bruno_collection.json`) and native Bruno project folder (`bruno/`).

---

## Deepgram Model Recommendation for Hinglish

| Use Case | Model Parameter | Language Parameter | Behavior |
| :--- | :--- | :--- | :--- |
| **English-Hindi Code-Switching (Hinglish)** *(Default)* | `model="nova-3"` | `language="multi"` | **Recommended.** Natively switches between English and Hindi mid-sentence. Formats mixed transcripts accurately. |
| **English Clinical / Healthcare** | `model="nova-3-medical"` | `language="en"` | Specialized clinical terminology, **strictly English-only**. Hindi speech is not recognized. |
| **Monolingual Hindi** | `model="nova-3"` | `language="hi"` | Optimized for standard Hindi. |
| **Auto-Detect Dominant Language** | `model="nova-3"` | `language="auto"` | Detects single dominant language across entire audio. |

---

## Getting Started

### 1. Setup Virtual Environment & Install Dependencies

```bash
# Create virtual environment (if not already created)
python3 -m venv .venv
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Configure Your Deepgram API Key

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and replace `YOUR_DEEPGRAM_API_KEY_HERE` with your actual Deepgram key:
   ```ini
   DEEPGRAM_API_KEY=your_actual_deepgram_api_key
   HOST=0.0.0.0
   PORT=8000
   DEFAULT_MODEL=nova-3
   DEFAULT_LANGUAGE=multi
   ```

### 3. Run the Application

```bash
# Start the FastAPI server using uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Interactive Web UI:** Open [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs:** Open [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check Endpoint:** [http://localhost:8000/health](http://localhost:8000/health)

---

## Using with Bruno API Client

You can use the included Bruno collection in two ways:

### Method A: Single-File Import (Recommended)
1. Open the **Bruno** application.
2. Click **Collection** in the top navigation or the **"+" (plus)** icon.
3. Select **Import Collection** -> **Bruno Collection**.
4. Choose the `bruno_collection.json` file located in the root of this repository.
5. In Bruno, select the **Local** environment (which pre-configures `{{baseUrl}} = http://localhost:8000`).

### Method B: Open Collection Folder Directly
1. Open Bruno and choose **Open Collection**.
2. Select the `bruno/` directory inside this repository.

### Included Bruno Requests:
1. `01_health_check`: Validates server status and API key presence.
2. `02_transcribe_audio_file_hinglish`: Multipart upload of `cost_of_care.mp3` with `nova-3` + `multi`.
3. `03_transcribe_audio_file_medical`: Multipart upload with `nova-3-medical` + `en`.
4. `04_transcribe_raw_stream`: Binary audio stream upload.
5. `05_transcribe_url`: Remote audio URL transcription.
6. `06_get_sample_audio`: Streams local test audio from backend.

---

## API Usage Examples

### 1. Transcribe Audio File (cURL)

```bash
curl -X POST "http://localhost:8000/api/v1/transcribe" \
  -F "file=@cost_of_care.mp3" \
  -F "model=nova-3" \
  -F "language=multi" \
  -F "diarize=true" \
  -F "smart_format=true"
```

### 2. Transcribe Audio File (Python `requests` / `httpx`)

```python
import httpx

url = "http://localhost:8000/api/v1/transcribe"
with open("cost_of_care.mp3", "rb") as f:
    files = {"file": ("cost_of_care.mp3", f, "audio/mpeg")}
    data = {
        "model": "nova-3",
        "language": "multi",
        "diarize": "true",
        "smart_format": "true"
    }
    response = httpx.post(url, files=files, data=data, timeout=60.0)
    result = response.json()

print("Full Transcript:\n", result["transcript"])
print("\nDiarized Utterances:")
for utterance in result["diarized_utterances"]:
    print(f"Speaker {utterance['speaker']} [{utterance['start']:.1f}s - {utterance['end']:.1f}s]: {utterance['text']}")
```

### 3. Transcribe Remote Audio URL

```bash
curl -X POST "http://localhost:8000/api/v1/transcribe/url" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://static.deepgram.com/examples/Bueller-Life-moves-pretty-fast.wav",
    "model": "nova-3",
    "language": "multi",
    "diarize": true
  }'
```

---

## Project Structure

```
cognitive_module/
├── app/
│   ├── __init__.py
│   ├── config.py              # Settings & .env loading (Pydantic Settings)
│   ├── schemas.py             # Pydantic schemas for requests & responses
│   ├── main.py                # FastAPI routes, CORS, and endpoint handlers
│   ├── services/
│   │   ├── __init__.py
│   │   └── deepgram_service.py # Async Deepgram client wrapper & response parser
│   └── static/
│       └── index.html         # Modern single-page interactive UI
├── bruno/                     # Native Bruno directory
│   ├── bruno.json
│   ├── environments/
│   │   └── local.bru
│   ├── 01_health_check.bru
│   ├── 02_transcribe_audio_file_hinglish.bru
│   ├── 03_transcribe_audio_file_medical.bru
│   ├── 04_transcribe_raw_stream.bru
│   ├── 05_transcribe_url.bru
│   └── 06_get_sample_audio.bru
├── bruno_collection.json      # Importable Bruno Collection JSON
├── tests/
│   ├── __init__.py
│   └── test_api.py            # Unit & integration test suite
├── .env.example               # Template environment configuration
├── .env                       # Local environment file
├── .gitignore                 # Git ignore rules
├── pytest.ini                 # Pytest configuration
├── requirements.txt           # Python package dependencies
├── cost_of_care.mp3           # Sample audio file
└── README.md                  # Documentation and API reference
```

---

## Running the Automated Test Suite

```bash
source .venv/bin/activate
pytest tests/ -v
```
