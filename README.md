# Natural Language Google Sheets Dashboard (FastAPI)

This app lets non-technical users query a Google Sheet using natural language, view results in a table, and download CSVs.

## Features
- Natural language to structured filters using OpenAI or heuristics
- Read-only access to a shared Google Sheet; write support for new contacts
- Web dashboard (HTML) to enter queries and render tabular results
- Download results as CSV or Excel (.xlsx)
- Simple in-memory cache with manual invalidation on writes

## Requirements
- Python 3.10+
- A Google service account with access to the sheet (share the sheet with the service account email)
- OpenAI API key

## Setup
1. Clone repo and create virtualenv
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure environment:
   - Copy `.env.example` to `.env` and fill values
   - Ensure your Google Sheet is shared with the service account email
   - Optionally set `INGEST_TOKEN` for secure POST ingestion
4. Run the server:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
5. Open `http://localhost:8000` in your browser.

## Environment Variables
- `OPENAI_API_KEY`: OpenAI API key
- `GOOGLE_SHEET_ID`: Spreadsheet ID from Google Sheets URL
- `GOOGLE_SERVICE_ACCOUNT_FILE`: Absolute path to service account JSON (local dev)
- `GOOGLE_SERVICE_ACCOUNT_JSON_BASE64`: Base64 JSON for service account (deployment)
- `INGEST_TOKEN`: Shared secret for `/api/contacts/new`

## Ingestion API (for GPT workflows)
Send contacts to be appended to the `New contacts` worksheet. The worksheet is created automatically if missing.

- Endpoint: `POST /api/contacts/new`
- Header: `X-INGEST-TOKEN: <INGEST_TOKEN>`
- Body example:
```json
{
  "worksheet": "New contacts",
  "contacts": [
    {
      "Event name": "Example Event",
      "Contact Name": "Jane Doe",
      "Email": "jane@example.com",
      "Position": "Booker",
      "Linkedin contact": "https://linkedin.com/in/jane"
    }
  ]
}
```
- Response: `{ "inserted": 1, "worksheet": "New contacts" }`

## Exports
- CSV: `POST /download` (uses the same query form)
- Excel: `POST /download-xlsx`

## Cache
- Reads are cached per worksheet. Any ingestion write invalidates that worksheet cache.
