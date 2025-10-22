import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, Form, Header, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader, select_autoescape
import io
import pandas as pd

from .core.config import get_settings
from .services.sheets import GoogleSheetsClient
from .services.nl_query import NLQueryTranslator, NLQuerySpec
from .services.query_engine import apply_query_spec
from .services.contact_extractor import extract_contact_fields, SHEET_HEADERS
from .services.data_processor import data_processor
from .services.generic_handler import generic_handler

load_dotenv()

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

if not os.path.exists(TEMPLATES_DIR):
    os.makedirs(TEMPLATES_DIR, exist_ok=True)
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)

jinja_env = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    autoescape=select_autoescape(["html", "xml"]),
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

settings = get_settings()
sheets_client = GoogleSheetsClient()

# Force read OpenAI key from .env file, ignoring shell environment
from dotenv import dotenv_values
env_values = dotenv_values(".env")
openai_key_from_env = env_values.get("OPENAI_API_KEY")
translator = NLQueryTranslator(api_key=openai_key_from_env)

# Simple in-memory cache with manual invalidation
_CACHE: Dict[str, pd.DataFrame] = {}

def _cache_key(sheet_id: str, worksheet: Optional[str]) -> str:
    return f"{sheet_id}:{worksheet or 'sheet1'}"


def _read_dataframe_cached(sheet_id: str, worksheet: Optional[str]) -> pd.DataFrame:
    key = _cache_key(sheet_id, worksheet)
    if key in _CACHE:
        return _CACHE[key]
    df = sheets_client.read_dataframe(sheet_id, worksheet)
    _CACHE[key] = df
    return df


def _invalidate_cache(sheet_id: str, worksheet: Optional[str]):
    key = _cache_key(sheet_id, worksheet)
    _CACHE.pop(key, None)


class QueryRequest(BaseModel):
    query: str
    worksheet: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    worksheet_names = sheets_client.list_worksheets(settings.google_sheet_id)
    template = jinja_env.get_template("index.html")
    html = template.render(worksheet_names=worksheet_names)
    return HTMLResponse(html)


@app.post("/query", response_class=HTMLResponse)
async def run_query(request: Request, query: str = Form(...), worksheet: Optional[str] = Form(None)) -> HTMLResponse:
    worksheet_names = sheets_client.list_worksheets(settings.google_sheet_id)
    
    result_df = None
    selected_ws = None
    no_results_message = ""
    
    # If a specific worksheet is selected, search only in that worksheet
    if worksheet and worksheet.strip() in [name.strip() for name in worksheet_names]:
        # Find the exact worksheet name (handle spaces and case)
        exact_worksheet_name = None
        for name in worksheet_names:
            if name.strip().lower() == worksheet.strip().lower():
                exact_worksheet_name = name
                break
        
        if exact_worksheet_name:
            try:
                # Get columns dynamically for the selected worksheet
                available_columns = sheets_client.get_columns(settings.google_sheet_id, exact_worksheet_name)
                spec: NLQuerySpec = translator.translate(
                    user_query=query,
                    available_columns=available_columns,
                    default_worksheet=exact_worksheet_name,
                )
                df = _read_dataframe_cached(settings.google_sheet_id, exact_worksheet_name)
                result_df = apply_query_spec(df, spec)
                selected_ws = exact_worksheet_name
                
                # If no results from structured query, try generic search
                if result_df is None or result_df.empty:
                    try:
                        df = _read_dataframe_cached(settings.google_sheet_id, exact_worksheet_name)
                        if df is not None and not df.empty:
                            result_df = generic_handler.enhanced_generic_search(df, query)
                            selected_ws = exact_worksheet_name
                    except Exception:
                        pass
                
                # If still no results, try enhanced search
                if result_df is None or result_df.empty:
                    try:
                        df = _read_dataframe_cached(settings.google_sheet_id, exact_worksheet_name)
                        if df is not None and not df.empty:
                            enhanced_results = data_processor.enhanced_search(df, query)
                            if not enhanced_results.empty:
                                result_df = enhanced_results.head(10)  # Limit to 10 results
                                selected_ws = exact_worksheet_name
                    except Exception:
                        pass
                
                # If no results found in the selected worksheet
                if result_df is None or result_df.empty:
                    no_results_message = f"No records found in '{exact_worksheet_name}' worksheet for the query: '{query}'"
                    
            except Exception as e:
                no_results_message = f"Error searching in '{exact_worksheet_name}' worksheet: {str(e)}"
        else:
            no_results_message = f"Worksheet '{worksheet}' not found. Available worksheets: {', '.join(worksheet_names)}"
    
    # If no worksheet is selected, search all worksheets (fallback behavior)
    elif not worksheet:
        # Search all worksheets dynamically
        query_lower = query.lower()
        priority_sheets = []
        
        # First, try sheets that might be most relevant based on query content
        for ws_name in worksheet_names:
            ws_lower = ws_name.lower()
            # Check if worksheet name contains any terms from the query
            query_terms = query_lower.split()
            relevance_score = sum(1 for term in query_terms if term in ws_lower)
            if relevance_score > 0:
                priority_sheets.append((ws_name, relevance_score))
        
        # Sort by relevance score (highest first)
        priority_sheets.sort(key=lambda x: x[1], reverse=True)
        priority_sheets = [ws[0] for ws in priority_sheets]
        
        # Add remaining sheets that weren't prioritized
        for ws_name in worksheet_names:
            if ws_name not in priority_sheets:
                priority_sheets.append(ws_name)
        
        for ws_name in priority_sheets:
            try:
                # Try structured query first
                available_columns = sheets_client.get_columns(settings.google_sheet_id, ws_name)
                spec_try: NLQuerySpec = translator.translate(
                    user_query=query,
                    available_columns=available_columns,
                    default_worksheet=ws_name,
                )
                df_try = _read_dataframe_cached(settings.google_sheet_id, spec_try.worksheet)
                result_try = apply_query_spec(df_try, spec_try)
                if result_try is not None and not result_try.empty:
                    result_df = result_try
                    selected_ws = ws_name
                    break
            except Exception:
                # If structured query fails, try generic search
                try:
                    df_try = _read_dataframe_cached(settings.google_sheet_id, ws_name)
                    if df_try is not None and not df_try.empty:
                        result_try = generic_handler.enhanced_generic_search(df_try, query)
                        if result_try is not None and not result_try.empty:
                            result_df = result_try
                            selected_ws = ws_name
                            break
                except Exception:
                    continue
        
        # If still no results, try enhanced search across all worksheets
        if result_df is None or result_df.empty:
            for ws_name in worksheet_names:
                try:
                    # Use enhanced search with data processor
                    df_try = _read_dataframe_cached(settings.google_sheet_id, ws_name)
                    if df_try is not None and not df_try.empty:
                        # Enhanced search using data processor
                        enhanced_results = data_processor.enhanced_search(df_try, query)
                        if not enhanced_results.empty:
                            result_df = enhanced_results.head(10)  # Limit to 10 results
                            selected_ws = ws_name
                            break
                except Exception:
                    continue
        
        # If no results found across all worksheets
        if result_df is None or result_df.empty:
            no_results_message = f"No records found across all worksheets for the query: '{query}'"
    
    # If worksheet is specified but doesn't exist
    else:
        no_results_message = f"Worksheet '{worksheet}' not found. Available worksheets: {', '.join(worksheet_names)}"

    # Render table
    template = jinja_env.get_template("index.html")
    html_table = result_df.to_html(classes="result-table", index=False, border=0) if result_df is not None and not result_df.empty else ""
    html = template.render(
        worksheet_names=worksheet_names,
        selected_worksheet=worksheet,
        query_text=query,
        table_html=html_table,
        no_results_message=no_results_message,
    )
    return HTMLResponse(html)


@app.post("/download")
async def download_csv(query: str = Form(...), worksheet: Optional[str] = Form(None)):
    spec: NLQuerySpec = translator.translate(
        user_query=query,
        available_columns=sheets_client.get_columns(settings.google_sheet_id, worksheet),
        default_worksheet=worksheet,
    )
    df = _read_dataframe_cached(settings.google_sheet_id, spec.worksheet)
    result_df = apply_query_spec(df, spec)

    buf = io.StringIO()
    result_df.to_csv(buf, index=False)
    buf.seek(0)

    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=results.csv"},
    )


# Excel export
@app.post("/download-xlsx")
async def download_xlsx(query: str = Form(...), worksheet: Optional[str] = Form(None)):
    spec: NLQuerySpec = translator.translate(
        user_query=query,
        available_columns=sheets_client.get_columns(settings.google_sheet_id, worksheet),
        default_worksheet=worksheet,
    )
    df = _read_dataframe_cached(settings.google_sheet_id, spec.worksheet)
    result_df = apply_query_spec(df, spec)

    import xlsxwriter  # requires dependency installed
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        result_df.to_excel(writer, index=False, sheet_name="Results")
    output.seek(0)

    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=results.xlsx"},
    )


# Ingestion endpoint: send new contacts from GPT workflow
class IngestRequest(BaseModel):
    contacts: List[Dict[str, Any]]
    worksheet: Optional[str] = "New contacts"


@app.post("/api/contacts/new")
async def add_new_contacts(payload: IngestRequest, x_ingest_token: Optional[str] = Header(None)):
    expected = os.environ.get("INGEST_TOKEN")
    if not expected or x_ingest_token != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

    worksheet = payload.worksheet or "New contacts"
    rows = payload.contacts or []
    if not rows:
        return {"inserted": 0}

    # Normalize keys to match sheet headers casing as-is
    normalized_rows: List[Dict[str, Any]] = []
    for r in rows:
        normalized = {k.replace(" ", "_").strip(): v for k, v in r.items()}
        normalized.update(r)
        normalized_rows.append(normalized)

    sheets_client.append_rows(settings.google_sheet_id, worksheet, normalized_rows)

    _invalidate_cache(settings.google_sheet_id, worksheet)

    return {"inserted": len(rows), "worksheet": worksheet}


# Add contact via NL text; appends one row to "New contacts"
class AddContactRequest(BaseModel):
    query: str


@app.post("/add_contact")
async def add_contact(data: AddContactRequest):
    text = data.query
    fields = extract_contact_fields(translator, text)

    # Ensure worksheet and append in the canonical headers order
    worksheet = "New contacts"
    row = {k: fields.get(k, "") for k in SHEET_HEADERS}
    sheets_client.append_rows(settings.google_sheet_id, worksheet, [row])

    _invalidate_cache(settings.google_sheet_id, worksheet)

    return JSONResponse({"status": "success", "message": "Contact added successfully!"})


# Data quality endpoint
@app.get("/api/data-quality/{worksheet}")
async def get_data_quality(worksheet: str):
    """Get data quality metrics for a specific worksheet."""
    try:
        df = _read_dataframe_cached(settings.google_sheet_id, worksheet)
        if df is None or df.empty:
            return JSONResponse({"error": "Worksheet not found or empty"}, status_code=404)
        
        metrics = data_processor.get_data_quality_metrics(df)
        return JSONResponse(metrics)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# Enhanced search endpoint
@app.post("/api/search")
async def enhanced_search(query: str = Form(...), worksheet: Optional[str] = Form(None)):
    """Enhanced search endpoint with better matching algorithms."""
    try:
        if not worksheet:
            worksheet_names = sheets_client.list_worksheets(settings.google_sheet_id)
            # Search all worksheets
            all_results = []
            for ws_name in worksheet_names:
                try:
                    df = _read_dataframe_cached(settings.google_sheet_id, ws_name)
                    if df is not None and not df.empty:
                        results = data_processor.enhanced_search(df, query)
                        if not results.empty:
                            results['worksheet'] = ws_name
                            all_results.append(results)
                except Exception:
                    continue
            
            if all_results:
                combined_results = pd.concat(all_results, ignore_index=True)
                return JSONResponse({
                    "results": combined_results.to_dict('records'),
                    "total_count": len(combined_results)
                })
            else:
                return JSONResponse({"results": [], "total_count": 0})
        else:
            # Search specific worksheet
            df = _read_dataframe_cached(settings.google_sheet_id, worksheet)
            if df is None or df.empty:
                return JSONResponse({"error": "Worksheet not found or empty"}, status_code=404)
            
            results = data_processor.enhanced_search(df, query)
            return JSONResponse({
                "results": results.to_dict('records'),
                "total_count": len(results)
            })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
