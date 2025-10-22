import base64
import json
import os
from typing import List, Optional, Dict, Any

import gspread
import pandas as pd
from google.oauth2.service_account import Credentials

from ..core.config import get_settings


SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]


class GoogleSheetsClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._gc = self._authorize()

    def _authorize(self):
        creds: Optional[Credentials] = None
        if self.settings.service_account_json_b64:
            data = base64.b64decode(self.settings.service_account_json_b64).decode("utf-8")
            info = json.loads(data)
            creds = Credentials.from_service_account_info(info, scopes=SCOPES)
        elif self.settings.service_account_file and os.path.exists(self.settings.service_account_file):
            creds = Credentials.from_service_account_file(self.settings.service_account_file, scopes=SCOPES)
        else:
            raise RuntimeError(
                "Google service account credentials not configured. Set GOOGLE_SERVICE_ACCOUNT_FILE or GOOGLE_SERVICE_ACCOUNT_JSON_BASE64."
            )
        return gspread.authorize(creds)

    def _open_sheet(self, sheet_id: str):
        return self._gc.open_by_key(sheet_id)

    def list_worksheets(self, sheet_id: str) -> List[str]:
        sh = self._open_sheet(sheet_id)
        return [ws.title for ws in sh.worksheets()]

    def get_columns(self, sheet_id: str, worksheet: Optional[str]) -> List[str]:
        sh = self._open_sheet(sheet_id)
        if worksheet:
            # Try exact match first
            try:
                ws = sh.worksheet(worksheet)
            except gspread.exceptions.WorksheetNotFound:
                # Try case-insensitive and trimmed match
                for ws_obj in sh.worksheets():
                    if ws_obj.title.strip().lower() == worksheet.strip().lower():
                        ws = ws_obj
                        break
                else:
                    raise gspread.exceptions.WorksheetNotFound(f"Worksheet '{worksheet}' not found")
        else:
            ws = sh.sheet1
        values = ws.get_all_values()
        return values[0] if values else []

    def read_dataframe(self, sheet_id: str, worksheet: Optional[str]) -> pd.DataFrame:
        sh = self._open_sheet(sheet_id)
        if worksheet:
            # Try exact match first
            try:
                ws = sh.worksheet(worksheet)
            except gspread.exceptions.WorksheetNotFound:
                # Try case-insensitive and trimmed match
                for ws_obj in sh.worksheets():
                    if ws_obj.title.strip().lower() == worksheet.strip().lower():
                        ws = ws_obj
                        break
                else:
                    raise gspread.exceptions.WorksheetNotFound(f"Worksheet '{worksheet}' not found")
        else:
            ws = sh.sheet1
        data = ws.get_all_records()
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        return df

    # Write support
    def ensure_worksheet(self, sheet_id: str, title: str, headers: Optional[List[str]] = None):
        sh = self._open_sheet(sheet_id)
        try:
            ws = sh.worksheet(title)
            if headers:
                first_row = ws.row_values(1)
                if not first_row and headers:
                    ws.append_row(headers)
            return ws
        except gspread.exceptions.WorksheetNotFound:
            # Try case-insensitive match before creating
            lowered = title.lower()
            for ws in sh.worksheets():
                if ws.title.lower() == lowered:
                    if headers:
                        first_row = ws.row_values(1)
                        if not first_row:
                            ws.update("A1", [headers])
                    return ws
            rows = 100
            cols = max(10, len(headers) if headers else 10)
            ws = sh.add_worksheet(title=title, rows=rows, cols=cols)
            if headers:
                ws.update("A1", [headers])
            return ws

    def append_rows(self, sheet_id: str, worksheet: str, rows: List[Dict[str, Any]]):
        if not rows:
            return
        ws = self.ensure_worksheet(sheet_id, worksheet, headers=list(rows[0].keys()))
        # Ensure columns order
        headers = ws.row_values(1)
        if not headers:
            headers = list(rows[0].keys())
            ws.update("A1", [headers])
        values = []
        for r in rows:
            values.append([r.get(h, "") for h in headers])
        ws.append_rows(values, value_input_option="RAW")
