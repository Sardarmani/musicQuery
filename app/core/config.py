import os
from dataclasses import dataclass


@dataclass
class Settings:
    google_sheet_id: str
    service_account_file: str | None
    service_account_json_b64: str | None


def get_settings() -> Settings:
    return Settings(
        google_sheet_id=os.environ.get("GOOGLE_SHEET_ID", ""),
        service_account_file=os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE"),
        service_account_json_b64=os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON_BASE64"),
    )
