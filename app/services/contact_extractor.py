from __future__ import annotations
from typing import Dict, Any, Optional
import re
import os

from .nl_query import NLQueryTranslator
from .data_processor import data_processor


# Canonical headers as present in the Google Sheet
SHEET_HEADERS = [
    "Event name",
    "Contact Name",
    "Position",
    "Linkedin contact",
    "Status",
    "Email",
    "Instagram page",
    "Instagram followers",
    "Club / Promoter / Festival",
    "Active?",
    "Month (if festival)",
    "Website/RA page",
    "Continent",
    "Country",
    "City",
]


def extract_contact_fields(translator: NLQueryTranslator, text: str) -> Dict[str, Any]:
    # Try LLM first if available
    if getattr(translator, "client", None):
        try:
            system = (
                "Extract contact details from the user text and return ONLY minified JSON with these exact keys: "
                + ", ".join(SHEET_HEADERS)
                + ". Missing fields must be empty strings."
            )
            user = f"Text: {text}"
            comp = translator.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.1,
            )
            raw = comp.choices[0].message.content if comp.choices else ""
            if os.environ.get("OPENAI_DEBUG") == "1":
                print("[OPENAI DEBUG][CONTACT] Raw output:", raw)
            data = _first_json_object(raw or "")
            if os.environ.get("OPENAI_DEBUG") == "1":
                print("[OPENAI DEBUG][CONTACT] Parsed JSON:", data)
            if data:
                # Normalize to require all keys
                return {k: data.get(k, "") for k in SHEET_HEADERS}
        except Exception as e:
            if os.environ.get("OPENAI_DEBUG") == "1":
                print("[OPENAI DEBUG][CONTACT] Exception:", repr(e))
            pass
    # Heuristic fallback
    return _heuristic_extract(text)


def _first_json_object(text: str) -> Optional[Dict[str, Any]]:
    import json
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _heuristic_extract(text: str) -> Dict[str, Any]:
    # Use enhanced data processor for better extraction
    extracted = data_processor.extract_contact_info(text)
    
    result: Dict[str, Any] = {k: "" for k in SHEET_HEADERS}
    
    # Map extracted data to sheet headers
    if 'email' in extracted:
        result["Email"] = extracted['email']
    if 'linkedin' in extracted:
        result["Linkedin contact"] = extracted['linkedin']
    if 'website' in extracted:
        result["Website/RA page"] = extracted['website']
    if 'instagram' in extracted:
        result["Instagram page"] = extracted['instagram']
    if 'name' in extracted:
        result["Contact Name"] = extracted['name']

    # Enhanced position extraction
    position_patterns = [
        r"(?i)\b(?:as|is|,)?\s*(?P<pos>DJ|Booker|Programmer|Owner|Founder|Talent Buyer|Event Producer|Artistic Director|Co-owner|Resident DJ)\b",
        r"(?i)\b(?P<pos>Manager|Director|Producer|Artist|Promoter)\b"
    ]
    
    for pattern in position_patterns:
        mpos = re.search(pattern, text)
        if mpos:
            result["Position"] = mpos.group("pos")
            break

    # Enhanced location extraction
    location_patterns = [
        r"(?i) in ([A-Z][A-Za-z .'-]+)$",
        r"(?i) from ([A-Z][A-Za-z .'-]+)",
        r"(?i) based in ([A-Z][A-Za-z .'-]+)"
    ]
    
    for pattern in location_patterns:
        mcity = re.search(pattern, text.strip())
        if mcity:
            location = mcity.group(1).strip()
            # Try to determine if it's a city or country
            if any(country in location.lower() for country in ['usa', 'united states', 'uk', 'united kingdom', 'germany', 'france', 'spain', 'italy']):
                result["Country"] = location
            else:
                result["City"] = location
            break

    return result


def _match_one(pattern: str, text: str) -> Optional[str]:
    m = re.search(pattern, text)
    return m.group(0) if m else None
