import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
from difflib import get_close_matches
from openai import OpenAI

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------------------------------------------------
# Utility mappings and normalization helpers
# ------------------------------------------------------------------

GENERIC_EMAIL_PREFIXES = (
    "info", "contact", "hello", "support", "admin", "sales",
    "team", "noreply", "no-reply", "do-not-reply"
)

NATIONALITY_TO_COUNTRY = {
    "irish": ["ireland", "dublin", "cork", "galway"],
    "french": ["france", "paris", "lyon", "marseille"],
    "italian": ["italy", "milan", "turin", "rome"],
    "portuguese": ["portugal", "lisbon", "porto"],
    "german": ["germany", "berlin", "hamburg", "munich"],
    "spanish": ["spain", "madrid", "barcelona", "valencia"],
}

MONTH_ALIASES = {
    "january": 1, "jan": 1, "february": 2, "feb": 2,
    "march": 3, "mar": 3, "april": 4, "apr": 4, "may": 5,
    "june": 6, "jun": 6, "july": 7, "jul": 7, "juky": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

ROLE_KEYWORDS = {
    "dj": ["dj", "disc jockey"],
    "event": ["event", "festival", "concert"],
    "media": ["media", "press", "magazine", "blog"],
    "artist": ["artist", "musician", "performer"],
}

def _norm(s: Any) -> str:
    return str(s).strip().casefold() if pd.notna(s) else ""

def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    cols_norm = [c.casefold() for c in cols]
    for cand in candidates:
        c = cand.casefold()
        for i, coln in enumerate(cols_norm):
            if coln == c or c in coln:
                return cols[i]
    match = get_close_matches(cand, cols_norm, n=1, cutoff=0.8)
    if match:
        return cols[cols_norm.index(match[0])]
    return None

def _extract_letter(text: str) -> Optional[str]:
    m = re.search(r"name\s+contains\s+['\"]?([a-zA-Z])['\"]?", text, re.IGNORECASE)
    return m.group(1) if m else None

def _extract_nationality(text: str) -> Optional[str]:
    t = text.casefold()
    for nat in NATIONALITY_TO_COUNTRY.keys():
        if nat in t:
            return nat
    for nat, countries in NATIONALITY_TO_COUNTRY.items():
        for c in countries:
            if c in t:
                return nat
    return None

def _month_from_query(text: str) -> Optional[int]:
    t = text.casefold()
    for word in re.findall(r"[a-z]+", t):
        if word in MONTH_ALIASES:
            return MONTH_ALIASES[word]
    return None

def _has_word(text: str, word: str) -> bool:
    return re.search(rf"\b{word}\b", text, re.IGNORECASE) is not None

@dataclass
class ParsedQuery:
    nationality: Optional[str]
    role: Optional[str]
    month: Optional[int]
    name_contains: Optional[str]
    require_linkedin: bool
    direct_email_only: bool
    high_ig: bool
    high_yt: bool

def parse_query(q: str) -> ParsedQuery:
    ql = q.lower()
    return ParsedQuery(
        nationality=_extract_nationality(q),
        role=next((r for r, kws in ROLE_KEYWORDS.items() if any(k in ql for k in kws)), None),
        month=_month_from_query(q),
        name_contains=_extract_letter(q),
        require_linkedin="linkedin" in ql,
        direct_email_only="direct email" in ql or "generic email" in ql,
        high_ig="instagram" in ql and "high" in ql,
        high_yt="youtube" in ql and "high" in ql,
    )

# ------------------------------------------------------------------
# Main Analyzer
# ------------------------------------------------------------------

class DataStructureAnalyzer:
    def __init__(self):
        try:
            from dotenv import dotenv_values
            env_values = dotenv_values(".env")
            api_key = env_values.get("OPENAI_API_KEY")
            self.client = OpenAI(api_key=api_key) if api_key else None
        except Exception as e:
            self.client = None
            logger.error(f"Failed to initialize OpenAI client: {e}")

    # ------------------ Original-style method ------------------
    def get_gpt_query_analysis(self, user_query: str, data_structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Works like the old method name but uses new optimized filtering logic.
        """
        df = data_structure.get("dataframe")
        if df is None or df.empty:
            return self._fallback_analysis(user_query, data_structure)

        parsed = parse_query(user_query)
        local_filtered = self._local_prefilter(df, parsed)

        if len(local_filtered) <= 150:
            return {
                "intent": "Local filtering",
                "search_strategy": "local_only",
                "filtered_data": local_filtered.to_dict("records"),
                "return_full_rows": True,
                "confidence": 0.9,
            }

        # If more rows, refine with GPT
        refined = self._refine_with_gpt(user_query, local_filtered, list(df.columns))
        return {
            "intent": "Refined GPT filtering",
            "search_strategy": "direct_gpt",
            "filtered_data": refined,
            "return_full_rows": True,
            "confidence": 0.97,
        }

    # ------------------ Local filtering logic ------------------
    def _local_prefilter(self, df: pd.DataFrame, q: ParsedQuery) -> pd.DataFrame:
        df2 = df.copy()
        cols = [c.casefold() for c in df.columns]

        # Guess columns
        name_col = _find_column(df2, ["name"])
        country_col = _find_column(df2, ["country", "location", "nationality"])
        linkedin_col = _find_column(df2, ["linkedin", "linkedin_url"])
        email_col = _find_column(df2, ["email"])
        month_col = _find_column(df2, ["month"])
        ig_col = next((c for c in df.columns if "instagram" in c.lower()), None)
        yt_col = next((c for c in df.columns if "youtube" in c.lower()), None)

        # Country/nationality
        if q.nationality and country_col:
            countries = NATIONALITY_TO_COUNTRY.get(q.nationality, [])
            mask = pd.Series(False, index=df2.index)
            for c in countries:
                mask |= df2[country_col].astype(str).str.contains(c, case=False, na=False)
            df2 = df2[mask]

        # Role
        if q.role:
            mask = pd.Series(False, index=df2.index)
            for col in df2.columns:
                mask |= df2[col].astype(str).str.contains(q.role, case=False, na=False)
            df2 = df2[mask]

        # Month
        if q.month and month_col:
            df2 = df2[df2[month_col].astype(str).str.contains(str(q.month), na=False)]

        # Name contains
        if q.name_contains and name_col:
            df2 = df2[df2[name_col].astype(str).str.contains(q.name_contains, case=False, na=False)]

        # LinkedIn required
        if q.require_linkedin and linkedin_col:
            df2 = df2[df2[linkedin_col].astype(str).str.len() > 2]

        # Direct email
        if q.direct_email_only and email_col:
            df2 = df2[~df2[email_col].astype(str).apply(lambda x: any(p in x.lower() for p in GENERIC_EMAIL_PREFIXES))]

        # High followers
        if q.high_ig and ig_col:
            df2 = df2[pd.to_numeric(df2[ig_col], errors="coerce") >= df2[ig_col].astype(float).quantile(0.75)]
        if q.high_yt and yt_col:
            df2 = df2[pd.to_numeric(df2[yt_col], errors="coerce") >= df2[yt_col].astype(float).quantile(0.75)]

        return df2

    # ------------------ GPT refinement ------------------
    def _refine_with_gpt(self, user_query: str, df_candidates: pd.DataFrame, columns: List[str]) -> List[Dict[str, Any]]:
        if self.client is None:
            return df_candidates.to_dict("records")

        data = df_candidates.head(150).to_dict("records")

        system_prompt = f"""You are a precise data filtering engine.
Columns: {columns}
You will receive a user query and a set of records. Return ONLY records matching all conditions.
Apply these rules:
- Respect nationality mapping: {json.dumps(NATIONALITY_TO_COUNTRY)}
- Apply LinkedIn, name contains, month, and role filters exactly.
- For 'direct email', exclude generic prefixes: {', '.join(GENERIC_EMAIL_PREFIXES)}.
Return ONLY a valid JSON array of matching records.
"""

        user_prompt = f"""DATA:
{json.dumps(data, ensure_ascii=False)}

QUERY: "{user_query}"
Return only the JSON array of matching records."""

        try:
            resp = self.client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=4000
            )
            content = resp.choices[0].message.content
            match = re.search(r"\[.*\]", content, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception as e:
            logger.error(f"GPT refine failed: {e}")
        return df_candidates.to_dict("records")

    # ------------------ Fallback ------------------
    def _fallback_analysis(self, user_query: str, data_structure: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "intent": f"Fallback for {user_query}",
            "search_strategy": "fallback",
            "filtered_data": [],
            "return_full_rows": True,
            "confidence": 0.5,
        }

    # ------------------ Backward compatibility ------------------
    def apply_gpt_analysis(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> pd.DataFrame:
        """Convert filtered_data (dict) back to DataFrame, same as old version."""
        if df is None or df.empty:
            return df
        filtered_data = analysis.get("filtered_data", [])
        return pd.DataFrame(filtered_data) if filtered_data else pd.DataFrame()

# Global instance
data_analyzer = DataStructureAnalyzer()
