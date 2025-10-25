# data_analyzer.py
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
from difflib import get_close_matches

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -----------------------------
# Helpers & configuration
# -----------------------------

GENERIC_EMAIL_PREFIXES = (
    "info", "contact", "hello", "support", "admin", "sales", "team",
    "noreply", "no-reply", "do-not-reply"
)

NATIONALITY_TO_COUNTRY = {
    "irish": ["ireland", "dublin", "cork", "galway", "limerick"],
    "french": ["france", "paris", "lyon", "marseille", "toulouse", "bordeaux"],
    "italian": ["italy", "milan", "torino", "turin", "rome", "naples", "napoli"],
    "portuguese": ["portugal", "lisbon", "porto", "coimbra", "braga"],
    "german": ["germany", "berlin", "hamburg", "munich", "münchen", "frankfurt"],
    "spanish": ["spain", "madrid", "barcelona", "valencia", "sevilla", "seville"],
    "english": ["england", "london", "manchester", "birmingham", "leeds"],
    "scottish": ["scotland", "edinburgh", "glasgow", "aberdeen"],
    "welsh": ["wales", "cardiff", "swansea"],
}

MONTH_ALIASES = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7, "jule": 7, "julyy": 7, "juky": 7, "juky": 7,  # catch common typos
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

ROLE_KEYWORDS = {
    "dj": ["dj", "disc jockey"],
    "event": ["event", "festival", "concert", "show", "gig"],
    "festival": ["festival", "fest"],
    "media": ["media", "press", "magazine", "blog", "newspaper", "outlet", "journalist"],
    "artist": ["artist", "musician", "performer"],
}

FOLLOWER_COLUMNS_CANDIDATES = [
    "instagram_followers", "instagram", "ig_followers", "ig",
    "youtube_followers", "youtube_subscribers", "youtube", "yt_followers", "yt_subscribers"
]

EMAIL_COLUMNS_CANDIDATES = ["email", "e-mail", "mail", "contact_email", "direct_email"]
LINKEDIN_COLUMNS_CANDIDATES = ["linkedin", "linkedin_url", "linkedin profile", "li"]
NAME_COLUMNS_CANDIDATES = ["name", "full_name", "contact_name", "artist_name", "dj_name"]
COUNTRY_COLUMNS_CANDIDATES = ["country", "location", "nationality", "region", "city", "country_city"]
ROLE_COLUMNS_CANDIDATES = ["role", "position", "title", "type", "category", "tags"]
EVENT_COLUMNS_CANDIDATES = ["event", "festival", "event_name", "festival_name"]
MONTH_COLUMNS_CANDIDATES = ["month", "event_month", "date_month", "month_name"]

def _norm(s: Any) -> str:
    return str(s).strip().casefold() if pd.notna(s) else ""

def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    cols_norm = [c.casefold() for c in cols]
    # exact or contains match
    for cand in candidates:
        c = cand.casefold()
        for i, coln in enumerate(cols_norm):
            if coln == c or c in coln:
                return cols[i]
    # fuzzy (last resort)
    for cand in candidates:
        match = get_close_matches(cand.casefold(), cols_norm, n=1, cutoff=0.75)
        if match:
            return cols[cols_norm.index(match[0])]
    return None

def _month_from_query(text: str) -> Optional[int]:
    t = text.casefold()
    tokens = re.findall(r"[a-zA-Z]+", t)
    for token in tokens:
        if token in MONTH_ALIASES:
            return MONTH_ALIASES[token]
    # fuzzy month fix
    m = get_close_matches("july", tokens, n=1, cutoff=0.75)
    if m:
        return 7
    return None

def _contains_letter_query(text: str) -> Optional[str]:
    # matches "name contains M" or "name contains 'm'"
    m = re.search(r"name\s+contains\s+['\"]?([a-zA-Z])['\"]?", text, re.IGNORECASE)
    if m:
        return m.group(1)
    return None

def _extract_nationality(text: str) -> Optional[str]:
    t = text.casefold()
    for nat in NATIONALITY_TO_COUNTRY.keys():
        if nat in t:
            return nat
    # phrases like "from france", "in france"
    for nat, countries in NATIONALITY_TO_COUNTRY.items():
        for c in countries:
            if re.search(rf"\b{re.escape(c)}\b", t):
                return nat
    return None

def _needs_linkedin(text: str) -> bool:
    return "linkedin" in text.casefold()

def _wants_direct_email(text: str) -> bool:
    t = text.casefold()
    return ("direct email" in t) or ("direct emails" in t) or ("disregard" in t and "generic" in t)

def _wants_role(text: str) -> Optional[str]:
    t = text.casefold()
    for role, kws in ROLE_KEYWORDS.items():
        for kw in kws:
            if re.search(rf"\b{re.escape(kw)}\b", t):
                return role
    return None

def _high_followers_request(text: str) -> Tuple[bool, bool]:
    t = text.casefold()
    ig = ("instagram" in t or "ig" in t) and ("high" in t)
    yt = ("youtube" in t or "yt" in t) and ("high" in t)
    return ig, yt

def _is_direct_email(addr: str) -> bool:
    a = _norm(addr)
    if not a or "@" not in a:
        return False
    local = a.split("@", 1)[0]
    return not any(local.startswith(p) or p in local for p in GENERIC_EMAIL_PREFIXES)

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
    return ParsedQuery(
        nationality=_extract_nationality(q),
        role=_wants_role(q),
        month=_month_from_query(q),
        name_contains=_contains_letter_query(q),
        require_linkedin=_needs_linkedin(q),
        direct_email_only=_wants_direct_email(q),
        high_ig=_high_followers_request(q)[0],
        high_yt=_high_followers_request(q)[1],
    )

# -----------------------------
# Main Analyzer
# -----------------------------

class DataStructureAnalyzer:
    def __init__(self):
        self.client = None
        try:
            from dotenv import dotenv_values
            env_values = dotenv_values(".env")
            api_key = env_values.get("OPENAI_API_KEY")
            if api_key:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")

    def analyze_data_structure(self, df: pd.DataFrame, worksheet_name: str) -> Dict[str, Any]:
        if df is None or df.empty:
            return {}
        return {
            "worksheet_name": worksheet_name,
            "columns": list(df.columns),
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "dataframe": df
        }

    # ---------- Local filtering first (deterministic, cheap) ----------
    def _local_prefilter(self, df: pd.DataFrame, q: ParsedQuery) -> pd.DataFrame:
        if df is None or df.empty:
            return df.copy()

        df2 = df.copy()

        # Detect columns
        name_col     = _find_column(df2, NAME_COLUMNS_CANDIDATES)
        email_col    = _find_column(df2, EMAIL_COLUMNS_CANDIDATES)
        linkedin_col = _find_column(df2, LINKEDIN_COLUMNS_CANDIDATES)
        country_col  = _find_column(df2, COUNTRY_COLUMNS_CANDIDATES)
        role_col     = _find_column(df2, ROLE_COLUMNS_CANDIDATES)
        event_col    = _find_column(df2, EVENT_COLUMNS_CANDIDATES)
        month_col    = _find_column(df2, MONTH_COLUMNS_CANDIDATES)

        # Followers
        ig_col = None
        yt_col = None
        for c in df2.columns:
            cn = c.casefold()
            if any(k in cn for k in ["instagram", "ig"]) and any(k in cn for k in ["follower", "sub", "count"]):
                ig_col = c
            if any(k in cn for k in ["youtube", "yt"]) and any(k in cn for k in ["follower", "sub", "count"]):
                yt_col = c
        if not ig_col:
            ig_col = _find_column(df2, FOLLOWER_COLUMNS_CANDIDATES[:4])
        if not yt_col:
            yt_col = _find_column(df2, FOLLOWER_COLUMNS_CANDIDATES[4:])

        # Nationality -> country matching
        if q.nationality and country_col:
            countries = NATIONALITY_TO_COUNTRY.get(q.nationality, [])
            mask = pd.Series(False, index=df2.index)
            for ckw in countries:
                mask = mask | df2[country_col].astype(str).str.contains(ckw, case=False, na=False)
            df2 = df2[mask]

        # Role filtering
        if q.role:
            # If DJ: prefer role_col contains 'dj', else use event_col/other hints
            if role_col:
                df2 = df2[df2[role_col].astype(str).str.contains(q.role, case=False, na=False)]
            else:
                # Try to infer via event/name columns
                cols_to_search = []
                if event_col: cols_to_search.append(event_col)
                if name_col: cols_to_search.append(name_col)
                any_mask = pd.Series(False, index=df2.index)
                for c in cols_to_search:
                    any_mask = any_mask | df2[c].astype(str).str.contains(q.role, case=False, na=False)
                df2 = df2[any_mask] if not any_mask.empty else df2

        # Month filtering
        if q.month and month_col:
            # Normalize month col to numeric if possible
            col_vals = df2[month_col].astype(str).str.strip()
            # Map textual month -> number
            def to_month_num(x: str) -> Optional[int]:
                xcf = x.casefold()
                if xcf in MONTH_ALIASES:
                    return MONTH_ALIASES[xcf]
                match = get_close_matches("july", [xcf], n=1, cutoff=0.75)
                if match:
                    return 7
                try:
                    v = int(x)
                    if 1 <= v <= 12:
                        return v
                except:
                    pass
                return None
            mnums = col_vals.apply(to_month_num)
            df2 = df2[mnums == q.month]

        # Name contains letter
        if q.name_contains and name_col:
            letter = q.name_contains
            df2 = df2[df2[name_col].astype(str).str.contains(letter, case=False, na=False)]

        # Require LinkedIn
        if q.require_linkedin and linkedin_col:
            df2 = df2[df2[linkedin_col].astype(str).str.len() > 2]
            df2 = df2[~df2[linkedin_col].astype(str).str.fullmatch(r"(?i)na|n/a|-|none|null|^$", na=False)]

        # Direct email only
        if q.direct_email_only and email_col:
            df2 = df2[df2[email_col].astype(str).apply(_is_direct_email)]

        # High followers logic: use 75th percentile of available column(s)
        if q.high_ig and ig_col and pd.api.types.is_numeric_dtype(pd.to_numeric(df2[ig_col], errors="coerce")):
            col = pd.to_numeric(df2[ig_col], errors="coerce")
            thr = col.quantile(0.75)
            df2 = df2[col >= thr]
        if q.high_yt and yt_col and pd.api.types.is_numeric_dtype(pd.to_numeric(df2[yt_col], errors="coerce")):
            col = pd.to_numeric(df2[yt_col], errors="coerce")
            thr = col.quantile(0.75)
            df2 = df2[col >= thr]

        return df2

    # ---------- Optional GPT refinement ----------
    def _refine_with_gpt(
        self,
        user_query: str,
        df_candidates: pd.DataFrame,
        columns: List[str],
        max_rows_to_send: int = 150,
        retries: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Sends a small candidate set to GPT for precise filtering.
        Returns a list of dicts (records). Safe fallback to candidates if GPT fails.
        """
        if self.client is None or df_candidates.empty:
            return df_candidates.to_dict("records")

        data = df_candidates.head(max_rows_to_send).to_dict("records")

        system = f"""You are a strict data-filtering engine.
You will receive:
- Column headers: {columns}
- Candidate records (JSON array)
- A user query

Your job:
- Return ONLY the subset of candidate records that match ALL conditions in the user query.
- Do NOT invent fields or values. Only return rows from the provided candidate list.
- If query requires: nationality, roles (DJ/event/media/artist), month, LinkedIn present, name contains letter, direct email only, or high follower thresholds—apply them strictly.
- "Direct email" excludes generic prefixes: {', '.join(GENERIC_EMAIL_PREFIXES)}.
- Nationality mapping examples: {json.dumps(NATIONALITY_TO_COUNTRY)}
- Month understanding (typos allowed): {list(MONTH_ALIASES.keys())}
- Output MUST be a valid JSON array of records (objects). If nothing matches, return [].
"""

        user = f"""CANDIDATE_DATA (JSON):
{json.dumps(data, ensure_ascii=False)}

QUERY:
{user_query}

Return only the matching JSON array. No explanations—just the JSON array."""

        backoff = 1.5
        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model="gpt-5-pro",  # or your best available model
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user}
                    ],
                    temperature=0,
                    max_completion_tokens=4000,
                    response_format={"type": "json_object"}  # Encourage valid JSON
                )
                content = resp.choices[0].message.content

                # Some clients wrap it; ensure we extract the array
                arr_match = re.search(r"\[\s*{.*}\s*\]", content, flags=re.DOTALL)
                if arr_match:
                    return json.loads(arr_match.group(0))
                # If response is a JSON object with a key like "results"
                try:
                    obj = json.loads(content)
                    if isinstance(obj, list):
                        return obj
                    if isinstance(obj, dict):
                        # find first list value
                        for v in obj.values():
                            if isinstance(v, list):
                                return v
                except Exception:
                    pass

                # If parse failed, try again
                time.sleep(backoff ** (attempt + 1))
            except Exception as e:
                logger.warning(f"GPT refine attempt {attempt+1} failed: {e}")
                time.sleep(backoff ** (attempt + 1))

        # Fallback: return the local candidates if GPT keeps failing
        return data

    def get_query_results(self, df: pd.DataFrame, worksheet_name: str, user_query: str,
                          enable_gpt_refine: bool = True) -> pd.DataFrame:
        """
        Main entry: parse -> local filter -> (optional) GPT refine -> DataFrame result.
        """
        if df is None or df.empty:
            return pd.DataFrame()

        parsed = parse_query(user_query)
        pre = self._local_prefilter(df, parsed)

        # If local result is small enough, we can return directly.
        # Else, optionally ask GPT to refine the candidate set for extra precision.
        if not enable_gpt_refine or len(pre) <= 150:
            return pre.reset_index(drop=True)

        refined = self._refine_with_gpt(
            user_query=user_query,
            df_candidates=pre,
            columns=list(df.columns),
            max_rows_to_send=150
        )
        return pd.DataFrame(refined).reset_index(drop=True)

# ------------- Convenience global -------------
data_analyzer = DataStructureAnalyzer()

# ------------- Example usage (comment out in production) -------------
if __name__ == "__main__":
    # Example:
    # df = pd.read_csv("contacts.csv")
    # query = "Give me a list of all the French contacts you have here but only those whose name contains an M (DJ list)"
    # results = data_analyzer.get_query_results(df, "Contacts", query, enable_gpt_refine=True)
    # print(results.head())
    pass
