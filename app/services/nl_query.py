from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import re
from difflib import SequenceMatcher

from openai import OpenAI
from .data_processor import data_processor


@dataclass
class FilterClause:
    column: str
    op: str  # one of: eq, neq, contains, icontains, gt, gte, lt, lte, in
    value: Any


@dataclass
class NLQuerySpec:
    worksheet: Optional[str] = None
    select_columns: List[str] = field(default_factory=list)
    filters: List[FilterClause] = field(default_factory=list)
    sort_by: List[Dict[str, str]] = field(default_factory=list)  # [{"column": ..., "direction": "asc|desc"}]
    limit: Optional[int] = None


class NLQueryTranslator:
    def __init__(self, api_key: Optional[str]) -> None:
        # If api_key is None, let SDK read from env; else use provided key
        if api_key is None:
            try:
                self.client = OpenAI()
            except Exception:
                self.client = None
        elif api_key:
            try:
                self.client = OpenAI(api_key=api_key)
            except Exception:
                self.client = None
        else:
            self.client = None

    def translate(self, user_query: str, available_columns: List[str], default_worksheet: Optional[str]) -> NLQuerySpec:
        # First try LLM if available
        if self.client:
            spec = self._try_llm(user_query, available_columns, default_worksheet)
            if spec and (spec.filters or spec.select_columns or spec.sort_by or spec.limit):
                return spec
        # Fallback to heuristic parsing
        heuristic = self._heuristic_parse(user_query, available_columns, default_worksheet)
        if heuristic:
            return heuristic
        # Final fallback: no-op filter
        return NLQuerySpec(worksheet=default_worksheet)

    def _try_llm(self, user_query: str, available_columns: List[str], default_worksheet: Optional[str]) -> Optional[NLQuerySpec]:
        import os
        system_prompt = (
            "You are an expert at converting natural language queries into precise database searches. "
            "Convert user requests into JSON specifications for Google Sheets data. "
            "Only use columns that exist. If columns are ambiguous, pick the closest reasonable match. "
            "JSON keys: worksheet (string|null), select_columns (string[]), filters ({column, op, value}[]), "
            "sort_by ({column, direction}[]), limit (int|null). "
            "Operators: eq, neq, contains, icontains, gt, gte, lt, lte, in, generic_email, specific_email, fuzzy_contains, regex. "
            "For 'generic email addresses' queries, use op='generic_email' with value=true. "
            "For fuzzy matching, use op='fuzzy_contains'. For regex patterns, use op='regex'. "
            "For event queries with location and time, use 'icontains' for location and month filters. "
            "For event queries, prefer the EVENTS worksheet. "
            "For contact searches, prefer INSTANTLY 12/10 worksheet. "
            "For DJ/Label searches, prefer DJs / LABELS / MGMT / BOOKING worksheet. "
            "Always prioritize the most relevant worksheet based on query context. "
            "If user does not specify columns, leave select_columns empty. "
            "Be intelligent about understanding user intent and context."
        )

        user_prompt = (
            f"Available columns: {available_columns}\n"
            f"Default worksheet: {default_worksheet or 'sheet1'}\n"
            f"User query: {user_query}\n"
            "Return ONLY minified JSON with the spec."
        )

        try:
            comp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
            )
            text = comp.choices[0].message.content if comp.choices else ""
            if os.environ.get("OPENAI_DEBUG") == "1":
                print("[OPENAI DEBUG] Raw output:", text)
            data = self._safe_json(text or "")
            if os.environ.get("OPENAI_DEBUG") == "1":
                print("[OPENAI DEBUG] Parsed JSON:", data)
            
            # Post-process to fix common issues
            spec = self._post_process_spec(data, available_columns, default_worksheet, user_query)
            return spec
        except Exception as e:
            if os.environ.get("OPENAI_DEBUG") == "1":
                print("[OPENAI DEBUG] Exception:", repr(e))
            return None

    def _post_process_spec(self, data: Dict[str, Any], available_columns: List[str], default_worksheet: Optional[str], user_query: str) -> Optional[NLQuerySpec]:
        """Post-process the LLM output to fix common issues"""
        if not data:
            return None
            
        # Generic worksheet selection based on query relevance
        query_lower = user_query.lower()
        # Let the system automatically find the most relevant worksheet
        # No hardcoded worksheet preferences
        
        # Fix generic email queries
        filters = data.get("filters", [])
        if filters and len(filters) > 1:
            # Check if this looks like a generic email query (multiple email domain filters)
            email_filters = [f for f in filters if f.get("column", "").lower() == "email" and f.get("op") == "contains"]
            if len(email_filters) >= 2 and all("@" in f.get("value", "") for f in email_filters):
                # Replace with single generic_email filter
                data["filters"] = [{"column": "Email", "op": "generic_email", "value": True}]
                if os.environ.get("OPENAI_DEBUG") == "1":
                    print("[OPENAI DEBUG] Fixed multiple email filters to generic_email")
        
        return self._parse_spec(data, fallback_worksheet=default_worksheet)

    def _heuristic_parse(self, user_query: str, available_columns: List[str], default_worksheet: Optional[str]) -> Optional[NLQuerySpec]:
        # Handle simple forms and targeted field-of-person queries
        q = (user_query or "").strip().strip(",.;")
        if not q:
            return None
        q_lower = q.lower()

        # Build mapping of normalized column names to actual names
        norm_map: Dict[str, str] = {}
        for c in (available_columns or []):
            norm_map[c.lower().strip()] = c
        
        # Enhanced column matching using data processor
        def find_column_like_enhanced(token: str) -> Optional[str]:
            # First try the data processor's enhanced matching
            best_match = data_processor.find_best_column_match(token, available_columns)
            if best_match:
                return best_match
            
            # Fallback to original logic
            return find_column_like(token)

        # Enhanced synonyms mapping based on Excel structure
        def find_column_like(token: str) -> Optional[str]:
            t = token.lower().strip()
            # normalize common variants
            t = t.replace("linkedin", "linkedin").replace("linked in", "linkedin").replace("linked contact", "linkedin contact").replace("linked", "linkedin")
            # attempt direct contains match over available columns
            for k, actual in norm_map.items():
                if t in k:
                    return actual
            # Enhanced synonyms based on Excel columns
            synonyms = {
                "linkedin": ["linkedin contact", "linkedin", "linkedin link", "linkedin url", "linkedin profile"],
                "email": ["email", "e-mail", "email name"],
                "instagram": ["instagram page", "instagram", "ig", "instagram handle"],
                "website": ["website/ra page", "website", "ra page", "ra"],
                "status": ["status"],
                "position": ["position", "role"],
                "name": ["contact name", "dj name", "name", "email name"],
                "event": ["event name", "event", "venue"],
                "club": ["club", "promoter", "festival", "club\npromoter\nfestival"],
                "month": ["month", "month (for festivals)"],
                "country": ["country", "country.city", "continent"],
                "city": ["city", "country.city"],
                "continent": ["continent"],
            }
            for key, variants in synonyms.items():
                if key in t:
                    for v in variants:
                        for k, actual in norm_map.items():
                            if v in k:
                                return actual
            return None

        # Special handling for client examples
        import re
        
        # Pattern 1: "Show me a full list with only generic email addresses"
        if "generic email" in q_lower or "generic email addresses" in q_lower:
            email_col = find_column_like_enhanced("email")
            if email_col:
                # Find name-like columns dynamically
                name_cols = [col for col in available_columns if any(term in col.lower() for term in ['name', 'contact', 'person'])]
                name_col = name_cols[0] if name_cols else None
                
                select_cols = [email_col]
                if name_col:
                    select_cols.insert(0, name_col)
                
                return NLQuerySpec(
                    worksheet=default_worksheet,
                    select_columns=select_cols,
                    filters=[FilterClause(column=email_col, op="generic_email", value=True)],
                )
        
        # Pattern 2: "Give me a LinkedIn contact for [name/venue]"
        linkedin_for_venue = re.search(r"linkedin contact for (.+)", q_lower)
        if linkedin_for_venue:
            venue_name = linkedin_for_venue.group(1).strip()
            linkedin_col = find_column_like_enhanced("linkedin")
            # Find any column that might contain the venue/name
            venue_cols = [col for col in available_columns if any(term in col.lower() for term in ['name', 'venue', 'event', 'title', 'item'])]
            venue_col = venue_cols[0] if venue_cols else None
            
            if linkedin_col and venue_col:
                # Find name-like columns dynamically
                name_cols = [col for col in available_columns if any(term in col.lower() for term in ['name', 'contact', 'person'])]
                name_col = name_cols[0] if name_cols else None
                
                select_cols = [venue_col, linkedin_col]
                if name_col:
                    select_cols.insert(1, name_col)
                
                return NLQuerySpec(
                    worksheet=default_worksheet,
                    select_columns=select_cols,
                    filters=[FilterClause(column=venue_col, op="icontains", value=venue_name)],
                    limit=10,
                )
        
        # Pattern 3: "events in [location] in [month]" - specific pattern for events
        events_location_month_pattern = re.search(r"(?:gave\s+me\s+all\s+)?events?\s+(?:which\s+)?(?:happened|happeded|took\s+place|occurred|took\s+place)\s+in\s+(.+?)\s+in\s+(?:the\s+month\s+of\s+)?(.+)", q_lower)
        if events_location_month_pattern:
            location = events_location_month_pattern.group(1).strip()
            month = events_location_month_pattern.group(2).strip()
            
            # Find relevant columns for events
            event_cols = [col for col in available_columns if any(term in col.lower() for term in ['event', 'name', 'title', 'festival', 'club'])]
            event_col = event_cols[0] if event_cols else None
            
            location_cols = [col for col in available_columns if any(term in col.lower() for term in ['country', 'city', 'location', 'place', 'country.city'])]
            location_col = location_cols[0] if location_cols else None
            
            month_cols = [col for col in available_columns if any(term in col.lower() for term in ['month', 'time', 'date', 'period', 'month (for festivals)'])]
            month_col = month_cols[0] if month_cols else None
            
            filters = []
            if location_col:
                filters.append(FilterClause(column=location_col, op="icontains", value=location))
            if month_col:
                filters.append(FilterClause(column=month_col, op="icontains", value=month))
            
            if filters:
                select_cols = []
                if event_col:
                    select_cols.append(event_col)
                if location_col:
                    select_cols.append(location_col)
                if month_col:
                    select_cols.append(month_col)
                
                return NLQuerySpec(
                    worksheet=default_worksheet,
                    select_columns=select_cols,
                    filters=filters,
                )
        
        # Pattern 3.4: Specific pattern for "Italian events in August" type queries
        italian_events_pattern = re.search(r"(?:gave\s+me\s+|show\s+me\s+|give\s+me\s+|find\s+)?(?:list\s+of\s+)?(?:all\s+)?(.+?)\s+events?\s+(?:in\s+|during\s+|for\s+)?(?:the\s+month\s+of\s+)?(.+)", q_lower)
        if italian_events_pattern and ("event" in q_lower or "italian" in q_lower or "august" in q_lower):
            location = italian_events_pattern.group(1).strip()
            time_period = italian_events_pattern.group(2).strip()
            
            # Find relevant columns for events
            event_cols = [col for col in available_columns if any(term in col.lower() for term in ['event', 'name', 'title', 'festival', 'club'])]
            event_col = event_cols[0] if event_cols else None
            
            location_cols = [col for col in available_columns if any(term in col.lower() for term in ['country', 'city', 'location', 'place', 'country.city'])]
            location_col = location_cols[0] if location_cols else None
            
            time_cols = [col for col in available_columns if any(term in col.lower() for term in ['month', 'time', 'date', 'period', 'month (for festivals)'])]
            time_col = time_cols[0] if time_cols else None
            
            filters = []
            if location_col:
                filters.append(FilterClause(column=location_col, op="icontains", value=location))
            if time_col:
                filters.append(FilterClause(column=time_col, op="icontains", value=time_period))
            
            if filters:
                select_cols = []
                if event_col:
                    select_cols.append(event_col)
                if location_col:
                    select_cols.append(location_col)
                if time_col:
                    select_cols.append(time_col)
                
                return NLQuerySpec(
                    worksheet=default_worksheet,
                    select_columns=select_cols,
                    filters=filters,
                )
        
        # Pattern 3.5: General event queries with location and time
        general_event_pattern = re.search(r"(?:show\s+me\s+|give\s+me\s+|find\s+)?(?:all\s+)?events?\s+(?:in\s+|from\s+)?(.+?)\s+(?:in\s+|during\s+|for\s+)?(?:the\s+month\s+of\s+)?(.+)", q_lower)
        if general_event_pattern and ("event" in q_lower or "italian" in q_lower or "august" in q_lower):
            location = general_event_pattern.group(1).strip()
            time_period = general_event_pattern.group(2).strip()
            
            # Find relevant columns for events
            event_cols = [col for col in available_columns if any(term in col.lower() for term in ['event', 'name', 'title', 'festival', 'club'])]
            event_col = event_cols[0] if event_cols else None
            
            location_cols = [col for col in available_columns if any(term in col.lower() for term in ['country', 'city', 'location', 'place', 'country.city'])]
            location_col = location_cols[0] if location_cols else None
            
            time_cols = [col for col in available_columns if any(term in col.lower() for term in ['month', 'time', 'date', 'period', 'month (for festivals)'])]
            time_col = time_cols[0] if time_cols else None
            
            filters = []
            if location_col:
                filters.append(FilterClause(column=location_col, op="icontains", value=location))
            if time_col:
                filters.append(FilterClause(column=time_col, op="icontains", value=time_period))
            
            if filters:
                select_cols = []
                if event_col:
                    select_cols.append(event_col)
                if location_col:
                    select_cols.append(location_col)
                if time_col:
                    select_cols.append(time_col)
                
                return NLQuerySpec(
                    worksheet=default_worksheet,
                    select_columns=select_cols,
                    filters=filters,
                )
        
        # Pattern 4: "Give me a full list of items in [location] in [time period]"
        location_time_pattern = re.search(r"(.+?) in (.+?) in (.+)", q_lower)
        if location_time_pattern:
            item_type = location_time_pattern.group(1).strip()
            location = location_time_pattern.group(2).strip()
            time_period = location_time_pattern.group(3).strip()
            
            # Find relevant columns dynamically
            item_cols = [col for col in available_columns if any(term in col.lower() for term in ['name', 'title', 'item', 'event'])]
            item_col = item_cols[0] if item_cols else None
            
            location_cols = [col for col in available_columns if any(term in col.lower() for term in ['country', 'city', 'location', 'place'])]
            location_col = location_cols[0] if location_cols else None
            
            time_cols = [col for col in available_columns if any(term in col.lower() for term in ['month', 'time', 'date', 'period'])]
            time_col = time_cols[0] if time_cols else None
            
            filters = []
            if location_col:
                filters.append(FilterClause(column=location_col, op="icontains", value=location))
            if time_col:
                filters.append(FilterClause(column=time_col, op="icontains", value=time_period))
            
            if item_col and filters:
                # Find name-like columns dynamically
                name_cols = [col for col in available_columns if any(term in col.lower() for term in ['name', 'contact', 'person'])]
                name_col = name_cols[0] if name_cols else None
                
                select_cols = [item_col]
                if name_col:
                    select_cols.append(name_col)
                if location_col:
                    select_cols.append(location_col)
                if time_col:
                    select_cols.append(time_col)
                
                return NLQuerySpec(
                    worksheet=default_worksheet,
                    select_columns=select_cols,
                    filters=filters,
                )
        
        # Pattern 6: "events in [country]" (without month)
        country_only_pattern = re.search(r"events in (.+?)(?:\s+in\s|$)", q_lower)
        if country_only_pattern and "month" not in q_lower:
            country = country_only_pattern.group(1).strip()
            event_col = find_column_like("event")
            country_col = find_column_like("country")
            if event_col and country_col:
                return NLQuerySpec(
                    worksheet=default_worksheet,
                    select_columns=[event_col, "Contact name", country_col],
                    filters=[FilterClause(column=country_col, op="icontains", value=country)],
                )

        # Detect an email specific query first
        import re
        email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", q)
        if email_match:
            email_value = email_match.group(0)
            email_col = None
            for k, actual in norm_map.items():
                if "email" in k:
                    email_col = actual
                    break
            if email_col:
                return NLQuerySpec(
                    worksheet=default_worksheet,
                    filters=[FilterClause(column=email_col, op="eq", value=email_value)],
                    limit=10,
                )

        # Pattern: "give me <field> of <name>" or "show <field> for <name>"
        patterns = [
            r"give me\s+(?P<field>.+?)\s+of\s+(?P<who>.+)$",
            r"show\s+(?P<field>.+?)\s+for\s+(?P<who>.+)$",
            r"what is\s+(?P<field>.+?)\s+for\s+(?P<who>.+)$",
        ]
        for pat in patterns:
            m = re.search(pat, q_lower)
            if m:
                field_raw = m.group("field").strip().strip(".?")
                who_raw = m.group("who").strip().strip(".?")
                target_col = find_column_like(field_raw)
                if not target_col:
                    # Try direct lookup
                    target_col = norm_map.get(field_raw)
                if target_col:
                    # Find name-like column to match person/entity
                    name_cols = [k for k in norm_map.values() if any(tok in k.lower() for tok in ["contact name", "dj name", "name", "event name"]) ]
                    name_col = name_cols[0] if name_cols else None
                    filters = []
                    if name_col:
                        filters.append(FilterClause(column=name_col, op="icontains", value=who_raw))
                    # Prefer showing target and name column
                    select_cols = [target_col] + ([name_col] if name_col else [])
                    return NLQuerySpec(
                        worksheet=default_worksheet,
                        select_columns=select_cols,
                        filters=filters,
                        limit=5,
                    )

        # Previous equality/assignment patterns
        patterns_simple = [
            r"where\s+(?P<col>[^=]+?)\s+is\s+(?P<val>.+)",
            r"(?P<col>[^=]+?)\s+is\s+(?P<val>.+)",
            r"(?P<col>[^=]+?)\s*=\s*(?P<val>.+)",
            r"where\s+(?P<col>[^=]+?)\s+(?P<val>[\w.+%-]+@[\w.-]+\.[A-Za-z]{2,})",
            r"(?P<col>[^=]+?)\s+(?P<val>[\w.+%-]+@[\w.-]+\.[A-Za-z]{2,})",
        ]
        for pat in patterns_simple:
            m = re.search(pat, q_lower)
            if not m:
                continue
            col_raw = m.group("col").strip()
            val_raw = m.group("val").strip().strip(",.;")
            # Map to actual column name by best lower-case match
            col_actual = norm_map.get(col_raw)
            if not col_actual:
                token = col_raw.replace("the ", "").replace("emails", "email").strip()
                for k, v in norm_map.items():
                    if token in k:
                        col_actual = v
                        break
            if not col_actual:
                continue
            return NLQuerySpec(
                worksheet=default_worksheet,
                filters=[FilterClause(column=col_actual, op="eq", value=val_raw)],
            )
        return None

    def _safe_json(self, text: str) -> Dict[str, Any]:
        import json, re
        # Extract first JSON object from text
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}

    def _parse_spec(self, data: Dict[str, Any], fallback_worksheet: Optional[str]) -> NLQuerySpec:
        worksheet = data.get("worksheet") or fallback_worksheet
        select_columns = data.get("select_columns") or []
        sort_by = data.get("sort_by") or []
        limit = data.get("limit")

        raw_filters = data.get("filters") or []
        filters: List[FilterClause] = []
        for f in raw_filters:
            col = f.get("column")
            op = (f.get("op") or "eq").lower()
            val = f.get("value")
            if not col:
                continue
            filters.append(FilterClause(column=col, op=op, value=val))

        return NLQuerySpec(
            worksheet=worksheet,
            select_columns=select_columns,
            filters=filters,
            sort_by=sort_by,
            limit=limit if isinstance(limit, int) else None,
        )
