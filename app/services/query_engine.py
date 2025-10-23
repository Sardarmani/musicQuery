from __future__ import annotations
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
import re

from .nl_query import NLQuerySpec, FilterClause
from .data_processor import data_processor


def apply_query_spec(df: pd.DataFrame, spec: NLQuerySpec) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    # Enhanced data preprocessing
    result = data_processor.normalize_dataframe(df)

    # Filters
    for flt in spec.filters:
        result = _apply_filter(result, flt)
        if result.empty:
            break

    # Select columns
    if spec.select_columns:
        existing = [c for c in spec.select_columns if c in result.columns]
        if existing:
            result = result[existing]

    # Enhanced result ranking and quality scoring
    result = _rank_results(result, spec)

    # Sorting
    if spec.sort_by:
        by: List[str] = []
        ascending: List[bool] = []
        for item in spec.sort_by:
            col = item.get("column")
            direction = (item.get("direction") or "asc").lower()
            if col in result.columns:
                by.append(col)
                ascending.append(direction != "desc")
        if by:
            result = result.sort_values(by=by, ascending=ascending)

    # Limit
    if spec.limit is not None and spec.limit > 0:
        result = result.head(spec.limit)

    return result.reset_index(drop=True)


def _rank_results(df: pd.DataFrame, spec: NLQuerySpec) -> pd.DataFrame:
    """
    Rank results by relevance and quality to ensure consistent, excellent results.
    """
    if df is None or df.empty:
        return df
    
    # Create a relevance score for each row
    relevance_scores = []
    
    for idx, row in df.iterrows():
        score = 0
        
        # Base score for having data
        non_null_count = row.notna().sum()
        score += non_null_count * 0.1
        
        # Bonus for complete records (more non-null values)
        completeness_ratio = non_null_count / len(row)
        score += completeness_ratio * 10
        
        # Bonus for having email addresses
        email_cols = [col for col in df.columns if 'email' in col.lower()]
        for col in email_cols:
            if pd.notna(row[col]) and '@' in str(row[col]):
                score += 5
        
        # Bonus for having contact information
        contact_cols = [col for col in df.columns if any(term in col.lower() for term in ['name', 'contact', 'linkedin', 'instagram'])]
        for col in contact_cols:
            if pd.notna(row[col]) and str(row[col]).strip():
                score += 2
        
        # Penalty for empty or placeholder values
        for col in df.columns:
            if pd.notna(row[col]):
                value = str(row[col]).strip().lower()
                if value in ['', 'n/a', 'na', 'null', 'none', '-', 'tbd', 'tba']:
                    score -= 1
        
        relevance_scores.append(score)
    
    # Add relevance score as a column and sort by it
    df_with_scores = df.copy()
    df_with_scores['_relevance_score'] = relevance_scores
    
    # Sort by relevance score (descending) and then by original index
    df_with_scores = df_with_scores.sort_values(['_relevance_score', df_with_scores.index], ascending=[False, True])
    
    # Remove the relevance score column before returning
    return df_with_scores.drop('_relevance_score', axis=1)


def _apply_filter(df: pd.DataFrame, flt: FilterClause) -> pd.DataFrame:
    col = flt.column
    if col not in df.columns:
        # Try to find best matching column
        best_match = data_processor.find_best_column_match(col, df.columns.tolist())
        if best_match:
            col = best_match
        else:
            return df

    op = flt.op
    val = flt.value

    series = df[col]

    # Normalize string values for comparison
    if isinstance(val, str):
        val_norm = val.strip()
    else:
        val_norm = val

    if op == "eq":
        if series.dtype == object:
            return df[series.str.strip().str.casefold() == str(val_norm).strip().casefold()]
        return df[series == val_norm]
    if op == "neq":
        if series.dtype == object:
            return df[series.str.strip().str.casefold() != str(val_norm).strip().casefold()]
        return df[series != val_norm]
    if op == "contains":
        return df[series.astype(str).str.contains(str(val_norm), na=False)]
    if op == "icontains":
        return df[series.astype(str).str.contains(str(val_norm), case=False, na=False)]
    if op == "generic_email":
        # Enhanced generic email filtering
        email_mask = series.astype(str).str.contains('@', na=False)
        domain_mask = pd.Series([False] * len(df), index=df.index)
        for email in series[email_mask]:
            if email and '@' in str(email):
                domain = str(email).split('@')[1].lower()
                if data_processor.categorize_email_domain(str(email)) == 'generic':
                    domain_mask |= series.astype(str) == email
        return df[email_mask & domain_mask]
    if op == "specific_email":
        # Enhanced specific email filtering
        email_mask = series.astype(str).str.contains('@', na=False)
        domain_mask = pd.Series([False] * len(df), index=df.index)
        for email in series[email_mask]:
            if email and '@' in str(email):
                if data_processor.categorize_email_domain(str(email)) == 'specific':
                    domain_mask |= series.astype(str) == email
        return df[email_mask & domain_mask]
    if op == "fuzzy_contains":
        # Enhanced fuzzy matching
        return df[series.astype(str).apply(
            lambda x: any(SequenceMatcher(None, str(val_norm).lower(), word.lower()).ratio() > 0.7 
                        for word in str(x).split() if word.strip())
        )]
    if op == "regex":
        # Regular expression matching
        try:
            pattern = re.compile(str(val_norm), re.IGNORECASE)
            return df[series.astype(str).str.contains(pattern, na=False)]
        except re.error:
            return df
    if op == "gt":
        return df[_coerce_numeric(series) > _to_number(val_norm)]
    if op == "gte":
        return df[_coerce_numeric(series) >= _to_number(val_norm)]
    if op == "lt":
        return df[_coerce_numeric(series) < _to_number(val_norm)]
    if op == "lte":
        return df[_coerce_numeric(series) <= _to_number(val_norm)]
    if op == "in":
        values = val_norm if isinstance(val_norm, list) else [val_norm]
        if series.dtype == object:
            norm_series = series.str.strip().str.casefold()
            norm_values = [str(v).strip().casefold() for v in values]
            return df[norm_series.isin(norm_values)]
        return df[series.isin(values)]

    return df


def _to_number(value):
    try:
        return float(value)
    except Exception:
        return float("nan")


def _coerce_numeric(series):
    return pd.to_numeric(series, errors="coerce")
