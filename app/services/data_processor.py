"""
Enhanced data processing module for improved CSV/Google Sheets data extraction.
This module provides optimized data cleaning, normalization, and search capabilities.
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from difflib import SequenceMatcher
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Enhanced data processor for music database queries.
    Provides improved data cleaning, normalization, and search capabilities.
    """
    
    def __init__(self):
        self.column_synonyms = {
            'email': ['email', 'e-mail', 'email name', 'email address'],
            'name': ['name', 'contact name', 'dj name', 'contact', 'person'],
            'linkedin': ['linkedin', 'linkedin contact', 'linkedin link', 'linkedin url', 'linkedin profile'],
            'instagram': ['instagram', 'instagram page', 'ig', 'instagram handle', 'instagram followers'],
            'website': ['website', 'website/ra page', 'ra page', 'ra', 'url'],
            'event': ['event name', 'event', 'venue', 'festival', 'club'],
            'position': ['position', 'role', 'title', 'job'],
            'status': ['status', 'active', 'active?'],
            'country': ['country', 'country.city', 'location'],
            'city': ['city', 'location'],
            'continent': ['continent', 'region'],
            'month': ['month', 'month (for festivals)', 'month (if festival)'],
            'club': ['club', 'promoter', 'festival', 'club\npromoter\nfestival', 'club / promoter / festival'],
            'safe': ['safe?', 'safe', 'ok_for_all', 'ok']
        }
        
        # Common data cleaning patterns
        self.cleaning_patterns = {
            'email': r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',
            'phone': r'[\+]?[1-9]?[0-9]{7,15}',
            'url': r'https?://[^\s,]+',
            'instagram': r'@\w+|https?://(www\.)?instagram\.com/[^\s,]+',
            'linkedin': r'https?://(www\.)?linkedin\.com/[^\s,]+'
        }
    
    def normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize and clean a DataFrame for better query performance.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned and normalized DataFrame
        """
        if df is None or df.empty:
            return df
            
        result = df.copy()
        
        # Clean column names
        result.columns = [self._clean_column_name(col) for col in result.columns]
        
        # Normalize string data
        for col in result.columns:
            if result[col].dtype == 'object':
                result[col] = result[col].apply(self._normalize_string)
        
        # Remove completely empty rows
        result = result.dropna(how='all')
        
        # Clean email addresses
        email_cols = self._find_columns_by_type(result, 'email')
        for col in email_cols:
            result[col] = result[col].apply(self._clean_email)
        
        # Clean URLs
        url_cols = self._find_columns_by_type(result, 'url')
        for col in url_cols:
            result[col] = result[col].apply(self._clean_url)
        
        return result
    
    def _clean_column_name(self, name: str) -> str:
        """Clean and normalize column names."""
        if pd.isna(name):
            return "unnamed"
        return str(name).strip().replace('\n', ' ').replace('\r', ' ')
    
    def _normalize_string(self, value: Any) -> str:
        """Normalize string values."""
        if pd.isna(value) or value is None:
            return ""
        return str(value).strip()
    
    def _clean_email(self, email: str) -> str:
        """Clean and validate email addresses."""
        if pd.isna(email) or not email:
            return ""
        email = str(email).strip().lower()
        if re.match(self.cleaning_patterns['email'], email):
            return email
        return ""
    
    def _clean_url(self, url: str) -> str:
        """Clean and validate URLs."""
        if pd.isna(url) or not url:
            return ""
        url = str(url).strip()
        if re.match(self.cleaning_patterns['url'], url):
            return url
        return ""
    
    def _find_columns_by_type(self, df: pd.DataFrame, col_type: str) -> List[str]:
        """Find columns that likely contain specific data types."""
        matching_cols = []
        for col in df.columns:
            col_lower = col.lower()
            if col_type in col_lower or any(syn in col_lower for syn in self.column_synonyms.get(col_type, [])):
                matching_cols.append(col)
        return matching_cols
    
    def find_best_column_match(self, target: str, available_columns: List[str]) -> Optional[str]:
        """
        Find the best matching column for a given target using fuzzy matching.
        
        Args:
            target: Target column name to match
            available_columns: List of available column names
            
        Returns:
            Best matching column name or None
        """
        target_lower = target.lower().strip()
        
        # Direct exact match
        for col in available_columns:
            if col.lower() == target_lower:
                return col
        
        # Synonym matching
        for col in available_columns:
            col_lower = col.lower()
            for key, synonyms in self.column_synonyms.items():
                if key in target_lower and any(syn in col_lower for syn in synonyms):
                    return col
                if any(syn in target_lower for syn in synonyms) and key in col_lower:
                    return col
        
        # Fuzzy matching
        best_match = None
        best_score = 0.0
        
        for col in available_columns:
            score = SequenceMatcher(None, target_lower, col.lower()).ratio()
            if score > best_score and score > 0.6:  # Minimum threshold
                best_score = score
                best_match = col
        
        return best_match
    
    def enhanced_search(self, df: pd.DataFrame, query: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Enhanced search functionality with better matching algorithms.
        
        Args:
            df: DataFrame to search
            query: Search query
            columns: Specific columns to search (if None, searches all string columns)
            
        Returns:
            Filtered DataFrame with matching rows
        """
        if df is None or df.empty:
            return df
        
        query_lower = query.lower().strip()
        query_terms = [term.strip() for term in query_lower.split() if term.strip()]
        
        if not query_terms:
            return df
        
        # Determine columns to search
        if columns is None:
            search_columns = [col for col in df.columns if df[col].dtype == 'object']
        else:
            search_columns = [col for col in columns if col in df.columns]
        
        # Create search mask - start with False for all rows
        mask = pd.Series([False] * len(df), index=df.index)
        
        for col in search_columns:
            col_mask = pd.Series([False] * len(df), index=df.index)
            
            for term in query_terms:
                # Exact match (case-insensitive)
                exact_match = df[col].astype(str).str.lower().str.contains(re.escape(term), na=False, regex=True)
                col_mask |= exact_match
                
                # Fuzzy match for better results (only if exact match fails)
                if not col_mask.any():  # Only apply fuzzy if no exact matches
                    fuzzy_match = df[col].astype(str).apply(
                        lambda x: any(SequenceMatcher(None, term, word.lower()).ratio() > 0.7 
                                    for word in str(x).split() if word.strip())
                    )
                    col_mask |= fuzzy_match
            
            # Only add to main mask if this column has matches
            if col_mask.any():
                mask |= col_mask
        
        # Return only rows that have at least one match
        return df[mask] if mask.any() else pd.DataFrame()
    
    def extract_contact_info(self, text: str) -> Dict[str, Any]:
        """
        Enhanced contact information extraction from text.
        
        Args:
            text: Input text to extract information from
            
        Returns:
            Dictionary with extracted contact information
        """
        result = {}
        
        # Extract email
        email_match = re.search(self.cleaning_patterns['email'], text)
        if email_match:
            result['email'] = email_match.group(0).lower()
        
        # Extract LinkedIn
        linkedin_match = re.search(self.cleaning_patterns['linkedin'], text)
        if linkedin_match:
            result['linkedin'] = linkedin_match.group(0)
        
        # Extract Instagram
        instagram_match = re.search(self.cleaning_patterns['instagram'], text)
        if instagram_match:
            result['instagram'] = instagram_match.group(0)
        
        # Extract website
        website_match = re.search(self.cleaning_patterns['url'], text)
        if website_match:
            result['website'] = website_match.group(0)
        
        # Extract name (simple heuristic)
        name_patterns = [
            r'(?:contact|name)[:\s]+([A-Z][A-Za-z\s\'-]+)',
            r'([A-Z][A-Za-z\s\'-]+)\s+(?:is|as|,)\s+(?:a|the)',
            r'add\s+([A-Z][A-Za-z\s\'-]+)'
        ]
        
        for pattern in name_patterns:
            name_match = re.search(pattern, text, re.IGNORECASE)
            if name_match:
                result['name'] = name_match.group(1).strip()
                break
        
        return result
    
    def categorize_email_domain(self, email: str) -> str:
        """
        Categorize email domain as generic or specific.
        
        Args:
            email: Email address to categorize
            
        Returns:
            'generic' or 'specific'
        """
        if not email or '@' not in email:
            return 'unknown'
        
        domain = email.split('@')[1].lower()
        generic_domains = [
            'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 
            'live.com', 'aol.com', 'icloud.com', 'protonmail.com'
        ]
        
        return 'generic' if domain in generic_domains else 'specific'
    
    def get_data_quality_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get data quality metrics for a DataFrame.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with quality metrics
        """
        if df is None or df.empty:
            return {}
        
        metrics = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'empty_rows': df.isnull().all(axis=1).sum(),
            'duplicate_rows': df.duplicated().sum(),
            'column_completeness': {}
        }
        
        for col in df.columns:
            non_null_count = df[col].notna().sum()
            metrics['column_completeness'][col] = {
                'non_null_count': int(non_null_count),
                'completeness_ratio': float(non_null_count / len(df)) if len(df) > 0 else 0.0
            }
        
        return metrics


# Global instance for use across the application
data_processor = DataProcessor()
