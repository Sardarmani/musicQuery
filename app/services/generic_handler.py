"""
Generic worksheet handler for dynamic tab support.
This module provides functionality to work with any worksheet structure dynamically.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import re
from difflib import SequenceMatcher
import logging

logger = logging.getLogger(__name__)


class GenericWorksheetHandler:
    """
    Generic handler for any worksheet structure.
    Automatically adapts to different column structures and data types.
    """
    
    def __init__(self):
        self.common_patterns = {
            'email': ['email', 'e-mail', 'email address', 'email name', 'mail'],
            'name': ['name', 'contact name', 'contact', 'person', 'full name', 'dj name'],
            'phone': ['phone', 'telephone', 'mobile', 'cell', 'number'],
            'linkedin': ['linkedin', 'linkedin profile', 'linkedin url', 'linkedin contact'],
            'instagram': ['instagram', 'ig', 'instagram handle', 'instagram page', 'instagram profile'],
            'website': ['website', 'url', 'web', 'site', 'homepage', 'ra page'],
            'event': ['event', 'event name', 'venue', 'festival', 'club', 'party'],
            'location': ['location', 'city', 'country', 'address', 'place', 'venue location'],
            'date': ['date', 'time', 'when', 'schedule', 'start date', 'end date'],
            'status': ['status', 'active', 'inactive', 'current', 'state'],
            'category': ['category', 'type', 'genre', 'style', 'classification'],
            'description': ['description', 'details', 'info', 'notes', 'comments', 'about'],
            'price': ['price', 'cost', 'fee', 'amount', 'rate', 'ticket price'],
            'capacity': ['capacity', 'size', 'max', 'maximum', 'seats', 'attendance']
        }
        
        # Data type detection patterns
        self.data_patterns = {
            'email': r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',
            'phone': r'[\+]?[1-9]?[0-9]{7,15}',
            'url': r'https?://[^\s,]+',
            'date': r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}',
            'price': r'\$?\d+\.?\d*',
            'instagram': r'@\w+|https?://(www\.)?instagram\.com/[^\s,]+',
            'linkedin': r'https?://(www\.)?linkedin\.com/[^\s,]+'
        }
    
    def analyze_worksheet_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze a worksheet to understand its structure and data types.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with structure analysis
        """
        if df is None or df.empty:
            return {'error': 'Empty worksheet'}
        
        analysis = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': {},
            'data_types': {},
            'suggested_mappings': {},
            'quality_metrics': {}
        }
        
        # Analyze each column
        for col in df.columns:
            col_analysis = self._analyze_column(df[col], col)
            analysis['columns'][col] = col_analysis
            
            # Suggest column type based on content
            suggested_type = self._suggest_column_type(df[col], col)
            if suggested_type:
                analysis['suggested_mappings'][col] = suggested_type
        
        # Overall quality metrics
        analysis['quality_metrics'] = self._calculate_quality_metrics(df)
        
        return analysis
    
    def _analyze_column(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Analyze a single column for its characteristics."""
        analysis = {
            'non_null_count': series.notna().sum(),
            'null_count': series.isna().sum(),
            'unique_count': series.nunique(),
            'completeness_ratio': series.notna().sum() / len(series) if len(series) > 0 else 0,
            'data_type': str(series.dtype),
            'sample_values': series.dropna().head(5).tolist(),
            'detected_patterns': []
        }
        
        # Detect data patterns
        for pattern_name, pattern in self.data_patterns.items():
            matches = series.astype(str).str.contains(pattern, na=False).sum()
            if matches > 0:
                analysis['detected_patterns'].append({
                    'pattern': pattern_name,
                    'matches': int(matches),
                    'percentage': float(matches / len(series)) if len(series) > 0 else 0
                })
        
        return analysis
    
    def _suggest_column_type(self, series: pd.Series, column_name: str) -> Optional[str]:
        """Suggest the most likely column type based on content and name."""
        col_lower = column_name.lower()
        
        # Check column name patterns
        for type_name, patterns in self.common_patterns.items():
            if any(pattern in col_lower for pattern in patterns):
                return type_name
        
        # Check content patterns
        for pattern_name, pattern in self.data_patterns.items():
            matches = series.astype(str).str.contains(pattern, na=False).sum()
            if matches > len(series) * 0.3:  # 30% threshold
                return pattern_name
        
        # Check for numeric data
        if pd.api.types.is_numeric_dtype(series):
            return 'numeric'
        
        # Check for date-like data
        if series.dtype == 'object':
            try:
                pd.to_datetime(series.dropna().head(10))
                return 'date'
            except:
                pass
        
        return 'text'
    
    def _calculate_quality_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate overall data quality metrics."""
        return {
            'completeness_score': df.notna().sum().sum() / (len(df) * len(df.columns)) if len(df) > 0 else 0,
            'duplicate_rows': df.duplicated().sum(),
            'empty_rows': df.isnull().all(axis=1).sum(),
            'columns_with_high_completeness': [
                col for col in df.columns 
                if df[col].notna().sum() / len(df) > 0.8
            ],
            'columns_with_low_completeness': [
                col for col in df.columns 
                if df[col].notna().sum() / len(df) < 0.3
            ]
        }
    
    def find_columns_by_type(self, df: pd.DataFrame, target_type: str) -> List[str]:
        """
        Find columns that match a specific data type.
        
        Args:
            df: DataFrame to search
            target_type: Type to search for (email, name, phone, etc.)
            
        Returns:
            List of matching column names
        """
        matching_columns = []
        
        # Check column names
        for col in df.columns:
            col_lower = col.lower()
            if target_type in col_lower:
                matching_columns.append(col)
                continue
            
            # Check against common patterns
            if target_type in self.common_patterns:
                for pattern in self.common_patterns[target_type]:
                    if pattern in col_lower:
                        matching_columns.append(col)
                        break
        
        # Check content patterns if no name matches
        if not matching_columns and target_type in self.data_patterns:
            pattern = self.data_patterns[target_type]
            for col in df.columns:
                if df[col].astype(str).str.contains(pattern, na=False).sum() > len(df) * 0.1:
                    matching_columns.append(col)
        
        return matching_columns
    
    def create_generic_search_query(self, df: pd.DataFrame, query: str) -> Dict[str, Any]:
        """
        Create a generic search query that works with any worksheet structure.
        
        Args:
            df: DataFrame to search
            query: Search query
            
        Returns:
            Dictionary with search configuration
        """
        query_lower = query.lower()
        query_terms = [term.strip() for term in query_lower.split() if term.strip()]
        
        # Analyze the worksheet structure
        analysis = self.analyze_worksheet_structure(df)
        
        # Determine search strategy based on content
        search_config = {
            'query_terms': query_terms,
            'search_columns': [],
            'filters': [],
            'suggested_columns': []
        }
        
        # Find relevant columns based on query terms
        for term in query_terms:
            # Look for exact column matches
            for col in df.columns:
                if term in col.lower():
                    search_config['search_columns'].append(col)
            
            # Look for content-based matches
            for col in df.columns:
                if df[col].dtype == 'object':
                    matches = df[col].astype(str).str.contains(term, case=False, na=False).sum()
                    if matches > 0:
                        search_config['suggested_columns'].append({
                            'column': col,
                            'matches': int(matches),
                            'relevance': float(matches / len(df))
                        })
        
        # Remove duplicates and sort by relevance
        search_config['search_columns'] = list(set(search_config['search_columns']))
        search_config['suggested_columns'].sort(key=lambda x: x['relevance'], reverse=True)
        
        return search_config
    
    def enhanced_generic_search(self, df: pd.DataFrame, query: str) -> pd.DataFrame:
        """
        Enhanced generic search that works with any worksheet structure.
        
        Args:
            df: DataFrame to search
            query: Search query
            
        Returns:
            Filtered DataFrame with matching rows
        """
        if df is None or df.empty:
            return df
        
        query_lower = query.lower().strip()
        query_terms = [term.strip() for term in query_lower.split() if term.strip()]
        
        if not query_terms:
            return df
        
        # Create search mask - start with False for all rows
        mask = pd.Series([False] * len(df), index=df.index)
        
        # For event queries, prioritize location and time columns
        location_terms = ['italian', 'italy', 'french', 'france', 'german', 'germany', 'spanish', 'spain', 'august', 'july', 'june', 'september']
        has_location_terms = any(term in query_lower for term in location_terms)
        
        # Search in all string columns
        for col in df.columns:
            if df[col].dtype == 'object':
                col_mask = pd.Series([False] * len(df), index=df.index)
                col_lower = col.lower()
                
                # Prioritize location and time columns for event queries
                is_location_col = any(term in col_lower for term in ['country', 'city', 'location', 'place'])
                is_time_col = any(term in col_lower for term in ['month', 'time', 'date', 'period'])
                is_event_col = any(term in col_lower for term in ['event', 'name', 'title', 'festival', 'club'])
                
                for term in query_terms:
                    # Exact match (case-insensitive)
                    exact_match = df[col].astype(str).str.lower().str.contains(re.escape(term), na=False, regex=True)
                    col_mask |= exact_match
                    
                    # For location terms, be more strict about matching
                    if term in location_terms and is_location_col:
                        # Require higher similarity for location terms
                        fuzzy_match = df[col].astype(str).apply(
                            lambda x: any(SequenceMatcher(None, term, word.lower()).ratio() > 0.8 
                                        for word in str(x).split() if word.strip())
                        )
                        col_mask |= fuzzy_match
                    elif not col_mask.any():  # Only apply fuzzy if no exact matches
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
    
    def get_worksheet_summary(self, df: pd.DataFrame, worksheet_name: str) -> Dict[str, Any]:
        """
        Get a comprehensive summary of a worksheet.
        
        Args:
            df: DataFrame to summarize
            worksheet_name: Name of the worksheet
            
        Returns:
            Dictionary with worksheet summary
        """
        if df is None or df.empty:
            return {'error': f'Worksheet {worksheet_name} is empty'}
        
        analysis = self.analyze_worksheet_structure(df)
        
        summary = {
            'worksheet_name': worksheet_name,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'column_types': {},
            'data_quality': analysis['quality_metrics'],
            'suggested_use_cases': []
        }
        
        # Categorize columns by type
        for col, col_analysis in analysis['columns'].items():
            suggested_type = self._suggest_column_type(df[col], col)
            if suggested_type:
                if suggested_type not in summary['column_types']:
                    summary['column_types'][suggested_type] = []
                summary['column_types'][suggested_type].append(col)
        
        # Suggest use cases based on column types
        if 'email' in summary['column_types']:
            summary['suggested_use_cases'].append('Contact management')
        if 'event' in summary['column_types']:
            summary['suggested_use_cases'].append('Event tracking')
        if 'location' in summary['column_types']:
            summary['suggested_use_cases'].append('Geographic analysis')
        if 'date' in summary['column_types']:
            summary['suggested_use_cases'].append('Time-based analysis')
        
        return summary


# Global instance for use across the application
generic_handler = GenericWorksheetHandler()
