"""
Data structure analyzer that uses GPT to understand data format and user queries.
This provides more accurate search by first analyzing the actual data structure.
"""

import pandas as pd
import json
from typing import Dict, List, Optional, Any, Tuple
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)


class DataStructureAnalyzer:
    """
    Analyzes data structure and uses GPT to understand user queries better.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        if api_key:
            try:
                self.client = OpenAI(api_key=api_key)
            except Exception:
                self.client = None
        else:
            try:
                self.client = OpenAI()
            except Exception:
                self.client = None
    
    def analyze_data_structure(self, df: pd.DataFrame, worksheet_name: str) -> Dict[str, Any]:
        """
        Analyze the data structure to understand column types and content.
        
        Args:
            df: DataFrame to analyze
            worksheet_name: Name of the worksheet
            
        Returns:
            Dictionary with data structure analysis
        """
        if df is None or df.empty:
            return {}
        
        # Get exactly 6 sample rows for GPT analysis
        sample_data = df.head(6).to_dict('records')
        
        # Analyze column types and content
        column_analysis = {}
        for col in df.columns:
            col_data = df[col].dropna()
            if not col_data.empty:
                # Get more sample values for better analysis
                sample_values = col_data.head(5).tolist()
                
                # Analyze column type
                column_type = self._analyze_column_type(col, col_data)
                
                column_analysis[col] = {
                    'type': column_type,
                    'sample_values': sample_values,
                    'non_null_count': len(col_data),
                    'total_count': len(df),
                    'completeness': len(col_data) / len(df) if len(df) > 0 else 0
                }
        
        return {
            'worksheet_name': worksheet_name,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': column_analysis,
            'sample_data': sample_data
        }
    
    def _analyze_column_type(self, col_name: str, col_data: pd.Series) -> str:
        """Analyze what type of data a column contains."""
        col_lower = col_name.lower()
        
        # Check for common patterns
        if any(term in col_lower for term in ['email', 'e-mail']):
            return 'email'
        elif any(term in col_lower for term in ['name', 'contact', 'person']):
            return 'name'
        elif any(term in col_lower for term in ['linkedin']):
            return 'linkedin'
        elif any(term in col_lower for term in ['instagram']):
            return 'instagram'
        elif any(term in col_lower for term in ['website', 'url']):
            return 'website'
        elif any(term in col_lower for term in ['event', 'festival', 'club']):
            return 'event'
        elif any(term in col_lower for term in ['country', 'location', 'place']):
            return 'location'
        elif any(term in col_lower for term in ['city']):
            return 'city'
        elif any(term in col_lower for term in ['month', 'time', 'date']):
            return 'time'
        elif any(term in col_lower for term in ['position', 'role', 'title']):
            return 'position'
        elif any(term in col_lower for term in ['status', 'active']):
            return 'status'
        else:
            # Analyze content to determine type
            sample_values = col_data.head(10).astype(str).tolist()
            
            # Check for email patterns
            if any('@' in str(val) for val in sample_values):
                return 'email'
            # Check for URL patterns
            elif any('http' in str(val).lower() for val in sample_values):
                return 'website'
            # Check for date patterns
            elif any(any(month in str(val).lower() for month in ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']) for val in sample_values):
                return 'time'
            else:
                return 'text'
    
    def get_gpt_query_analysis(self, user_query: str, data_structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simple, direct approach: Send sample rows + query to GPT-5 and ask how to extract data.
        
        Args:
            user_query: User's search query
            data_structure: Analyzed data structure with sample data
            
        Returns:
            Dictionary with GPT analysis and search recommendations
        """
        if not self.client:
            return self._fallback_analysis(user_query, data_structure)
        
        try:
            # Get the sample data (first 5-6 rows)
            sample_data = data_structure.get('sample_data', [])[:6]  # Take first 6 rows
            
            if not sample_data:
                return self._fallback_analysis(user_query, data_structure)
            
            system_prompt = """You are a data extraction expert. I will give you sample data from a database and a user query. 

Your job is to tell me EXACTLY how to find the data the user wants.

Look at the sample data and the user query, then tell me:
1. Which column(s) to search in
2. What to search for
3. How to search (exact match, contains, etc.)

Return ONLY a JSON response with this format:
{
  "search_column": "exact_column_name_from_sample_data",
  "search_value": "what_to_search_for",
  "search_type": "exact" or "contains"
}

Be very specific and use the EXACT column names from the sample data."""

            user_prompt = f"""SAMPLE DATA (first 6 rows from the database):
{json.dumps(sample_data, indent=2)}

USER QUERY: "{user_query}"

From this sample data and user query, tell me EXACTLY how to extract the data the user wants.
Look at the sample data to see what columns exist and what data looks like.
Then tell me which column to search and what to search for.

Return ONLY the JSON response."""

            response = self.client.chat.completions.create(
                model="gpt-5",  # Using GPT-5 as requested
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            result = response.choices[0].message.content
            return self._parse_simple_gpt_response(result)
            
        except Exception as e:
            logger.error(f"GPT analysis failed: {e}")
            return self._fallback_analysis(user_query, data_structure)
    
    def _parse_simple_gpt_response(self, response: str) -> Dict[str, Any]:
        """Parse simple GPT response into structured data."""
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                gpt_result = json.loads(json_str)
                
                # Convert simple response to our format
                search_column = gpt_result.get('search_column', '')
                search_value = gpt_result.get('search_value', '')
                search_type = gpt_result.get('search_type', 'contains')
                
                # Create filter based on GPT response
                filters = []
                if search_column and search_value:
                    operator = 'eq' if search_type == 'exact' else 'icontains'
                    filters.append({
                        'column': search_column,
                        'operator': operator,
                        'value': search_value
                    })
                
                return {
                    'intent': f'Search for {search_value} in {search_column}',
                    'search_strategy': 'simple',
                    'filters': filters,
                    'return_full_rows': True,
                    'confidence': 0.9
                }
            else:
                return self._fallback_analysis("", {})
        except Exception:
            return self._fallback_analysis("", {})
    
    def _parse_gpt_response(self, response: str) -> Dict[str, Any]:
        """Parse GPT response into structured data."""
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                return self._fallback_analysis("", {})
        except Exception:
            return self._fallback_analysis("", {})
    
    def _fallback_analysis(self, user_query: str, data_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced fallback analysis when GPT is not available."""
        query_lower = user_query.lower()
        
        # More comprehensive keyword matching
        relevant_columns = []
        filters = []
        
        # Look for location terms with better matching
        location_terms = ['italian', 'italy', 'french', 'france', 'german', 'germany', 'spanish', 'spain', 'uk', 'usa', 'united states', 'united kingdom']
        if any(term in query_lower for term in location_terms):
            for col, info in data_structure.get('columns', {}).items():
                col_lower = col.lower()
                # Look for location-related columns
                if any(loc_term in col_lower for loc_term in ['country', 'city', 'location', 'place', 'nation']):
                    relevant_columns.append(col)
                    # Create specific filters based on query terms
                    if 'italian' in query_lower or 'italy' in query_lower:
                        filters.append({
                            'column': col,
                            'operator': 'icontains',
                            'value': 'italy'
                        })
                    elif 'french' in query_lower or 'france' in query_lower:
                        filters.append({
                            'column': col,
                            'operator': 'icontains',
                            'value': 'france'
                        })
                    elif 'german' in query_lower or 'germany' in query_lower:
                        filters.append({
                            'column': col,
                            'operator': 'icontains',
                            'value': 'germany'
                        })
        
        # Look for time terms with better matching
        time_terms = ['august', 'july', 'june', 'september', 'october', 'november', 'december', 'january', 'february', 'march', 'april', 'may']
        if any(term in query_lower for term in time_terms):
            for col, info in data_structure.get('columns', {}).items():
                col_lower = col.lower()
                # Look for time-related columns
                if any(time_term in col_lower for time_term in ['month', 'time', 'date', 'period', 'when']):
                    relevant_columns.append(col)
                    # Create specific filters for each month
                    for month in time_terms:
                        if month in query_lower:
                            filters.append({
                                'column': col,
                                'operator': 'icontains',
                                'value': month
                            })
                            break
        
        # Look for event terms with better matching
        event_terms = ['event', 'events', 'festival', 'festivals', 'club', 'clubs', 'promoter', 'promoters']
        if any(term in query_lower for term in event_terms):
            for col, info in data_structure.get('columns', {}).items():
                col_lower = col.lower()
                # Look for event-related columns
                if any(event_term in col_lower for event_term in ['event', 'festival', 'club', 'promoter', 'venue', 'name']):
                    relevant_columns.append(col)
        
        return {
            'intent': f'Search for {user_query}',
            'relevant_columns': relevant_columns,
            'search_strategy': 'enhanced',
            'filters': filters,
            'return_full_rows': True,
            'confidence': 0.8
        }
    
    def apply_gpt_analysis(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply GPT analysis to filter and return full matching rows with enhanced precision.
        
        Args:
            df: DataFrame to search
            analysis: GPT analysis results
            
        Returns:
            Filtered DataFrame with full rows
        """
        if df is None or df.empty:
            return df
        
        result_df = df.copy()
        
        # Apply filters with enhanced precision
        for filter_info in analysis.get('filters', []):
            col = filter_info.get('column')
            operator = filter_info.get('operator', 'icontains')
            value = filter_info.get('value')
            
            if col in result_df.columns and value:
                try:
                    if operator == 'icontains':
                        # Case-insensitive contains with better handling
                        mask = result_df[col].astype(str).str.lower().str.contains(str(value).lower(), na=False, regex=False)
                        result_df = result_df[mask]
                    elif operator == 'eq':
                        # Exact match (case-insensitive)
                        mask = result_df[col].astype(str).str.lower() == str(value).lower()
                        result_df = result_df[mask]
                    elif operator == 'contains':
                        # Case-sensitive contains
                        mask = result_df[col].astype(str).str.contains(str(value), na=False, regex=False)
                        result_df = result_df[mask]
                    elif operator == 'startswith':
                        # Starts with
                        mask = result_df[col].astype(str).str.lower().str.startswith(str(value).lower(), na=False)
                        result_df = result_df[mask]
                    elif operator == 'endswith':
                        # Ends with
                        mask = result_df[col].astype(str).str.lower().str.endswith(str(value).lower(), na=False)
                        result_df = result_df[mask]
                except Exception as e:
                    # If filtering fails, continue with other filters
                    logger.warning(f"Filter failed for column {col}: {e}")
                    continue
        
        # Always return full rows (don't select specific columns)
        # The user wants complete matching records
        
        # Limit results to prevent overwhelming output
        limit = analysis.get('limit', 100)  # Increased limit for better results
        if limit and len(result_df) > limit:
            result_df = result_df.head(limit)
        
        return result_df


# Global instance
data_analyzer = DataStructureAnalyzer()
