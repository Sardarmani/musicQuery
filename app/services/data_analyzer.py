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
        
        # Get more sample data (first 10 rows) for better GPT analysis
        sample_data = df.head(10).to_dict('records')
        
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
        Use GPT to analyze the user query against the data structure.
        
        Args:
            user_query: User's search query
            data_structure: Analyzed data structure
            
        Returns:
            Dictionary with GPT analysis and search recommendations
        """
        if not self.client:
            return self._fallback_analysis(user_query, data_structure)
        
        try:
            # Prepare comprehensive data structure summary for GPT
            columns_info = []
            for col, info in data_structure.get('columns', {}).items():
                columns_info.append({
                    'name': col,
                    'type': info['type'],
                    'sample_values': info['sample_values'],  # Show all sample values
                    'completeness': info['completeness']
                })
            
            system_prompt = """You are an expert at analyzing data structures and understanding user queries for database searches.

Your task is to:
1. Analyze the user's query to understand what they're looking for
2. Identify which columns in the data structure are relevant for filtering
3. Determine the best search strategy
4. Provide specific filters to find matching rows

IMPORTANT: Always return FULL ROWS, not just specific columns. The user wants complete matching records.

Return a JSON response with:
- intent: What the user is looking for
- relevant_columns: List of column names that are relevant for filtering
- search_strategy: How to search (enhanced, fuzzy, or structured)
- filters: List of filters to apply (column, operator, value)
- return_full_rows: true (always return complete rows)
- confidence: How confident you are (0-1)

Be precise and focus on finding the most relevant rows for the user's query."""

            user_prompt = f"""
Data Structure Analysis:
- Worksheet: {data_structure.get('worksheet_name', 'Unknown')}
- Total Rows: {data_structure.get('total_rows', 0)}
- Total Columns: {data_structure.get('total_columns', 0)}

Columns with Sample Data:
{json.dumps(columns_info, indent=2)}

Sample Data (first 10 rows):
{json.dumps(data_structure.get('sample_data', []), indent=2)}

User Query: "{user_query}"

Analyze this query against the actual data structure and provide search recommendations.
Focus on finding rows that match the user's intent.
"""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            result = response.choices[0].message.content
            return self._parse_gpt_response(result)
            
        except Exception as e:
            logger.error(f"GPT analysis failed: {e}")
            return self._fallback_analysis(user_query, data_structure)
    
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
        """Fallback analysis when GPT is not available."""
        query_lower = user_query.lower()
        
        # Simple keyword matching
        relevant_columns = []
        filters = []
        
        # Look for location terms
        if any(term in query_lower for term in ['italian', 'italy', 'french', 'france', 'german', 'germany']):
            for col, info in data_structure.get('columns', {}).items():
                if info['type'] in ['location', 'city', 'country']:
                    relevant_columns.append(col)
                    if 'italian' in query_lower or 'italy' in query_lower:
                        filters.append({
                            'column': col,
                            'operator': 'icontains',
                            'value': 'italy'
                        })
        
        # Look for time terms
        if any(term in query_lower for term in ['august', 'july', 'june', 'september']):
            for col, info in data_structure.get('columns', {}).items():
                if info['type'] == 'time':
                    relevant_columns.append(col)
                    if 'august' in query_lower:
                        filters.append({
                            'column': col,
                            'operator': 'icontains',
                            'value': 'august'
                        })
        
        # Look for event terms
        if 'event' in query_lower:
            for col, info in data_structure.get('columns', {}).items():
                if info['type'] == 'event':
                    relevant_columns.append(col)
        
        return {
            'intent': 'Search for events',
            'relevant_columns': relevant_columns,
            'search_strategy': 'enhanced',
            'filters': filters,
            'return_full_rows': True,  # Always return full rows
            'confidence': 0.7
        }
    
    def apply_gpt_analysis(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply GPT analysis to filter and return full matching rows.
        
        Args:
            df: DataFrame to search
            analysis: GPT analysis results
            
        Returns:
            Filtered DataFrame with full rows
        """
        if df is None or df.empty:
            return df
        
        result_df = df.copy()
        
        # Apply filters
        for filter_info in analysis.get('filters', []):
            col = filter_info.get('column')
            operator = filter_info.get('operator', 'icontains')
            value = filter_info.get('value')
            
            if col in result_df.columns and value:
                if operator == 'icontains':
                    mask = result_df[col].astype(str).str.contains(str(value), case=False, na=False)
                    result_df = result_df[mask]
                elif operator == 'eq':
                    mask = result_df[col].astype(str).str.lower() == str(value).lower()
                    result_df = result_df[mask]
                elif operator == 'contains':
                    mask = result_df[col].astype(str).str.contains(str(value), na=False)
                    result_df = result_df[mask]
        
        # Always return full rows (don't select specific columns)
        # The user wants complete matching records
        
        # Limit results to prevent overwhelming output
        limit = analysis.get('limit', 50)  # Increased limit for better results
        if limit and len(result_df) > limit:
            result_df = result_df.head(limit)
        
        return result_df


# Global instance
data_analyzer = DataStructureAnalyzer()
