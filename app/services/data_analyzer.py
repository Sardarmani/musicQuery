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
            'sample_data': sample_data,
            'dataframe': df  # Include the full DataFrame for direct GPT approach
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
        Direct OpenAI approach: Send ALL sheet data + user query to GPT-5 and get filtered results.
        This is much simpler and more effective than complex filtering logic.
        """
        if not self.client:
            return self._fallback_analysis(user_query, data_structure)
        
        try:
            # Get ALL the data from the worksheet
            df = data_structure.get('dataframe')
            if df is None or df.empty:
                return self._fallback_analysis(user_query, data_structure)
            
            # Convert ALL data to JSON for GPT
            all_data = df.to_dict('records')
            
            system_prompt = """You are a data analysis expert. I will give you:
1. ALL the data from a database/spreadsheet
2. A user query

CRITICAL: You must match ALL conditions mentioned in the user query EXACTLY. Do NOT ignore any part of the query.

LOCATION VARIATIONS - IMPORTANT:
- "Italian" matches "Italy", "Italy, Turin", "Italy, Milan", etc.
- "French" matches "France", "France, Paris", "France, Lyon", etc.
- "Portuguese" matches "Portugal", "Portugal, Lisbon", "Portugal, Porto", etc.
- "Irish" matches "Ireland", "Ireland, Dublin", "Ireland, Cork", etc.
- "German" matches "Germany", "Germany, Berlin", "Germany, Munich", etc.
- "Spanish" matches "Spain", "Spain, Madrid", "Spain, Barcelona", etc.
- "British" matches "UK", "United Kingdom", "England", "Scotland", "Wales", etc.

COMPLEX QUERY PATTERNS:
- "French contacts with name containing M" → Find French records where name has "M"
- "Irish DJs" → Find Irish records where type is DJ
- "Portuguese festivals in July with LinkedIn" → Find Portuguese festival records in July with LinkedIn data
- "Media with high Instagram followers" → Find records with high Instagram follower counts
- "French events with direct email addresses" → Find French events with specific email patterns (exclude info@, contact@, etc.)

IMPORTANT RULES:
1. Look for ALL keywords/conditions in the user query
2. A record must match ALL keywords/conditions to be included
3. Handle location variations (Italian = Italy, French = France, etc.)
4. Handle partial matches in location fields (Italy, Turin = Italian)
5. If a query mentions LinkedIn, the record must have actual LinkedIn data (not empty/null)
6. Be VERY strict about matching ALL conditions
7. Check for actual data presence - don't include records with empty fields when specific data is requested
8. Handle name pattern matching (name contains M, etc.)
9. Handle email pattern matching (exclude generic emails like info@)

STRICT MATCHING EXAMPLES:
- Query "Italian events" → Include "Italy, Turin", "Italy, Milan", etc.
- Query "French contacts with name containing M" → Include French records where name has "M"
- Query "Portuguese festivals in July with LinkedIn" → Include Portugal records in July WITH LinkedIn data
- Query "French events with direct email" → Include French events with specific emails (exclude info@)

Return ONLY a JSON array of records that match ALL the conditions mentioned in the user query.
If no records match ALL conditions, return an empty array [].

Example response format:
[
  {"Name": "John Smith", "Country": "Italy, Turin", "Event": "Festival", "Month": "July", "LinkedIn": "linkedin.com/in/johnsmith"},
  {"Name": "Jane Doe", "Country": "France, Paris", "Event": "Club", "Month": "July", "LinkedIn": "linkedin.com/in/janedoe"}
]"""

            user_prompt = f"""ALL DATA FROM THE DATABASE:
{json.dumps(all_data, indent=2)}

USER QUERY: "{user_query}"

TASK: Find records that match ALL conditions mentioned in the user query EXACTLY.

ANALYSIS REQUIRED:
1. Identify ALL conditions in the user query (location, time, event type, specific data fields, etc.)
2. Look for records that contain ALL of these conditions
3. A record must match ALL conditions to be included in results
4. Be VERY strict - if a record is missing ANY condition, exclude it
5. Check for actual data presence - if a query asks for specific data (like LinkedIn), the record must have actual data (not empty/null)

LOCATION VARIATIONS - CRITICAL:
- "Italian" matches "Italy", "Italy, Turin", "Italy, Milan", etc.
- "French" matches "France", "France, Paris", "France, Lyon", etc.
- "Portuguese" matches "Portugal", "Portugal, Lisbon", "Portugal, Porto", etc.
- "Irish" matches "Ireland", "Ireland, Dublin", "Ireland, Cork", etc.
- "German" matches "Germany", "Germany, Berlin", "Germany, Munich", etc.
- "Spanish" matches "Spain", "Spain, Madrid", "Spain, Barcelona", etc.
- "British" matches "UK", "United Kingdom", "England", "Scotland", "Wales", etc.

CLIENT QUERY EXAMPLES:
- "French contacts with name containing M" → Find French records where name has "M"
- "Irish DJs" → Find Irish records where type is DJ
- "Portuguese festivals in July with LinkedIn" → Find Portuguese festival records in July with LinkedIn data
- "Media with high Instagram followers" → Find records with high Instagram follower counts
- "French events with direct email addresses" → Find French events with specific email patterns (exclude info@, contact@, etc.)

STRICT VALIDATION RULES:
- If query mentions "Italian", include "Italy, Turin", "Italy, Milan", etc.
- If query mentions "French", include "France, Paris", "France, Lyon", etc.
- If query mentions "Portuguese", include "Portugal, Lisbon", "Portugal, Porto", etc.
- If query mentions "July", ONLY include records from July - EXCLUDE other months
- If query mentions LinkedIn, ONLY include records with actual LinkedIn URLs/data - EXCLUDE empty LinkedIn fields
- If query mentions "name contains M", ONLY include records where name has "M"
- If query mentions "direct email", EXCLUDE generic emails like info@, contact@, hello@, etc.

EXCLUSION EXAMPLES:
- Query "Italian events" → INCLUDE "Italy, Turin", "Italy, Milan" - EXCLUDE other countries
- Query "French contacts with name containing M" → INCLUDE French records with "M" in name - EXCLUDE non-French, EXCLUDE names without "M"
- Query "Portuguese festivals in July with LinkedIn" → INCLUDE Portugal festivals in July WITH LinkedIn - EXCLUDE other countries, EXCLUDE other months, EXCLUDE records without LinkedIn

Return ONLY a JSON array of records that match ALL conditions.
If no records match ALL conditions, return an empty array [].

Return ONLY the JSON array, no other text."""

            response = self.client.chat.completions.create(
                model="gpt-5",  # Using GPT-5 as requested
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=6000  # More tokens for better analysis of large datasets
            )
            
            result = response.choices[0].message.content
            return self._parse_direct_gpt_response(result)
            
        except Exception as e:
            logger.error(f"GPT analysis failed: {e}")
            return self._fallback_analysis(user_query, data_structure)
    
    def _parse_direct_gpt_response(self, response: str) -> Dict[str, Any]:
        """Parse direct GPT response that returns filtered data."""
        try:
            # Extract JSON array from response
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                filtered_data = json.loads(json_str)
                
                return {
                    'intent': 'Direct GPT filtering',
                    'search_strategy': 'direct_gpt',
                    'filtered_data': filtered_data,
                    'return_full_rows': True,
                    'confidence': 0.95
                }
            else:
                return self._fallback_analysis("", {})
        except Exception:
            return self._fallback_analysis("", {})
    
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
        Apply GPT analysis - now handles direct GPT filtering results.
        
        Args:
            df: DataFrame to search (not used for direct GPT approach)
            analysis: GPT analysis results
            
        Returns:
            Filtered DataFrame with full rows
        """
        if df is None or df.empty:
            return df
        
        # Check if this is direct GPT filtering
        if analysis.get('search_strategy') == 'direct_gpt':
            filtered_data = analysis.get('filtered_data', [])
            if filtered_data:
                # Convert filtered data back to DataFrame
                return pd.DataFrame(filtered_data)
            else:
                # Return empty DataFrame if no matches
                return pd.DataFrame()
        
        # Fallback to old filtering logic for backward compatibility
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
