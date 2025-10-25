import json
import logging
from typing import Dict, Any, Optional
import pandas as pd
from openai import OpenAI

logger = logging.getLogger(__name__)

class DataStructureAnalyzer:
    def __init__(self):
        self.client = None
        try:
            from dotenv import dotenv_values
            env_values = dotenv_values(".env")
            api_key = env_values.get("OPENAI_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")

    def analyze_data_structure(self, df: pd.DataFrame, worksheet_name: str) -> Dict[str, Any]:
        """Simple data structure analysis - just return the DataFrame."""
        if df is None or df.empty:
            return {}
        print(f"Data structure analysis: {df.head()}")
        return {
            'worksheet_name': worksheet_name,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'dataframe': df  # Include the full DataFrame for direct GPT approach
        }

    def get_gpt_query_analysis(self, user_query: str, data_structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        SIMPLIFIED: Send ALL sheet data + user query to GPT-5 and get filtered results.
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
            
            # SIMPLIFIED SYSTEM PROMPT
            system_prompt = """You are a data filtering expert. I will give you:
1. ALL the data from a database/spreadsheet
2. A user query

Your job: Return ONLY records that match ALL conditions in the user query.

IMPORTANT RULES:
- "Italian" matches "Italy", "Italy, Turin", "Italy, Milan"
- "French" matches "France", "France, Paris", "France, Lyon"  
- "Portuguese" matches "Portugal", "Portugal, Lisbon", "Portugal, Porto"
- "Irish" matches "Ireland", "Ireland, Dublin", "Ireland, Cork"
- If query mentions LinkedIn, record must have actual LinkedIn data (not empty)
- If query mentions "name contains M", record must have "M" in the name
- If query mentions "direct email", exclude generic emails like info@, contact@

Return ONLY a JSON array of matching records. If no matches, return empty array [].

Example:
[
  {"Name": "John Smith", "Country": "Italy, Turin", "Event": "Festival", "Month": "July", "LinkedIn": "linkedin.com/in/johnsmith"}
]"""

            # SIMPLIFIED USER PROMPT
            user_prompt = f"""DATA:
{json.dumps(all_data, indent=2)}

QUERY: "{user_query}"

Find records that match ALL conditions in the query. Return ONLY the JSON array."""

            print("=" * 80)
            print("SYSTEM PROMPT:")
            print(system_prompt)
            print("=" * 80)
            print("USER PROMPT:")
            print(user_prompt)
            print("=" * 80)
            print("SENDING TO GPT-5...")
            
            response = self.client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=6000
            )
            
            result = response.choices[0].message.content
            print("GPT RESPONSE:")
            print(result)
            print("=" * 80)
            
            return self._parse_direct_gpt_response(result)
            
        except Exception as e:
            logger.error(f"GPT analysis failed: {e}")
            print(f"ERROR: {e}")
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

    def _fallback_analysis(self, user_query: str, data_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Simple fallback when GPT is not available."""
        return {
            'intent': f'Fallback search for {user_query}',
            'search_strategy': 'fallback',
            'filters': [],
            'return_full_rows': True,
            'confidence': 0.5
        }

    def apply_gpt_analysis(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply GPT analysis - handles direct GPT filtering results.
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
        
        # Fallback to old filtering logic
        return df

# Global instance
data_analyzer = DataStructureAnalyzer()