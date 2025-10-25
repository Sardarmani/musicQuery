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
            
            # ENHANCED SYSTEM PROMPT FOR STRICT FILTERING
            system_prompt = """You are a data filtering expert. I will give you:
1. ALL the data from a database/spreadsheet
2. A user query

CRITICAL: You must FILTER the data and return ONLY records that match the user query.

FILTERING RULES - BE VERY STRICT:
- "Irish" means ONLY records where Country field contains "Ireland" or "Irish"
- "Italian" means ONLY records where Country field contains "Italy" or "Italian"  
- "French" means ONLY records where Country field contains "France" or "French"
- "Portuguese" means ONLY records where Country field contains "Portugal" or "Portuguese"
- "German" means ONLY records where Country field contains "Germany" or "German"
- "Spanish" means ONLY records where Country field contains "Spain" or "Spanish"
- "British" means ONLY records where Country field contains "UK", "United Kingdom", "England", "Scotland", "Wales"

STRICT FILTERING EXAMPLES:
- Query "give me a list of all irish" → ONLY return records where Country = "Ireland" or contains "Irish"
- Query "Italian events" → ONLY return records where Country = "Italy" or contains "Italian"
- Query "French contacts" → ONLY return records where Country = "France" or contains "French"

IMPORTANT: 
- Do NOT return records with empty Country fields
- Do NOT return records with "?" in Country field
- Do NOT return records that don't match the country criteria
- If no records match, return empty array []

Return ONLY a JSON array of records that match the country criteria. If no matches, return empty array [].

Example for "Irish" query:
[
  {"Name": "John Smith", "Country": "Ireland", "Email": "john@email.com"},
  {"Name": "Jane Doe", "Country": "Ireland, Dublin", "Email": "jane@email.com"}
]"""

            # ENHANCED USER PROMPT FOR STRICT FILTERING
            user_prompt = f"""DATA:
{json.dumps(all_data, indent=2)}

QUERY: "{user_query}"

TASK: Filter the data and return ONLY records that match the query.

FILTERING INSTRUCTIONS:
- Look at the Country field in each record
- If query mentions "Irish", ONLY include records where Country contains "Ireland" or "Irish"
- If query mentions "Italian", ONLY include records where Country contains "Italy" or "Italian"
- If query mentions "French", ONLY include records where Country contains "France" or "French"
- EXCLUDE records with empty Country fields
- EXCLUDE records with "?" in Country field
- EXCLUDE records that don't match the country criteria

Return ONLY the JSON array of matching records. If no matches, return empty array []."""

            print("=" * 80)
            print("SYSTEM PROMPT:")
            print(system_prompt)
            print("=" * 80)
            print("USER PROMPT:")
            print(user_prompt)
            print("=" * 80)
            print("SENDING TO GPT-5 PRO...")
            
            response = self.client.chat.completions.create(
                model="gpt-5-pro",  # Upgraded to GPT-5 Pro
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