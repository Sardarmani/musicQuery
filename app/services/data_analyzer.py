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
        ENHANCED: Send ALL sheet data + user query to GPT-5 Pro with context handling.
        """
        if not self.client:
            return self._fallback_analysis(user_query, data_structure)
        
        try:
            # Get ALL the data from the worksheet
            df = data_structure.get('dataframe')
            if df is None or df.empty:
                return self._fallback_analysis(user_query, data_structure)
            
            print(f"📊 DATASET INFO: {len(df)} rows, {len(df.columns)} columns")
            print(f"📋 COLUMNS: {list(df.columns)}")
            
            # Convert ALL data to JSON for GPT
            all_data = df.to_dict('records')
            
            # Check data size and handle context limitations
            data_json = json.dumps(all_data, indent=2)
            data_size = len(data_json)
            print(f"📏 DATA SIZE: {data_size:,} characters")
            
            # If data is too large, we might need to chunk it
            if data_size > 100000:  # 100k characters limit
                print("⚠️  Large dataset detected, using first 1000 rows for analysis")
                all_data = df.head(1000).to_dict('records')
                data_json = json.dumps(all_data, indent=2)
                print(f"📏 REDUCED DATA SIZE: {len(data_json):,} characters")
            
            # ENHANCED SYSTEM PROMPT FOR CLIENT QUERIES
            system_prompt = """You are a data filtering expert. I will give you:
1. ALL the data from a database/spreadsheet
2. A user query

CRITICAL: You must FILTER the data and return ONLY records that match ALL conditions in the user query.

CLIENT QUERY PATTERNS - BE VERY STRICT:
- "French contacts with name containing M" → ONLY French records where Name field contains "M"
- "Irish DJs" → ONLY Irish records where Type/Job field contains "DJ"
- "Portuguese festivals in July with LinkedIn" → ONLY Portuguese festival records in July WITH LinkedIn data
- "Media with high Instagram followers" → ONLY records with high Instagram follower counts

FILTERING RULES:
- "French" = Country field contains "France" or "French"
- "Irish" = Country field contains "Ireland" or "Irish"  
- "Portuguese" = Country field contains "Portugal" or "Portuguese"
- "Italian" = Country field contains "Italy" or "Italian"
- "German" = Country field contains "Germany" or "German"
- "Spanish" = Country field contains "Spain" or "Spanish"
- "British" = Country field contains "UK", "United Kingdom", "England", "Scotland", "Wales"

STRICT CONDITIONS:
- "name contains M" → Name field must contain letter "M"
- "DJs" → Type/Job field must contain "DJ"
- "festivals" → Type/Event field must contain "festival"
- "July" → Month field must contain "July" or "07"
- "LinkedIn" → LinkedIn field must have actual data (not empty)
- "high Instagram followers" → Instagram followers must be high numbers

IMPORTANT: 
- Do NOT return records with empty required fields
- Do NOT return records with "?" in required fields
- Do NOT return records that don't match ALL conditions
- If no records match ALL conditions, return empty array []

Return ONLY a JSON array of records that match ALL conditions. If no matches, return empty array [].

Example for "French contacts with name containing M":
[
  {"Name": "Martin", "Country": "France", "Email": "martin@email.com"},
  {"Name": "Marie", "Country": "France, Paris", "Email": "marie@email.com"}
]"""

            # ENHANCED USER PROMPT FOR CLIENT QUERIES
            user_prompt = f"""DATA:
{json.dumps(all_data, indent=2)}

QUERY: "{user_query}"

TASK: Filter the data and return ONLY records that match ALL conditions in the query.

FILTERING INSTRUCTIONS FOR CLIENT QUERIES:
- "French contacts with name containing M" → Find French records where Name contains "M"
- "Irish DJs" → Find Irish records where Type/Job contains "DJ"
- "Portuguese festivals in July with LinkedIn" → Find Portuguese festival records in July WITH LinkedIn
- "Media with high Instagram followers" → Find records with high Instagram follower counts

STRICT FILTERING RULES:
- Look at ALL relevant fields (Country, Name, Type, Month, LinkedIn, Instagram followers, etc.)
- If query mentions "French", ONLY include records where Country contains "France" or "French"
- If query mentions "Irish", ONLY include records where Country contains "Ireland" or "Irish"
- If query mentions "Portuguese", ONLY include records where Country contains "Portugal" or "Portuguese"
- If query mentions "name contains M", ONLY include records where Name contains "M"
- If query mentions "DJs", ONLY include records where Type/Job contains "DJ"
- If query mentions "festivals", ONLY include records where Type/Event contains "festival"
- If query mentions "July", ONLY include records where Month contains "July" or "07"
- If query mentions "LinkedIn", ONLY include records with actual LinkedIn data
- If query mentions "high Instagram followers", ONLY include records with high follower counts

EXCLUSION RULES:
- EXCLUDE records with empty required fields
- EXCLUDE records with "?" in required fields
- EXCLUDE records that don't match ALL conditions

Return ONLY the JSON array of records that match ALL conditions. If no matches, return empty array []."""

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