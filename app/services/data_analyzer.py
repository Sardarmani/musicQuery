import json
import logging
import pandas as pd
from typing import Dict, Any
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
        if df is None or df.empty:
            return {}
        return {
            'worksheet_name': worksheet_name,
            'columns': list(df.columns),
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'dataframe': df
        }

    def get_gpt_query_analysis(self, user_query: str, data_structure: Dict[str, Any]) -> Dict[str, Any]:
        if not self.client:
            return self._fallback_analysis(user_query, data_structure)

        try:
            df = data_structure.get('dataframe')
            if df is None or df.empty:
                return self._fallback_analysis(user_query, data_structure)

            # Convert full data to JSON
            all_data = df.to_dict('records')

            # ✅ NEW: Add column context
            column_names = data_structure.get('columns', [])
            column_info = ", ".join(column_names)

            # ✅ IMPROVED SYSTEM PROMPT
            system_prompt = f"""
You are a data filtering expert. I will provide:
1. A list of records from a spreadsheet (as JSON)
2. The column headers: {column_info}
3. A user query

Your job: Return ONLY records that match ALL conditions in the user query.

RULES:
- Match nationalities smartly:
  "Irish" → Ireland, Dublin, Cork
  "Italian" → Italy, Milan, Turin
  "French" → France, Paris, Lyon
  "Portuguese" → Portugal, Lisbon, Porto
  "German" → Germany, Berlin, Hamburg
  "Spanish" → Spain, Madrid, Barcelona
- "DJ" or "Event" or "Artist" should match related role or profession fields.
- If query mentions "LinkedIn", only include rows where LinkedIn is not empty.
- If query mentions "direct email", exclude generic ones like info@, contact@.
- Do not generate new data — only filter existing records.
- Return ONLY a JSON array of matching records. If none match, return [].
"""

            # ✅ USER PROMPT with structured context
            user_prompt = f"""
DATA (JSON):
{json.dumps(all_data, indent=2)}

QUERY: "{user_query}"

Return ONLY a JSON array of records that match ALL conditions in the query.
"""

            print("=" * 80)
            print("SYSTEM PROMPT:")
            print(system_prompt)
            print("=" * 80)
            print("USER PROMPT:")
            print(user_prompt)
            print("=" * 80)

            response = self.client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=6000
            )

            result = response.choices[0].message.content
            return self._parse_direct_gpt_response(result)

        except Exception as e:
            logger.error(f"GPT analysis failed: {e}")
            print(f"ERROR: {e}")
            return self._fallback_analysis(user_query, data_structure)

    def _parse_direct_gpt_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON array from GPT response"""
        import re
        try:
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                filtered_data = json.loads(json_str)
                return {
                    'intent': 'Direct GPT filtering',
                    'search_strategy': 'direct_gpt',
                    'filtered_data': filtered_data,
                    'return_full_rows': True,
                    'confidence': 0.97
                }
            else:
                return self._fallback_analysis("", {})
        except Exception as e:
            logger.error(f"Error parsing GPT response: {e}")
            return self._fallback_analysis("", {})

    def _fallback_analysis(self, user_query: str, data_structure: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'intent': f'Fallback search for {user_query}',
            'search_strategy': 'fallback',
            'filters': [],
            'return_full_rows': True,
            'confidence': 0.5
        }

    def apply_gpt_analysis(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        if analysis.get('search_strategy') == 'direct_gpt':
            filtered_data = analysis.get('filtered_data', [])
            if filtered_data:
                return pd.DataFrame(filtered_data)
            return pd.DataFrame()
        return df


# ✅ Global instance
data_analyzer = DataStructureAnalyzer()
