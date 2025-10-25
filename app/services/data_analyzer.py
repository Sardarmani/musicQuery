import json
import logging
import re
import time
from typing import Dict, Any, List, Optional

import pandas as pd
from openai import OpenAI

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class DataStructureAnalyzer:
    def __init__(self, api_key: Optional[str] = None):
        if api_key:
            self.client = OpenAI(api_key=api_key)
            print(f"✅ OpenAI client initialized with API key")
        else:
            # Try to get API key from environment
            import os
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.client = OpenAI(api_key=api_key)
                print(f"✅ OpenAI client initialized from environment")
            else:
                self.client = None
                print(f"❌ No OpenAI API key found")

    def analyze_data_structure(self, df: pd.DataFrame, worksheet_name: str) -> Dict[str, Any]:
        """
        Analyzes the structure of a DataFrame, including column types and sample data.
        Includes the full DataFrame for direct GPT processing.
        """
        if df is None or df.empty:
            return {}

        sample_data = df.head(6).to_dict('records') # Get exactly 6 sample rows

        column_analysis = {}
        for col in df.columns:
            col_data = df[col].dropna()
            sample_values = col_data.sample(min(5, len(col_data))).tolist() if not col_data.empty else []
            column_analysis[col] = {
                'type': str(df[col].dtype),
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

    def get_gpt_query_analysis(self, user_query: str, data_structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        SMART APPROACH: First understand data structure, then apply filters.
        """
        print("🚀 ENTERING SMART GPT QUERY ANALYSIS...")
        print(f"📝 User query: {user_query}")
        print(f"🔑 Client available: {self.client is not None}")
        
        if not self.client:
            print("❌ No OpenAI client, using fallback")
            return self._fallback_analysis(user_query, data_structure)
        
        try:
            # Get the data from the worksheet
            df = data_structure.get('dataframe')
            if df is None or df.empty:
                return self._fallback_analysis(user_query, data_structure)
            
            print(f"📊 DATASET INFO: {len(df)} rows, {len(df.columns)} columns")
            print(f"📋 COLUMNS: {list(df.columns)}")
            
            # STEP 1: Send sample data to GPT to understand structure and get filtering strategy
            sample_data = df.head(20).to_dict('records')  # First 20 rows for structure understanding
            
            structure_prompt = f"""You are a data analysis expert. I will give you:
1. Sample data from a spreadsheet (first 20 rows)
2. A user query

TASK: Analyze the data structure and create a filtering strategy.

SAMPLE DATA:
{json.dumps(sample_data, indent=2)}

USER QUERY: "{user_query}"

ANALYZE:
1. What columns are available in this dataset?
2. What does the user query want to find?
3. What filters should be applied to get the correct results?
4. Are there any specific patterns in the data I should be aware of?

Return your analysis as JSON with this structure:
{{
  "columns_available": ["list", "of", "columns"],
  "query_intent": "what the user wants",
  "filtering_strategy": "how to filter the data",
  "key_fields": ["most", "important", "fields"],
  "data_patterns": "any patterns you notice"
}}"""

            print("=" * 80)
            print("STEP 1: ANALYZING DATA STRUCTURE...")
            print("=" * 80)
            print("🔍 SAMPLE DATA BEING SENT TO GPT:")
            print(f"📊 Sample size: {len(sample_data)} rows")
            print(f"📋 Sample data: {json.dumps(sample_data[:3], indent=2)}...")  # Show first 3 rows
            print("=" * 80)
            print("🤖 GPT STRUCTURE PROMPT:")
            print(structure_prompt)
            print("=" * 80)
            print("🚀 SENDING TO GPT-5 FOR STRUCTURE ANALYSIS...")
            
            structure_response = self.client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": "You are a data analysis expert. Analyze the data structure and provide filtering strategy."},
                    {"role": "user", "content": structure_prompt}
                ],
                max_completion_tokens=1000
            )
            
            structure_analysis = structure_response.choices[0].message.content
            print("✅ GPT STRUCTURE ANALYSIS RESPONSE:")
            print(structure_analysis)
            print("=" * 80)
            
            # STEP 2: Apply the filtering strategy to get the actual results
            # Get a larger sample for filtering (but still manageable)
            filter_sample_size = min(200, len(df))
            filter_data = df.head(filter_sample_size).to_dict('records')
            
            print(f"STEP 2: APPLYING FILTERS TO {filter_sample_size} ROWS...")
            print("=" * 80)
            print("🔍 FILTER DATA BEING SENT TO GPT:")
            print(f"📊 Filter data size: {len(filter_data)} rows")
            print(f"📋 Sample filter data: {json.dumps(filter_data[:3], indent=2)}...")  # Show first 3 rows
            print("=" * 80)
            
            # STEP 2: Apply smart filtering based on structure analysis
            filtering_prompt = f"""You are a data filtering expert. Based on the structure analysis, filter the data.

DATA TO FILTER:
{json.dumps(filter_data, indent=2)}

USER QUERY: "{user_query}"

STRUCTURE ANALYSIS FROM STEP 1:
{structure_analysis}

TASK: Apply the filtering strategy and return ONLY records that match the user query.

FILTERING INSTRUCTIONS:
1. Use the structure analysis to understand the data
2. Apply ALL conditions from the user query
3. Be very strict about matching criteria
4. Return ONLY records that match ALL conditions

IMPORTANT FILTERING RULES:
- "Irish" = Country field contains "Ireland" or "Irish"
- "French" = Country field contains "France" or "French"  
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

EXCLUSION RULES:
- EXCLUDE records with empty required fields
- EXCLUDE records with "?" in required fields
- EXCLUDE records that don't match ALL conditions

Return ONLY a JSON array of records that match ALL conditions. If no matches, return empty array []."""

            print("🤖 GPT FILTERING PROMPT:")
            print(filtering_prompt)
            print("=" * 80)
            print("🚀 SENDING TO GPT-4O FOR FILTERING...")
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a data filtering expert. Apply strict filtering based on user query and data structure analysis."},
                    {"role": "user", "content": filtering_prompt}
                ],
                max_completion_tokens=4000
            )
            
            result = response.choices[0].message.content
            print("✅ GPT FILTERING RESPONSE:")
            print(result)
            print("=" * 80)
            print("🔍 PARSING GPT RESPONSE...")
            parsed_result = self._parse_direct_gpt_response(result)
            print(f"📊 PARSED RESULT: {parsed_result}")
            print("=" * 80)
            return parsed_result

        except Exception as e:
            logger.error(f"GPT analysis failed: {e}")
            return self._fallback_analysis(user_query, data_structure)

    def _parse_direct_gpt_response(self, response: str) -> Dict[str, Any]:
        """Parse direct GPT response that returns filtered data."""
        print("🔍 PARSING GPT RESPONSE...")
        print(f"📝 Raw response: {response[:500]}...")  # Show first 500 chars
        
        try:
            # Extract JSON array from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                print(f"📋 Found JSON array: {json_str[:200]}...")  # Show first 200 chars
                filtered_data = json.loads(json_str)
                print(f"✅ Successfully parsed {len(filtered_data)} records")

                return {
                    'intent': 'Direct GPT filtering',
                    'search_strategy': 'direct_gpt',
                    'filtered_data': filtered_data,
                    'return_full_rows': True,
                    'confidence': 0.95
                }
            else:
                print("❌ No JSON array found in response")
                return self._fallback_analysis("", {})
        except Exception as e:
            print(f"❌ Error parsing JSON: {e}")
            return self._fallback_analysis("", {})

    def _fallback_analysis(self, user_query: str, data_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced fallback analysis when GPT is not available."""
        # Simplified fallback for now
        return {
            'intent': f'Fallback search for {user_query}',
            'search_strategy': 'enhanced',
            'filters': [],
            'return_full_rows': True,
            'confidence': 0.5
        }

    def apply_gpt_analysis(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply GPT analysis - now handles direct GPT filtering results.
        """
        print("🔍 APPLYING GPT ANALYSIS...")
        print(f"📊 Input DataFrame shape: {df.shape if df is not None else 'None'}")
        print(f"📋 Analysis strategy: {analysis.get('search_strategy', 'unknown')}")
        
        if df is None or df.empty:
            print("❌ Input DataFrame is empty or None")
            return df

        # Check if this is direct GPT filtering
        if analysis.get('search_strategy') == 'direct_gpt':
            filtered_data = analysis.get('filtered_data', [])
            print(f"📊 Filtered data from GPT: {len(filtered_data)} records")
            
            if filtered_data:
                # Convert filtered data back to DataFrame
                result_df = pd.DataFrame(filtered_data)
                print(f"✅ Successfully created DataFrame with {len(result_df)} rows")
                print(f"📋 Result columns: {list(result_df.columns)}")
                return result_df
            else:
                # Return empty DataFrame if no matches
                print("❌ No filtered data from GPT, returning empty DataFrame")
                return pd.DataFrame()

        # Fallback to old filtering logic for backward compatibility (should not be reached with direct_gpt)
        print("⚠️ Using fallback logic")
        return pd.DataFrame() # Return empty if not direct_gpt

# Global instance - try to get API key from environment
import os
api_key = os.getenv('OPENAI_API_KEY')
data_analyzer = DataStructureAnalyzer(api_key=api_key)
