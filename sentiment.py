"""
SEC Filing Sentiment Analyzer

This script analyzes SEC filings for sentiment and classifies them as positive, negative, or neutral.
It monitors the processed_data directory for new files and performs sentiment analysis using OpenRouter API.

Usage:
    python sentiment.py              # Monitor mode (checks every 10 seconds)
    python sentiment.py --once       # Analyze existing files once
    python sentiment.py --interval 30  # Change polling interval to 30 seconds

Setup:
    1. Set your OpenRouter API key as an environment variable:
       - Create a .env file with: OPENROUTER_API_KEY=your_key_here
       - Or set it in your terminal: 
         - Windows: set OPENROUTER_API_KEY=your_key_here
         - Linux/Mac: export OPENROUTER_API_KEY=your_key_here
    2. Run the script: python sentiment.py

    You can also pass the API key as an argument: python sentiment.py --api-key your_key_here
"""

import os
import time
import logging
import json
from typing import Dict, List, Set, Optional, Any
import argparse
from dotenv import load_dotenv
import requests
import re

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sentiment_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SECSentimentAnalyzer:
    def __init__(self, api_key=None, model="meta-llama/llama-3.3-70b-instruct:free"):
        """Initialize the sentiment analyzer"""
        # Use API key from environment or passed parameter
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable or pass as argument.")
        
        self.model = model
        self.processed_files: Set[str] = set()
        self.data_dir = "processed_data"
        
        # Create the sentiment results directory if it doesn't exist
        os.makedirs("sentiment_results", exist_ok=True)
        
        # System prompt for consistent sentiment analysis
        self.system_prompt = """
        You are a financial sentiment analyzer. Your task is to analyze SEC filings and 
        determine if the sentiment is positive, negative, or neutral.
        
        Focus on key financial indicators, management tone, risk factors, 
        and forward-looking statements.
        
        Your response must be EXACTLY ONE WORD, either "positive", "negative", or "neutral".
        Do not include any explanations, details, or additional text.
        """

    def get_new_files(self) -> List[str]:
        """Get list of new processed files that haven't been analyzed yet"""
        if not os.path.exists(self.data_dir):
            logger.warning(f"Directory {self.data_dir} does not exist.")
            return []
            
        all_files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) 
                    if f.endswith('.txt')]
        new_files = [f for f in all_files if f not in self.processed_files]
        return new_files

    def extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract a JSON object from text that might contain other content"""
        # First try direct parsing
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Next try to find JSON object with regex
        try:
            json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
            match = re.search(json_pattern, text)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
        except (json.JSONDecodeError, re.error):
            pass
        
        # Try to find json with basic pattern specific to our expected format
        try:
            basic_pattern = r'\{\s*"sentiment"\s*:\s*"[^"]+"\s*,.*?\}'
            match = re.search(basic_pattern, text)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
        except (json.JSONDecodeError, re.error):
            pass
        
        return None

    def analyze_sentiment(self, text: str) -> dict:
        """Analyze the sentiment of a document using OpenRouter API"""
        try:
            # Truncate text if too long to avoid token limits
            max_text_length = 20000  # Conservative limit for context window
            if len(text) > max_text_length:
                logger.info(f"Truncating text from {len(text)} to {max_text_length} characters")
                text = text[:max_text_length]
            
            system_prompt = """
            You are a senior equity analyst specializing in SEC filings analysis.
            Given an SEC filing text, analyze it and return ONLY a JSON object with the following keys and specified formats:
            
            {
                "sentiment": "positive" or "negative" or "neutral",
                "guidance_change": "raise" or "lower" or "maintain" or "none",
                "forward_looking_sentiment": float between 0.0-1.0,
                "risk_factor_level": float between 0.0-1.0,
                "mna_intent": "acquisition" or "divestiture" or "merger" or "spin-off" or "none",
                "executive_tone": "positive" or "neutral" or "negative",
                "uncertainty_level": float between 0.0-1.0
            }
            
            Return ONLY the valid JSON object without any additional text, explanation, or markdown formatting.
            """
            
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://trading-sec-filings.com",  # Replace with your site URL or localhost
                "X-Title": "SEC Filing Sentiment Analyzer"
            }
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze the following SEC filing for sentiment and trading signals:\n\n{text}"}
                ],
                "response_format": {"type": "json_object"} if "anthropic" in self.model or "gpt" in self.model else None,
                "temperature": 0.1,
                "max_tokens": 500  # Increased for the JSON response
            }
            
            # Remove response_format if it's None
            if payload.get("response_format") is None:
                del payload["response_format"]
            
            logger.debug(f"Sending request to OpenRouter API with model: {self.model}")
            response = requests.post(url, headers=headers, json=payload)
            
            # Check for API errors
            if response.status_code != 200:
                error_info = response.json() if response.headers.get('content-type') == 'application/json' else response.text
                logger.error(f"OpenRouter API error (status {response.status_code}): {error_info}")
                if response.status_code == 401:
                    logger.error("Authentication error. Please check your API key.")
                elif response.status_code == 404:
                    logger.error(f"Model '{self.model}' not found. Try another model like 'meta-llama/llama-3.3-70b-instruct:free'")
                return {"sentiment": "neutral", "error": str(error_info)}
                
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            
            # Log the raw response for debugging
            logger.debug(f"Raw LLM response: {content}")
            
            # Try multiple methods to extract valid JSON
            analysis = self.extract_json_from_text(content)
            
            if not analysis:
                logger.error(f"Failed to parse JSON from response: {content}")
                # Try fallback extraction of key fields if JSON parsing failed
                fallback_analysis = self._fallback_extract_fields(content)
                if fallback_analysis:
                    logger.info("Successfully extracted fields via fallback method")
                    analysis = fallback_analysis
                else:
                    return {"sentiment": "neutral", "error": "JSON parsing failed"}
            
            # Validate required fields
            required_fields = ["sentiment", "guidance_change", "forward_looking_sentiment", 
                            "risk_factor_level", "mna_intent", "executive_tone", "uncertainty_level"]
            
            for field in required_fields:
                if field not in analysis:
                    logger.warning(f"Missing required field '{field}' in analysis. Adding default value.")
                    if field in ["sentiment", "executive_tone"]:
                        analysis[field] = "neutral"
                    elif field in ["guidance_change", "mna_intent"]:
                        analysis[field] = "none"
                    elif field in ["forward_looking_sentiment", "risk_factor_level", "uncertainty_level"]:
                        analysis[field] = 0.5
            
            # Basic validation for float fields
            float_fields = ["forward_looking_sentiment", "risk_factor_level", "uncertainty_level"]
            for field in float_fields:
                if not isinstance(analysis.get(field), (int, float)):
                    logger.warning(f"Field '{field}' is not a number. Setting to default value 0.5.")
                    analysis[field] = 0.5
                
                # Ensure values are between 0 and 1
                analysis[field] = max(0.0, min(1.0, float(analysis[field])))
            
            logger.info(f"Analysis completed: {json.dumps(analysis, indent=2)}")
            return analysis
                
        except requests.exceptions.ConnectionError:
            logger.error("Connection error when calling OpenRouter API. Please check your internet connection.")
            return {"sentiment": "neutral", "error": "Connection error"}
        except requests.exceptions.Timeout:
            logger.error("Request to OpenRouter API timed out.")
            return {"sentiment": "neutral", "error": "Request timeout"}
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return {"sentiment": "neutral", "error": str(e)}
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {"sentiment": "neutral", "error": str(e)}

    def _fallback_extract_fields(self, text: str) -> Optional[Dict[str, Any]]:
        """Attempt to extract fields when JSON parsing fails"""
        try:
            # Default values
            analysis = {
                "sentiment": "neutral", 
                "guidance_change": "none",
                "forward_looking_sentiment": 0.5,
                "risk_factor_level": 0.5,
                "mna_intent": "none",
                "executive_tone": "neutral",
                "uncertainty_level": 0.5
            }
            
            # Extract sentiment (positive, negative, neutral)
            sentiment_pattern = r'sentiment"?\s*:?\s*"?(positive|negative|neutral)'
            sentiment_match = re.search(sentiment_pattern, text.lower())
            if sentiment_match:
                analysis["sentiment"] = sentiment_match.group(1)
                
            # Extract guidance change
            guidance_pattern = r'guidance_change"?\s*:?\s*"?(raise|lower|maintain|none)'
            guidance_match = re.search(guidance_pattern, text.lower())
            if guidance_match:
                analysis["guidance_change"] = guidance_match.group(1)
                
            # Extract numeric values
            numeric_fields = {
                "forward_looking_sentiment": r'forward.?looking.?sentiment"?\s*:?\s*([0-9.]+)',
                "risk_factor_level": r'risk.?factor.?level"?\s*:?\s*([0-9.]+)',
                "uncertainty_level": r'uncertainty.?level"?\s*:?\s*([0-9.]+)'
            }
            
            for field, pattern in numeric_fields.items():
                match = re.search(pattern, text.lower())
                if match:
                    try:
                        value = float(match.group(1))
                        analysis[field] = max(0.0, min(1.0, value))
                    except ValueError:
                        pass
                        
            # Extract text values
            text_fields = {
                "executive_tone": r'executive.?tone"?\s*:?\s*"?(positive|negative|neutral)',
                "mna_intent": r'mna.?intent"?\s*:?\s*"?(acquisition|divestiture|merger|spin-off|none)'
            }
            
            for field, pattern in text_fields.items():
                match = re.search(pattern, text.lower())
                if match:
                    analysis[field] = match.group(1)
                    
            return analysis
            
        except Exception as e:
            logger.error(f"Error in fallback extraction: {e}")
            return None

    def process_new_files(self):
        """Process any new files found in the processed_data directory"""
        new_files = self.get_new_files()
        
        if not new_files:
            return
            
        logger.info(f"Found {len(new_files)} new files to analyze")
        
        for file_path in new_files:
            try:
                file_name = os.path.basename(file_path)
                logger.info(f"Analyzing {file_name}")
                
                # Extract metadata from filename
                # Format: {company_name}_{cik}_{filing_type}_{date}.txt
                parts = file_name.rsplit('_', 3)
                if len(parts) >= 4:
                    company_name = parts[0]
                    filing_type = parts[-2]
                else:
                    company_name = "Unknown"
                    filing_type = "Unknown"
                
                # Read the file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Analyze sentiment
                analysis = self.analyze_sentiment(content)
                
                # Add metadata
                analysis["company"] = company_name
                analysis["filing_type"] = filing_type
                analysis["file_name"] = file_name
                analysis["timestamp"] = time.time()
                
                # Output result
                simple_sentiment = analysis.get("sentiment", "neutral")
                result = f"{company_name} ({filing_type}): {simple_sentiment}"
                print(result)
                
                # Save result to file
                result_file = os.path.join("sentiment_results", f"{file_name}.json")
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(analysis, f, indent=2)
                
                # Mark as processed
                self.processed_files.add(file_path)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

    def monitor(self, interval=10):
        """Continuously monitor for new files at a specified interval"""
        logger.info("Starting sentiment analysis monitoring...")
        
        try:
            while True:
                self.process_new_files()
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")

def main():
    parser = argparse.ArgumentParser(description="SEC Filing Sentiment Analyzer")
    parser.add_argument("--api-key", help="OpenRouter API Key (or set OPENROUTER_API_KEY env var)")
    parser.add_argument("--model", default="meta-llama/llama-3.3-70b-instruct:free", help="OpenRouter model to use")
    parser.add_argument("--interval", type=int, default=10, help="Polling interval in seconds")
    parser.add_argument("--once", action="store_true", help="Run once instead of monitoring")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    try:
        # Check if API key is available
        api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            logger.error("""
OpenRouter API key not found! Please set it using one of these methods:
1. Create a .env file with: OPENROUTER_API_KEY=your_key_here
2. Set an environment variable:
   - PowerShell: $env:OPENROUTER_API_KEY = "your_key_here"
   - Command Prompt: set OPENROUTER_API_KEY=your_key_here
   - Linux/Mac: export OPENROUTER_API_KEY=your_key_here
3. Pass it as a parameter: python sentiment.py --api-key your_key_here

Get an API key at: https://openrouter.ai/keys
            """)
            return 1
            
        # Print startup information
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║               SEC Filing Sentiment Analyzer                  ║
╚══════════════════════════════════════════════════════════════╝

• Model: {args.model}
• Mode: {'One-time analysis' if args.once else f'Continuous monitoring (every {args.interval}s)'}
• API: OpenRouter

Starting...
""")
        
        analyzer = SECSentimentAnalyzer(api_key=args.api_key, model=args.model)
        
        if args.once:
            analyzer.process_new_files()
            print("\nOne-time analysis complete!")
        else:
            analyzer.monitor(interval=args.interval)
            
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main()) 