"""
SEC Filing Agentic Trader

This script reads sentiment analysis results from the sentiment_results directory
and uses an LLM to generate trading signals (BUY, SHORT, HOLD).

Usage:
    python agentictrader.py             # Process all new analysis files
    python agentictrader.py --watch     # Continuously watch for new analysis files
"""

import os
import json
import time
import logging
import argparse
import requests
import re
from typing import Dict, List, Set, Optional, Any
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agentictrader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AgenticTrader:
    def __init__(self, api_key=None, model="meta-llama/llama-3.3-70b-instruct:free"):
        """Initialize the agentic trader"""
        # Use API key from environment or passed parameter
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable or pass as argument.")
        
        self.model = model
        self.processed_files: Set[str] = set()
        self.data_dir = "sentiment_results"
        
        # Create trade signals directory
        self.signals_dir = "trade_signals"
        os.makedirs(self.signals_dir, exist_ok=True)
        
        # System prompt for trading agent - improved formatting guidance
        self.system_prompt = """You are a financial trading agent that makes decisions based on SEC filing analysis.

Your ONLY task is to output a valid JSON object with this exact structure:
{
    "decision": "BUY" or "SHORT" or "HOLD",
    "confidence": float between 0.0-1.0,
    "reasoning": "brief explanation"
}

RULES:
1. NEVER include ANY explanatory text before or after the JSON
2. NEVER include markdown formatting
3. NEVER include ```json``` or any other code block markers
4. ONLY output the raw JSON object

DECISION CRITERIA:
- BUY: Strong positive outlook, raised guidance, low risk, positive tone
- SHORT: Negative outlook, lowered guidance, high risk, negative tone
- HOLD: Mixed or uncertain signals

EXAMPLE VALID OUTPUT:
{"decision":"BUY","confidence":0.8,"reasoning":"Strong forward-looking statements and positive guidance."}"""
    
    def get_new_files(self) -> List[str]:
        """Get list of new analysis files that haven't been processed yet"""
        if not os.path.exists(self.data_dir):
            logger.warning(f"Directory {self.data_dir} does not exist.")
            return []
            
        all_files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) 
                    if f.endswith('.json')]
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
        
        # Try to find the most basic pattern with exact keys we need
        try:
            basic_pattern = r'\{\s*"decision"\s*:\s*"[^"]+"\s*,\s*"confidence"\s*:\s*[0-9.]+\s*,\s*"reasoning"\s*:\s*"[^"]+"\s*\}'
            match = re.search(basic_pattern, text)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
        except (json.JSONDecodeError, re.error):
            pass
        
        return None
    
    def generate_signal(self, analysis: Dict) -> Dict:
        """Generate a trading signal using the LLM"""
        try:
            # Format analysis to send to LLM
            input_json = {
                "sentiment": analysis.get("sentiment", "neutral"),
                "guidance_change": analysis.get("guidance_change", "none"),
                "forward_looking_sentiment": analysis.get("forward_looking_sentiment", 0.5),
                "risk_factor_level": analysis.get("risk_factor_level", 0.5),
                "mna_intent": analysis.get("mna_intent", "none"),
                "executive_tone": analysis.get("executive_tone", "neutral"),
                "uncertainty_level": analysis.get("uncertainty_level", 0.5)
            }
            
            # Create a more structured prompt with clear instructions
            input_text = f"""Based on the following SEC filing analysis, make a trading decision:

Company: {analysis.get('company', 'Unknown')}
Filing Type: {analysis.get('filing_type', 'Unknown')}

ANALYSIS DATA:
- Sentiment: {input_json["sentiment"]}
- Guidance Change: {input_json["guidance_change"]}
- Forward-Looking Score: {input_json["forward_looking_sentiment"]}
- Risk Level: {input_json["risk_factor_level"]}
- M&A Intent: {input_json["mna_intent"]}
- Executive Tone: {input_json["executive_tone"]}
- Uncertainty: {input_json["uncertainty_level"]}

Respond with ONLY a valid JSON object containing your trading decision.
DO NOT include any explanatory text or markdown formatting.
"""
            
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://trading-sec-filings.com",
                "X-Title": "SEC Filing Agentic Trader"
            }
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": input_text}
                ],
                "response_format": {"type": "json_object"} if "anthropic" in self.model or "gpt" in self.model else None,
                "temperature": 0.1,
                "max_tokens": 150
            }
            
            # Remove response_format if it's None
            if payload["response_format"] is None:
                del payload["response_format"]
                
            logger.debug(f"Sending request to OpenRouter API with model: {self.model}")
            response = requests.post(url, headers=headers, json=payload)
            
            # Check for API errors
            if response.status_code != 200:
                error_info = response.json() if response.headers.get('content-type') == 'application/json' else response.text
                logger.error(f"OpenRouter API error (status {response.status_code}): {error_info}")
                return {
                    "decision": "HOLD",
                    "confidence": 0.0,
                    "reasoning": "API error occurred"
                }
                
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            
            # Log the raw response for debugging
            logger.debug(f"Raw LLM response: {content}")
            
            # Try multiple methods to extract valid JSON
            trading_decision = self.extract_json_from_text(content)
            
            # If we still couldn't get JSON, fall back to heuristic extraction
            if not trading_decision:
                logger.warning(f"Could not parse response as JSON: {content}")
                content_lower = content.lower()
                
                # Extract decision using keyword matching
                if "buy" in content_lower and ("sell" not in content_lower and "short" not in content_lower):
                    decision = "BUY"
                elif "short" in content_lower or "sell" in content_lower:
                    decision = "SHORT"
                else:
                    decision = "HOLD"
                
                # Try to find confidence with regex
                confidence = 0.5
                confidence_match = re.search(r'confidence[^\d]*([0-9.]+)', content_lower)
                if confidence_match:
                    try:
                        confidence = float(confidence_match.group(1))
                        confidence = max(0.0, min(1.0, confidence))
                    except ValueError:
                        pass
                
                # Get reasoning from the content
                truncated_content = content[:100] + "..." if len(content) > 100 else content
                
                trading_decision = {
                    "decision": decision,
                    "confidence": confidence,
                    "reasoning": f"Decision extracted from non-JSON response: {truncated_content}"
                }
                
                logger.warning(f"Fallback extraction resulted in: {trading_decision}")
            
            # Validate the decision
            if "decision" not in trading_decision:
                trading_decision["decision"] = "HOLD"
                
            decision = trading_decision.get("decision", "").upper()
            if decision not in ["BUY", "SHORT", "HOLD"]:
                logger.warning(f"Invalid decision: {decision}. Defaulting to HOLD.")
                trading_decision["decision"] = "HOLD"
            else:
                trading_decision["decision"] = decision
            
            if "confidence" not in trading_decision:
                trading_decision["confidence"] = 0.5
            elif not isinstance(trading_decision["confidence"], (int, float)):
                # Try to convert to float if it's a string
                try:
                    trading_decision["confidence"] = float(trading_decision["confidence"])
                except:
                    trading_decision["confidence"] = 0.5
            
            # Ensure confidence is between 0 and 1
            trading_decision["confidence"] = max(0.0, min(1.0, float(trading_decision["confidence"])))
            
            if "reasoning" not in trading_decision:
                trading_decision["reasoning"] = "No explanation provided"
            
            return trading_decision
                
        except Exception as e:
            logger.error(f"Error generating trading signal: {e}")
            return {
                "decision": "HOLD",
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}"
            }
    
    def process_analysis(self, file_path: str):
        """Process a sentiment analysis file and generate a trading signal"""
        try:
            file_name = os.path.basename(file_path)
            
            # Load the analysis
            with open(file_path, 'r', encoding='utf-8') as f:
                analysis = json.load(f)
            
            # Extract company info
            company = analysis.get("company", "Unknown")
            filing_type = analysis.get("filing_type", "Unknown")
            
            # Generate trading signal using LLM
            trading_decision = self.generate_signal(analysis)
            signal = trading_decision.get("decision", "HOLD")
            confidence = trading_decision.get("confidence", 0.0)
            reasoning = trading_decision.get("reasoning", "No explanation provided")
            
            # Create signal data
            signal_data = {
                "company": company,
                "filing_type": filing_type,
                "signal": signal,
                "confidence": confidence,
                "reasoning": reasoning,
                "timestamp": time.time(),
                "analysis": analysis,
                "source": "agentic_trader",
                "model": self.model
            }
            
            # Output signal
            logger.info(f"SIGNAL: {signal} ({confidence:.2f}) for {company} ({filing_type})")
            print(f"\n{'='*60}")
            print(f"TRADING SIGNAL: {signal} {company}")
            print(f"{'='*60}")
            print(f"Confidence: {confidence:.2f}")
            print(f"Analysis: {filing_type} filing")
            print(f"Reasoning: {reasoning}")
            print(f"Model: {self.model}")
            print(f"{'='*60}\n")
            
            # Save signal to file
            signal_file = os.path.join(self.signals_dir, f"agentic_{file_name}")
            with open(signal_file, 'w', encoding='utf-8') as f:
                json.dump(signal_data, f, indent=2)
            
            # Mark as processed
            self.processed_files.add(file_path)
            
        except Exception as e:
            logger.error(f"Error processing analysis {file_path}: {e}")
    
    def process_new_files(self):
        """Process all new analysis files"""
        new_files = self.get_new_files()
        
        if not new_files:
            return 0
            
        logger.info(f"Found {len(new_files)} new analysis files to process")
        
        for file_path in new_files:
            self.process_analysis(file_path)
            
        return len(new_files)
    
    def watch(self, interval=5):
        """Continuously watch for new analysis files"""
        logger.info("Starting agentic trader monitoring...")
        
        try:
            while True:
                num_processed = self.process_new_files()
                if num_processed:
                    logger.info(f"Processed {num_processed} new files")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")

def main():
    parser = argparse.ArgumentParser(description="SEC Filing Agentic Trader")
    parser.add_argument("--api-key", help="OpenRouter API Key (or set OPENROUTER_API_KEY env var)")
    parser.add_argument("--model", default="meta-llama/llama-3.3-70b-instruct:free", 
                        help="OpenRouter model to use. Options: 'meta-llama/llama-3.3-70b-instruct:free', 'anthropic/claude-3-haiku:free', etc.")
    parser.add_argument("--watch", action="store_true", help="Continuously watch for new analysis files")
    parser.add_argument("--interval", type=int, default=5, help="Polling interval in seconds")
    
    args = parser.parse_args()
    
    try:
        trader = AgenticTrader(api_key=args.api_key, model=args.model)
        
        if args.watch:
            print(f"""
╔══════════════════════════════════════════════════════════════╗
║                SEC Filing Agentic Trader                     ║
╚══════════════════════════════════════════════════════════════╝

• Model: {args.model}
• Mode: Continuous monitoring (every {args.interval}s)
• Strategy: LLM-based trading decisions
• Watching: sentiment_results/*.json

Starting...
""")
            trader.watch(interval=args.interval)
        else:
            num_processed = trader.process_new_files()
            print(f"Processed {num_processed} files")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main()) 