"""
SEC Filing Rule-Based Algorithmic Trader

This script reads sentiment analysis results from the sentiment_results directory
and generates trading signals (BUY, SHORT, HOLD) based on predefined rules.

Usage:
    python algotrader.py             # Process all new analysis files
    python algotrader.py --watch     # Continuously watch for new analysis files
"""

import os
import json
import time
import logging
import argparse
from typing import Dict, List, Set
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('algotrader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RuleBasedTrader:
    def __init__(self):
        """Initialize the rule-based algorithmic trader"""
        self.processed_files: Set[str] = set()
        self.data_dir = "sentiment_results"
        
        # Create trade signals directory
        self.signals_dir = "trade_signals"
        os.makedirs(self.signals_dir, exist_ok=True)
        
    def get_new_files(self) -> List[str]:
        """Get list of new analysis files that haven't been processed yet"""
        if not os.path.exists(self.data_dir):
            logger.warning(f"Directory {self.data_dir} does not exist.")
            return []
            
        all_files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) 
                    if f.endswith('.json')]
        new_files = [f for f in all_files if f not in self.processed_files]
        return new_files
    
    def generate_signal(self, analysis: Dict) -> str:
        """
        Generate a trading signal based on predefined rules
        
        Rules:
        - BUY if:
          - guidance_change is "raise" OR
          - forward_looking_sentiment > 0.7 AND risk_factor_level < 0.3 AND executive_tone is "positive"
        
        - SHORT if:
          - guidance_change is "lower" OR
          - risk_factor_level > 0.7 AND forward_looking_sentiment < 0.3 AND executive_tone is "negative"
        
        - HOLD otherwise
        """
        # Extract metrics
        guidance = analysis.get("guidance_change", "none")
        forward_sentiment = analysis.get("forward_looking_sentiment", 0.5)
        risk_level = analysis.get("risk_factor_level", 0.5)
        exec_tone = analysis.get("executive_tone", "neutral")
        uncertainty = analysis.get("uncertainty_level", 0.5)
        
        # Rule 1: Strong BUY signal
        if (guidance == "raise" or 
            (forward_sentiment > 0.7 and risk_level < 0.3 and exec_tone == "positive")):
            return "BUY"
            
        # Rule 2: Strong SHORT signal
        elif (guidance == "lower" or 
              (risk_level > 0.7 and forward_sentiment < 0.3 and exec_tone == "negative")):
            return "SHORT"
            
        # Rule 3: HOLD (neutral or mixed signals)
        else:
            return "HOLD"
    
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
            
            # Generate trading signal
            signal = self.generate_signal(analysis)
            
            # Create signal data
            signal_data = {
                "company": company,
                "filing_type": filing_type,
                "signal": signal,
                "timestamp": time.time(),
                "analysis": analysis,
                "metrics_used": {
                    "guidance_change": analysis.get("guidance_change"),
                    "forward_looking_sentiment": analysis.get("forward_looking_sentiment"),
                    "risk_factor_level": analysis.get("risk_factor_level"),
                    "executive_tone": analysis.get("executive_tone"),
                    "uncertainty_level": analysis.get("uncertainty_level")
                },
                "source": "rule_based_algo"
            }
            
            # Generate a readable explanation
            explanation = self._generate_explanation(signal, analysis)
            signal_data["explanation"] = explanation
            
            # Output signal
            logger.info(f"SIGNAL: {signal} for {company} ({filing_type})")
            print(f"\n{'='*60}")
            print(f"TRADING SIGNAL: {signal} {company}")
            print(f"{'='*60}")
            print(f"Analysis: {filing_type} filing")
            print(f"Explanation: {explanation}")
            print(f"Guidance: {analysis.get('guidance_change', 'none')}")
            print(f"Forward-looking Sentiment: {analysis.get('forward_looking_sentiment', 0.5):.2f}")
            print(f"Risk Level: {analysis.get('risk_factor_level', 0.5):.2f}")
            print(f"Executive Tone: {analysis.get('executive_tone', 'neutral')}")
            print(f"Uncertainty: {analysis.get('uncertainty_level', 0.5):.2f}")
            print(f"{'='*60}\n")
            
            # Save signal to file
            signal_file = os.path.join(self.signals_dir, f"algo_{file_name}")
            with open(signal_file, 'w', encoding='utf-8') as f:
                json.dump(signal_data, f, indent=2)
            
            # Mark as processed
            self.processed_files.add(file_path)
            
        except Exception as e:
            logger.error(f"Error processing analysis {file_path}: {e}")
    
    def _generate_explanation(self, signal: str, analysis: Dict) -> str:
        """Generate a human-readable explanation for the trading signal"""
        guidance = analysis.get("guidance_change", "none")
        forward_sentiment = analysis.get("forward_looking_sentiment", 0.5)
        risk_level = analysis.get("risk_factor_level", 0.5)
        exec_tone = analysis.get("executive_tone", "neutral")
        
        if signal == "BUY":
            if guidance == "raise":
                return "Company has raised guidance, indicating positive outlook."
            else:
                return f"Strong forward-looking sentiment ({forward_sentiment:.2f}) combined with low risk ({risk_level:.2f}) and positive executive tone."
                
        elif signal == "SHORT":
            if guidance == "lower":
                return "Company has lowered guidance, indicating negative outlook."
            else:
                return f"High risk factors ({risk_level:.2f}) combined with weak forward-looking sentiment ({forward_sentiment:.2f}) and negative executive tone."
                
        else:  # HOLD
            return "Mixed signals or neutral outlook."
    
    def process_new_files(self):
        """Process all new analysis files"""
        new_files = self.get_new_files()
        
        if not new_files:
            return
            
        logger.info(f"Found {len(new_files)} new analysis files to process")
        
        for file_path in new_files:
            self.process_analysis(file_path)
    
    def watch(self, interval=5):
        """Continuously watch for new analysis files"""
        logger.info("Starting algorithmic trader monitoring...")
        
        try:
            while True:
                self.process_new_files()
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")

def main():
    parser = argparse.ArgumentParser(description="SEC Filing Algorithmic Trader")
    parser.add_argument("--watch", action="store_true", help="Continuously watch for new analysis files")
    parser.add_argument("--interval", type=int, default=5, help="Polling interval in seconds")
    
    args = parser.parse_args()
    
    try:
        trader = RuleBasedTrader()
        
        if args.watch:
            print(f"""
╔══════════════════════════════════════════════════════════════╗
║             SEC Filing Rule-Based Algo Trader                ║
╚══════════════════════════════════════════════════════════════╝

• Mode: Continuous monitoring (every {args.interval}s)
• Strategy: Rule-based trading signals
• Watching: sentiment_results/*.json

Starting...
""")
            trader.watch(interval=args.interval)
        else:
            trader.process_new_files()
            
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main()) 