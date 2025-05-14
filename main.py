"""
SEC Filing Analysis Pipeline

This script chains together all components of the SEC filing pipeline:
1. Fetching SEC filings (fetch.py)
2. Analyzing sentiment (sentiment.py)
3. Generating rule-based trading signals (algotrader.py)
4. Generating LLM-based trading signals (agentictrader.py)

Usage:
    python main.py                    # Run the full pipeline once
    python main.py --continuous       # Run continuously
    python main.py --component fetch  # Run only specific component(s)
"""

import os
import sys
import time
import argparse
import logging
import importlib.util
import subprocess
from typing import List, Dict, Any
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Component configurations
COMPONENTS = {
    "fetch": {
        "module": "fetch",
        "class": "EDGARFetcher",
        "method": "monitor_real_time_filings",
        "script": "fetch.py",
        "args": [],
        "description": "SEC filing fetcher - extracts documents from EDGAR"
    },
    "sentiment": {
        "module": "sentiment",
        "class": "SECSentimentAnalyzer",
        "method": "process_new_files",
        "script": "sentiment.py",
        "args": ["--once"],
        "description": "Sentiment analyzer - processes filings with LLM"
    },
    "algotrader": {
        "module": "algotrader",
        "class": "RuleBasedTrader",
        "method": "process_new_files",
        "script": "algotrader.py",
        "args": [],
        "description": "Rule-based trader - generates signals based on predefined rules"
    },
    "agentictrader": {
        "module": "agentictrader",
        "class": "AgenticTrader",
        "method": "process_new_files",
        "script": "agentictrader.py",
        "args": [],
        "description": "Agentic trader - generates signals using LLM"
    }
}

def run_component(component_name: str, as_module: bool = False) -> int:
    """Run a specific component either by importing module or subprocess"""
    if component_name not in COMPONENTS:
        logger.error(f"Unknown component: {component_name}")
        return 1
    
    component = COMPONENTS[component_name]
    logger.info(f"Running component: {component_name} - {component['description']}")
    
    try:
        if as_module:
            # Import and run as module
            module_path = Path(component["script"])
            module_name = module_path.stem
            
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                logger.error(f"Could not load module: {module_name}")
                return 1
                
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Get the class and call method
            cls = getattr(module, component["class"])
            instance = cls()
            method = getattr(instance, component["method"])
            result = method()
            
            return 0
        else:
            # Run as subprocess
            cmd = [sys.executable, component["script"]] + component["args"]
            logger.info(f"Running command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, check=True)
            return result.returncode
            
    except Exception as e:
        logger.error(f"Error running component {component_name}: {e}")
        return 1

def run_pipeline(components: List[str], interval: int = 0) -> int:
    """Run the specified components of the pipeline"""
    if not components:
        logger.error("No components specified to run")
        return 1
    
    # If not continuous, run each component once
    if interval == 0:
        for component in components:
            result = run_component(component)
            if result != 0:
                logger.warning(f"Component {component} failed with code {result}")
        return 0
    
    # Run continuously with interval
    try:
        logger.info(f"Starting continuous pipeline with {interval}s interval...")
        while True:
            for component in components:
                try:
                    run_component(component)
                except Exception as e:
                    logger.error(f"Error in continuous pipeline - component {component}: {e}")
            
            logger.info(f"Pipeline cycle completed. Waiting {interval} seconds...")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        logger.info("Pipeline stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        return 1

def display_pipeline_info(components: List[str] = None) -> None:
    """Display information about the pipeline components"""
    if components is None:
        components = COMPONENTS.keys()
        
    print("\n╔════════════════════════════════════════════════════════════════╗")
    print("║             SEC Filing Analysis Pipeline                       ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")
    
    print("Pipeline Components:")
    for i, component in enumerate(components, 1):
        info = COMPONENTS.get(component)
        if info:
            print(f"{i}. {component}: {info['description']}")
    print("\nData Flow: SEC EDGAR → raw_data/ → processed_data/ → sentiment_results/ → trade_signals/")
    print("\nEach component outputs to a different directory and logs information to its own log file.")

def create_test_files() -> None:
    """Create directories if they don't exist"""
    dirs = ['raw_data', 'processed_data', 'sentiment_results', 'trade_signals']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)

def main() -> int:
    parser = argparse.ArgumentParser(description="SEC Filing Analysis Pipeline")
    parser.add_argument("--component", nargs="*", choices=COMPONENTS.keys(),
                        help="Specific component(s) to run. Default is all.")
    parser.add_argument("--continuous", action="store_true", 
                        help="Run pipeline continuously")
    parser.add_argument("--interval", type=int, default=60,
                        help="Interval between pipeline runs in seconds (for continuous mode)")
    parser.add_argument("--info", action="store_true",
                        help="Display information about the pipeline")
    
    args = parser.parse_args()
    
    # Create necessary directories
    create_test_files()
    
    # Handle info display
    if args.info:
        display_pipeline_info(args.component)
        return 0
    
    # Determine components to run
    components = args.component if args.component else list(COMPONENTS.keys())
    
    # Display startup info
    display_pipeline_info(components)
    
    if args.continuous:
        print(f"\nRunning in continuous mode with {args.interval}s interval.")
        print("Press Ctrl+C to stop.\n")
    else:
        print("\nRunning pipeline once.\n")
    
    # Run the pipeline
    return run_pipeline(components, args.interval if args.continuous else 0)

if __name__ == "__main__":
    sys.exit(main()) 