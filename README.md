# SEC EDGAR Trading Signal Generator

This system (still a WIP) monitors the SEC EDGAR RSS feeds for new filings from all publicly traded companies, analyzes their sentiment, and generates trading signals. The system watches for new SEC filings (10-K, 10-Q, 8-K, 6-K), figures out what they mean, and generates buy, sell, or hold signals.

## Getting Started

### Requirements

- Python 3.8 or higher
- An OpenRouter API key (I use the free tier)

### Quick Setup

1. Clone this repository to your computer
2. Copy `config.yaml.sample` to `config.yaml` and add your contact email (SEC requires this)
3. Create a `.env` file with your OpenRouter API key:
   ```
   OPENROUTER_API_KEY=your_key_here
   ```
4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the full pipeline with a single command:
   ```bash
   python main.py
   ```

### OpenRouter Setup

You'll need an OpenRouter API key to use the sentiment analysis and agentic trading features:

1. Create a free account at [OpenRouter](https://openrouter.ai)
2. Navigate to the [API Keys section](https://openrouter.ai/keys)
3. Create a new API key
4. Add it to your `.env` file or set it as an environment variable

By default, the system uses Llama 3.3 70B (free tier) for analysis and trading decisions.

## Running Options

### The All-in-One Approach

Run the entire pipeline with a single command:

```bash
# Run everything once
python main.py

# Keep it running continuously
python main.py --continuous

# Run continuously with checks every 2 minutes
python main.py --continuous --interval 120
```

### Cherry-Pick Components

Only interested in specific parts? No problem:

```bash
# Just fetch new SEC filings
python main.py --component fetch

# Just analyze sentiment of existing filings
python main.py --component sentiment

# Get both rule-based and AI trading signals for analyzed filings
python main.py --component algotrader agentictrader
```

### Get Information About the Pipeline

```bash
# See what each component does
python main.py --info
```

## How It All Works

The system has four main components that work together:

1. **EDGAR Fetcher** (`fetch.py`): Monitors the SEC EDGAR RSS feeds for all recent 10-K, 10-Q, 8-K, and 6-K filings from any company
2. **Sentiment Analyzer** (`sentiment.py`): Analyzes filing text using LLMs to extract key metrics like sentiment, guidance changes, and risk levels
3. **Rule-Based Trader** (`algotrader.py`): Applies predefined rules to generate trading signals based on the sentiment analysis
4. **Agentic Trader** (`agentictrader.py`): Uses LLMs to make trading decisions by interpreting the sentiment analysis results

Data flows through the system like this:
- SEC EDGAR API → raw_data/ → processed_data/ → sentiment_results/ → trade_signals/

Each component logs its activities to its own log file, making troubleshooting easy.

## Advanced Options

Each component can be fine-tuned and run individually:

```bash
# Fetch SEC filings
python fetch.py

# Analyze sentiment with a different model
python sentiment.py --model anthropic/claude-3-sonnet:free

# Generate rule-based trading signals
python algotrader.py --watch

# Generate AI-based trading signals with a specific model
python agentictrader.py --model meta-llama/llama-3.3-70b-instruct:free --watch
```

## Staying Within Limits

This tool automatically respects SEC's rate limits:
- Keeps requests under 10 per second
- Maintains minimum 0.1s between requests
- Includes your contact information with requests

## Troubleshooting

If something doesn't seem right:

1. Check the relevant log file for specific errors (each component has its own log)
2. Make sure your internet connection is stable
3. Verify your OpenRouter API key is correct and has available credits
4. For SEC fetching issues, ensure your contact information is valid in `config.yaml`

## Data Privacy

The system only sends filing text to OpenRouter for analysis. No personal data is shared beyond what's necessary for API functionality. Your API key is kept locally in the `.env` file.
