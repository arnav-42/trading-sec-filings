import os
import requests
from bs4 import BeautifulSoup
import yaml
import logging
from datetime import datetime
import re
import time
import json
from pathlib import Path
import feedparser
from typing import Set, Dict, Any
from queue import Queue
from threading import Thread, Lock

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('edgar_fetcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EDGARFetcher:
    def __init__(self, config_path='config.yaml'):
        """Initialize the EDGAR fetcher with configuration"""
        self.config = self._load_config(config_path)
        self.base_url = "https://www.sec.gov/Archives"
        self.headers = {
            'User-Agent': self.config.get('user_agent', 'EDGARFetcher 1.0')
        }
        self.check_interval = self.config.get('check_interval_seconds', 0.2)  # Default to 0.2 seconds
        self.processed_ids: Set[str] = set()
        self.request_times = []
        self.request_lock = Lock()
        self.processing_queue = Queue()
        self.num_worker_threads = 5
        self.last_progress_update = time.time()
        self.filings_processed = 0
        
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            raise

    def _create_directories(self):
        """Create necessary directories for data storage"""
        dirs = ['raw_data', 'processed_data']
        for dir_name in dirs:
            os.makedirs(dir_name, exist_ok=True)

    def _respect_rate_limit(self):
        """Implement rate limiting to respect SEC guidelines"""
        with self.request_lock:
            current_time = time.time()
            # Keep only the last 10 seconds of requests
            self.request_times = [t for t in self.request_times if current_time - t < 10]
            
            # If we've made 10 requests in the last 10 seconds, wait
            if len(self.request_times) >= 100:  # Allow for 10 requests per second
                sleep_time = 10 - (current_time - self.request_times[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            self.request_times.append(current_time)
            time.sleep(0.1)  # Minimum delay between requests

    def process_filing_worker(self):
        """Worker thread to process filings from the queue"""
        while True:
            try:
                entry, filing_type = self.processing_queue.get()
                if entry is None:  # Poison pill
                    break
                self._process_filing_from_feed(entry, filing_type)
                self.processing_queue.task_done()
            except Exception as e:
                logger.error(f"Error in worker thread: {e}")

    def monitor_real_time_filings(self):
        """Monitor SEC's RSS feed for real-time filing updates"""
        logger.info("Starting real-time monitoring of SEC filings")
        self._create_directories()
        start_time = time.time()

        # Start worker threads
        workers = []
        for _ in range(self.num_worker_threads):
            worker = Thread(target=self.process_filing_worker)
            worker.daemon = True
            worker.start()
            workers.append(worker)

        # SEC's RSS feed URLs for different filing types
        feeds = [
            ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=10-K&company=&dateb=&owner=include&start=0&count=100&output=atom", "10-K"),
            ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=10-Q&company=&dateb=&owner=include&start=0&count=100&output=atom", "10-Q"),
            ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&company=&dateb=&owner=include&start=0&count=100&output=atom", "8-K"),
            ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=6-K&company=&dateb=&owner=include&start=0&count=100&output=atom", "6-K")
        ]

        while True:
            try:
                current_time = time.time()
                
                # Print progress update every 60 seconds
                if current_time - self.last_progress_update >= 60:
                    runtime = current_time - start_time
                    hours = int(runtime // 3600)
                    minutes = int((runtime % 3600) // 60)
                    logger.info(f"Status Update - Runtime: {hours}h {minutes}m | "
                              f"Filings Processed: {self.filings_processed} | "
                              f"Queue Size: {self.processing_queue.qsize()} | "
                              f"Unique Filings: {len(self.processed_ids)}")
                    self.last_progress_update = current_time

                for feed_url, filing_type in feeds:
                    self._check_feed(feed_url, filing_type)
                
                logger.debug(f"Sleeping for {self.check_interval} seconds")
                time.sleep(self.check_interval)
            
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1)  # Wait 1 second before retrying on error

        # Clean up workers
        for _ in workers:
            self.processing_queue.put((None, None))
        for worker in workers:
            worker.join()

    def _check_feed(self, feed_url: str, filing_type: str):
        """Check SEC RSS feed for new filings"""
        try:
            self._respect_rate_limit()
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries:
                # Check if we've already processed this filing
                if entry.id in self.processed_ids:
                    continue
                
                # Add to processing queue
                self.processing_queue.put((entry, filing_type))
                self.processed_ids.add(entry.id)
                
                # Keep processed_ids from growing too large
                if len(self.processed_ids) > 1000:
                    self.processed_ids = set(list(self.processed_ids)[-1000:])
                
        except Exception as e:
            logger.error(f"Error checking feed: {e}")

    def _process_filing_from_feed(self, entry: Dict[str, Any], filing_type: str):
        """Process a filing from the RSS feed"""
        try:
            # Extract company info from the entry
            company_name = entry.get('title', '').split(' - ')[0].strip()
            cik_match = re.search(r'CIK=(\d+)', entry.link)
            if not cik_match:
                logger.error("Could not extract CIK from link")
                return
            cik = cik_match.group(1)
            
            # Extract accession number
            acc_match = re.search(r'/(\d{10}-\d{2}-\d{6})/', entry.link)
            if not acc_match:
                logger.error("Could not extract accession number from link")
                return
            
            accession_number = acc_match.group(1)
            accession_no_dashes = accession_number.replace('-', '')
            
            # Construct the URL to fetch documents
            base_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession_no_dashes}"
            
            # Try to get the document list from index.json
            self._respect_rate_limit()
            index_url = f"{base_url}/index.json"
            
            try:
                index_response = requests.get(index_url, headers=self.headers)
                
                if index_response.status_code == 200:
                    # Get the list of documents
                    documents_data = index_response.json()
                    
                    # Find the primary document
                    main_doc = None
                    for file in documents_data.get('directory', {}).get('item', []):
                        file_name = file.get('name', '')
                        if file_name.lower().endswith('.htm') and not file_name.startswith('R'):
                            # Look for the main filing document
                            if filing_type.lower() in file_name.lower() or 'index' not in file_name.lower():
                                main_doc = file_name
                                break
                    
                    # If we didn't find a specific match, use any .htm file
                    if not main_doc:
                        for item in documents_data.get('directory', {}).get('item', []):
                            name = item.get('name', '')
                            if name.endswith('.htm') and not name.startswith('R'):
                                main_doc = name
                                break
                    
                    if main_doc:
                        doc_url = f"{base_url}/{main_doc}"
                    else:
                        # Fallback to common naming patterns
                        doc_url = f"{base_url}/{filing_type.lower()}.htm"
                else:
                    # Fallback if we can't get the index.json
                    doc_url = f"{base_url}/{filing_type.lower()}.htm"
                    
            except Exception as e:
                logger.warning(f"Error fetching document list: {e}")
                # Fallback to common naming pattern
                doc_url = f"{base_url}/{filing_type.lower()}.htm"
            
            # Fetch the document
            self._respect_rate_limit()
            response = requests.get(doc_url, headers=self.headers)
            
            # Check if we got the XBRL viewer or an error
            if response.status_code != 200 or 'XBRL Viewer' in response.text:
                # Try alternative URLs
                alternative_urls = [
                    f"{base_url}/{filing_type.lower()}",
                    f"{base_url}/{accession_number}.txt",
                    f"{base_url}/primary-document.htm",
                    f"{base_url}/FilingSummary.xml"
                ]
                
                for alt_url in alternative_urls:
                    self._respect_rate_limit()
                    response = requests.get(alt_url, headers=self.headers)
                    if response.status_code == 200 and 'XBRL Viewer' not in response.text:
                        break
            
            # If we still couldn't get a valid document, log and return
            if response.status_code != 200 or 'XBRL Viewer' in response.text:
                logger.warning(f"Could not retrieve document for {company_name} {filing_type}")
                return
            
            raw_text = response.text
            
            # Generate a timestamp for the file names
            filing_date = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save raw data
            raw_filename = f"raw_data/{company_name}_{cik}_{filing_type}_{filing_date}.txt"
            with open(raw_filename, 'w', encoding='utf-8') as f:
                f.write(raw_text)
            
            # Clean and process the data
            processed_text = self._clean_text(raw_text)
            processed_filename = f"processed_data/{company_name}_{cik}_{filing_type}_{filing_date}.txt"
            with open(processed_filename, 'w', encoding='utf-8') as f:
                f.write(processed_text)
            
            logger.info(f"Successfully processed {filing_type} for {company_name}")
            self.filings_processed += 1
            
        except Exception as e:
            logger.error(f"Error processing filing: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _clean_text(self, text: str) -> str:
        """Clean and process the raw text"""
        # Use BeautifulSoup for better HTML handling
        soup = BeautifulSoup(text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text
        text = soup.get_text(separator=' ')
        
        # Remove multiple newlines
        text = re.sub(r'\n+', '\n', text)
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove HTML artifacts
        text = re.sub(r'&nbsp;|&lt;|&gt;|&amp;|&quot;|&apos;', ' ', text)
        # Remove non-ASCII characters
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        return text.strip()

if __name__ == "__main__":
    fetcher = EDGARFetcher()
    fetcher.monitor_real_time_filings()
