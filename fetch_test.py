import os
import requests
from bs4 import BeautifulSoup
import logging
from datetime import datetime
import re
import time
from typing import Optional, Tuple
import json

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('edgar_fetcher_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EDGARTester:
    def __init__(self, user_agent: str):
        """Initialize the EDGAR tester"""
        self.headers = {
            'User-Agent': user_agent
        }
        self._create_directories()

    def _create_directories(self):
        """Create necessary directories for data storage"""
        dirs = ['raw_data', 'processed_data']
        for dir_name in dirs:
            os.makedirs(dir_name, exist_ok=True)

    def _respect_rate_limit(self):
        """Implement rate limiting to respect SEC guidelines"""
        time.sleep(0.1)  # Ensure at least 0.1s between requests

    def fetch_latest_filing(self, company_cik: str, filing_type: str = "10-K") -> Optional[Tuple[str, str]]:
        """
        Fetch the latest filing of specified type for a company using SEC's API
        
        Args:
            company_cik: The CIK number of the company (can be with or without leading zeros)
            filing_type: The type of filing to fetch (10-K, 10-Q, 8-K, etc.)
            
        Returns:
            Tuple of (raw_filename, processed_filename) if successful, None if failed
        """
        try:
            # Ensure CIK is properly formatted (remove leading zeros)
            cik = company_cik.lstrip('0')
            logger.info(f"Fetching latest {filing_type} for CIK: {cik}")
            
            # Step 1: Use SEC's REST API to get company submissions
            submissions_url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
            self._respect_rate_limit()
            logger.debug(f"Fetching company submissions from: {submissions_url}")
            
            response = requests.get(submissions_url, headers=self.headers)
            if response.status_code != 200:
                logger.error(f"Failed to get company submissions: {response.status_code}")
                return None
                
            submissions_data = response.json()
            
            # Get company name
            company_name = submissions_data.get('name', f"Company_{cik}")
            
            # Step 2: Find the latest filing of the specified type
            filings = None
            accession_number = None
            filing_date = None
            
            # Look in recent filings first
            recent_filings = submissions_data.get('filings', {}).get('recent', {})
            if recent_filings:
                for i, form in enumerate(recent_filings.get('form', [])):
                    if form == filing_type:
                        accession_number = recent_filings.get('accessionNumber', [])[i]
                        filing_date = recent_filings.get('filingDate', [])[i]
                        break
            
            # If not found in recent, try to check filing history
            if not accession_number and 'filings' in submissions_data:
                # We might need to make additional API calls for historical filings
                # This is a simplified approach for the test
                logger.debug("Filing not found in recent filings, would need to check historical filings")
            
            # If we still didn't find the filing, return
            if not accession_number:
                logger.error(f"No {filing_type} filing found for {company_name}")
                return None
                
            logger.debug(f"Found filing: Accession {accession_number}, Date {filing_date}")
            
            # Step 3: Construct the URL to the filing documents
            # Format accession number for URL (remove dashes)
            accession_no_dashes = accession_number.replace('-', '')
            
            # Get the list of documents in this filing
            documents_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/index.json"
            self._respect_rate_limit()
            logger.debug(f"Fetching documents list from: {documents_url}")
            
            response = requests.get(documents_url, headers=self.headers)
            if response.status_code != 200:
                logger.error(f"Failed to get documents list: {response.status_code}")
                # Try an alternative approach without the JSON endpoint
                # Sometimes the files are accessible without going through the index
                doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/{filing_type.lower()}.htm"
            else:
                # Process the JSON to find the main document
                documents_data = response.json()
                
                # Find the primary document
                main_doc = None
                for file in documents_data.get('directory', {}).get('item', []):
                    file_name = file.get('name', '')
                    if file_name.lower().endswith('.htm') and not file_name.startswith('R'):
                        # Look for the main filing document
                        if filing_type.lower() in file_name.lower() or 'index' not in file_name.lower():
                            main_doc = file_name
                            break
                
                if not main_doc:
                    # Try common naming patterns
                    for item in documents_data.get('directory', {}).get('item', []):
                        name = item.get('name', '')
                        if name.endswith('.htm') and not name.startswith('R'):
                            main_doc = name
                            break
                
                if not main_doc:
                    logger.error("Could not find the main document in the filing")
                    return None
                    
                doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/{main_doc}"
            
            # Step 4: Fetch the actual document
            self._respect_rate_limit()
            logger.debug(f"Fetching document from: {doc_url}")
            
            response = requests.get(doc_url, headers=self.headers)
            if response.status_code != 200:
                logger.error(f"Failed to fetch document: {response.status_code}")
                return None
                
            # Check if we got the XBRL viewer instead of the actual document
            if 'XBRL Viewer' in response.text:
                logger.warning("Received XBRL Viewer instead of actual document, trying alternative approaches")
                
                # Try alternative URLs
                alternative_found = False
                alternative_urls = [
                    f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/{filing_type.lower()}",
                    f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/{accession_number}.txt",
                    f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/primary-document.htm"
                ]
                
                for alt_url in alternative_urls:
                    self._respect_rate_limit()
                    logger.debug(f"Trying alternative URL: {alt_url}")
                    response = requests.get(alt_url, headers=self.headers)
                    if response.status_code == 200 and 'XBRL Viewer' not in response.text:
                        logger.debug(f"Success with alternative URL: {alt_url}")
                        alternative_found = True
                        break
                
                if not alternative_found:
                    logger.error("All alternative URLs failed to get the actual document")
                    return None
            
            raw_text = response.text
            
            # Format the date
            if filing_date:
                formatted_date = datetime.strptime(filing_date, '%Y-%m-%d').strftime('%Y%m%d')
            else:
                formatted_date = datetime.now().strftime('%Y%m%d')
            
            # Save raw data
            raw_filename = f"raw_data/{company_name}_{cik}_{filing_type}_{formatted_date}.txt"
            with open(raw_filename, 'w', encoding='utf-8') as f:
                f.write(raw_text)
            
            # Clean and process the data
            processed_text = self._clean_text(raw_text)
            processed_filename = f"processed_data/{company_name}_{cik}_{filing_type}_{formatted_date}.txt"
            with open(processed_filename, 'w', encoding='utf-8') as f:
                f.write(processed_text)
            
            logger.info(f"Successfully fetched {filing_type} for {company_name}")
            return raw_filename, processed_filename
            
        except Exception as e:
            logger.error(f"Error fetching filing: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _clean_text(self, text: str) -> str:
        """Clean and process the raw text"""
        # Convert HTML to text more effectively
        soup = BeautifulSoup(text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text and clean it
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

def main():
    # Example usage
    user_agent = "akalekar@purdue.edu"  # Replace with your contact info
    tester = EDGARTester(user_agent)
    
    # Example: Fetch Apple's latest 10-K
    result = tester.fetch_latest_filing("0000320193", "10-K")
    if result:
        raw_file, processed_file = result
        logger.info(f"Files saved:\nRaw: {raw_file}\nProcessed: {processed_file}")

if __name__ == "__main__":
    main() 