from langchain_community.document_loaders import PyMuPDFLoader
import requests
from bs4 import BeautifulSoup
import fitz
import re
from typing import List, Tuple, Dict
import random

def extract_pdf_content(pdf_path):
    """Extract text content from PDF"""
    try:
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        return documents
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return []

def extract_hyperlinks(pdf_path, max_links=5):
    """Extract hyperlinks from PDF with a limit on the number of links returned.
    
    Args:
        pdf_path: Path to the PDF file
        max_links: Maximum number of external links to extract
        
    Returns:
        List of (page_num, url) tuples
    """
    links = []
    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            link_info = page.get_links()
            for link in link_info:
                if "uri" in link:
                    links.append((page_num, link["uri"]))
        
        # Prioritize links - can implement a more sophisticated selection here
        # For example, prioritize certain domains or keywords
        if len(links) > max_links:
            # Option 1: Take first max_links links
            # links = links[:max_links]
            
            # Option 2: Random sample (better for demo diversity)
            links = random.sample(links, max_links)
            
        print(f"Extracted {len(links)} hyperlinks from PDF (limited to {max_links})")
        return links
    except Exception as e:
        print(f"Error extracting hyperlinks: {e}")
        return []

def fetch_url_content(url):
    """Fetch content from a URL."""
    try:
        print(f"Fetching content from: {url}")
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            text = soup.get_text()
            # Clean text
            lines = (line.strip() for line in text.splitlines())
            text = "\n".join(line for line in lines if line)
            return text
        else:
            print(f"Failed to fetch {url}: {response.status_code}")
            return ""
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

pdf_path = "Curated Learning Resources.pdf"
hyperlinks = extract_hyperlinks(pdf_path)

