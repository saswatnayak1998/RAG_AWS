import time
import csv
import random
import faiss
import numpy as np
import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from sentence_transformers import SentenceTransformer

def get_driver(headless=True):
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--start-maximized')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

def scrape_aws_page(url, index, model, metadata, visited_links, driver, max_depth=2, current_depth=0):
    try:
        if url in visited_links or current_depth > max_depth:
            return  

        driver.get(url)
        time.sleep(random.uniform(2, 5))

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        title = soup.title.string if soup.title else "AWS Documentation"
        main_content = soup.find('div', {'id': 'main-content'}) or soup.find('div', {'class': 'main-content'})
        content_text = main_content.get_text(separator=' ', strip=True) if main_content else "No main content found."

        embedding = model.encode([content_text])[0].astype('float32')

        index.add(np.array([embedding]))
        
        metadata.append({
            "title": title,
            "url": url,
            "content": content_text
        })

        print(f"Scraped and stored: {title} | {url}")

        visited_links.add(url)

        internal_links = soup.find_all('a', href=True)
        for link in internal_links:
            href = link['href']
            full_url = urljoin(url, href)
            if "docs.aws.amazon.com" in full_url and full_url not in visited_links:
                scrape_aws_page(full_url, index, model, metadata, visited_links, driver, max_depth, current_depth + 1)

    except Exception as e:
        print(f"Failed to scrape {url}: {e}")

def initialize_faiss_index(embedding_size=384):
    index = faiss.IndexFlatL2(embedding_size)  # Initialize FAISS index for L2 distance
    return index

def save_faiss_index(index, index_file, metadata, metadata_file):
    faiss.write_index(index, index_file)
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f)
    print(f"Vector store saved to {index_file} and metadata to {metadata_file}.")

def load_faiss_index(index_file, metadata_file):
    index = faiss.read_index(index_file)
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    return index, metadata

def main():
    start_url = 'https://docs.aws.amazon.com/'  # Starting URL for AWS documentation
    index_file = 'aws_docs_faiss.index'  # File to store FAISS index
    metadata_file = 'aws_metadata.json'  # File to store metadata

    # Initialize Vector Store (FAISS Index)
    index = initialize_faiss_index(embedding_size=384)

    # Load Pre-trained Embedding Model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    metadata = []
    visited_links = set()  

    driver = get_driver(headless=True)

    scrape_aws_page(start_url, index, model, metadata, visited_links, driver, max_depth=2)

    save_faiss_index(index, index_file, metadata, metadata_file)

    driver.quit()

if __name__ == "__main__":
    main()
