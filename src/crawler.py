import requests
from bs4 import BeautifulSoup
import time

BASE_URL = "https://www.comune.arezzo.it"


def fetch(url):
    """Scarica una singola pagina e restituisce {url, title, content}."""
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print("[ERR] Request error:", url, e)
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Titolo Drupal
    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else ""

    # Contenuto principale (Drupal: .node, .content, article)
    selectors = [
        ".node__content",
        "article",
        ".region-content",
    ]

    content = ""
    for sel in selectors:
        block = soup.select_one(sel)
        if block:
            content = block.get_text(" ", strip=True)
            break

    # Fallback
    if not content:
        content = soup.get_text(" ", strip=True)

    return {
        "url": url,
        "title": title,
        "content": content
    }


def crawl_comune_arezzo(max_pages=1500, delay=0.2):
    """Crawling Drupal con gestione link e dedup."""
    to_visit = {BASE_URL}
    visited = set()
    pages = []

    session = requests.Session()

    while to_visit and len(pages) < max_pages:
        url = to_visit.pop()

        if url in visited:
            continue
        visited.add(url)

        print(f"[CRAWL] {len(visited)} → {url}")

        page = fetch(url)
        if page:
            pages.append(page)

        # Trova nuovi link Drupal
        try:
            resp = session.get(url, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"]

                # Manteniamo solo link interni Drupal
                if href.startswith("/"):
                    href = BASE_URL + href
                if not href.startswith(BASE_URL):
                    continue

                if href not in visited:
                    to_visit.add(href)

        except:
            pass

        time.sleep(delay)

    return pages
