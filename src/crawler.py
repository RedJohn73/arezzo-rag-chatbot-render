
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


def crawl_comune_arezzo(
    root_url: str = "https://www.comune.arezzo.it",
    max_pages: int = 1500,
    max_depth: int = 8,
    delay_seconds: float = 0.15,
):
    """Crawling semplice del sito del Comune di Arezzo.

    Raccoglie pagine HTML, estrae il testo principale da <p> e <li>.
    Ritorna una lista di dict: {url, title, content}.
    """
    session = requests.Session()
    session.headers = {"User-Agent": "ArezzoCrawlerBot/1.0"}

    to_visit = [(root_url, 0)]
    visited = set()
    pages = []

    while to_visit and len(pages) < max_pages:
        url, depth = to_visit.pop(0)

        if url in visited:
            continue
        if depth > max_depth:
            continue

        visited.add(url)

        try:
            resp = session.get(url, timeout=10)
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                continue
        except Exception:
            continue

        soup = BeautifulSoup(resp.text, "html.parser")

        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""

        text_blocks = []
        for tag in soup.find_all(["p", "li"]):
            t = tag.get_text(" ", strip=True)
            if t:
                text_blocks.append(t)

        content = "\n".join(text_blocks)

        pages.append(
            {
                "url": url,
                "title": title,
                "content": content,
            }
        )

        for a in soup.find_all("a", href=True):
            href = a["href"]
            absolute = urljoin(url, href)

            parsed = urlparse(absolute)
            if parsed.netloc and "comune.arezzo.it" not in parsed.netloc:
                continue

            if absolute not in visited:
                to_visit.append((absolute, depth + 1))

        time.sleep(delay_seconds)

    return pages
