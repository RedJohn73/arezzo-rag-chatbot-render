import requests
from bs4 import BeautifulSoup
import time

BASE_URL = "https://www.comune.arezzo.it"

def extract_text_from_drupal(html):
    """Estrae contenuto testo dai layout Drupal (classi tipiche: .node, .content, .field)."""
    soup = BeautifulSoup(html, "html.parser")

    # Blocchi di testo comuni nelle installazioni Drupal
    selectors = [
        ".node__content",
        ".content",
        ".field--name-body",
        ".field__item",
        ".block-content",
        "article",
        ".paragraph",
        ".text-formatted"
    ]

    texts = []

    for sel in selectors:
        for el in soup.select(sel):
            t = el.get_text(separator=" ", strip=True)
            if t and len(t) > 30:
                texts.append(t)

    # fallback assoluto
    if not texts:
        all_text = soup.get_text(separator=" ", strip=True)
        return all_text

    return "\n\n".join(texts)


def crawl_comune_arezzo(max_pages=2000):
    """Crawl ricorsivo molto semplice per Drupal."""
    visited = set()
    to_visit = [BASE_URL]
    pages = []

    while to_visit and len(pages) < max_pages:
        url = to_visit.pop(0)

        if url in visited:
            continue
        visited.add(url)

        try:
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                continue
        except:
            continue

        html = r.text
        text = extract_text_from_drupal(html)

        pages.append({
            "url": url,
            "text": text
        })

        # trova nuovi link Drupal
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]

            # Ignora link esterni
            if href.startswith("http"):
                if BASE_URL not in href:
                    continue
                new_url = href
            else:
                new_url = BASE_URL + href

            # Filtri molto utili
            if any(x in new_url for x in [
                ".pdf", ".jpg", ".png", ".zip", "login", "user"
            ]):
                continue

            if new_url not in visited and new_url not in to_visit:
                to_visit.append(new_url)

        time.sleep(0.3)

    return pages
