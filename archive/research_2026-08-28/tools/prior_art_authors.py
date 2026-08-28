"""Fetch author names for the papers to be cited in the whitepaper."""
import re
import time

import requests

IDS = {
    "dropbear": "2205.15757",
    "goldengrain": "2011.06458",
    "opml": "2401.17555",
    "svip": "2410.22307",
    "hadagent": "2604.18614",
    "bittensor_critique": "2507.02951",
    "token_inflation": "2605.30040",
    "toploc": "2501.16007",
}

for key, aid in IDS.items():
    url = f"https://arxiv.org/abs/{aid}"
    try:
        text = requests.get(url, timeout=30).text
    except requests.RequestException as exc:
        print(f"{key}: FETCH FAILED {exc}")
        continue
    authors = re.findall(
        r'<meta name="citation_author" content="([^"]+)"', text)
    title = re.findall(
        r'<meta name="citation_title" content="([^"]+)"', text)
    print(f"{key}: {aid}")
    print(f"  title: {title[0] if title else '?'}")
    print(f"  authors: {', '.join(authors)}")
    time.sleep(3)
