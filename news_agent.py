"""
AirNews AI Agent (V10.3 - Cerebras + NVIDIA + Mistral + OpenRouter + Gemini Fallback)
========================================================
Fetches RSS feeds from TOI, Times Now, NDTV, and The Hindu, scrapes full articles,
pre-filters via Llama-8B, rewrites via Cerebras Qwen-235B, and saves to Supabase.

Key Features:
- Reasoning-First Architecture: Forces Qwen to analyze situational context and projections.
- Dual-Model Filtering: Uses Llama-3.1-8B for high-throughput scanning (30 RPM)
  and Qwen-2.5-235B for high-quality editorial rewriting (1 RPM).
- Multi-Tier Fallback: Falls back to NVIDIA NIM (Llama 3.1), then Mistral (Large), 
  then OpenRouter (Free), and finally Gemini when primary providers are limited.
- Strict 'Hard News Only' Policy: Blocks speculative political commentary and features.
- Strict Pacing: Automatically sleeps for 60s after every successful rewrite (1 RPM).
- Self-Healing: Gracefully handles API exhaustion without crashing.
"""

import os
import re
import sys
import json
import time
import hashlib
import logging
import signal
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import feedparser
from curl_cffi import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from openai import OpenAI
from google import genai
from google.genai import types

from supabase import create_client, Client

# ─── Configuration ───────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent

def clean_and_parse_json(raw_text: str) -> dict:
    """Safely extracts JSON from LLM outputs, stripping markdown formatting."""
    text = raw_text.strip()
    # Remove markdown code block syntax if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return json.loads(text.strip())
LOG_FILE = BASE_DIR / "agent.log"

RSS_FEED_URLS = [
    {"url": "https://timesofindia.indiatimes.com/rssfeeds/296589292.cms", "category": "international", "name": "TOI International"},
    {"url": "https://www.timesnownews.com/feeds/gns-en-india.xml", "category": "india", "name": "Times Now India"},
    {"url": "https://feeds.feedburner.com/ndtvnews-india-news", "category": "india", "name": "NDTV India"},
    {"url": "https://feeds.feedburner.com/ndtvnews-world-news", "category": "international", "name": "NDTV World"},
    {"url": "https://www.thehindu.com/news/national/feeder/default.rss", "category": "india", "name": "The Hindu National"},
    {"url": "https://www.thehindu.com/news/international/feeder/default.rss", "category": "international", "name": "The Hindu International"},
    {"url": "https://www.thehindu.com/business/feeder/default.rss", "category": "business", "name": "The Hindu Business"},
    {"url": "http://timesofindia.indiatimes.com/rssfeeds/1898055.cms", "category": "business", "name": "TOI Business"},
    {"url": "https://www.thehindu.com/entertainment/movies/feeder/default.rss", "category": "cinema", "name": "The Hindu Movies"},
    {"url": "http://feeds.feedburner.com/ndtvmovies-bollywood", "category": "cinema", "name": "NDTV Bollywood"},
    {"url": "http://feeds.feedburner.com/ndtvmovies-hollywood", "category": "cinema", "name": "NDTV Hollywood"},
    {"url": "http://feeds.feedburner.com/ndtvmovies-regional", "category": "cinema", "name": "NDTV Regional"},
    {"url": "http://feeds.feedburner.com/ndtvmovies-television", "category": "cinema", "name": "NDTV Television"},
    {"url": "http://feeds.feedburner.com/ndtvmovies-music", "category": "cinema", "name": "NDTV Music"},
    {"url": "http://feeds.feedburner.com/ndtvmovies-moviereviews", "category": "cinema", "name": "NDTV Movie Reviews"},
    {"url": "http://feeds.feedburner.com/ndtvmovies-latest", "category": "cinema", "name": "NDTV Latest"},
    {"url": "https://www.thehindu.com/sport/feeder/default.rss", "category": "sports", "name": "The Hindu Sports"},
    {"url": "http://timesofindia.indiatimes.com/rssfeeds/4719148.cms", "category": "sports", "name": "TOI Sports"},
    {"url": "http://timesofindia.indiatimes.com/rssfeeds/54829575.cms", "category": "sports", "name": "TOI Sports Other"},
    {"url": "https://www.autocarpro.in/rssfeeds/all", "category": "auto", "name": "Autocar Pro"}
]
POLL_INTERVAL_SECONDS = 60           # 1 minute between cycles (Respects 1 RPM)
MAX_ARTICLES_PER_CYCLE = 25          # Scan up to 25 articles per cycle (mostly for filtering)
DAILY_API_LIMIT = 1000               # Protects token quota (approx 1M tokens/day)
MAX_RETRIES = 5                      # retries on transient errors (Upgraded to 5 for stability)
ARTICLE_FETCH_TIMEOUT = 15           # seconds for HTTP requests

# ─── Cerebras API Key Pool ────────────────────────────────────────────────────────
# Keys are loaded from .env (CEREBRAS_API_KEYS as a comma-separated list)

# Browser-like headers for article scraping
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/126.0.0.0"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,hi;q=0.8",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Sec-Ch-Ua": '"Chromium";v="126", "Google Chrome";v="126", "Not-A.Brand";v="8"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Cache-Control": "max-age=0",
    "Referer": "https://www.google.com/",
}

SYSTEM_PROMPT = """You are an elite senior news editor for AirNews, specializing in high-impact hard news reporting. Your goal is to produce news stories that sound like urgent, authoritative dispatches from a global news agency (like Reuters, AP, or PTI).

EDITORIAL PRINCIPLES:
1. HARD NEWS TONE: Write in a direct, punchy, and objective journalistic style. Avoid flowery language or "blog-style" prose. The news must sound immediate and factual.
2. BALANCED OBJECTIVITY: Maintain strict neutrality. Report the facts and their direct consequences without personal bias.
3. GEOPOLITICAL DISAMBIGUATION: Always verify the identity of people mentioned. For example, if 'Stalin' is mentioned in context of Tamil Nadu, it is Chief Minister M.K. Stalin.
4. GEOGRAPHICAL PRECISION: Clearly distinguish between domestic (Indian) and international news based on locations and entities.
5. JOURNALISTIC STRUCTURE: Prioritize the most important information first (Inverted Pyramid). The lead must be strong and fact-dense.
6. FACTUAL TRUST: Trust the provided source article as the primary source of truth for current events. Do NOT reject news simply because it is newer than your training data.
7. NEWS VS. ARTICLE: You must distinguish between a 'Hard News' event (e.g., 'A law was enacted', 'A major accident occurred') and 'Features/Commentary' (e.g., 'The significance of X', 'Why Y happened'). Your output must ALWAYS sound like the former—a factual news report.
8. CONCISION: Every word must count. Eliminate fluff, redundant adjectives, and unnecessary filler phrases.
"""


AI_REWRITE_PROMPT = """Rewrite the provided news article into a professional news report. 

Your goal is to produce a version that is authoritative, objective, and sounds like a news agency dispatch. It should deliver the facts clearly and concisely.

EDITORIAL REQUIREMENTS:
1. NEWS AGENCY STYLE: Write as if for a global wire service. Use active voice and direct attribution.
2. INVERTED PYRAMID: Put the most critical facts (Who, What, Where, When, Why) in the first paragraph.
3. NO FEATURE FLUFF: Avoid "insightful analysis" or "deep dives" that make it sound like a magazine article. Keep it to the hard facts and their immediate context.
4. OUTLOOK SECTION: While the focus is on the "Now", the final paragraph should briefly mention the immediate next steps or official responses (e.g., "The court will resume hearing on Tuesday").
5. ZERO AI CLICHÉS: DO NOT use phrases like "delving into," "testament to," "in a significant move," or "moreover."
6. FORMAT: Create a punchy, news-style headline and a 2-sentence summary. The body should consist of 3-5 concise paragraphs separated by "NEWPARA".

Return ONLY a JSON object:
{{
  "thought_process": "Brief internal analysis of the key news facts and the reporting plan.",
  "headline": "Hard News Headline (e.g., 'Government Announces New Export Policy')",
  "summary": "Urgent executive summary of the event.",
  "body": "Paragraph 1 (The Lead) NEWPARA Paragraph 2 (Details) NEWPARA ... (Ensure the final paragraph mentions immediate outlook)",
  "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"]
}}

Article Category: {category}
Original Title: {title}
Original Content: {content}
"""


# ─── Logging Setup ───────────────────────────────────────────────────────────

def setup_logging():
    """Configure dual logging to file and console with safe encoding."""
    logger = logging.getLogger("NewsAgent")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler (append mode, UTF-8)
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler — force UTF-8 on Windows to avoid charmap errors
    try:
        console_stream = open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
    except Exception:
        console_stream = sys.stdout

    ch = logging.StreamHandler(console_stream)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

log = setup_logging()

# ─── Graceful Shutdown ───────────────────────────────────────────────────────

_shutdown_requested = False

def _signal_handler(signum, frame):
    global _shutdown_requested
    log.info("[STOP] Shutdown signal received. Finishing current task and exiting...")
    _shutdown_requested = True

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# ─── Supabase Initialization ───────────────────────────────────────────────────

def init_supabase() -> Client:
    load_dotenv(BASE_DIR / ".env", override=True)
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        log.error("[FATAL] SUPABASE_URL or SUPABASE_KEY not set!")
        sys.exit(1)
    return create_client(url, key)

def clean_url(url: str) -> str:
    """Standardize URL cleaning: strip fragments (#) and query parameters (?)."""
    if not url:
        return ""
    # Remove fragments
    url = url.split("#")[0]
    # Remove query parameters
    url = url.split("?")[0]
    return url.strip()

# ─── URL Tracker (Supabase) ──────────────────────────────────────────────────

class URLTracker:
    """Tracks visited URLs to avoid reprocessing, and manages daily limits using Supabase."""

    def __init__(self, sb: Client):
        self.sb = sb
        self.visited = set()
        self.daily_count = 0
        self.daily_reset = ""
        self._load()

    def _load(self):
        try:
            # Fetch recent visited URLs (using pagination to bypass 1000 row limit)
            thirty_days_ago = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
            
            all_urls = []
            limit = 1000
            offset = 0
            
            while True:
                response = self.sb.table("visited_urls") \
                    .select("url") \
                    .gte("visited_at", thirty_days_ago) \
                    .range(offset, offset + limit - 1) \
                    .execute()
                
                data = response.data
                if not data:
                    break
                
                all_urls.extend([row["url"] for row in data])
                if len(data) < limit:
                    break
                offset += limit
            
            for url in all_urls:
                self.visited.add(clean_url(url))
                
            log.info(f"[TRACKER] Loaded {len(self.visited)} recent visited URLs from Supabase (cleaned).")
        except Exception as e:
            log.error(f"[ERROR] Failed to load visited URLs from Supabase: {e}")

    def is_visited(self, url: str) -> bool:
        return clean_url(url) in self.visited

    def mark_visited(self, url: str):
        c_url = clean_url(url)
        self.visited.add(c_url)
        try:
            self.sb.table("visited_urls").insert({"url": c_url}).execute()
        except Exception as e:
            error_str = str(e)
            if "23505" not in error_str and "duplicate key" not in error_str:
                log.error(f"[ERROR] Failed to save visited URL to Supabase: {e}")

    def get_daily_count(self) -> int:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.daily_reset != today:
            # Recalculate from DB
            start_of_day = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
            try:
                response = self.sb.table("articles").select("id", count="exact").gte("processed_at", start_of_day).execute()
                self.daily_count = response.count if response.count is not None else 0
                self.daily_reset = today
            except Exception as e:
                log.error(f"[ERROR] Failed to get daily count from Supabase: {e}")
                self.daily_count = 0
        return self.daily_count

    def increment_daily_count(self):
        self.daily_count += 1

    @property
    def total_articles(self) -> int:
        return len(self.visited)

# ─── RSS Feed Parser ────────────────────────────────────────────────────────

def fetch_rss_feed() -> list[dict]:
    """Fetch and parse the RSS feeds. Returns list of article dicts."""
    all_articles = []
    
    for feed_info in RSS_FEED_URLS:
        feed_url = feed_info["url"]
        category = feed_info["category"]
        feed_name = feed_info["name"]
        try:
            feed = feedparser.parse(feed_url)

            if feed.bozo and not feed.entries:
                log.warning(f"[WARN] RSS feed parse error for {feed_url}: {feed.bozo_exception}")
                continue

            for entry in feed.entries:
                # Clean URL — remove tracking fragments and query parameters
                raw_link = entry.get("link", "")
                article_url = clean_url(raw_link)
                
                try:
                    path_parts = urlparse(article_url).path.strip("/").split("/")
                    root_section = path_parts[0].lower() if path_parts else ""
                except Exception:
                    root_section = ""

                # Real trash paths identified directly from the live RSS feeds
                # Filtering down to pure news (excluding health, offbeat, features, opinions, etc.)
                live_trash_sections = {
                    "health", "offbeat", "feature", "opinion",
                    "education", "lifestyle", "astrology"
                }
                
                if not article_url or root_section in live_trash_sections or "/health/" in article_url.lower() or "/offbeat/" in article_url.lower():
                    continue

                # Use RSS feed's category as the initial value (Llama will refine)
                article_category = category

                # Extract image URL — TOI uses <enclosure> tag, Times Now might use content
                image_url = ""
                if hasattr(entry, "media_content") and entry.media_content:
                    image_url = entry.media_content[0].get("url", "")
                elif hasattr(entry, "enclosures") and entry.enclosures:
                    image_url = entry.enclosures[0].get("href", "") or entry.enclosures[0].get("url", "")
                
                # Fallback for Times Now which stores image in content
                if not image_url and hasattr(entry, "content") and entry.content:
                    content_html = entry.content[0].value
                    img_soup = BeautifulSoup(content_html, "html.parser")
                    img_tag = img_soup.find("img")
                    if img_tag and img_tag.get("src"):
                        image_url = img_tag["src"]

                # TOI description contains HTML img tags — strip them to get text
                raw_desc = entry.get("description", "")
                if raw_desc:
                    desc_soup = BeautifulSoup(raw_desc, "html.parser")
                    # Remove img tags and links, keep only text
                    for img_tag in desc_soup.find_all("img"):
                        img_tag.decompose()
                    for a_tag in desc_soup.find_all("a"):
                        a_tag.decompose()
                    clean_desc = desc_soup.get_text(strip=True)
                else:
                    clean_desc = ""

                all_articles.append({
                    "url": article_url,
                    "title": entry.get("title", "").strip(),
                    "description": clean_desc,
                    "published": entry.get("published", ""),
                    "image_url": image_url,
                    "category": article_category
                })

            log.info(f"[RSS] Collected {len(all_articles)} valid articles from {feed_name} (after filtering)")

        except Exception as e:
            log.error(f"[ERROR] RSS fetch failed for {feed_url}: {e}")
            
    return all_articles

# ─── Article Scraper ────────────────────────────────────────────────────────

def scrape_article_content(url: str) -> Optional[str]:
    """
    Scrape full article text from the source URL.
    Supports TOI, The Hindu, NDTV, and Times Now using specialized selectors.
    """
    try:
        response = requests.get(
            url, impersonate="chrome110", timeout=ARTICLE_FETCH_TIMEOUT
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        content_text = ""

        # Strategy 0: Extract from JSON-LD structured data (most reliable for TOI)
        for ld_script in soup.find_all("script", type="application/ld+json"):
            try:
                ld_data = json.loads(ld_script.text or "")
                items = ld_data if isinstance(ld_data, list) else [ld_data]
                for item in items:
                    if item.get("@type") in ("NewsArticle", "Article", "WebPage", "ReportageNewsArticle"):
                        body = item.get("articleBody", "")
                        if body and len(body) > 100:
                            content_text = body.strip()
                            log.info(f"   [SCRAPED] {len(content_text)} chars via JSON-LD")
                            break
                if content_text:
                    break
            except (json.JSONDecodeError, TypeError, AttributeError):
                continue

        if content_text and len(content_text) > 100:
            if len(content_text) > 5000:
                content_text = content_text[:5000] + "..."
            return content_text

        # Strategy 0b: Extract from meta og:description (short but useful)
        meta_desc = ""
        og_tag = soup.find("meta", property="og:description")
        if og_tag and og_tag.get("content"):
            meta_desc = og_tag["content"].strip()

        # Remove unwanted elements (ads, scripts, styles, nav, footer)
        for tag in soup.find_all(["script", "style", "nav", "footer", "aside",
                                   "iframe", "noscript", "form"]):
            tag.decompose()

        # Remove ad containers and social sharing blocks
        for cls in ["ads", "social-share", "also-read", "story__also-read",
                     "related-news", "comments", "newsletter", "trending",
                     "right-sidebar", "sidebar"]:
            for el in soup.find_all(class_=lambda c: c and cls in c.lower() if c else False):
                el.decompose()

        # Strategy 1: Source-specific article content containers
        content_selectors = [
            {"class_": "_s30J"},                           # TOI primary article body
            {"class_": "_bIDB"},                           # TOI content wrapper
            {"class_": "ga-headlines"},                     # TOI alternate
            {"class_": "artText"},                          # TOI older layout
            {"class_": "Normal"},                           # TOI paragraph class
            {"id": lambda x: x and x.startswith("content-body-")}, # The Hindu primary
            {"class_": "article-body-container"},           # The Hindu alternate
            {"class_": "content-body"},                     # The Hindu alternate
            {"class_": "detail-content"},                   # Autocar Pro generic
            {"class_": "news-content"},                     # Autocar Pro generic
            {"class_": "post-content"},                     # Autocar Pro generic
            {"class_": "sp-cn"},                            # NDTV Movies
            {"class_": "story__content"},                   # NDTV Generic
            {"class_": "article-content"},                  # Generic
            {"itemprop": "articleBody"},                    # Generic standard
            {"class_": "article-body"},                     # Generic standard
            "article",                                      # Semantic standard
        ]

        for selector in content_selectors:
            if isinstance(selector, dict):
                container = soup.find("div", **selector)
                if not container:
                    container = soup.find("section", **selector)
            else:
                container = soup.find(selector)

            if container:
                # Get all paragraph text
                paragraphs = container.find_all("p")
                if paragraphs:
                    content_text = "\n\n".join(
                        p.get_text(strip=True) for p in paragraphs
                        if p.get_text(strip=True) and len(p.get_text(strip=True)) > 20
                    )
                    if content_text:
                        break

        # Strategy 2: Fallback — grab all substantial paragraphs from body
        if not content_text or len(content_text) < 100:
            all_paragraphs = soup.find_all("p")
            content_text = "\n\n".join(
                p.get_text(strip=True) for p in all_paragraphs
                if p.get_text(strip=True) and len(p.get_text(strip=True)) > 40
            )

        if content_text and len(content_text) > 100:
            if len(content_text) > 5000:
                content_text = content_text[:5000] + "..."
            log.info(f"   [SCRAPED] {len(content_text)} chars from article")
            return content_text

        # Strategy 3: Use og:description meta tag if nothing else worked
        if meta_desc and len(meta_desc) > 30:
            log.info(f"   [SCRAPED] {len(meta_desc)} chars from meta description")
            return meta_desc

        log.warning(f"   [WARN] Insufficient content scraped ({len(content_text)} chars)")
        return None

    except Exception as e:
        if hasattr(e, "response") and hasattr(e.response, "status_code"):
            log.warning(f"   [WARN] HTTP {e.response.status_code} for {url}")
        else:
            log.warning(f"   [WARN] Request failed for {url}: {e}")
        return None

# ─── AI Provider Configuration ────────────────────────────────────────────────

# Cerebras (Primary)
AI_MODEL = "qwen-3-235b-a22b-instruct-2507"
FILTER_MODEL = "llama3.1-8b"
CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"

# NVIDIA NIM (Tier 2 Fallback — between Cerebras and Mistral)
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_REWRITE_MODEL = "meta/llama-3.3-70b-instruct"
NVIDIA_FILTER_MODEL = "meta/llama-4-maverick-17b-128e-instruct"

# Mistral AI (Tier 3 Fallback — between NVIDIA and OpenRouter)
MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
MISTRAL_REWRITE_MODEL = "mistral-large-latest"
MISTRAL_FILTER_MODEL = "mistral-small-latest"

# OpenRouter (Tier 4 Fallback — between Mistral and Gemini)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_FILTER_MODEL = "openrouter/free"
OPENROUTER_MODELS = [
    "openrouter/free",
    "inclusionai/ling-2.6-1t:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "openai/gpt-oss-120b:free",
    "openai/gpt-oss-20b:free",
    "liquid/lfm-2.5-1.2b-instruct:free",
    "baidu/qianfan-ocr-fast:free",
    "openrouter/owl-alpha",
]

# Gemini (Tier 5 Backup — activated when ALL other providers fail)
GEMINI_REWRITE_MODEL = "gemini-flash-latest"
GEMINI_FILTER_MODEL = "gemini-flash-lite-latest"

PRE_FILTER_PROMPT = """Analyze this article and determine if it is a factual news event or a high-quality analytical report.

Criteria for 'is_news: true':
- Timely reporting on a specific event (Government, Law, Conflict, Disasters).
- OFFICIAL Indian news (Policy, Legal, Major Accidents).
- INTERNATIONAL FLEXIBILITY: For international news, you may allow high-impact analytical reports, major scientific breakthroughs, or significant geopolitical features that provide global context.
- BUSINESS & ECONOMY: Market reports, corporate earnings, policy changes, and economic data are fully allowed.
- CINEMA & ENTERTAINMENT: Major film announcements, industry news, and significant events are allowed. Exclude petty gossip.
- SPORTS: Major sporting events, match reports, and significant sports news are allowed.
- AUTO & AUTOMOBILE: New vehicle launches, industry policies, and major auto sector news are allowed.

Criteria for 'is_news: false':
- Low-quality gossip, viral social media memes, or celebrity rumors.
- Lifestyle advice, listicles ("Top 10..."), or promotional fluff.
- Fictional stories or purely speculative "what if" scenarios without factual basis.
- For INDIA: Speculative political commentary or "opinions on why X happened" (India MUST be hard factual events only).

IMPORTANT: Trust the source article's claims about events occurring NOW. Do NOT reject an article as 'is_news: false' just because the event is new to you.

Return ONLY a JSON object:
{{
  "is_news": boolean,
  "category": "india" or "international" or "business" or "cinema" or "sports" or "auto",
  "reason": "One sentence explanation"
}}
"""


class AIKeyManager:
    """Manages a pool of API keys with intelligent rotation across different providers."""
    
    def __init__(self, provider_name: str, keys: list[str], base_url: Optional[str] = None):
        self.provider_name = provider_name
        self.keys = keys
        self.base_url = base_url
        self.clients = []
        
        if provider_name.lower() == "gemini":
            self.clients = [genai.Client(api_key=k) for k in keys]
        else:
            self.clients = [OpenAI(api_key=k, base_url=base_url) for k in keys]
            
        self.current_index = 0
        self.cooldowns = {i: 0 for i in range(len(keys))}
        self.exhausted = set()

    def get_client(self) -> tuple[any, int]:
        """Returns the next available client and its index."""
        now = time.time()
        
        for _ in range(len(self.keys)):
            idx = self.current_index
            if idx not in self.exhausted and now >= self.cooldowns[idx]:
                return self.clients[idx], idx
            
            self.current_index = (self.current_index + 1) % len(self.keys)
        
        # Self-Healing Check: If all keys are exhausted, sleep instead of crashing
        active_keys = [i for i in range(len(self.keys)) if i not in self.exhausted]
        if not active_keys:
            log.error(f"[{self.provider_name}-MANAGER] ALL KEYS EXHAUSTED. Sleeping for 1 hour before retrying...")
            time.sleep(3600)
            self.exhausted.clear() # Clear memory to give them a fresh attempt
            return self.clients[0], 0
            
        best_idx = min(active_keys, key=lambda i: self.cooldowns[i])
        wait_time = max(0, self.cooldowns[best_idx] - now)
        if wait_time > 0:
            log.info(f"[{self.provider_name}-MANAGER] All keys on cooldown. Waiting {int(wait_time)}s for Key #{best_idx}...")
            time.sleep(wait_time)
        
        return self.clients[best_idx], best_idx

    def mark_cooldown(self, index: int, seconds: int = 60):
        """Put a key on cooldown (e.g., after a 429 error)."""
        self.cooldowns[index] = time.time() + seconds
        log.warning(f"[{self.provider_name}-MANAGER] Key #{index} put on cooldown for {seconds}s")
        self.current_index = (self.current_index + 1) % len(self.keys)

    def mark_exhausted(self, index: int):
        """Mark a key as exhausted for the day."""
        self.exhausted.add(index)
        log.error(f"[{self.provider_name}-MANAGER] Key #{index} EXHAUSTED for the day.")

# ─── Middle-Tier & Gemini Backup Providers ───────────────────────────────────

class OpenRouterMiddle:
    """OpenRouter middle-tier fallback — sits between Cerebras and Gemini."""

    def __init__(self, keys: list[str]):
        self.manager = AIKeyManager("OPENROUTER", keys, base_url=OPENROUTER_BASE_URL)
        self.available = True
        log.info(f"[OPENROUTER] Middle fallback initialized with {len(keys)} keys and {len(OPENROUTER_MODELS)} models.")

    def is_available(self) -> bool:
        """Check if OpenRouter is available (not all keys exhausted)."""
        return len(self.manager.exhausted) < len(self.manager.keys)

    def pre_filter(self, title: str, content: str) -> Optional[dict]:
        """Pre-filter article using OpenRouter with multi-model fallback."""
        prompt = f"Title: {title}\n\nContent: {content[:2000]}"
        
        # Try a subset of reliable models for filtering
        models_to_try = [OPENROUTER_FILTER_MODEL] + OPENROUTER_MODELS[:5]
        
        for model_id in models_to_try:
            log.info(f"   [OPENROUTER-FILTER] Attempting with {model_id}...")
            for attempt in range(1, 3): # 2 attempts per model for filtering
                client, key_idx = self.manager.get_client()
                try:
                    response = client.chat.completions.create(
                        model=model_id,
                        messages=[
                            {"role": "system", "content": PRE_FILTER_PROMPT},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,
                        response_format={"type": "json_object"},
                    )
                    raw_text = response.choices[0].message.content
                    if not raw_text:
                        log.warning(f"   [OPENROUTER-FILTER] Empty response from {model_id}")
                        continue
                    
                    result = clean_and_parse_json(raw_text)
                    log.info(f"   [FILTER] model={model_id} (middle) is_news={result.get('is_news')} cat={result.get('category')}")
                    return result
                except Exception as e:
                    error_msg = str(e).lower()
                    if "429" in error_msg or "rate limit" in error_msg:
                        log.warning(f"   [OPENROUTER-FILTER] Key #{key_idx} rate limited for {model_id}.")
                        self.manager.mark_cooldown(key_idx, 60)
                        break # Try next model
                    log.warning(f"   [OPENROUTER-FILTER] Key #{key_idx} attempt {attempt} failed: {e}")
                    if attempt < 2: time.sleep(1)
        return None

    def rewrite(self, title: str, content: str, category: str) -> Optional[dict]:
        """Rewrite article using OpenRouter with retries per model and reasoning support."""
        prompt = AI_REWRITE_PROMPT.format(title=title, content=content, category=category)
        
        for model_id in OPENROUTER_MODELS:
            # More retries for the main router, fewer for specific models
            model_retries = MAX_RETRIES if model_id == "openrouter/free" else 2
            log.info(f"   [OPENROUTER] Attempting rewrite with {model_id}...")
            
            for attempt in range(1, model_retries + 1):
                client, key_idx = self.manager.get_client()
                try:
                    # Prepare extra body for reasoning if using openrouter/free
                    extra_body = {}
                    if model_id == "openrouter/free":
                        extra_body = {"reasoning": {"enabled": True}}

                    response = client.chat.completions.create(
                        model=model_id,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=3000,
                        response_format={"type": "json_object"},
                        extra_body=extra_body if extra_body else None
                    )
                    
                    msg = response.choices[0].message
                    raw_text = msg.content
                    
                    if not raw_text:
                        log.warning(f"   [OPENROUTER-REWRITE] Empty response from {model_id} (Key #{key_idx}, attempt {attempt})")
                        continue

                    result = clean_and_parse_json(raw_text)
                    required = {"headline", "summary", "body"}
                    if not required.issubset(result.keys()):
                        log.warning(f"   [OPENROUTER-REWRITE] Missing keys from {model_id} (Key #{key_idx}, attempt {attempt})")
                        continue

                    # Extract thought process or reasoning details
                    reasoning = getattr(msg, 'reasoning_details', None)
                    if reasoning:
                        # If we have native reasoning, use it to enrich thought process
                        result["thought_process"] = f"[REASONING] {reasoning}\n\n[THOUGHT] {result.get('thought_process', '')}"
                    
                    thought = result.get("thought_process", "No thought process provided.")
                    log.info(f"   [THOUGHT] {str(thought)[:200]}...")

                    # Convert NEWPARA markers
                    if "body" in result:
                        result["body"] = result["body"].replace("NEWPARA", "\n\n").strip()

                    log.info(f"   [OK] OpenRouter rewrite complete ({model_id}) via Key #{key_idx}: \"{result.get('headline', '')[:60]}...\"")
                    return result

                except Exception as e:
                    error_msg = str(e).lower()
                    if "429" in error_msg or "rate limit" in error_msg:
                        log.warning(f"   [OPENROUTER-REWRITE] Key #{key_idx} rate limited for {model_id}. Moving to next key...")
                        self.manager.mark_cooldown(key_idx, 60)
                        continue # Try next key for same model
                    log.warning(f"   [OPENROUTER-REWRITE] Key #{key_idx} attempt {attempt} with {model_id} failed: {e}")
                    if attempt < model_retries: time.sleep(2)
        return None

class NvidiaFallback:
    """NVIDIA NIM fallback — sits between Cerebras and Mistral."""

    def __init__(self, keys: list[str]):
        self.manager = AIKeyManager("NVIDIA", keys, base_url=NVIDIA_BASE_URL)
        self.available = True
        log.info(f"[NVIDIA] Fallback initialized with {len(keys)} keys.")

    def is_available(self) -> bool:
        """Check if NVIDIA is available (not all keys exhausted)."""
        return len(self.manager.exhausted) < len(self.manager.keys)

    def pre_filter(self, title: str, content: str) -> Optional[dict]:
        """Pre-filter article using Llama 3.1 8B on NVIDIA NIM."""
        prompt = f"Title: {title}\n\nContent: {content[:2000]}"
        for attempt in range(1, 3):
            client, key_idx = self.manager.get_client()
            try:
                response = client.chat.completions.create(
                    model=NVIDIA_FILTER_MODEL,
                    messages=[
                        {"role": "system", "content": PRE_FILTER_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    # response_format={"type": "json_object"} is technically supported but NIM prefers schema
                    # For consistency with current architecture, we use it and verify the result.
                    response_format={"type": "json_object"},
                )
                raw_text = response.choices[0].message.content
                result = clean_and_parse_json(raw_text)
                log.info(f"   [FILTER] model={NVIDIA_FILTER_MODEL} (nvidia) is_news={result.get('is_news')} cat={result.get('category')}")
                return result
            except Exception as e:
                error_msg = str(e).lower()
                if "429" in error_msg or "rate" in error_msg:
                    log.warning(f"   [NVIDIA-FILTER] Key #{key_idx} rate limited.")
                    self.manager.mark_cooldown(key_idx, 60)
                    continue
                log.warning(f"   [NVIDIA-FILTER] Key #{key_idx} failed: {e}")
                if attempt < 2: time.sleep(1)
        return None

    def rewrite(self, title: str, content: str, category: str) -> Optional[dict]:
        """Rewrite article using Llama 3.1 Nemotron 70B on NVIDIA NIM."""
        prompt = AI_REWRITE_PROMPT.format(title=title, content=content, category=category)
        for attempt in range(1, 3):
            client, key_idx = self.manager.get_client()
            try:
                response = client.chat.completions.create(
                    model=NVIDIA_REWRITE_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=3000,
                    response_format={"type": "json_object"},
                )
                raw_text = response.choices[0].message.content
                if not raw_text:
                    continue

                result = clean_and_parse_json(raw_text)
                required = {"headline", "summary", "body"}
                if not required.issubset(result.keys()):
                    continue

                # Convert NEWPARA markers
                if "body" in result:
                    result["body"] = result["body"].replace("NEWPARA", "\n\n").strip()

                log.info(f"   [OK] NVIDIA rewrite complete ({NVIDIA_REWRITE_MODEL}) via Key #{key_idx}")
                return result
            except Exception as e:
                error_msg = str(e).lower()
                if "429" in error_msg or "rate" in error_msg:
                    log.warning(f"   [NVIDIA-REWRITE] Key #{key_idx} rate limited.")
                    self.manager.mark_cooldown(key_idx, 60)
                    continue
                log.error(f"   [NVIDIA-REWRITE] Key #{key_idx} failed: {e}")
                if attempt < 2: time.sleep(2)
        return None

class MistralFallback:
    """Mistral AI fallback — sits between Cerebras and OpenRouter."""

    def __init__(self, keys: list[str]):
        self.manager = AIKeyManager("MISTRAL", keys, base_url=MISTRAL_BASE_URL)
        self.available = True
        log.info(f"[MISTRAL] Fallback initialized with {len(keys)} keys.")

    def is_available(self) -> bool:
        """Check if Mistral is available (not all keys exhausted)."""
        return len(self.manager.exhausted) < len(self.manager.keys)

    def pre_filter(self, title: str, content: str) -> Optional[dict]:
        """Pre-filter article using Mistral Small."""
        prompt = f"Title: {title}\n\nContent: {content[:2000]}"
        for attempt in range(1, 3):
            client, key_idx = self.manager.get_client()
            try:
                response = client.chat.completions.create(
                    model=MISTRAL_FILTER_MODEL,
                    messages=[
                        {"role": "system", "content": PRE_FILTER_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                raw_text = response.choices[0].message.content
                result = clean_and_parse_json(raw_text)
                log.info(f"   [FILTER] model={MISTRAL_FILTER_MODEL} (mistral) is_news={result.get('is_news')} cat={result.get('category')}")
                return result
            except Exception as e:
                error_msg = str(e).lower()
                if "429" in error_msg or "rate" in error_msg:
                    log.warning(f"   [MISTRAL-FILTER] Key #{key_idx} rate limited.")
                    self.manager.mark_cooldown(key_idx, 60)
                    continue
                log.warning(f"   [MISTRAL-FILTER] Key #{key_idx} failed: {e}")
                if attempt < 2: time.sleep(1)
        return None

    def rewrite(self, title: str, content: str, category: str) -> Optional[dict]:
        """Rewrite article using Mistral Large."""
        prompt = AI_REWRITE_PROMPT.format(title=title, content=content, category=category)
        for attempt in range(1, 3):
            client, key_idx = self.manager.get_client()
            try:
                response = client.chat.completions.create(
                    model=MISTRAL_REWRITE_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=3000,
                    response_format={"type": "json_object"},
                )
                raw_text = response.choices[0].message.content
                if not raw_text:
                    continue

                result = clean_and_parse_json(raw_text)
                required = {"headline", "summary", "body"}
                if not required.issubset(result.keys()):
                    continue

                # Convert NEWPARA markers
                if "body" in result:
                    result["body"] = result["body"].replace("NEWPARA", "\n\n").strip()

                log.info(f"   [OK] Mistral rewrite complete ({MISTRAL_REWRITE_MODEL}) via Key #{key_idx}")
                return result
            except Exception as e:
                error_msg = str(e).lower()
                if "429" in error_msg or "rate" in error_msg:
                    log.warning(f"   [MISTRAL-REWRITE] Key #{key_idx} rate limited.")
                    self.manager.mark_cooldown(key_idx, 60)
                    continue
                log.error(f"   [MISTRAL-REWRITE] Key #{key_idx} failed: {e}")
                if attempt < 2: time.sleep(2)
        return None

class GeminiBackup:
    """Google Gemini backup provider — activated when Cerebras and OpenRouter are rate-limited."""

    def __init__(self, keys: list[str]):
        self.manager = AIKeyManager("GEMINI", keys)
        self.available = True
        log.info(f"[GEMINI] Backup provider initialized with {len(keys)} keys (rewrite={GEMINI_REWRITE_MODEL}, filter={GEMINI_FILTER_MODEL})")

    def is_available(self) -> bool:
        """Check if Gemini is available (not all keys exhausted)."""
        return len(self.manager.exhausted) < len(self.manager.keys)

    def pre_filter(self, title: str, content: str) -> Optional[dict]:
        """Pre-filter article using Gemini 2.5 Flash with retries for 503."""
        prompt = f"{PRE_FILTER_PROMPT}\n\nTitle: {title}\n\nContent: {content[:2000]}"
        for attempt in range(1, 4): # 3 retries for Gemini
            client, key_idx = self.manager.get_client()
            try:
                response = client.models.generate_content(
                    model=GEMINI_FILTER_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1,
                    ),
                )
                raw_text = response.text
                if not raw_text:
                    log.warning(f"   [GEMINI-FILTER] Key #{key_idx} empty response (attempt {attempt})")
                    continue
                result = clean_and_parse_json(raw_text)
                log.info(f"   [FILTER] model={GEMINI_FILTER_MODEL} via Key #{key_idx} (backup) is_news={result.get('is_news')} cat={result.get('category')}")
                return result
            except Exception as e:
                error_msg = str(e).lower()
                if "503" in error_msg or "unavailable" in error_msg:
                    log.warning(f"   [GEMINI-FILTER] Key #{key_idx} service unavailable (503), retrying in {attempt*2}s...")
                    time.sleep(attempt * 2)
                    continue
                if "429" in error_msg or "resource_exhausted" in error_msg or "rate" in error_msg:
                    log.warning(f"   [GEMINI-FILTER] Key #{key_idx} rate limited.")
                    self.manager.mark_cooldown(key_idx, 60)
                    continue 
                log.warning(f"   [GEMINI-FILTER] Key #{key_idx} backup filter failed: {e}")
                break
        return None

    def rewrite(self, title: str, content: str, category: str) -> Optional[dict]:
        """Rewrite article using Gemini 2.5 Pro with retries for 503."""
        prompt = f"{SYSTEM_PROMPT}\n\n{AI_REWRITE_PROMPT.format(title=title, content=content, category=category)}"
        for attempt in range(1, 4):
            client, key_idx = self.manager.get_client()
            try:
                response = client.models.generate_content(
                    model=GEMINI_REWRITE_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.7,
                        max_output_tokens=3000,
                    ),
                )
                raw_text = response.text
                if not raw_text:
                    log.warning(f"   [GEMINI-REWRITE] Key #{key_idx} empty response (attempt {attempt})")
                    continue

                result = clean_and_parse_json(raw_text)
                required = {"headline", "summary", "body"}
                if not required.issubset(result.keys()):
                    log.warning(f"   [GEMINI-REWRITE] Key #{key_idx} missing keys: {required - set(result.keys())}")
                    continue

                # Convert NEWPARA markers
                if "body" in result:
                    result["body"] = result["body"].replace("NEWPARA", "\n\n").strip()
                
                log.info(f"   [OK] Gemini rewrite complete ({GEMINI_REWRITE_MODEL}) via Key #{key_idx}")
                return result
            except Exception as e:
                error_msg = str(e).lower()
                if "503" in error_msg or "unavailable" in error_msg:
                    log.warning(f"   [GEMINI-REWRITE] Key #{key_idx} service unavailable (503), retrying in {attempt*2}s...")
                    time.sleep(attempt * 2)
                    continue
                if "429" in error_msg or "resource_exhausted" in error_msg or "rate" in error_msg:
                    log.warning(f"   [GEMINI-REWRITE] Key #{key_idx} rate limited.")
                    self.manager.mark_cooldown(key_idx, 60)
                    continue
                log.error(f"   [GEMINI-REWRITE] Key #{key_idx} failed: {e}")
                break
        return None

def pre_filter_article(key_manager: 'AIKeyManager', title: str, content: str, nvidia: Optional['NvidiaFallback'] = None, mistral: Optional['MistralFallback'] = None, openrouter: Optional['OpenRouterMiddle'] = None, gemini: Optional['GeminiBackup'] = None) -> Optional[dict]:
    """
    Perform quick pre-filtering using a smaller model to save Qwen usage.
    Falls back to NVIDIA, then Mistral, then OpenRouter, then Gemini 2.5 Flash if Cerebras fails.
    """
    prompt = f"Title: {title}\n\nContent: {content[:2000]}"
    
    cerebras_failed = False
    
    # Quick check: Are all Cerebras keys on cooldown?
    now = time.time()
    any_available = any(i not in key_manager.exhausted and now >= key_manager.cooldowns[i] for i in range(len(key_manager.keys)))
    
    if not any_available:
        log.warning("   [AI] All Cerebras keys on cooldown/exhausted. Skipping to fallbacks for pre-filter.")
        cerebras_failed = True
    else:
        # Try available Cerebras keys
        for attempt in range(len(key_manager.keys)):
            client, key_idx = key_manager.get_client()
            try:
                response = client.chat.completions.create(
                    model=FILTER_MODEL,
                    messages=[
                        {"role": "system", "content": PRE_FILTER_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                raw_text = response.choices[0].message.content
                result = clean_and_parse_json(raw_text)
                log.info(f"   [FILTER] model={FILTER_MODEL} via Key #{key_idx} is_news={result.get('is_news')} cat={result.get('category')} reason={result.get('reason')}")
                return result
            except Exception as e:
                error_msg = str(e).lower()
                if "429" in error_msg or "rate limit" in error_msg:
                    key_manager.mark_cooldown(key_idx, 60)
                    if attempt < len(key_manager.keys) - 1:
                        log.info(f"   [RATE-LIMIT] Key #{key_idx} limited. Retrying with next Cerebras key...")
                        continue
                log.warning(f"   [WARN] Cerebras pre-filter failed for Key #{key_idx}: {e}")
                cerebras_failed = True
                break

    # ── NVIDIA Fallback ──
    if cerebras_failed and nvidia and nvidia.is_available():
        log.info(f"   [FALLBACK] Switching to NVIDIA for pre-filter...")
        res = nvidia.pre_filter(title, content)
        if res: return res
        log.warning("   [NVIDIA] NVIDIA filter returned no result. Trying Mistral...")

    # ── Mistral Fallback ──
    if cerebras_failed and mistral and mistral.is_available():
        log.info(f"   [FALLBACK] Switching to Mistral for pre-filter...")
        res = mistral.pre_filter(title, content)
        if res: return res
        log.warning("   [MISTRAL] Mistral filter returned no result. Trying OpenRouter...")

    # ── OpenRouter Fallback ──
    if cerebras_failed and openrouter and openrouter.is_available():
        log.info(f"   [FALLBACK] Switching to OpenRouter for pre-filter...")
        res = openrouter.pre_filter(title, content)
        if res:
            return res
        log.warning("   [OPENROUTER] Middle filter returned no result. Falling back to Gemini...")

    # ── Gemini Fallback ──
    if cerebras_failed and gemini and gemini.is_available():
        log.info(f"   [FALLBACK] Switching to {GEMINI_FILTER_MODEL} for pre-filter...")
        return gemini.pre_filter(title, content)

    log.warning("   [WARN] No backup available. Proceeding to rewrite for safety.")
    return None

def init_ai() -> tuple['AIKeyManager', Optional['NvidiaFallback'], Optional['MistralFallback'], Optional['OpenRouterMiddle'], Optional['GeminiBackup']]:
    """Initialize Cerebras, NVIDIA, Mistral, OpenRouter, and Gemini managers with multiple API keys."""
    load_dotenv(BASE_DIR / ".env", override=True)
    
    # Cerebras
    c_keys_str = os.getenv("CEREBRAS_API_KEYS", "")
    c_keys = [k.strip() for k in c_keys_str.split(",") if k.strip()]
    if not c_keys:
        log.error("[FATAL] No CEREBRAS_API_KEYS defined in .env!")
        sys.exit(1)
    cerebras_manager = AIKeyManager("CEREBRAS", c_keys, base_url=CEREBRAS_BASE_URL)
    log.info(f"[AI] Cerebras initialized with {len(c_keys)} keys. Primary model: {AI_MODEL}")

    # NVIDIA
    nv_keys_str = os.getenv("Nvidia_AI_Key", "")
    nv_keys = [k.strip() for k in nv_keys_str.split(",") if k.strip()]
    nvidia = None
    if nv_keys:
        try:
            nvidia = NvidiaFallback(nv_keys)
        except Exception as e:
            log.warning(f"[WARN] NVIDIA fallback init failed: {e}")
    else:
        log.warning("[WARN] No Nvidia_AI_Key in .env. Running without NVIDIA fallback.")

    # Mistral
    m_keys_str = os.getenv("Mistral_Key", "") # Key from user .env
    m_keys = [k.strip() for k in m_keys_str.split(",") if k.strip()]
    mistral = None
    if m_keys:
        try:
            mistral = MistralFallback(m_keys)
        except Exception as e:
            log.warning(f"[WARN] Mistral fallback init failed: {e}")
    else:
        log.warning("[WARN] No Mistral_Key in .env. Running without Mistral fallback.")

    # OpenRouter
    or_keys_str = os.getenv("OPENROUTER_API_KEYS", "")
    or_keys = [k.strip() for k in or_keys_str.split(",") if k.strip()]
    openrouter = None
    if or_keys:
        try:
            openrouter = OpenRouterMiddle(or_keys)
        except Exception as e:
            log.warning(f"[WARN] OpenRouter middle fallback init failed: {e}")
    else:
        log.warning("[WARN] No OPENROUTER_API_KEYS in .env. Running without OpenRouter middle fallback.")

    # Gemini
    g_keys_str = os.getenv("GEMINI_API_KEYS", "")
    g_keys = [k.strip() for k in g_keys_str.split(",") if k.strip()]
    gemini = None
    if g_keys:
        try:
            gemini = GeminiBackup(g_keys)
        except Exception as e:
            log.warning(f"[WARN] Gemini backup init failed: {e}. Continuing without backup.")
    else:
        log.warning("[WARN] No GEMINI_API_KEYS in .env. Running without Gemini backup.")

    return cerebras_manager, nvidia, mistral, openrouter, gemini

def rewrite_article(key_manager: 'AIKeyManager', title: str, content: str, category: str, nvidia: Optional['NvidiaFallback'] = None, mistral: Optional['MistralFallback'] = None, openrouter: Optional['OpenRouterMiddle'] = None, gemini: Optional['GeminiBackup'] = None) -> Optional[dict]:
    """
    Send article to Cerebras for analysis and rewriting.
    Falls back to NVIDIA, then Mistral, then OpenRouter Qwen-235B/Free, then Gemini 2.5 Pro if all fail.
    """
    prompt = AI_REWRITE_PROMPT.format(title=title, content=content, category=category)

    cerebras_failed = False
    
    # Quick check: Are all Cerebras keys on cooldown?
    now = time.time()
    any_available = any(i not in key_manager.exhausted and now >= key_manager.cooldowns[i] for i in range(len(key_manager.keys)))
    
    if not any_available:
        log.warning("   [AI] All Cerebras keys on cooldown/exhausted. Skipping to fallbacks.")
        cerebras_failed = True
    else:
        # Try all available keys if needed
        max_attempts = max(MAX_RETRIES, len(key_manager.keys))
        for attempt in range(1, max_attempts + 1):
            client, key_idx = key_manager.get_client()
            
            try:
                response = client.chat.completions.create(
                    model=AI_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=3000,
                    response_format={"type": "json_object"},
                )

                raw_text = response.choices[0].message.content
                if not raw_text:
                    log.warning(f"   [WARN] Empty response (Key #{key_idx}, attempt {attempt})")
                    continue

                result = clean_and_parse_json(raw_text)
                required = {"headline", "summary", "body"}
                if not required.issubset(result.keys()):
                    log.warning(f"   [WARN] Missing keys in response: {required - set(result.keys())}")
                    continue

                # Log thought process
                thought = result.get("thought_process", "No thought process provided.")
                log.info(f"   [THOUGHT] {thought[:200]}...")

                # Convert NEWPARA markers
                if "body" in result:
                    result["body"] = result["body"].replace("NEWPARA", "\n\n").strip()

                log.info(f"   [OK] Rewrite complete (Key #{key_idx}): \"{result.get('headline', '')[:60]}...\"")
                return result

            except Exception as e:
                error_msg = str(e).lower()
                if "429" in error_msg or "rate limit" in error_msg:
                    log.warning(f"   [RATE-LIMIT] Key #{key_idx} hit limit. Switching...")
                    wait_time = 60
                    if "retry-after" in error_msg:
                        try:
                            parts = error_msg.split("retry-after")[-1].split()
                            for p in parts:
                                if p.isdigit():
                                    wait_time = int(p)
                                    break
                        except: pass
                    
                    key_manager.mark_cooldown(key_idx, wait_time)
                    cerebras_failed = True
                    if (nvidia and nvidia.is_available()) or (mistral and mistral.is_available()) or (openrouter and openrouter.is_available()) or (gemini and gemini.is_available()):
                        log.info(f"   [FALLBACK] Backup available — skipping Cerebras cooldown wait.")
                        break
                    continue 
                
                # Handle exhaustion
                if "daily limit" in error_msg:
                    key_manager.mark_exhausted(key_idx)
                    cerebras_failed = True

                    # If backup is available, skip waiting — fallback NOW
                    if (nvidia and nvidia.is_available()) or (mistral and mistral.is_available()) or (openrouter and openrouter.is_available()) or (gemini and gemini.is_available()):
                        log.info(f"   [FALLBACK] Backup available — skipping exhausted Cerebras keys.")
                        break
                    continue

                log.error(f"   [ERROR] Cerebras error (Key #{key_idx}, attempt {attempt}): {e}")
                cerebras_failed = True
                if attempt < MAX_RETRIES:
                    time.sleep(2 * attempt)

    # ── NVIDIA Fallback ──
    if cerebras_failed and nvidia and nvidia.is_available():
        log.info(f"   [FALLBACK] All Cerebras keys failed. Switching to NVIDIA...")
        res = nvidia.rewrite(title, content, category)
        if res: return res

    # ── Mistral Fallback ──
    if cerebras_failed and mistral and mistral.is_available():
        log.info(f"   [FALLBACK] Switching to Mistral...")
        res = mistral.rewrite(title, content, category)
        if res: return res

    # ── OpenRouter Fallback ──
    if cerebras_failed and openrouter and openrouter.is_available():
        log.info(f"   [FALLBACK] Switching to OpenRouter...")
        res = openrouter.rewrite(title, content, category)
        if res: return res

    # ── Gemini Fallback ──
    if cerebras_failed and gemini and gemini.is_available():
        log.info(f"   [FALLBACK] All previous options failed. Switching to {GEMINI_REWRITE_MODEL}...")
        return gemini.rewrite(title, content, category)

    return None

# ─── Supabase Writer ─────────────────────────────────────────────────────────

def save_article_supabase(sb: Client, article_data: dict, rewritten: dict) -> bool:
    """Save the rewritten article to Supabase."""
    url_hash = hashlib.md5(article_data["url"].encode()).hexdigest()[:10]
    
    output = {
        "id": url_hash,
        "original_url": article_data["url"],
        "original_title": article_data["title"],
        "published_date": article_data["published"],
        "image_url": article_data.get("image_url", ""),
        "rewritten_headline": rewritten["headline"],
        "rewritten_summary": rewritten["summary"],
        "rewritten_body": rewritten["body"],
        "tags": rewritten.get("tags", []),
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "source": "AirNews",
        "category": article_data.get("category", "india")
    }

    try:
        sb.table("articles").insert(output).execute()
        log.info(f"   [SAVED] {rewritten['headline'][:50]}... to Supabase")
        return True
    except Exception as e:
        error_str = str(e)
        if "23505" in error_str or "duplicate key" in error_str:
            log.warning(f"   [SKIP] Article already exists in Supabase (Hash collision or duplicate).")
            return True # Return true so it gets tracked as visited
        log.error(f"   [ERROR] Failed to save to Supabase: {e}")
        return False

def save_ignored_article_supabase(sb: Client, article: dict, rewritten: dict):
    """Save an article that was filtered out as non-news to the ignored_articles table."""
    try:
        data = {
            "original_url": article["url"],
            "original_title": article["title"],
            "category": article["category"],
            "thought_process": rewritten.get("thought_process", ""),
            "processed_at": datetime.now(timezone.utc).isoformat()
        }
        sb.table("ignored_articles").upsert(data, on_conflict="original_url").execute()
        log.info(f"   [LOGGED] Recorded ignored article in DB")
    except Exception as e:
        log.error(f"[ERROR] Failed to save to ignored_articles: {e}")

# ─── Data Cleanup ────────────────────────────────────────────────────────────

def cleanup_old_data(sb: Client):
    """Delete articles and visited_urls older than 30 days to maintain DB size."""
    try:
        thirty_days_ago = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        
        sb.table("articles").delete().lt("processed_at", thirty_days_ago).execute()
        sb.table("visited_urls").delete().lt("visited_at", thirty_days_ago).execute()
        sb.table("ignored_articles").delete().lt("processed_at", thirty_days_ago).execute()
    except Exception as e:
        log.error(f"[ERROR] Cleanup failed: {e}")

# ─── Main Processing Loop ───────────────────────────────────────────────────

def process_cycle(key_manager: 'AIKeyManager', sb: Client, tracker: URLTracker, nvidia: Optional['NvidiaFallback'] = None, mistral: Optional['MistralFallback'] = None, openrouter: Optional['OpenRouterMiddle'] = None, gemini: Optional['GeminiBackup'] = None) -> tuple[int, bool]:
    """
    One full processing cycle:
    1. Fetch RSS feed and filter for new content
    2. Scrape full article body
    3. Analyze and rewrite using Cerebras/NVIDIA/Mistral/OpenRouter/Gemini AI (Reasoning Step)
    4. Determine category (India vs International)
    5. Save to Supabase and track usage
    
    Returns (articles_processed, rewrite_model_was_used).
    """
    daily_count = tracker.get_daily_count()
    if daily_count >= DAILY_API_LIMIT:
        log.info(f"[QUOTA] Daily limit reached ({daily_count}/{DAILY_API_LIMIT}). Waiting for reset.")
        return 0, False

    remaining = DAILY_API_LIMIT - daily_count

    articles = fetch_rss_feed()
    if not articles:
        return 0, False

    new_articles = [a for a in articles if not tracker.is_visited(a["url"])]

    if not new_articles:
        log.info("[OK] No new articles to process")
        return 0, False

    log.info(f"[NEW] Found {len(new_articles)} new articles (daily budget: {remaining} remaining)")

    batch = new_articles[:min(MAX_ARTICLES_PER_CYCLE, remaining)]
    processed = 0
    qwen_used = False

    for i, article in enumerate(batch, 1):
        if _shutdown_requested:
            log.info("[STOP] Shutdown requested, stopping processing")
            break

        log.info(f"\n{'='*60}")
        log.info(f"[ARTICLE {i}/{len(batch)}] {article['title'][:70]}...")
        log.info(f"   [URL] {article['url']}")

        # Step 1: Scrape full article content
        content = scrape_article_content(article["url"])

        if not content:
            content = article.get("description", "")
            if len(content) < 50:
                log.warning("   [SKIP] No content available")
                tracker.mark_visited(article["url"]) 
                continue
            log.info("   [INFO] Using RSS description as fallback content")

        # Step 2: Pre-filter with smaller model (Cerebras Llama → NVIDIA → Mistral → OpenRouter Llama → Gemini Flash fallback)
        filter_result = pre_filter_article(key_manager, article["title"], content, nvidia=nvidia, mistral=mistral, openrouter=openrouter, gemini=gemini)
        
        if filter_result:
            raw_is_news = filter_result.get("is_news", True)
            if isinstance(raw_is_news, str):
                is_news = str(raw_is_news).strip().lower() == "true"
            else:
                is_news = bool(raw_is_news)

            if not is_news:
                log.info(f"   [IGNORE] Non-news detected by pre-filter. Skipping...")
                save_ignored_article_supabase(sb, article, {"thought_process": filter_result.get("reason", "")})
                tracker.mark_visited(article["url"])
                # SAFETY DELAY: Prevents hitting Llama 30 RPM limit if batch is full of junk
                time.sleep(2) 
                continue 
            
            article["category"] = filter_result.get("category", "india")

        # Step 3: Rewrite with high-quality AI (Cerebras Qwen → NVIDIA → Mistral → OpenRouter Qwen → Gemini Pro fallback)
        qwen_used = True 
        rewritten = rewrite_article(key_manager, article["title"], content, article["category"], nvidia=nvidia, mistral=mistral, openrouter=openrouter, gemini=gemini)
        
        if not rewritten:
            log.warning("   [SKIP] Rewrite failed. Breaking cycle to protect Qwen rate limits.")
            break 

        # Step 4: Save to Supabase
        save_article_supabase(sb, article, rewritten)

        # Step 5: Track and End Cycle
        tracker.mark_visited(article["url"])
        tracker.increment_daily_count()
        processed += 1
        
        log.info(f"   [DONE] News processed. Ending cycle to respect 1 RPM limit.")
        break

    cleanup_old_data(sb)

    return processed, qwen_used


def main():
    """Main entry point - runs the agent loop 24/7."""

    log.info("=" * 60)
    log.info(">>> AirNews AI Agent (V10.3 - Cerebras + NVIDIA + Mistral + OpenRouter + Gemini) Starting")
    log.info("=" * 60)
    
    log.info(f"[CONFIG] Poll interval: {POLL_INTERVAL_SECONDS}s ({POLL_INTERVAL_SECONDS//60} min)")
    log.info(f"[CONFIG] Daily limit:   {DAILY_API_LIMIT} articles")
    log.info(f"[CONFIG] Per-cycle max: {MAX_ARTICLES_PER_CYCLE} articles")
    log.info(f"[CONFIG] Fallback chain: Cerebras -> NVIDIA -> Mistral -> OpenRouter -> Gemini")
    log.info("")

    key_manager, nvidia, mistral, openrouter, gemini = init_ai()
    sb = init_supabase()
    tracker = URLTracker(sb)

    log.info(f"[STATS] Previously processed: {tracker.total_articles} articles")
    log.info(f"[STATS] Today's API calls:    {tracker.get_daily_count()}/{DAILY_API_LIMIT}")
    log.info("")

    cycle_count = 0

    while not _shutdown_requested:
        cycle_count += 1
        log.info(f"\n{'---'*20}")
        log.info(f"[CYCLE #{cycle_count}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log.info(f"   Daily API usage: {tracker.get_daily_count()}/{DAILY_API_LIMIT}")
        log.info(f"{'---'*20}")

        try:
            processed, qwen_used = process_cycle(key_manager, sb, tracker, nvidia=nvidia, mistral=mistral, openrouter=openrouter, gemini=gemini)
            log.info(f"[DONE] Cycle #{cycle_count}: {processed} articles processed")
        except Exception as e:
            log.error(f"[ERROR] Cycle #{cycle_count}: {e}")
            log.error(traceback.format_exc())
            qwen_used = True 

        if _shutdown_requested:
            break

        daily_count = tracker.get_daily_count()
        if daily_count >= DAILY_API_LIMIT:
            now = datetime.now(timezone.utc)
            tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0)
            if tomorrow <= now:
                tomorrow = tomorrow + timedelta(days=1)
            sleep_time = min((tomorrow - now).total_seconds() + 60, 3600)
            log.info(f"[SLEEP] Daily limit reached. Sleeping {int(sleep_time)}s until quota reset...")
        else:
            sleep_time = POLL_INTERVAL_SECONDS if qwen_used else 30
            model_waited = "Qwen (60s)" if qwen_used else "Llama Scanner (30s)"
            log.info(f"[SLEEP] Next check in {sleep_time}s for {model_waited}...")

        for _ in range(int(sleep_time)):
            if _shutdown_requested:
                break
            time.sleep(1)

    log.info("\n" + "=" * 60)
    log.info(">>> Agent shut down gracefully")
    log.info(f"[STATS] Total articles processed: {tracker.total_articles}")
    log.info("=" * 60)

if __name__ == "__main__":
    main()
