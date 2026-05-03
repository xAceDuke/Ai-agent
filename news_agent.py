"""
AirNews AI Agent (V8.0 - Cerebras Migration)
========================================================
Fetches RSS feeds from TOI, Times Now, NDTV, and The Hindu, scrapes full articles,
rewrites them via Cerebras AI (Qwen 2.5 70B / 250B), and saves to Supabase.
This version enforces a strict 'Hard News Only' policy for Indian content,
blocking speculative political commentary, opinions, and features.

Key Features:
- Reasoning-First Architecture: Forces the AI to 'think' and disambiguate facts 
  before writing.
- Cerebras Integration: High-speed inference using the OpenAI-compatible SDK.
- Strict Pacing: Enforces 1 RPM limit to stay within free-tier quotas.
- Multi-Source Integration: Scrapes and parses content from major Indian news outlets.
- Strict India News Filter: Automatically skips non-news content for the India category.

Designed to run 24/7 within Cerebras API limits.
"""

import os
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

import feedparser
from curl_cffi import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from openai import OpenAI

from supabase import create_client, Client

# ─── Configuration ───────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
LOG_FILE = BASE_DIR / "agent.log"

RSS_FEED_URLS = [
    {"url": "https://timesofindia.indiatimes.com/rssfeeds/-2128936835.cms", "category": "india", "name": "TOI India"},
    {"url": "https://timesofindia.indiatimes.com/rssfeeds/296589292.cms", "category": "international", "name": "TOI International"},
    {"url": "https://www.timesnownews.com/feeds/gns-en-india.xml", "category": "india", "name": "Times Now India"},
    {"url": "https://feeds.feedburner.com/ndtvnews-india-news", "category": "india", "name": "NDTV India"},
    {"url": "https://feeds.feedburner.com/ndtvnews-world-news", "category": "international", "name": "NDTV World"},
    {"url": "https://www.thehindu.com/news/national/feeder/default.rss", "category": "india", "name": "The Hindu National"},
    {"url": "https://www.thehindu.com/news/international/feeder/default.rss", "category": "international", "name": "The Hindu International"}
]
POLL_INTERVAL_SECONDS = 180          # 3 minutes between cycles (Strict 1 RPM)
REQUEST_DELAY_SECONDS = 0            # No delay needed within cycle (1 article per cycle)
MAX_ARTICLES_PER_CYCLE = 1          # Only process 1 article at a time
DAILY_API_LIMIT = 1000               # Increased to reset blockage (previously 250)
MAX_RETRIES = 3                      # retries on transient errors
ARTICLE_FETCH_TIMEOUT = 15           # seconds for HTTP requests

# ─── Cerebras API Key Pool ────────────────────────────────────────────────────────
# Keys are loaded from .env (CEREBRAS_API_KEYS as a comma-separated list)

# Browser-like headers for article scraping
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
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

SYSTEM_PROMPT = """You are an elite senior news editor for AirNews, renowned for deep situational analysis and incisive reporting. Your goal is to produce reports that provide not just the facts, but the broader context and potential future implications.

EDITORIAL PRINCIPLES:
1. PROFESSIONAL CLARITY: Write in a sophisticated but accessible style. Use precise vocabulary but avoid overly dense or academic language.
2. BALANCED TONE: Maintain the objectivity of a global news agency while providing sharp, analytical insights.
3. GEOPOLITICAL DISAMBIGUATION: Always verify the identity of people mentioned. For example, if 'Stalin' is mentioned in the context of Tamil Nadu or India, it refers to the Chief Minister M.K. Stalin, NOT the historical Soviet leader Joseph Stalin.
4. GEOGRAPHICAL CONTEXT: Carefully distinguish between domestic (Indian) and international news based on locations, names, and organizations.
5. DEEP REASONING & PROJECTION: You must think holistically to understand the full scope of the situation. Identify:
   - Core Facts: Key entities, roles, locations, and the primary event.
   - Contextual Links: How does this relate to historical events or current global trends?
   - Logical Projections: What are the likely next steps, future implications, or potential outcomes of this news?
   - Situational Nuance: What are the secondary effects on society, markets, or geopolitics?
6. FACT FILTERING & NOISE REDUCTION: If you detect that the source article contains factual errors, logical contradictions, or clearly incorrect data, do NOT include those errors. Intelligently filter out the "noise" and focus only on the verified core of the story.
7. HARD NEWS FOCUS (INDIA): For Indian news, you must be extremely strict. Differentiate between a 'Hard News' event (e.g., 'A law was enacted', 'A major accident occurred', 'A policy was announced') and 'Articles/Features/Commentary' (e.g., 'The significance of X', 'Why Y happened', 'Opinions on Z'). Only classify factual, time-sensitive events as 'is_news' for India.
8. INTERNATIONAL FLEXIBILITY: For international news, you may allow slightly broader features or analytical articles if they provide significant global context, but still prioritize hard news.
"""

AI_REWRITE_PROMPT = """Perform an advanced editorial analysis of the provided news article and rewrite it into a comprehensive report.

Your task is to produce a version that is authoritative, objective, and deeply insightful. It should provide a clear narrative of the events while situating them within their broader context and highlighting potential future outcomes.

EDITORIAL REQUIREMENTS:
1. Content Classification: Set 'is_news' to true ONLY if the content reports on a verified factual event that has already occurred or an official announcement/data release. This is critical for Indian content. Set to false (Article) for:
   - Speculative political analysis (e.g., 'Entry sparks uncertainty', 'Future of X in doubt').
   - Opinion pieces, editorials, and commentary.
   - Stories focused on 'buzz', sentiment, or 'what-if' scenarios without a major new event.
   - Lifestyle, listicles, advice, or general interest features.
2. Balanced Vocabulary: Use professional, sophisticated English. Aim for clarity and precision.
3. Narrative Flow & Outlook: Use the 'inverted pyramid' style for the lead, but ensure the closing sections provide an outlook on future implications or the logical next steps in the situation.
4. Zero AI Clichés: DO NOT use repetitive AI phrases like "delving into," "testament to," "moreover," or "in conclusion."
5. Fact Integrity: Ensure 100% factual accuracy. If a detail in the source seems incorrect or suspicious, omit it.
6. Format: Create a strong, professional headline and a 2-3 sentence executive summary. The body should consist of 4+ distinct paragraphs separated by "NEWPARA".

You MUST return ONLY a valid JSON object.
Structure:
{{
  "is_news": boolean, (Strictly true for timely News, false for Features, Opinions, or general Articles. Be especially strict for India category)
  "category": "india" or "international", (Determine based on the locations and entities mentioned)
  "thought_process": "Your deep analysis of the situation: Core facts, historical/global context, logical projections, and plan for an insightful rewrite.",
  "headline": "Professional headline",
  "summary": "Executive summary with contextual insight",
  "body": "Paragraph 1 NEWPARA Paragraph 2 NEWPARA ... (Ensure the final paragraph provides a situational outlook)",
  "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"]
}}

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
            # Fetch only recent visited URLs to avoid infinite memory growth
            thirty_days_ago = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
            response = self.sb.table("visited_urls").select("url").gte("visited_at", thirty_days_ago).execute()
            for row in response.data:
                self.visited.add(row["url"])
            log.info(f"[TRACKER] Loaded {len(self.visited)} recent visited URLs from Supabase.")
        except Exception as e:
            log.error(f"[ERROR] Failed to load visited URLs from Supabase: {e}")

    def is_visited(self, url: str) -> bool:
        return url in self.visited

    def mark_visited(self, url: str):
        self.visited.add(url)
        try:
            self.sb.table("visited_urls").insert({"url": url}).execute()
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
                # Clean URL — remove tracking fragments
                raw_link = entry.get("link", "")
                clean_url = raw_link.split("#")[0].strip()
                
                try:
                    from urllib.parse import urlparse
                    path_parts = urlparse(clean_url).path.strip("/").split("/")
                    root_section = path_parts[0].lower() if path_parts else ""
                except Exception:
                    root_section = ""

                # Real trash paths identified directly from the live RSS feeds
                # Filtering down to pure news (excluding health, offbeat, features, opinions, etc.)
                live_trash_sections = {
                    "health", "offbeat", "feature", "opinion", "business", 
                    "business-economy", "education", "entertainment", "sports",
                    "cricket", "lifestyle", "astrology", "movies", "tv"
                }
                
                if not clean_url or root_section in live_trash_sections or "/health/" in clean_url.lower() or "/offbeat/" in clean_url.lower():
                    continue

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

                # All articles use AI category determination
                article_category = "needs_ai"

                all_articles.append({
                    "url": clean_url,
                    "title": entry.get("title", "").strip(),
                    "description": clean_desc,
                    "published": entry.get("published", ""),
                    "image_url": image_url,
                    "category": article_category
                })

            log.info(f"[RSS] Fetched {len(feed.entries)} articles from {feed_name} feed")

        except Exception as e:
            log.error(f"[ERROR] RSS fetch failed for {feed_url}: {e}")
            
    return all_articles

# ─── Article Scraper ────────────────────────────────────────────────────────

def scrape_article_content(url: str) -> Optional[str]:
    """
    Scrape full article text from a Times of India article page.
    Uses multiple fallback selectors to handle layout variations.
    """
    try:
        response = requests.get(
            url, impersonate="chrome110", timeout=ARTICLE_FETCH_TIMEOUT
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        content_text = ""

        # Strategy 0: Extract from JSON-LD structured data (most reliable for TOI)
        # TOI embeds full article text in <script type="application/ld+json">
        for ld_script in soup.find_all("script", type="application/ld+json"):
            try:
                ld_data = json.loads(ld_script.text or "")
                # Handle both single object and array
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
            # Truncate very long articles to save tokens
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
        # Check if e has response and status_code for HTTPError equivalent
        if hasattr(e, "response") and hasattr(e.response, "status_code"):
            log.warning(f"   [WARN] HTTP {e.response.status_code} for {url}")
        else:
            log.warning(f"   [WARN] Request failed for {url}: {e}")
        return None

# ─── Cerebras AI Rewriter ─────────────────────────────────────────────────────

AI_MODEL = "qwen-3-235b-a22b-instruct-2507"
CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"

class AIKeyManager:
    """Manages a pool of Cerebras API keys with intelligent rotation."""
    
    def __init__(self, keys: list[str]):
        self.keys = keys
        self.clients = [OpenAI(api_key=k, base_url=CEREBRAS_BASE_URL) for k in keys]
        self.current_index = 0
        self.cooldowns = {i: 0 for i in range(len(keys))}
        self.exhausted = set()

    def get_client(self) -> tuple[OpenAI, int]:
        """Returns the next available client and its index."""
        now = time.time()
        
        for _ in range(len(self.keys)):
            idx = self.current_index
            if idx not in self.exhausted and now >= self.cooldowns[idx]:
                return self.clients[idx], idx
            
            self.current_index = (self.current_index + 1) % len(self.keys)
        
        best_idx = min(self.cooldowns.keys(), key=lambda i: self.cooldowns[i])
        wait_time = max(0, self.cooldowns[best_idx] - now)
        if wait_time > 0:
            log.info(f"[KEY-MANAGER] All keys on cooldown. Waiting {int(wait_time)}s for Key #{best_idx}...")
            time.sleep(wait_time)
        
        return self.clients[best_idx], best_idx

    def mark_cooldown(self, index: int, seconds: int = 60):
        """Put a key on cooldown (e.g., after a 429 error)."""
        self.cooldowns[index] = time.time() + seconds
        log.warning(f"[KEY-MANAGER] Key #{index} put on cooldown for {seconds}s")
        self.current_index = (self.current_index + 1) % len(self.keys)

    def mark_exhausted(self, index: int):
        """Mark a key as exhausted for the day."""
        self.exhausted.add(index)
        log.error(f"[KEY-MANAGER] Key #{index} EXHAUSTED for the day.")

def init_ai() -> AIKeyManager:
    """Initialize Cerebras Key Manager with keys from .env."""
    load_dotenv(BASE_DIR / ".env", override=True)
    keys_str = os.getenv("CEREBRAS_API_KEYS", "")
    
    keys = [k.strip() for k in keys_str.split(",") if k.strip()]
    
    if not keys:
        log.error("[FATAL] No CEREBRAS_API_KEYS defined in .env!")
        sys.exit(1)
        
    manager = AIKeyManager(keys)
    log.info(f"[AI] Initialized with {len(keys)} keys from .env. Model: {AI_MODEL}")
    return manager


def rewrite_article(key_manager: AIKeyManager, title: str, content: str) -> Optional[dict]:
    """
    Send article to Cerebras for analysis and rewriting.
    Uses strict rate-limit handling and JSON response extraction.
    """
    prompt = AI_REWRITE_PROMPT.format(title=title, content=content)

    for attempt in range(1, MAX_RETRIES + 1):
        client, key_idx = key_manager.get_client()
        
        try:
            # Using OpenAI SDK standard call
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

            result = json.loads(raw_text)

            # Validate required keys
            required = {"headline", "summary", "body", "is_news", "category"}
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
            
            # Handle rate limiting (429)
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
                # Retry immediately with a different key
                continue 
            
            # Handle exhaustion (e.g., if we know RPD is 0)
            if "daily limit" in error_msg:
                key_manager.mark_exhausted(key_idx)
                continue

            log.error(f"   [ERROR] Groq error (Key #{key_idx}, attempt {attempt}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(2 * attempt)

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
        log.error(f"   [ERROR] Failed to save to ignored_articles: {e}")

# ─── Data Cleanup ────────────────────────────────────────────────────────────

def cleanup_old_data(sb: Client):
    """Delete articles and visited_urls older than 30 days to maintain DB size."""
    try:
        thirty_days_ago = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        
        # We don't have exact counts of deleted rows via supabase-py easily without returning them,
        # but we can execute the delete and rely on Supabase.
        sb.table("articles").delete().lt("processed_at", thirty_days_ago).execute()
        sb.table("visited_urls").delete().lt("visited_at", thirty_days_ago).execute()
        sb.table("ignored_articles").delete().lt("processed_at", thirty_days_ago).execute()
    except Exception as e:
        log.error(f"[ERROR] Cleanup failed: {e}")

# ─── Main Processing Loop ───────────────────────────────────────────────────

def process_cycle(key_manager: AIKeyManager, sb: Client, tracker: URLTracker) -> int:
    """
    One full processing cycle:
    1. Fetch RSS feed and filter for new content
    2. Scrape full article body
    3. Analyze and rewrite using Cerebras AI (Reasoning Step)
    4. Determine category (India vs International)
    5. Save to Supabase and track usage
    
    Returns number of articles processed.
    """
    # Check daily quota
    daily_count = tracker.get_daily_count()
    if daily_count >= DAILY_API_LIMIT:
        log.info(f"[QUOTA] Daily limit reached ({daily_count}/{DAILY_API_LIMIT}). Waiting for reset.")
        return 0

    remaining = DAILY_API_LIMIT - daily_count

    # Fetch RSS
    articles = fetch_rss_feed()
    if not articles:
        return 0

    # Filter new articles
    new_articles = [a for a in articles if not tracker.is_visited(a["url"])]

    if not new_articles:
        log.info("[OK] No new articles to process")
        return 0

    log.info(f"[NEW] Found {len(new_articles)} new articles (daily budget: {remaining} remaining)")

    # Limit per cycle
    batch = new_articles[:min(MAX_ARTICLES_PER_CYCLE, remaining)]
    processed = 0

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
            # Fallback: use RSS description if scraping fails
            content = article.get("description", "")
            if len(content) < 50:
                log.warning("   [SKIP] No content available")
                tracker.mark_visited(article["url"])  # mark to avoid retrying
                continue
            log.info("   [INFO] Using RSS description as fallback content")

        # Step 2: Rewrite with AI
        rewritten = rewrite_article(key_manager, article["title"], content)

        if not rewritten:
            log.warning("   [SKIP] Rewrite failed")
            # Don't mark as visited so we can retry next cycle
            continue

        # Use the AI-determined category
        ai_determined_cat = rewritten.get("category", "india").lower()
        if "international" in ai_determined_cat or "world" in ai_determined_cat:
            article["category"] = "international"
        else:
            article["category"] = "india"

        # Filter out non-news articles ONLY if category is india
        if not rewritten.get("is_news", True) and article["category"] == "india":
            log.info(f"   [IGNORE] Indian 'Article/Feature' detected. Skipping...")
            save_ignored_article_supabase(sb, article, rewritten)
            tracker.mark_visited(article["url"])
            continue

        if not rewritten.get("is_news", True) and article["category"] == "international":
            log.info(f"   [PASS] International Article/Feature detected. Allowing per updated rules.")


        # Step 3: Save to Supabase
        save_article_supabase(sb, article, rewritten)

        # Step 4: Track
        tracker.mark_visited(article["url"])
        tracker.increment_daily_count()
        processed += 1

        # Respect rate limits — wait between API calls
        if i < len(batch):
            log.info(f"   [WAIT] {REQUEST_DELAY_SECONDS}s before next article...")
            time.sleep(REQUEST_DELAY_SECONDS)

    # Clean up old data automatically (every cycle)
    cleanup_old_data(sb)

    return processed


def main():
    """Main entry point - runs the agent loop 24/7."""

    log.info("=" * 60)
    log.info(">>> AirNews Cerebras AI Agent (V8.0) Starting")
    log.info("=" * 60)
    
    log.info(f"[CONFIG] Poll interval: {POLL_INTERVAL_SECONDS}s ({POLL_INTERVAL_SECONDS//60} min)")
    log.info(f"[CONFIG] Daily limit:   {DAILY_API_LIMIT} articles")
    log.info(f"[CONFIG] Per-cycle max: {MAX_ARTICLES_PER_CYCLE} articles")
    log.info("")

    # Initialize
    key_manager = init_ai()
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
            processed = process_cycle(key_manager, sb, tracker)
            log.info(f"[DONE] Cycle #{cycle_count}: {processed} articles processed")
        except Exception as e:
            log.error(f"[ERROR] Cycle #{cycle_count}: {e}")
            log.error(traceback.format_exc())

        if _shutdown_requested:
            break

        # Smart sleep — longer if daily limit is reached
        daily_count = tracker.get_daily_count()
        if daily_count >= DAILY_API_LIMIT:
            # Calculate seconds until midnight UTC
            now = datetime.now(timezone.utc)
            tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0)
            if tomorrow <= now:
                tomorrow = tomorrow.replace(day=now.day + 1)
            sleep_time = min((tomorrow - now).total_seconds() + 60, 3600)
            log.info(f"[SLEEP] Daily limit reached. Sleeping {int(sleep_time)}s until quota reset...")
        else:
            sleep_time = POLL_INTERVAL_SECONDS
            log.info(f"[SLEEP] Next check in {sleep_time}s...")

        # Interruptible sleep (check shutdown flag every second)
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
