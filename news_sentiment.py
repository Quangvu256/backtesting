import gzip
import json
import logging
import re
from datetime import datetime
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import feedparser
import requests

from config import LMSTUDIO_BASE_URL, LMSTUDIO_MODEL, LLM_BACKEND, GEMINI_API_KEY
from database import DatabaseManager

logger = logging.getLogger("NewsSentiment")

SYMBOL_KEYWORDS = {
    "VNM": [r"\bvnm\b", "vinamilk", "sua viet nam", "sua viet", "vietnam dairy"],
    "FPT": [r"\bfpt\b", "tap doan fpt", "cong nghe fpt", "fpt software", "fpt digital", "fpt telecom"],
    "HPG": [r"\bhpg\b", "hoa phat", "thep hoa phat", "tran dinh long"],
    "VIC": [r"\bvic\b", "vingroup", "tap doan vic", "vinfast", "vinhomes", "vinpearl", "pham nhat vuong"],
    "VCB": [r"\bvcb\b", "vietcombank", "ngoai thuong viet nam", "ngan hang ngoai thuong"],
}

RSS_FEEDS = [
    {"source": "CafeF", "url": "https://cafef.vn/doanh-nghiep.rss"},
    {"source": "CafeF", "url": "https://cafef.vn/thi-truong-chung-khoan.rss"},
    {"source": "CafeF", "url": "https://cafef.vn/tai-chinh-ngan-hang.rss"},
    {"source": "CafeF", "url": "https://cafef.vn/vi-mo-dau-tu.rss"},
    {"source": "CafeF", "url": "https://cafef.vn/smart-money.rss"},
    {"source": "Vietstock", "url": "https://vietstock.vn/830/chung-khoan/co-phieu.rss"},
    {"source": "Vietstock", "url": "https://vietstock.vn/737/doanh-nghiep/hoat-dong-kinh-doanh.rss"},
    {"source": "Vietstock", "url": "https://vietstock.vn/757/tai-chinh/ngan-hang.rss"},
    {"source": "Vietstock", "url": "https://vietstock.vn/761/kinh-te/vi-mo.rss"},
    {"source": "Vietstock", "url": "https://vietstock.vn/1636/nhan-dinh-phan-tich/nhan-dinh-thi-truong.rss"},
    {"source": "Vietstock", "url": "https://vietstock.vn/582/nhan-dinh-phan-tich/phan-tich-co-ban.rss"},
    {"source": "VnEconomy", "url": "https://vneconomy.vn/tai-chinh.rss"},
    {"source": "VnEconomy", "url": "https://vneconomy.vn/chung-khoan.rss"},
    {"source": "VnEconomy", "url": "https://vneconomy.vn/nhip-cau-doanh-nghiep.rss"},
    {"source": "VnEconomy", "url": "https://vneconomy.vn/tieu-diem.rss"},
    {"source": "VnExpress", "url": "https://vnexpress.net/rss/kinh-doanh.rss"},
]

HTML_NEWS_PAGES = [
    {"source": "Tin nhanh chung khoan", "url": "https://www.tinnhanhchungkhoan.vn/"},
    {"source": "Tin nhanh chung khoan", "url": "https://www.tinnhanhchungkhoan.vn/chung-khoan/"},
    {"source": "Tin nhanh chung khoan", "url": "https://www.tinnhanhchungkhoan.vn/dai-hoi-co-dong/"},
    {"source": "Tin nhanh chung khoan", "url": "https://www.tinnhanhchungkhoan.vn/chuyen-kinh-doanh/"},
    {"source": "Tin nhanh chung khoan", "url": "https://www.tinnhanhchungkhoan.vn/thuong-truong/"},
]


class NewsSentimentAnalyzer:
    def __init__(self, db_manager=None):
        self.db_manager = db_manager if db_manager else DatabaseManager()
        self.api_enabled = False
        self.llm_backend = LLM_BACKEND  # 'lmstudio' hoặc 'gemini'

        if self.llm_backend == "lmstudio":
            self.lmstudio_url = LMSTUDIO_BASE_URL.rstrip("/") + "/chat/completions"
            self.lmstudio_model = LMSTUDIO_MODEL or ""  # rỗng = dùng model đang load
            try:
                # Health check: thử GET /v1/models
                models_url = LMSTUDIO_BASE_URL.rstrip("/") + "/models"
                resp = requests.get(models_url, timeout=5)
                if resp.status_code == 200:
                    models_data = resp.json().get("data", [])
                    loaded = [m.get("id", "unknown") for m in models_data[:3]]
                    if not self.lmstudio_model and loaded:
                        self.lmstudio_model = loaded[0]  # Auto-pick model đầu tiên
                    self.api_enabled = True
                    logger.info(f"[FACT] LMStudio server OK. Models: {loaded}. Using: {self.lmstudio_model}")
                else:
                    logger.warning(f"[WARNING] LMStudio server returned {resp.status_code}. Fallback rule-based.")
            except requests.ConnectionError:
                logger.warning("[WARNING] Khong ket noi duoc LMStudio server. Dung rule-based fallback.")
            except Exception as e:
                logger.error(f"[ERROR] LMStudio health check loi: {e}. Dung rule-based fallback.")
        elif self.llm_backend == "gemini":
            # Legacy Gemini path
            if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=GEMINI_API_KEY)
                    self.model = genai.GenerativeModel("gemma4-26b")
                    self.api_enabled = True
                    logger.info("[FACT] Da cau hinh thanh cong Gemini API de phan tich tin tuc.")
                except Exception as e:
                    logger.error(f"[ERROR] Khong the khoi tao Gemini API: {e}. Dung rule-based fallback.")
            else:
                logger.warning("[WARNING] Thieu GEMINI_API_KEY. Dung rule-based sentiment analyzer.")
        else:
            logger.warning(f"[WARNING] LLM_BACKEND='{self.llm_backend}' khong hop le. Dung rule-based.")

    def log_api_call(self, prompt_tokens, completion_tokens, status, error_message=None):
        query = """
        INSERT INTO gemini_api_log (timestamp, prompt_tokens, completion_tokens, status, error_message)
        VALUES (?, ?, ?, ?, ?)
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with self.db_manager.get_connection() as conn:
                conn.execute(query, (now, prompt_tokens, completion_tokens, status, error_message))
                conn.commit()
        except Exception as e:
            logger.error(f"Loi khi luu API log: {e}")

    def _call_lmstudio(self, prompt):
        """Gọi LMStudio OpenAI-compatible /v1/chat/completions endpoint."""
        payload = {
            "model": self.lmstudio_model,
            "messages": [
                {"role": "system", "content": "Ban la chuyen gia phan tich tai chinh chung khoan Viet Nam. Luon tra loi bang JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 512,
        }
        resp = requests.post(
            self.lmstudio_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        content = choice["message"]["content"].strip()
        usage = data.get("usage", {})
        return content, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

    def clean_html(self, text):
        if not text:
            return ""
        text = re.sub(r"<.*?>", " ", str(text))
        return re.sub(r"\s+", " ", text).strip()

    def strip_vietnamese_accents(self, text):
        mapping = {
            "àáạảãâầấậẩẫăằắặẳẵ": "a",
            "èéẹẻẽêềếệểễ": "e",
            "ìíịỉĩ": "i",
            "òóọỏõôồốộổỗơờớợởỡ": "o",
            "ùúụủũưừứựửữ": "u",
            "ỳýỵỷỹ": "y",
            "đ": "d",
        }
        result = str(text).lower()
        for chars, repl in mapping.items():
            for ch in chars:
                result = result.replace(ch, repl)
        return result

    def filter_article_for_symbols(self, title, summary):
        full_text = self.strip_vietnamese_accents(f"{title} {summary}")
        matched_symbols = []

        for symbol, patterns in SYMBOL_KEYWORDS.items():
            for pattern in patterns:
                if re.search(pattern, full_text):
                    matched_symbols.append(symbol)
                    break

        return matched_symbols

    def parse_entry_date(self, entry):
        parsed = entry.get("published_parsed") or entry.get("updated_parsed")
        if parsed:
            try:
                return datetime(*parsed[:6]).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def normalize_feed_entry(self, entry, source):
        title = self.clean_html(entry.get("title", ""))
        summary = self.clean_html(entry.get("summary", "") or entry.get("description", ""))
        url = entry.get("link", "")

        if not title or not url:
            return None

        return {
            "source": source,
            "title": title,
            "summary": summary,
            "url": url,
            "pub_date": self.parse_entry_date(entry),
        }

    def fetch_rss_entries(self, feed_meta):
        feed = feedparser.parse(feed_meta["url"])
        logger.info(f"Doc RSS {feed_meta['source']}: {feed_meta['url']} - {len(feed.entries)} bai.")
        return [
            article
            for article in (self.normalize_feed_entry(entry, feed_meta["source"]) for entry in feed.entries)
            if article
        ]

    def fetch_html_entries(self, page_meta, limit=30):
        request = Request(
            page_meta["url"],
            headers={"User-Agent": "Mozilla/5.0 (compatible; BacktestingNewsBot/1.0)"},
        )

        try:
            with urlopen(request, timeout=15) as response:
                raw_html = response.read()
                if raw_html.startswith(b"\x1f\x8b"):
                    raw_html = gzip.decompress(raw_html)
                html = raw_html.decode(response.headers.get_content_charset() or "utf-8", errors="ignore")
        except Exception as e:
            logger.warning(f"[UNVERIFIED] Khong doc duoc HTML source {page_meta['url']}: {e}")
            return []

        article_pattern = re.compile(
            r'<a[^>]+href=["\'](?P<href>[^"\']*post\d+\.html[^"\']*)["\'][^>]*>(?P<title>.*?)</a>',
            re.IGNORECASE | re.DOTALL,
        )

        articles = []
        seen_urls = set()
        for match in article_pattern.finditer(html):
            url = urljoin(page_meta["url"], match.group("href"))
            title = self.clean_html(match.group("title"))

            if not title or url in seen_urls:
                continue

            seen_urls.add(url)
            articles.append({
                "source": page_meta["source"],
                "title": title,
                "summary": title,
                "url": url,
                "pub_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

            if len(articles) >= limit:
                break

        logger.info(f"Doc HTML {page_meta['source']}: {page_meta['url']} - {len(articles)} bai.")
        return articles

    def analyze_sentiment_rule_based(self, title, content):
        text = self.strip_vietnamese_accents(f"{title} {content}")
        positive_words = [
            "tang truong", "vuot ke hoach", "lai lon", "loi nhuan tang", "ky luc",
            "khoi sac", "hop tac", "phat trien", "mo rong", "dai hoi thanh cong",
            "chia co tuc", "dat doanh thu", "mua vao", "khuyen nghi mua", "but pha",
        ]
        negative_words = [
            "sut giam", "thua lo", "giam sau", "bi phat", "vi pham", "thanh tra",
            "canh bao", "dinh chi", "ban thao", "khuyen nghi ban", "no xau",
            "lo rong", "tram lang", "anh huong tieu cuc", "khieu nai", "giam manh",
        ]

        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)

        if pos_count > neg_count:
            return {
                "sentiment_label": "GOOD",
                "sentiment_score": min(0.1 * (pos_count - neg_count), 1.0),
                "sentiment_reason": f"Rule-based: {pos_count} positive, {neg_count} negative keywords.",
                "confidence": 0.5,
                "impact_sectors": [],
                "time_horizon": "SHORT",
            }
        if neg_count > pos_count:
            return {
                "sentiment_label": "BAD",
                "sentiment_score": max(-0.1 * (neg_count - pos_count), -1.0),
                "sentiment_reason": f"Rule-based: {neg_count} negative, {pos_count} positive keywords.",
                "confidence": 0.5,
                "impact_sectors": [],
                "time_horizon": "SHORT",
            }

        return {
            "sentiment_label": "NEUTRAL",
            "sentiment_score": 0.0,
            "sentiment_reason": "Rule-based: khong co keyword sentiment ro rang.",
            "confidence": 0.5,
            "impact_sectors": [],
            "time_horizon": "SHORT",
        }

    def analyze_sentiment_llm(self, symbol, title, content):
        """Phân tích sentiment bằng LLM (LMStudio hoặc Gemini). Auto-fallback rule-based."""
        if not self.api_enabled:
            return self.analyze_sentiment_rule_based(title, content)

        prompt = f"""Hay phan tich tac dong cua tin tuc sau doi voi ma co phieu '{symbol}'.

Tieu de: {title}
Tom tat: {content}

Tra ve dung mot JSON object:
{{
  "sentiment_label": "GOOD" hoac "BAD" hoac "NEUTRAL",
  "sentiment_score": so thuc tu -1.0 den 1.0,
  "sentiment_reason": "giai thich ngan gon bang tieng Viet",
  "confidence": so thuc tu 0.0 den 1.0,
  "impact_sectors": ["cac nganh lien quan"],
  "time_horizon": "SHORT" hoac "MEDIUM" hoac "LONG"
}}"""

        try:
            if self.llm_backend == "lmstudio":
                raw_text, prompt_tokens, completion_tokens = self._call_lmstudio(prompt)
                # Parse JSON — LMStudio có thể trả về markdown code block
                json_match = re.search(r'```(?:json)?\s*(.+?)\s*```', raw_text, re.DOTALL)
                json_str = json_match.group(1) if json_match else raw_text
                result_json = json.loads(json_str.strip())
            else:
                # Legacy Gemini path
                response = self.model.generate_content(
                    prompt,
                    generation_config={"response_mime_type": "application/json"},
                )
                result_json = json.loads(response.text.strip())
                usage = getattr(response, "usage_metadata", None)
                prompt_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
                completion_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0

            self.log_api_call(prompt_tokens, completion_tokens, "SUCCESS")

            result_json["sentiment_score"] = max(
                -1.0,
                min(1.0, float(result_json.get("sentiment_score", 0.0))),
            )
            return result_json
        except Exception as e:
            logger.error(f"[ERROR] LLM ({self.llm_backend}) loi cho {symbol}: {e}. Fallback rule-based.")
            self.log_api_call(0, 0, "FAILED", str(e))
            return self.analyze_sentiment_rule_based(title, content)

    def save_article(self, symbol, article, sentiment):
        query = """
        INSERT OR IGNORE INTO news_articles (
            symbol, title, source, pub_date, content, url,
            sentiment_label, sentiment_score, sentiment_reason, analyzed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute(
                    query,
                    (
                        symbol,
                        article["title"],
                        article["source"],
                        article["pub_date"],
                        article["summary"],
                        article["url"],
                        sentiment.get("sentiment_label", "NEUTRAL"),
                        float(sentiment.get("sentiment_score", 0.0)),
                        sentiment.get("sentiment_reason", ""),
                        now,
                    ),
                )
                conn.commit()
                return cursor.rowcount == 1
        except Exception as e:
            logger.error(f"[ERROR] Loi khi luu bai viet '{article.get('title', '')}': {e}", exc_info=True)
            return False

    def fetch_and_analyze_news(self):
        logger.info("Bat dau cao tin tuc tu RSS/HTML sources...")

        articles = []
        source_errors = 0

        for feed_meta in RSS_FEEDS:
            try:
                articles.extend(self.fetch_rss_entries(feed_meta))
            except Exception as e:
                source_errors += 1
                logger.error(f"[ERROR] Loi khi xu ly RSS {feed_meta['url']}: {e}", exc_info=True)

        for page_meta in HTML_NEWS_PAGES:
            try:
                articles.extend(self.fetch_html_entries(page_meta))
            except Exception as e:
                source_errors += 1
                logger.error(f"[ERROR] Loi khi xu ly HTML {page_meta['url']}: {e}", exc_info=True)

        fetched_count = len(articles)
        matched_count = 0
        inserted_count = 0
        duplicate_count = 0

        for article in articles:
            symbols = self.filter_article_for_symbols(article["title"], article["summary"])
            if not symbols:
                continue

            matched_count += 1
            # DB hien tai co url UNIQUE, nen moi URL chi luu duoc 1 dong.
            symbol = symbols[0]
            sentiment = self.analyze_sentiment_llm(symbol, article["title"], article["summary"])
            inserted = self.save_article(symbol, article, sentiment)

            if inserted:
                inserted_count += 1
                logger.info(f"[FACT] Saved news {symbol}/{article['source']}: {article['title']}")
            else:
                duplicate_count += 1

        logger.info(
            "[FACT] News crawl summary: "
            f"fetched={fetched_count}, matched={matched_count}, "
            f"inserted={inserted_count}, duplicates={duplicate_count}, source_errors={source_errors}."
        )
        logger.info(f"[FACT] Hoan tat cao tin tuc. Luu moi {inserted_count} bai viet.")

    def is_article_exists(self, url):
        query = "SELECT 1 FROM news_articles WHERE url = ? LIMIT 1"
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (url,))
                return cursor.fetchone() is not None
        except Exception:
            return False

    def get_average_sentiment(self, symbol, hours_back=12):
        query = """
        SELECT AVG(sentiment_score) as avg_score, COUNT(id) as news_count
        FROM news_articles
        WHERE symbol = ? AND analyzed_at >= datetime('now', ?)
        """
        hours_str = f"-{hours_back} hours"
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (symbol, hours_str))
                row = cursor.fetchone()
                if row and row["news_count"] > 0:
                    return float(row["avg_score"]), int(row["news_count"])
        except Exception as e:
            logger.error(f"Loi khi tinh sentiment trung binh: {e}")
        return 0.0, 0
