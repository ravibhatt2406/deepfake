#!/usr/bin/env python3
"""
scripts/index_knowledge_base.py — Build Vector DB for RAG Fact-Checking

Indexes verified articles from trusted sources into ChromaDB for
semantic search during fact-checking.

Usage:
  python scripts/index_knowledge_base.py --source all
  python scripts/index_knowledge_base.py --source snopes
  python scripts/index_knowledge_base.py --source newsapi --query "vaccine"
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Dict, Any

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("index_kb")

CHROMA_DIR = "data/chroma_db"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"


class KnowledgeBaseIndexer:
    def __init__(self):
        self._embed_model = None
        self._collection = None

    def setup(self):
        from sentence_transformers import SentenceTransformer
        import chromadb

        logger.info("Loading embedding model: %s", EMBED_MODEL)
        self._embed_model = SentenceTransformer(EMBED_MODEL)

        Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        self._collection = client.get_or_create_collection(
            name="fact_articles",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB connected. Existing docs: %d", self._collection.count())

    def _embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._embed_model.encode(texts, normalize_embeddings=True, batch_size=32)
        return embeddings.tolist()

    def index_documents(self, docs: List[Dict[str, Any]]) -> int:
        """Index a list of documents. Each doc: {title, content, url, source, type}."""
        if not docs:
            return 0

        texts = [f"{d['title']} {d['content'][:300]}" for d in docs]
        embeddings = self._embed(texts)

        ids, metas, documents = [], [], []
        for doc, emb in zip(docs, embeddings):
            doc_id = hashlib.sha256(doc["url"].encode()).hexdigest()[:32]
            # Skip if already indexed
            existing = self._collection.get(ids=[doc_id])
            if existing["ids"]:
                continue
            ids.append(doc_id)
            documents.append(f"{doc['title']} {doc['content'][:500]}")
            metas.append({
                "title": doc["title"][:200],
                "url": doc["url"],
                "source": doc["source"],
                "type": doc.get("type", "NEWS"),
                "indexed_at": str(int(time.time())),
            })

        if ids:
            self._collection.add(
                ids=ids,
                embeddings=embeddings[:len(ids)],
                documents=documents,
                metadatas=metas,
            )
            logger.info("Indexed %d new documents", len(ids))

        return len(ids)

    async def fetch_fact_checks(self, client: httpx.AsyncClient) -> List[Dict]:
        """Fetch from Google Fact Check API."""
        api_key = os.getenv("GOOGLE_FACT_CHECK_API_KEY", "")
        if not api_key:
            logger.warning("GOOGLE_FACT_CHECK_API_KEY not set — skipping")
            return []

        docs = []
        queries = [
            "health misinformation", "political fact check",
            "covid vaccine", "climate change", "election fraud",
            "conspiracy theory", "fake news", "scientific study",
        ]
        for query in queries:
            try:
                resp = await client.get(
                    "https://factchecktools.googleapis.com/v1alpha1/claims:search",
                    params={"query": query, "key": api_key, "pageSize": 50},
                    timeout=10.0,
                )
                data = resp.json()
                for claim in data.get("claims", []):
                    for review in claim.get("claimReview", []):
                        url = review.get("url", "")
                        if not url:
                            continue
                        docs.append({
                            "title": review.get("title", claim.get("text", ""))[:200],
                            "content": (
                                f"Rating: {review.get('textualRating', '')}. "
                                f"Claim: {claim.get('text', '')}. "
                                f"Claimant: {claim.get('claimant', '')}."
                            ),
                            "url": url,
                            "source": review.get("publisher", {}).get("name", "Fact Checker"),
                            "type": "FACT-CHECK",
                        })
                await asyncio.sleep(0.5)  # rate limit
            except Exception as e:
                logger.warning("Fact check fetch error for '%s': %s", query, e)

        logger.info("Fetched %d fact-check documents", len(docs))
        return docs

    async def fetch_newsapi(self, client: httpx.AsyncClient, query: str = "") -> List[Dict]:
        """Fetch from NewsAPI — trusted domains only."""
        api_key = os.getenv("NEWSAPI_KEY", "")
        if not api_key:
            logger.warning("NEWSAPI_KEY not set — skipping")
            return []

        trusted_domains = "reuters.com,apnews.com,bbc.com,who.int,snopes.com,politifact.com,factcheck.org"
        docs = []
        queries = [query] if query else [
            "fake news fact check", "misinformation health",
            "political misinformation", "science fact check",
        ]

        for q in queries:
            try:
                resp = await client.get(
                    "https://newsapi.org/v2/everything",
                    params={
                        "q": q, "apiKey": api_key, "pageSize": 50,
                        "sortBy": "relevancy", "language": "en",
                        "domains": trusted_domains,
                    },
                    timeout=10.0,
                )
                data = resp.json()
                for article in data.get("articles", []):
                    url = article.get("url", "")
                    if not url:
                        continue
                    docs.append({
                        "title": article.get("title", "")[:200],
                        "content": (article.get("description") or article.get("content") or "")[:500],
                        "url": url,
                        "source": article.get("source", {}).get("name", ""),
                        "type": "NEWS",
                    })
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.warning("NewsAPI error for '%s': %s", q, e)

        logger.info("Fetched %d news articles", len(docs))
        return docs

    def load_local_dataset(self, dataset_path: str) -> List[Dict]:
        """Load a local JSONL dataset of verified facts."""
        if not os.path.exists(dataset_path):
            return []
        docs = []
        with open(dataset_path) as f:
            for line in f:
                try:
                    d = json.loads(line.strip())
                    if all(k in d for k in ["title", "content", "url", "source"]):
                        docs.append(d)
                except Exception:
                    pass
        logger.info("Loaded %d local documents from %s", len(docs), dataset_path)
        return docs


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["all", "google", "newsapi", "local"], default="all")
    parser.add_argument("--query", default="", help="Optional search query for NewsAPI")
    parser.add_argument("--local_file", default="data/verified_facts.jsonl")
    args = parser.parse_args()

    indexer = KnowledgeBaseIndexer()
    indexer.setup()

    total_indexed = 0

    async with httpx.AsyncClient(
        timeout=15.0,
        headers={"User-Agent": "TruthScanIndexer/2.0"},
        follow_redirects=True,
    ) as client:

        if args.source in ("all", "google"):
            docs = await indexer.fetch_fact_checks(client)
            total_indexed += indexer.index_documents(docs)

        if args.source in ("all", "newsapi"):
            docs = await indexer.fetch_newsapi(client, args.query)
            total_indexed += indexer.index_documents(docs)

        if args.source in ("all", "local"):
            docs = indexer.load_local_dataset(args.local_file)
            total_indexed += indexer.index_documents(docs)

    logger.info("🎉 Indexing complete. Total new documents indexed: %d", total_indexed)
    logger.info("Vector DB now contains: %d total documents", indexer._collection.count())


if __name__ == "__main__":
    asyncio.run(main())
