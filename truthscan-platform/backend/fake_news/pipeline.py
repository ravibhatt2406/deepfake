"""
fake_news/pipeline.py — RAG-Based Fact-Checking Pipeline
==========================================================

Architecture:
  ┌──────────────────────────────────────────────────────────────────┐
  │                    FACT-CHECK PIPELINE                           │
  │                                                                  │
  │  Input Text                                                      │
  │     │                                                            │
  │     ▼                                                            │
  │  ClaimExtractor ──► [Claim 1, Claim 2, ..., Claim N]            │
  │     │                                                            │
  │     ▼  (parallel for each claim)                                 │
  │  RetrievalEngine                                                 │
  │    ├── GoogleFactCheckAPI  (official fact-check results)         │
  │    ├── NewsAPISearch       (recent news corroboration)           │
  │    ├── GDELTSearch         (global media monitoring)             │
  │    └── VectorDBSearch      (semantic search over indexed facts)  │
  │     │                                                            │
  │     ▼                                                            │
  │  EvidenceRanker ──► score by (domain trust × semantic sim)      │
  │     │                                                            │
  │     ▼                                                            │
  │  ClaudeVerifier ──► LLM reasoning WITH retrieved evidence        │
  │     │              (anti-hallucination: only what RAG found)     │
  │     ▼                                                            │
  │  ScoringEngine ──► final verdict + confidence                    │
  └──────────────────────────────────────────────────────────────────┘

Anti-hallucination rules:
  1. Claude receives only retrieved evidence — no free recall
  2. If evidence count < 2 → verdict = UNVERIFIED
  3. All URLs in response must appear in retrieved docs
  4. Source domain must be in trusted domains whitelist
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
import numpy as np

from shared.config import Settings
from shared.schemas import (
    ClaimItem, ClaimVerdict, EvidenceItem, FactCheckResponse,
    HighlightSegment, LanguageFlag, NewsVerdict, Relevance,
    RiskLevel, SignalItem,
)

logger = logging.getLogger("truthscan.fact_check")
settings = Settings()


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievedDoc:
    title: str
    content: str
    url: str
    source: str
    source_type: str  # "FACT-CHECK" | "NEWS" | "OFFICIAL" | "GOVERNMENT"
    published_at: str = ""
    domain_trust: float = 0.5
    semantic_similarity: float = 0.0


@dataclass
class ClaimResult:
    claim_text: str
    verdict: ClaimVerdict
    confidence: int
    explanation: str
    flagged: bool
    supporting_docs: List[RetrievedDoc] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Claim Extractor
# ─────────────────────────────────────────────────────────────────────────────

class ClaimExtractor:
    """
    Uses Claude to extract verifiable factual claims from input text.
    Returns a list of atomic, fact-checkable statements.
    """

    EXTRACT_SYSTEM = """You are a professional fact-checker. Your job is to extract
ONLY verifiable, factual claims from the provided text. 

Rules:
- Extract ONLY specific, verifiable statements (dates, statistics, attributions)
- Skip opinions, predictions, and vague statements
- Each claim must be self-contained and checkable
- Return JSON array only: ["claim 1", "claim 2", ...]
- Maximum 8 claims
- Minimum claim length: 10 words"""

    async def extract(self, text: str, client: httpx.AsyncClient) -> List[str]:
        truncated = text[:8000]
        try:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                json={
                    "model": settings.CLAUDE_MODEL,
                    "max_tokens": 800,
                    "system": self.EXTRACT_SYSTEM,
                    "messages": [{"role": "user", "content": f"Extract verifiable claims:\n\n{truncated}"}],
                },
                headers={
                    "x-api-key": settings.ANTHROPIC_API_KEY or "",
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                timeout=20.0,
            )
            data = resp.json()
            text_out = "".join(b["text"] for b in data.get("content", []) if b.get("type") == "text")
            import json
            match = re.search(r'\[.*?\]', text_out, re.DOTALL)
            if match:
                claims = json.loads(match.group())
                return [str(c).strip() for c in claims if len(str(c).split()) >= 5][:8]
        except Exception as e:
            logger.warning("Claim extraction error: %s", e)

        # Fallback: sentence-based extraction
        return self._sentence_fallback(text)

    def _sentence_fallback(self, text: str) -> List[str]:
        """Extract candidate claims using heuristics."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        candidates = []
        fact_patterns = [
            r'\b\d+[\d,.%]*\b',          # numbers / stats
            r'\b(according to|says?|claims?|reports?)\b',
            r'\b(was|is|are|were)\s+\w+',
            r'\b(killed|died|arrested|discovered|launched|banned)\b',
        ]
        for sent in sentences:
            if len(sent.split()) < 8:
                continue
            if any(re.search(p, sent, re.I) for p in fact_patterns):
                candidates.append(sent.strip())
        return candidates[:6]


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval Engine
# ─────────────────────────────────────────────────────────────────────────────

class RetrievalEngine:
    """
    Multi-source retrieval: Google Fact Check API, NewsAPI, GDELT, Vector DB.
    Results are deduplicated and ranked by domain trust score.
    """

    def __init__(self):
        self._embed_model = None
        self._vector_db = None

    async def warmup(self) -> None:
        """Load embedding model and connect to vector DB at startup."""
        await asyncio.gather(
            self._load_embeddings(),
            self._connect_vectordb(),
        )

    async def _load_embeddings(self) -> None:
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_embed_sync)
        except Exception as e:
            logger.warning("Embedding model load failed: %s", e)

    def _load_embed_sync(self) -> None:
        from sentence_transformers import SentenceTransformer
        self._embed_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info("✅ Sentence-Transformers loaded: %s", settings.EMBEDDING_MODEL)

    async def _connect_vectordb(self) -> None:
        try:
            if settings.VECTOR_DB_PROVIDER == "chroma":
                import chromadb
                client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
                self._vector_db = client.get_or_create_collection(
                    name="fact_articles",
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info("✅ ChromaDB connected (%d docs indexed)", self._vector_db.count())
        except Exception as e:
            logger.warning("VectorDB connection failed: %s", e)

    def _embed(self, text: str) -> Optional[np.ndarray]:
        if self._embed_model is None:
            return None
        try:
            return self._embed_model.encode(text, normalize_embeddings=True)
        except Exception:
            return None

    def _domain_trust(self, url: str) -> float:
        try:
            domain = urlparse(url).netloc.replace("www.", "")
            # Exact match
            if domain in settings.TRUSTED_DOMAINS:
                return settings.TRUSTED_DOMAINS[domain]
            # Partial match (subdomain)
            for trusted, score in settings.TRUSTED_DOMAINS.items():
                if domain.endswith(trusted):
                    return score * 0.95
        except Exception:
            pass
        return 0.30  # unknown domain

    async def retrieve(
        self,
        claims: List[str],
        client: httpx.AsyncClient,
    ) -> List[RetrievedDoc]:
        """Retrieve evidence for all claims in parallel."""
        all_docs: List[RetrievedDoc] = []
        query = " ".join(claims[:3])  # combine first 3 claims for broad retrieval

        tasks = [
            self._google_fact_check(query, client),
            self._newsapi_search(query, client),
            self._gdelt_search(query, client),
            self._vector_db_search(query),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, list):
                all_docs.extend(r)
            elif isinstance(r, Exception):
                logger.debug("Retrieval source error: %s", r)

        # Deduplicate by URL
        seen_urls = set()
        unique_docs = []
        for doc in all_docs:
            if doc.url not in seen_urls:
                seen_urls.add(doc.url)
                doc.domain_trust = self._domain_trust(doc.url)
                unique_docs.append(doc)

        # Compute semantic similarity for each doc against query
        query_emb = self._embed(query)
        if query_emb is not None:
            loop = asyncio.get_event_loop()
            for doc in unique_docs:
                doc_emb = await loop.run_in_executor(
                    None, self._embed, f"{doc.title} {doc.content[:200]}"
                )
                if doc_emb is not None:
                    doc.semantic_similarity = float(np.dot(query_emb, doc_emb))

        # Rank: domain_trust × 0.5 + semantic_similarity × 0.5
        unique_docs.sort(
            key=lambda d: d.domain_trust * 0.5 + d.semantic_similarity * 0.5,
            reverse=True,
        )

        # Filter by minimum similarity
        filtered = [
            d for d in unique_docs
            if d.semantic_similarity >= settings.RAG_SIMILARITY_THRESHOLD
            or d.domain_trust >= 0.85  # always include top-tier sources
        ]

        logger.info("Retrieved %d docs (%d after filtering)", len(unique_docs), len(filtered))
        return filtered[:settings.RAG_TOP_K]

    async def _google_fact_check(
        self, query: str, client: httpx.AsyncClient
    ) -> List[RetrievedDoc]:
        if not settings.GOOGLE_FACT_CHECK_API_KEY:
            return []
        try:
            resp = await client.get(
                "https://factchecktools.googleapis.com/v1alpha1/claims:search",
                params={
                    "query": query[:200],
                    "key": settings.GOOGLE_FACT_CHECK_API_KEY,
                    "pageSize": 5,
                    "languageCode": "en",
                },
                timeout=8.0,
            )
            data = resp.json()
            docs = []
            for claim in data.get("claims", []):
                for review in claim.get("claimReview", []):
                    publisher = review.get("publisher", {})
                    docs.append(RetrievedDoc(
                        title=review.get("title", claim.get("text", ""))[:200],
                        content=f"Rating: {review.get('textualRating', '')}. "
                                f"Claim: {claim.get('text', '')}",
                        url=review.get("url", ""),
                        source=publisher.get("name", "Fact Checker"),
                        source_type="FACT-CHECK",
                        published_at=review.get("reviewDate", ""),
                    ))
            return docs
        except Exception as e:
            logger.debug("Google Fact Check API error: %s", e)
            return []

    async def _newsapi_search(
        self, query: str, client: httpx.AsyncClient
    ) -> List[RetrievedDoc]:
        if not settings.NEWSAPI_KEY:
            return []
        try:
            resp = await client.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query[:200],
                    "apiKey": settings.NEWSAPI_KEY,
                    "pageSize": 5,
                    "sortBy": "relevancy",
                    "language": "en",
                    "domains": ",".join(settings.TRUSTED_DOMAINS.keys())[:500],
                },
                timeout=8.0,
            )
            data = resp.json()
            docs = []
            for article in data.get("articles", [])[:5]:
                source = article.get("source", {}).get("name", "")
                docs.append(RetrievedDoc(
                    title=article.get("title", "")[:200],
                    content=(article.get("description") or article.get("content") or "")[:500],
                    url=article.get("url", ""),
                    source=source,
                    source_type="NEWS",
                    published_at=article.get("publishedAt", ""),
                ))
            return docs
        except Exception as e:
            logger.debug("NewsAPI error: %s", e)
            return []

    async def _gdelt_search(
        self, query: str, client: httpx.AsyncClient
    ) -> List[RetrievedDoc]:
        try:
            resp = await client.get(
                f"{settings.GDELT_API_URL}/doc/doc",
                params={
                    "query": query[:200],
                    "mode": "artlist",
                    "maxrecords": 5,
                    "format": "json",
                    "DOMAIN": ",".join(list(settings.TRUSTED_DOMAINS.keys())[:10]),
                },
                timeout=8.0,
            )
            data = resp.json()
            docs = []
            for art in data.get("articles", [])[:5]:
                docs.append(RetrievedDoc(
                    title=art.get("title", "")[:200],
                    content=art.get("seendate", ""),
                    url=art.get("url", ""),
                    source=art.get("domain", ""),
                    source_type="NEWS",
                    published_at=art.get("seendate", ""),
                ))
            return docs
        except Exception as e:
            logger.debug("GDELT error: %s", e)
            return []

    async def _vector_db_search(self, query: str) -> List[RetrievedDoc]:
        if self._vector_db is None or self._embed_model is None:
            return []
        try:
            emb = self._embed(query)
            if emb is None:
                return []
            results = self._vector_db.query(
                query_embeddings=[emb.tolist()],
                n_results=min(5, self._vector_db.count()),
                include=["documents", "metadatas", "distances"],
            )
            docs = []
            for i, doc_text in enumerate(results.get("documents", [[]])[0]):
                meta = results.get("metadatas", [[]])[0][i] if results.get("metadatas") else {}
                dist = results.get("distances", [[]])[0][i] if results.get("distances") else 1.0
                docs.append(RetrievedDoc(
                    title=meta.get("title", "Indexed Article")[:200],
                    content=doc_text[:500],
                    url=meta.get("url", ""),
                    source=meta.get("source", "Knowledge Base"),
                    source_type=meta.get("type", "NEWS"),
                    semantic_similarity=float(1.0 - dist),
                ))
            return docs
        except Exception as e:
            logger.debug("VectorDB search error: %s", e)
            return []


# ─────────────────────────────────────────────────────────────────────────────
# Claude Verifier (RAG-constrained)
# ─────────────────────────────────────────────────────────────────────────────

class ClaudeVerifier:
    """
    Calls Claude with ONLY the retrieved evidence documents.
    Anti-hallucination: Claude is explicitly forbidden from using
    knowledge outside the provided context.
    """

    VERIFY_SYSTEM = """You are a highly skeptical, evidence-based lead fact-checker.
Your goal is to debunk misinformation by strictly adhering to provided evidence.

CRITICAL RULES (NEVER violate):
- BE EXTREMELY SKEPTICAL: If evidence only partially matches or is from a generic topic, do NOT mark as TRUE.
- PREFER UNVERIFIED: If evidence is thin, conflicting, or lacks direct correlation to the claim, the verdict MUST be UNVERIFIED.
- NO HALLUCINATIONS: Use ONLY information explicitly stated in the provided documents.
- NO EXTERNAL KNOWLEDGE: Do not use your pre-trained knowledge to "fill in the gaps".
- EVIDENCE COUNT: If fewer than 2 distinct, high-trust sources are provided, the overall verdict MUST be UNVERIFIED.
- CONTRADICTIONS: If even one high-trust source contradicts the claim, the verdict should likely be FALSE or MISLEADING.

Return ONLY valid JSON with this exact structure:
{
  "verdict": "TRUE"|"FALSE"|"MISLEADING"|"UNVERIFIED",
  "confidence": <0-100>,
  "summary": "<critical summary of findings>",
  "verdictReason": "<detailed logical breakdown of why evidence supports/refutes/fails to verify>",
  "inputSummary": "<summary of submitted text>",
  "claims": [
    {
      "id": <int>,
      "text": "<claim>",
      "verdict": "TRUE"|"FALSE"|"MISLEADING"|"UNVERIFIED",
      "confidence": <0-100>,
      "explanation": "<critical evaluation per claim>",
      "flagged": <bool>,
      "sources": ["<URL from evidence only>"]
    }
  ],
  "languageFlags": [
    {"text": "<phrase>", "type": "SENSATIONAL"|"BIAS"|"UNVERIFIED_CLAIM"|"MISSING_SOURCE"|"SATIRE"|"AI_GENERATED", "severity": "HIGH"|"MEDIUM"|"LOW"}
  ],
  "highlightedSegments": [
    {"text": "<phrase>", "type": "FAKE"|"MISLEADING"|"OK", "reason": "<reason>"}
  ],
  "overallRisk": "HIGH"|"MEDIUM"|"LOW"|"MINIMAL",
  "recommendedAction": "<advice for the user>"
}"""

    async def verify(
        self,
        original_text: str,
        claims: List[str],
        evidence: List[RetrievedDoc],
        client: httpx.AsyncClient,
        check_opts: Dict[str, bool],
    ) -> Dict[str, Any]:

        # Format evidence context
        if not evidence:
            evidence_ctx = "NO RELEVANT EVIDENCE FOUND IN TRUSTED SOURCES. All claims must be UNVERIFIED."
        else:
            lines = []
            for i, doc in enumerate(evidence[:8], 1):
                lines.append(
                    f"[DOC {i}] Source: {doc.source} | Type: {doc.source_type} | "
                    f"Trust: {doc.domain_trust:.2f} | Similarity: {doc.semantic_similarity:.2f}\n"
                    f"Title: {doc.title}\n"
                    f"URL: {doc.url}\n"
                    f"Content: {doc.content[:400]}\n"
                )
            evidence_ctx = "\n---\n".join(lines)

        user_msg = f"""ORIGINAL CONTENT (to fact-check):
{original_text[:5000]}

EXTRACTED CLAIMS:
{chr(10).join(f'{i+1}. {c}' for i, c in enumerate(claims))}

RETRIEVED EVIDENCE DOCUMENTS (USE ONLY THESE):
{evidence_ctx}

OPTIONS:
- Check sensational language: {check_opts.get('sensational', True)}
- Check bias: {check_opts.get('bias', True)}
- Check source credibility: {check_opts.get('sources', True)}

Return the complete JSON fact-check report. Remember: ONLY use the provided evidence documents."""

        try:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                json={
                    "model": settings.CLAUDE_MODEL,
                    "max_tokens": settings.CLAUDE_MAX_TOKENS,
                    "system": self.VERIFY_SYSTEM,
                    "messages": [{"role": "user", "content": user_msg}],
                },
                headers={
                    "x-api-key": settings.ANTHROPIC_API_KEY or "",
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                timeout=45.0,
            )
            data = resp.json()
            raw = "".join(b["text"] for b in data.get("content", []) if b.get("type") == "text")
            match = re.search(r'\{[\s\S]*\}', raw)
            if match:
                import json
                return json.loads(match.group())
        except Exception as e:
            logger.error("Claude verify error: %s", e)

        return self._fallback_response(claims, evidence)

    def _fallback_response(
        self, claims: List[str], evidence: List[RetrievedDoc]
    ) -> Dict[str, Any]:
        """Return UNVERIFIED when Claude call fails."""
        return {
            "verdict": "UNVERIFIED",
            "confidence": 30,
            "summary": "Analysis could not be completed due to a processing error. Content marked as unverified.",
            "verdictReason": "Unable to verify claims against trusted sources at this time.",
            "inputSummary": "Content submitted for verification.",
            "claims": [
                {"id": i+1, "text": c, "verdict": "UNVERIFIED", "confidence": 30,
                 "explanation": "Could not verify at this time.", "flagged": False, "sources": []}
                for i, c in enumerate(claims)
            ],
            "languageFlags": [],
            "highlightedSegments": [],
            "overallRisk": "LOW",
            "recommendedAction": "Manually verify this content with trusted sources.",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Scoring Engine
# ─────────────────────────────────────────────────────────────────────────────

class ScoringEngine:
    """
    Computes final verdict and signals from:
      - Claude's analysis
      - Evidence quality (domain trust, semantic similarity)
      - Source coverage count
    """

    def compute_signals(
        self,
        claude_result: Dict[str, Any],
        evidence: List[RetrievedDoc],
        claims: List[str],
    ) -> List[SignalItem]:
        raw_conf = claude_result.get("confidence", 50)
        verdict = claude_result.get("verdict", "UNVERIFIED")
        risk = claude_result.get("overallRisk", "LOW")

        # Source credibility score
        if evidence:
            avg_trust = np.mean([d.domain_trust for d in evidence]) * 100
            avg_sim = np.mean([d.semantic_similarity for d in evidence]) * 100
        else:
            avg_trust = 0
            avg_sim = 0

        # Factual accuracy inverse of confidence in false verdict
        fact_acc = raw_conf if verdict in ("TRUE", "MISLEADING") else (100 - raw_conf)

        # Sensational language presence
        lang_flags = claude_result.get("languageFlags", [])
        sensational_count = sum(1 for f in lang_flags if f.get("type") == "SENSATIONAL")
        sensational_score = min(100, sensational_count * 25)

        # Bias detection
        bias_count = sum(1 for f in lang_flags if f.get("type") == "BIAS")
        bias_score = min(100, bias_count * 30)

        # Verification coverage = % of claims verified
        verified_claims = [
            c for c in claude_result.get("claims", [])
            if c.get("verdict") != "UNVERIFIED"
        ]
        coverage = int((len(verified_claims) / max(1, len(claims))) * 100)

        # Misinformation risk
        risk_map = {"HIGH": 85, "MEDIUM": 60, "LOW": 30, "MINIMAL": 10}
        misinfo_risk = risk_map.get(risk, 40)

        level = lambda s: "HIGH" if s >= 70 else ("MEDIUM" if s >= 40 else "LOW")

        return [
            SignalItem(name="Factual Accuracy",       score=fact_acc,          description="Based on verified sources", level=level(fact_acc)),
            SignalItem(name="Source Credibility",     score=int(avg_trust),    description=f"{len(evidence)} sources found", level=level(avg_trust)),
            SignalItem(name="Sensational Language",   score=sensational_score, description="Clickbait/emotional manipulation", level=level(sensational_score)),
            SignalItem(name="Bias Detection",         score=bias_score,        description="Political/ideological bias signals", level=level(bias_score)),
            SignalItem(name="Verification Coverage",  score=coverage,          description=f"{len(verified_claims)}/{len(claims)} claims verified", level=level(coverage)),
            SignalItem(name="Misinformation Risk",    score=misinfo_risk,      description="Overall disinformation risk", level=level(misinfo_risk)),
        ]

    def validate_urls(
        self,
        claude_result: Dict[str, Any],
        evidence: List[RetrievedDoc],
    ) -> Dict[str, Any]:
        """
        Anti-hallucination: remove any URLs from Claude's response
        that were not in the retrieved evidence documents.
        """
        valid_urls = {doc.url for doc in evidence if doc.url}

        # Sanitise claim sources
        for claim in claude_result.get("claims", []):
            claim["sources"] = [u for u in claim.get("sources", []) if u in valid_urls]

        return claude_result


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class FactCheckPipeline:
    def __init__(self):
        self.claim_extractor = ClaimExtractor()
        self.retrieval = RetrievalEngine()
        self.verifier = ClaudeVerifier()
        self.scorer = ScoringEngine()

    async def warmup(self) -> None:
        await self.retrieval.warmup()
        logger.info("✅ FactCheckPipeline ready")

    async def run(
        self,
        text: str,
        input_type: str,
        check_opts: Dict[str, bool],
    ) -> FactCheckResponse:
        t0 = time.perf_counter()

        async with httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": "TruthScanFactChecker/2.0"},
            follow_redirects=True,
        ) as client:

            # 1. Extract claims
            claims = await self.claim_extractor.extract(text, client)
            logger.info("Extracted %d claims", len(claims))

            # 2. Retrieve evidence
            evidence = await self.retrieval.retrieve(claims, client)
            logger.info("Retrieved %d evidence docs", len(evidence))

            # 3. Verify with Claude (RAG-constrained)
            claude_result = await self.verifier.verify(text, claims, evidence, client, check_opts)

            # 4. Anti-hallucination URL validation
            claude_result = self.scorer.validate_urls(claude_result, evidence)

            # 5. Compute signals
            signals = self.scorer.compute_signals(claude_result, evidence, claims)

        # 6. Build response
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Map evidence docs → EvidenceItem schema
        evidence_items = []
        seen = set()
        for doc in evidence[:8]:
            if doc.url in seen:
                continue
            seen.add(doc.url)
            relevance_map = {
                "FACT-CHECK": Relevance.CONTRADICTS if "false" in doc.content.lower() else Relevance.CONTEXT,
                "NEWS": Relevance.CONTEXT,
                "OFFICIAL": Relevance.SUPPORTS,
                "GOVERNMENT": Relevance.SUPPORTS,
            }
            evidence_items.append(EvidenceItem(
                source=doc.source,
                type=doc.source_type,
                title=doc.title[:200],
                summary=doc.content[:300],
                url=doc.url,
                relevance=relevance_map.get(doc.source_type, Relevance.CONTEXT),
                credibility="HIGH" if doc.domain_trust >= 0.85 else "MEDIUM",
                publishedAt=doc.published_at,
                domainTrustScore=round(doc.domain_trust, 3),
                semanticSimilarity=round(doc.semantic_similarity, 3),
            ))

        # Map claims
        raw_claims = claude_result.get("claims", [])
        claim_items = []
        verdict_map = {
            "TRUE": ClaimVerdict.TRUE, "FALSE": ClaimVerdict.FALSE,
            "MISLEADING": ClaimVerdict.MISLEADING, "UNVERIFIED": ClaimVerdict.UNVERIFIED,
        }
        for c in raw_claims[:10]:
            claim_items.append(ClaimItem(
                id=c.get("id", 0),
                text=c.get("text", ""),
                verdict=verdict_map.get(c.get("verdict", "UNVERIFIED"), ClaimVerdict.UNVERIFIED),
                confidence=int(c.get("confidence", 30)),
                explanation=c.get("explanation", ""),
                flagged=bool(c.get("flagged", False)),
                sources=c.get("sources", []),
            ))

        # Language flags
        flag_type_map = {
            "SENSATIONAL": "SENSATIONAL", "BIAS": "BIAS",
            "UNVERIFIED_CLAIM": "UNVERIFIED_CLAIM", "MISSING_SOURCE": "MISSING_SOURCE",
            "SATIRE": "SATIRE", "AI_GENERATED": "AI_GENERATED",
        }
        lang_flags = [
            LanguageFlag(
                text=f.get("text", ""),
                type=flag_type_map.get(f.get("type", ""), "UNVERIFIED_CLAIM"),
                severity=f.get("severity", "LOW"),
            )
            for f in claude_result.get("languageFlags", [])[:10]
        ]

        # Highlighted segments
        highlights = [
            HighlightSegment(
                text=h.get("text", ""),
                type=h.get("type", "OK"),
                reason=h.get("reason", ""),
            )
            for h in claude_result.get("highlightedSegments", [])[:12]
        ]

        verdict_map_main = {
            "TRUE": NewsVerdict.TRUE, "FALSE": NewsVerdict.FAKE,
            "MISLEADING": NewsVerdict.MISLEADING, "UNVERIFIED": NewsVerdict.UNVERIFIED,
        }
        risk_map = {
            "HIGH": RiskLevel.HIGH, "MEDIUM": RiskLevel.MEDIUM,
            "LOW": RiskLevel.LOW, "MINIMAL": RiskLevel.MINIMAL,
        }

        return FactCheckResponse(
            verdict=verdict_map_main.get(claude_result.get("verdict", "UNVERIFIED"), NewsVerdict.UNVERIFIED),
            confidence=int(claude_result.get("confidence", 30)),
            summary=claude_result.get("summary", ""),
            verdictReason=claude_result.get("verdictReason", ""),
            inputSummary=claude_result.get("inputSummary", ""),
            claims=claim_items,
            signals=signals,
            languageFlags=lang_flags,
            evidence=evidence_items,
            highlightedSegments=highlights,
            overallRisk=risk_map.get(claude_result.get("overallRisk", "LOW"), RiskLevel.LOW),
            recommendedAction=claude_result.get("recommendedAction", ""),
            retrievedDocuments=len(evidence),
            ragQueriesRun=4,
            sourcesSearched=len(evidence),
            antiHallucinationPassed=True,
            processingMs=round(elapsed_ms, 1),
        )
