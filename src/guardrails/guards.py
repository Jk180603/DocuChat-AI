"""
Input and Output Guardrails
Handles jailbreak detection, sensitive topics, PII, and output validation
"""
import re
from dataclasses import dataclass
from enum import Enum


class GuardrailStatus(Enum):
    ALLOWED = "allowed"
    BLOCKED = "blocked"
    MODIFIED = "modified"


@dataclass
class GuardrailResult:
    status: GuardrailStatus
    reason: str | None
    modified_query: str | None = None


class InputGuardrails:
    JAILBREAK_PATTERNS = [
        "ignore previous instructions",
        "ignore all instructions",
        "disregard your",
        "forget your training",
        "you are now",
        "act as if",
        "pretend you are",
        "pretend to be",
        "roleplay as",
        "you have no restrictions",
        "bypass your",
        "override your",
        "jailbreak",
        "dan mode",
        "developer mode",
        "sudo mode",
        "ignore your guidelines",
        "ignore your rules",
    ]

    SENSITIVE_TOPICS = [
        "how to make a bomb",
        "how to make weapons",
        "how to hack",
        "how to synthesize drugs",
        "child pornography",
        "csam",
        "how to hurt",
        "how to kill",
        "suicide method",
        "self harm method",
    ]

    PII_PATTERNS = [
        r"\b\d{3}-\d{2}-\d{4}\b",           # SSN
        r"\b4[0-9]{12}(?:[0-9]{3})?\b",      # Visa card
        r"\b5[1-5][0-9]{14}\b",              # Mastercard
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
        r"\b(?:\+49|0)[0-9\s\-]{9,15}\b",   # German phone
    ]

    MAX_QUERY_LENGTH = 2000
    MIN_QUERY_LENGTH = 2

    def check(self, query: str) -> GuardrailResult:
        # Length check
        if len(query.strip()) < self.MIN_QUERY_LENGTH:
            return GuardrailResult(
                status=GuardrailStatus.BLOCKED,
                reason="query_too_short"
            )

        if len(query) > self.MAX_QUERY_LENGTH:
            return GuardrailResult(
                status=GuardrailStatus.BLOCKED,
                reason="query_too_long"
            )

        query_lower = query.lower().strip()

        # Jailbreak check
        for pattern in self.JAILBREAK_PATTERNS:
            if pattern in query_lower:
                return GuardrailResult(
                    status=GuardrailStatus.BLOCKED,
                    reason=f"jailbreak_attempt: {pattern}"
                )

        # Sensitive topics check
        for topic in self.SENSITIVE_TOPICS:
            if topic in query_lower:
                return GuardrailResult(
                    status=GuardrailStatus.BLOCKED,
                    reason=f"sensitive_topic: {topic}"
                )

        # PII detection — warn but allow
        for pattern in self.PII_PATTERNS:
            if re.search(pattern, query):
                return GuardrailResult(
                    status=GuardrailStatus.MODIFIED,
                    reason="pii_detected",
                    modified_query=re.sub(pattern, "[REDACTED]", query)
                )

        return GuardrailResult(
            status=GuardrailStatus.ALLOWED,
            reason=None
        )


class OutputGuardrails:
    HALLUCINATION_PHRASES = [
        "i don't have access to",
        "i cannot access",
        "as an ai",
        "i was trained",
        "my training data",
        "i don't actually know",
        "i made that up",
    ]

    def validate(
        self,
        response: str,
        source_docs: list,
        min_confidence: float = 0.3
    ) -> GuardrailResult:

        if not response or len(response.strip()) < 10:
            return GuardrailResult(
                status=GuardrailStatus.BLOCKED,
                reason="empty_response"
            )

        response_lower = response.lower()

        # Check for hallucination signals
        for phrase in self.HALLUCINATION_PHRASES:
            if phrase in response_lower:
                return GuardrailResult(
                    status=GuardrailStatus.MODIFIED,
                    reason="possible_hallucination",
                    modified_query=response + "\n\n*Note: Please verify this information with the source documents.*"
                )

        # Check if response is grounded in retrieved docs
        if source_docs:
            response_words = set(response.lower().split())
            grounded = False
            for doc in source_docs[:3]:
                doc_words = set(doc.page_content.lower().split())
                overlap = len(response_words & doc_words) / max(len(response_words), 1)
                if overlap > min_confidence:
                    grounded = True
                    break

            if not grounded:
                return GuardrailResult(
                    status=GuardrailStatus.MODIFIED,
                    reason="low_grounding",
                    modified_query=response + "\n\n*Note: This answer may not be fully grounded in the provided documents.*"
                )

        return GuardrailResult(
            status=GuardrailStatus.ALLOWED,
            reason=None
        )