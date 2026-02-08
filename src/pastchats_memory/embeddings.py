from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import math
import re
from typing import Iterable


_WORD_RE = re.compile(r"[a-zA-Z0-9_]{2,}")


class EmbeddingProvider(ABC):
    model_name: str
    dim: int

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError


class LocalHashEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dim: int = 256) -> None:
        self.model_name = f"local-hash-{dim}"
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        lowered = text.lower()
        for token in _WORD_RE.findall(lowered):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign

        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return vec
        return [v / norm for v in vec]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str, api_key: str) -> None:
        self.model_name = model_name
        self._api_key = api_key
        self._client = None
        default_dims = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        self.dim = default_dims.get(model_name, 1536)

    def _client_or_raise(self):
        if self._client is not None:
            return self._client

        try:
            from openai import OpenAI  # pylint: disable=import-outside-toplevel
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "openai package is not installed. Run: pip install -e .[openai]"
            ) from exc

        self._client = OpenAI(api_key=self._api_key)
        return self._client

    def embed(self, text: str) -> list[float]:
        client = self._client_or_raise()
        response = client.embeddings.create(model=self.model_name, input=text)
        vector = list(response.data[0].embedding)
        self.dim = len(vector)
        return normalize(vector)


def normalize(values: Iterable[float]) -> list[float]:
    vec = [float(v) for v in values]
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0:
        return vec
    return [v / norm for v in vec]


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    l = list(left)
    r = list(right)
    if len(l) != len(r) or not l:
        return 0.0
    return sum(a * b for a, b in zip(l, r))
