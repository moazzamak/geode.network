"""LLM-powered semantic routing for the GEODE model network.

Registration (offline, runs once per model):
    A lightweight LLM is queried for semantic descriptors of each output class —
    synonyms, parent categories, and related task names.  Results are persisted
    in a JSON cache so subsequent runs are instant with no LLM calls.

Routing (online, cache-only):
    A query string is resolved against the cache using exact match, synonym
    lookup, and parent/ancestor walk.  Deterministic and sub-millisecond.

Supported backends (in recommended order):
    "ollama"        — calls a local Ollama server.  Install from https://ollama.ai
                      then run:  ollama pull phi3.5
                      Best option on Windows/AMD: Ollama handles device selection.
    "transformers"  — loads microsoft/Phi-3.5-mini-instruct via HuggingFace
                      transformers.  Downloads ~7 GB on first use; uses
                      device_map="auto" so it runs on whatever torch can see.
    "mock"          — returns empty descriptors (testing without LLM).

Quick start::

    router = SemanticRouter()          # defaults to ollama, phi3.5
    router.register(cat_model)         # queries LLM once per new class
    router.register(vehicle_model)

    hits = router.route("tabby", network)
    # → [("cat_node", 0.9), ("animal_node", 0.5)]
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field

from src.model_network import FittedModel, ModelNetwork


def _generate_label_id(task_name: str, class_name: str) -> str:
    """Return a stable 6-digit decimal ID for a (task, class) pair.

    Computed as the first 20 bits of SHA-256(task:class), giving a value in
    [000000, 999999] that is deterministic and unique enough for display and
    database use.  Collisions are astronomically unlikely within a single
    routing system.

    Example::

        _generate_label_id("animal_classifier", "jaguar")  # → "042831"
        _generate_label_id("vehicle_classifier", "jaguar")  # → "719405"
    """
    digest = hashlib.sha256(f"{task_name}:{class_name}".encode()).digest()
    numeric = int.from_bytes(digest[:3], "big") % 1_000_000
    return f"{numeric:06d}"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_BACKEND = "ollama"
_DEFAULT_CACHE_PATH = "data/semantic_cache.json"

_OLLAMA_HOST = "http://localhost:11434"
_OLLAMA_MODEL = "phi3.5"

_HF_MODEL = "microsoft/Phi-3.5-mini-instruct"

_SYSTEM_PROMPT = (
    "You are a semantic taxonomy assistant for a machine-learning routing system. "
    "Given a class label and the task it belongs to, respond with a single JSON "
    "object and nothing else. No explanation, no markdown fences, just the JSON."
)

_FEW_SHOT = """\
Example input:
  class: "cat"
  task:  "animal_classifier"
Example output:
  {"synonyms":["tabby","kitty","house cat","feline","domestic cat"],
   "parents":["feline","mammal","animal","pet"],
   "related_tasks":["pet_classifier","animal_detector","mammal_classifier"]}

Now answer for:"""


# ---------------------------------------------------------------------------
# SemanticDescriptor
# ---------------------------------------------------------------------------


@dataclass
class SemanticDescriptor:
    """Semantic metadata for a single class label, populated by the LLM.

    All string lists are lower-cased for case-insensitive matching.

    Attributes
    ----------
    label_id:
        Stable 6-digit decimal string derived from SHA-256(task:class).
        Unique per (task, class) pair — disambiguates homonyms like
        ``'jaguar'`` across different task domains and serves as a stable
        primary key for database storage.
        Format: ``"{class_name} ({task_name}) #{label_id}"``  e.g.
        ``"jaguar (animal_classifier) #042831"``.
    """

    class_name: str
    label_id: str = ""  # set by SemanticRouter.register() via _generate_label_id
    synonyms: list[str] = field(default_factory=list)
    parents: list[str] = field(default_factory=list)
    related_tasks: list[str] = field(default_factory=list)
    confidence: float = 1.0  # degraded when LLM output needed repair

    def qualified_name(self, task_name: str = "") -> str:
        """Human-readable disambiguated label, e.g. ``'jaguar (vehicle_classifier) #719405'``."""
        task_part = f" ({task_name})" if task_name else ""
        id_part = f" #{self.label_id}" if self.label_id else ""
        return f"{self.class_name}{task_part}{id_part}"

    def match_score(self, query: str) -> float:
        """Return a match score in [0, 1] for the given query string.

        1.0  exact class name
        0.9  synonym
        0.5  parent / ancestor
        0.3  related task
        0.0  no match
        """
        q = query.lower().strip()
        name = self.class_name.lower()

        if q == name:
            return 1.0
        if q in self.synonyms:
            return 0.9
        # Partial synonym match (e.g. "striped tabby" contains "tabby")
        if any(s in q or q in s for s in self.synonyms):
            return 0.8
        if q in self.parents:
            return 0.5
        if any(q in p or p in q for p in self.parents):
            return 0.4
        if any(q in rt or rt in q for rt in self.related_tasks):
            return 0.3
        return 0.0


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------


def _build_user_prompt(class_name: str, task_name: str) -> str:
    return (
        f"{_FEW_SHOT}\n"
        f'  class: "{class_name}"\n'
        f'  task:  "{task_name}"\n'
    )


def _parse_llm_response(content: str, class_name: str) -> SemanticDescriptor:
    """Extract JSON from LLM output, repairing common small-model quirks."""
    # Strip markdown code fences if present
    content = re.sub(r"```(?:json)?", "", content).strip()

    # Find the first balanced JSON object
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return SemanticDescriptor(class_name=class_name, confidence=0.0)

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return SemanticDescriptor(class_name=class_name, confidence=0.3)

    def _clean(lst) -> list[str]:
        return [str(x).lower().strip() for x in (lst or []) if str(x).strip()]

    return SemanticDescriptor(
        class_name=class_name,
        synonyms=_clean(data.get("synonyms", []))[:12],
        parents=_clean(data.get("parents", []))[:8],
        related_tasks=_clean(data.get("related_tasks", []))[:6],
        confidence=1.0,
    )


def _query_ollama(class_name: str, task_name: str) -> SemanticDescriptor:
    """Query the local Ollama server (phi3.5 by default)."""
    payload = json.dumps(
        {
            "model": _OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(class_name, task_name)},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1, "seed": 42, "num_predict": 300},
        }
    ).encode()

    req = urllib.request.Request(
        f"{_OLLAMA_HOST}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
        content = result["message"]["content"]
    except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Ollama request failed: {exc}\n"
            "Is Ollama running?  Start it with:  ollama serve\n"
            "Then pull the model:  ollama pull phi3.5"
        ) from exc

    return _parse_llm_response(content, class_name)


_hf_pipeline = None  # module-level cache — loaded once, reused


def _query_transformers(class_name: str, task_name: str) -> SemanticDescriptor:
    """Query Phi-3.5-mini-instruct via HuggingFace transformers."""
    global _hf_pipeline

    if _hf_pipeline is None:
        try:
            from transformers import pipeline  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "transformers is required for the 'transformers' backend.\n"
                "Install with:  pip install transformers accelerate"
            ) from exc

        print(f"Loading {_HF_MODEL} (first run only, ~7 GB download)...")
        _hf_pipeline = pipeline(
            "text-generation",
            model=_HF_MODEL,
            model_kwargs={"torch_dtype": "auto"},
            device_map="auto",
            trust_remote_code=True,
        )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(class_name, task_name)},
    ]
    out = _hf_pipeline(
        messages,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.1,
        pad_token_id=_hf_pipeline.tokenizer.eos_token_id,
    )
    content = out[0]["generated_text"][-1]["content"]
    return _parse_llm_response(content, class_name)


def _query_mock(class_name: str, task_name: str) -> SemanticDescriptor:
    """Return an empty descriptor — no LLM required (useful for unit tests)."""
    return SemanticDescriptor(class_name=class_name, confidence=0.0)


_BACKENDS = {
    "ollama": _query_ollama,
    "transformers": _query_transformers,
    "mock": _query_mock,
}


# ---------------------------------------------------------------------------
# SemanticRouter
# ---------------------------------------------------------------------------


class SemanticRouter:
    """Routes queries to the most semantically relevant nodes in a ModelNetwork.

    Cache layout (JSON on disk)::

        {
          "cat_detector": {
            "cat": {"class_name": "cat", "synonyms": [...], "parents": [...], ...}
          }
        }

    Parameters
    ----------
    backend:
        LLM backend to use for registration queries.  One of
        ``"ollama"`` (default), ``"transformers"``, or ``"mock"``.
    cache_path:
        Path to the JSON cache file.  Created automatically if absent.
    """

    def __init__(
        self,
        backend: str = _DEFAULT_BACKEND,
        cache_path: str = _DEFAULT_CACHE_PATH,
    ) -> None:
        if backend not in _BACKENDS:
            raise ValueError(
                f"Unknown backend {backend!r}. Choose from: {sorted(_BACKENDS)}"
            )
        self._backend = backend
        self._cache_path = cache_path
        # {task_name: {class_str: SemanticDescriptor}}
        self._cache: dict[str, dict[str, SemanticDescriptor]] = {}
        self._load_cache()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, model: FittedModel) -> None:
        """Populate semantic descriptors for all output classes of *model*.

        Skips classes that are already cached.  Saves the cache to disk after
        any new queries so progress is never lost.
        """
        task = model.fingerprint.task_name
        if task not in self._cache:
            self._cache[task] = {}

        pending = [
            str(cls)
            for cls in model.fingerprint.output_spec.classes
            if str(cls) not in self._cache[task]
        ]
        if not pending:
            return

        query_fn = _BACKENDS[self._backend]
        print(
            f"[SemanticRouter] Registering {len(pending)} class(es) for "
            f"'{task}' via {self._backend}..."
        )
        for cls_str in pending:
            desc = query_fn(cls_str, task)
            desc.label_id = _generate_label_id(task, cls_str)
            self._cache[task][cls_str] = desc
            syn_preview = ", ".join(desc.synonyms[:3]) or "—"
            par_preview = ", ".join(desc.parents[:3]) or "—"
            print(f"  {desc.qualified_name(task):40s}  synonyms=[{syn_preview}]  parents=[{par_preview}]")

        self._save_cache()

    def register_all(self, network: ModelNetwork) -> None:
        """Register every model currently in the network."""
        for node in network._nodes.values():
            self.register(node.model)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(
        self,
        query: str,
        network: ModelNetwork,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[tuple[str, float]]:
        """Return ranked (node_name, score) pairs relevant to *query*.

        Uses only the in-memory cache — no LLM calls at inference time.

        Parameters
        ----------
        query:
            A class name, synonym, or description (e.g. ``"tabby"``).
        network:
            The :class:`~src.model_network.ModelNetwork` to search.
        top_k:
            Maximum number of results to return.
        threshold:
            Minimum score to include in results.  Default 0 returns everything
            with any non-zero match.
        """
        results: list[tuple[str, float]] = []

        for node_name, node in network._nodes.items():
            task = node.model.fingerprint.task_name
            task_cache = self._cache.get(task, {})
            best = 0.0
            for desc in task_cache.values():
                score = desc.match_score(query) * desc.confidence
                if score > best:
                    best = score
            if best > threshold:
                results.append((node_name, best))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def explain(self, query: str, network: ModelNetwork) -> str:
        """Return a human-readable routing explanation for *query*."""
        hits = self.route(query, network, top_k=10, threshold=0.0)
        if not hits:
            return f"No registered nodes match '{query}'."

        lines = [f"Routing '{query}':"]
        score_labels = {1.0: "exact", 0.9: "synonym", 0.8: "partial-syn",
                        0.5: "parent", 0.4: "partial-parent", 0.3: "related-task"}
        for node_name, score in hits:
            label = score_labels.get(score, f"score={score:.2f}")
            lines.append(f"  {score:.2f}  {node_name:<20s}  [{label}]")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Cache persistence
    # ------------------------------------------------------------------

    def _load_cache(self) -> None:
        if not os.path.exists(self._cache_path):
            return
        try:
            with open(self._cache_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            for task, classes in raw.items():
                self._cache[task] = {
                    cls: SemanticDescriptor(**data)
                    for cls, data in classes.items()
                }
        except (json.JSONDecodeError, TypeError, KeyError):
            print(
                f"[SemanticRouter] Warning: cache at {self._cache_path} is corrupt; "
                "starting fresh."
            )
            self._cache = {}

    def _save_cache(self) -> None:
        os.makedirs(os.path.dirname(self._cache_path) or ".", exist_ok=True)
        serialisable = {
            task: {cls: asdict(desc) for cls, desc in classes.items()}
            for task, classes in self._cache.items()
        }
        tmp = self._cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(serialisable, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, self._cache_path)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_tasks = len(self._cache)
        n_classes = sum(len(v) for v in self._cache.values())
        return (
            f"SemanticRouter(backend={self._backend!r}, "
            f"tasks={n_tasks}, classes={n_classes}, "
            f"cache={self._cache_path!r})"
        )
