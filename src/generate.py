"""Stage 6 - Answer generation from retrieved evidence, with citations and abstention.

Three backends:
    hf          a local Hugging Face Transformers chat model writes the answer
    ollama      the same kind of model served by a local Ollama install in a
                compressed (quantized) format: faster on CPU, optional demo mode
    extractive  no language model; the answer is built from the best-matching
                sentences of the retrieved passages (fast, useful as a baseline)

`answer(question)` returns the project-wide common response:
    {"answer": str, "sources": [{"book_id", "page", "chunk_id"}], "insufficient_evidence": bool}
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

from src.citations import MARKER, format_citation, renumber_markers, resolve_sources, to_source
from src.config import load_config

INSUFFICIENT_TOKEN = "INSUFFICIENT_EVIDENCE"
MIN_ANSWER_WORDS = 4
INSUFFICIENT_MESSAGE = "The provided books do not contain enough evidence to answer this question."
QUOTE_PREFIX = "The books say: "

SYSTEM_PROMPT = (
    "You are a careful teaching assistant. You answer questions using only the numbered textbook "
    "passages you are given, never outside knowledge, and you cite the passage behind every sentence."
)

INSTRUCTIONS = (
    "Answer the question in 1 to 4 complete sentences, using only facts stated in the passages above.\n"
    "End each sentence with the number of the passage it comes from in square brackets.\n"
    "Example of the format only: Water boils at 100 degrees Celsius at sea level [2].\n"
    "Only use passage numbers listed above.\n"
    "If the passages do not discuss the topic of the question, or do not contain the answer, "
    f"write only: {INSUFFICIENT_TOKEN}"
)

STOPWORDS = set("""
a an and are as at be by can do does for from has have how in into is it its of on or
that the their there these this to was what when where which who why will with would
explain describe define give list state between difference compare
thereby each also such than then they them those while
""".split())


def build_prompt(question: str, evidence: list[dict]) -> str:
    blocks = [f"[{n}] {format_citation(p)}\n{p['text']}" for n, p in enumerate(evidence, start=1)]
    return "Passages:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\n\n{INSTRUCTIONS}"


def keywords(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in STOPWORDS}


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def support(sentence: str, passage_text: str) -> float:
    """Share of the sentence's content words that also appear in the passage (0 to 1)."""
    words = {_stem(w) for w in keywords(sentence)}
    if not words:
        return 0.0
    return len(words & {_stem(w) for w in keywords(passage_text)}) / len(words)


def ground_answer(raw: str, evidence: list[dict], min_support: float) -> str:
    """Keep only the sentences a passage supports, and cite each with that passage.

    A sentence keeps the passage it cited when that passage supports it; otherwise it is
    re-cited to the best-supporting passage, or dropped when no passage reaches `min_support`.
    Returns text with [n] markers in evidence numbering, or "" when nothing is supported.
    """
    lines = []
    for line in raw.splitlines():
        # Keep list markers ("1.", "-") out of sentence splitting and put them back afterwards.
        bullet = re.match(r"\s*((?:\d+[.)]|[-*•])\s+)", line)
        prefix = bullet.group(1) if bullet else ""
        kept = []
        for piece in re.split(r"(?<=[.!?])\s+", line[len(bullet.group(0)):] if bullet else line):
            cited = [int(n) - 1 for m in MARKER.finditer(piece) for n in m.group(1).split(",")
                     if 0 < int(n) <= len(evidence)]
            body = re.sub(r"\s+([.,;:!?])", r"\1", MARKER.sub("", piece)).strip()
            if not keywords(body):
                continue
            scores = [support(body, p["text"]) for p in evidence]
            supported_cites = [i for i in cited if scores[i] >= min_support]
            best = supported_cites[0] if supported_cites else max(range(len(evidence)), key=scores.__getitem__)
            if scores[best] < min_support:
                continue
            if body.endswith(":"):  # a lead-in such as "The five stages are:" needs no citation
                kept.append(body)
            elif body[-1] in ".!?":
                kept.append(f"{body[:-1]} [{best + 1}]{body[-1]}")
            else:
                kept.append(f"{body} [{best + 1}]")
        if kept:
            lines.append(prefix + " ".join(kept))
    return "\n".join(lines)


def extractive_answer(question: str, evidence: list[dict], max_sentences: int = 3) -> str:
    """Pick the passage sentences that share the most keywords with the question."""
    q = keywords(question)
    if not q:
        return INSUFFICIENT_TOKEN
    candidates = []
    for n, p in enumerate(evidence):
        text = p["text"]
        if p.get("section") and text.startswith(p["section"]):  # don't glue the heading onto the first sentence
            text = text[len(p["section"]):]
        for pos, sentence in enumerate(re.split(r"(?<=[.!?])\s+", text.strip())):
            overlap = len(q & keywords(sentence))
            if overlap and len(sentence.split()) >= 5:
                candidates.append((overlap / len(q) + p["score"], n, pos, sentence.strip()))
    if not candidates:
        return INSUFFICIENT_TOKEN
    best = sorted(candidates, key=lambda c: c[0], reverse=True)[:max_sentences]
    best.sort(key=lambda c: (c[1], c[2]))  # keep reading order
    return " ".join(f"{sentence} [{n + 1}]" for _, n, _, sentence in best)


class HFGenerator:
    """A local chat model loaded with Hugging Face Transformers at a pinned revision."""

    def __init__(self, model_name: str, revision: str | None = None, max_new_tokens: int = 300,
                 device: str = "cpu", dtype: str = "float32"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if revision and revision.startswith("REPLACE"):
            raise ValueError("Set generation.revision in config/config.yaml to a pinned commit hash.")
        self.torch = torch
        self.model_name, self.revision = model_name, revision
        self.max_new_tokens, self.device = max_new_tokens, device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        # bfloat16 halves memory use; float32 is faster on CPUs without native bfloat16 support.
        self.model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, dtype=getattr(torch, dtype))
        self.model.to(device).eval()

    def __call__(self, system: str, user: str, on_text=None) -> str:
        """Return the model's reply; if `on_text` is given, call it with the text so far as it is written."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.device)
        if on_text is None:
            with self.torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            new_tokens = output[0][inputs["input_ids"].shape[1]:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        import threading

        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        worker = threading.Thread(target=self.model.generate, kwargs={
            **inputs, "max_new_tokens": self.max_new_tokens, "do_sample": False, "streamer": streamer})
        worker.start()
        text = ""
        for piece in streamer:
            text += piece
            on_text(text)
        worker.join()
        return text.strip()


class OllamaGenerator:
    """A chat model served by a local Ollama install (https://ollama.com)."""

    def __init__(self, model: str, url: str = "http://localhost:11434", max_new_tokens: int = 300,
                 timeout: float = 300, keep_alive: str = "30m"):
        self.model_name, self.url = model, url.rstrip("/")
        self.max_new_tokens, self.timeout, self.keep_alive = max_new_tokens, timeout, keep_alive

    def warm_up(self) -> None:
        """Load the model into Ollama's memory now, so the first real answer is not slow."""
        self._post("/api/generate", {"model": self.model_name, "keep_alive": self.keep_alive})

    def __call__(self, system: str, user: str, on_text=None) -> str:
        """Return the model's reply; if `on_text` is given, call it with the text so far as it is written."""
        payload = {
            "model": self.model_name,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": on_text is not None,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0, "num_predict": self.max_new_tokens},
        }
        if on_text is None:
            return self._post("/api/chat", payload)["message"]["content"].strip()
        text = ""

        def read(response):
            nonlocal text
            for line in response:  # Ollama streams one JSON object per line
                if line.strip():
                    text += json.loads(line).get("message", {}).get("content", "")
                    on_text(text)

        self._post("/api/chat", payload, read=read)
        return text.strip()

    def _post(self, path: str, payload: dict, read=None):
        """POST JSON to Ollama; return the decoded reply, or pass the open response to `read`."""
        request = urllib.request.Request(f"{self.url}{path}", data=json.dumps(payload).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return read(response) if read else json.load(response)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Ollama could not run '{self.model_name}' ({e.code}). "
                               f"Download it with: ollama pull {self.model_name}") from e
        except (urllib.error.URLError, OSError) as e:
            raise RuntimeError(f"Ollama is not running at {self.url}. Install it from https://ollama.com, "
                               f"then run: ollama pull {self.model_name}") from e


def insufficient_response() -> dict:
    return {"answer": INSUFFICIENT_MESSAGE, "sources": [], "insufficient_evidence": True}


class Answerer:
    def __init__(self, retriever, cfg: dict, generator=None):
        self.retriever = retriever
        self.backend = cfg["generation"]["backend"]
        self.gen_cfg = cfg["generation"]
        self.top_k = cfg["retrieval"]["top_k"]
        self.min_score = cfg["retrieval"]["min_score"]
        self._generator = generator

    @property
    def generator(self):
        g = self.gen_cfg
        if self._generator is None and self.backend == "hf":
            self._generator = HFGenerator(g["model_name"], g.get("revision"), g.get("max_new_tokens", 300),
                                          g.get("device", "cpu"), g.get("dtype", "float32"))
        elif self._generator is None and self.backend == "ollama":
            self._generator = OllamaGenerator(g["ollama_model"], g.get("ollama_url", "http://localhost:11434"),
                                              g.get("max_new_tokens", 300), keep_alive=g.get("ollama_keep_alive", "30m"))
        return self._generator

    def run(self, question: str, top_k: int | None = None, on_draft=None) -> tuple[dict, list[dict]]:
        """Return (common response, cited passages in citation order).

        `on_draft`, if given, receives the model's unchecked text as it is written, so an interface can
        show progress; the returned response is the checked answer that replaces it.
        """
        hits = self.retriever.search(question, top_k or self.top_k)
        evidence = [h for h in hits if h["score"] >= self.min_score]
        if not evidence:
            return insufficient_response(), []

        if self.backend == "extractive":
            raw = extractive_answer(question, evidence)
        elif self.backend in ("hf", "ollama"):
            # The model reads only the best few passages: prompt length dominates CPU answer time.
            evidence = evidence[: self.gen_cfg.get("max_passages", len(evidence))]
            stream = {"on_text": on_draft} if on_draft else {}
            raw = self.generator(SYSTEM_PROMPT, build_prompt(question, evidence), **stream)
            if INSUFFICIENT_TOKEN not in raw:
                # The model may skip, misplace or invent citations, or add outside knowledge:
                # keep only sentences a retrieved passage actually supports.
                raw = ground_answer(raw, evidence, self.gen_cfg.get("min_support", 0.5))
                unverified = not MARKER.search(raw) or len(MARKER.sub("", raw).split()) < MIN_ANSWER_WORDS
                if unverified and self.gen_cfg.get("fallback_to_quotes", True):
                    # Relevant passages were found but the model's wording could not be checked
                    # against them: show the books' own sentences instead of refusing.
                    quotes = extractive_answer(question, evidence)
                    raw = quotes if INSUFFICIENT_TOKEN in quotes else QUOTE_PREFIX + quotes
        else:
            raise ValueError(f"Unknown generation backend: {self.backend}")

        if INSUFFICIENT_TOKEN in raw:
            return insufficient_response(), []
        text, order = renumber_markers(raw, len(evidence))
        # No checkable citation, or citations with no answer around them: nothing can be shown as supported.
        if not order or len(MARKER.sub("", text).split()) < MIN_ANSWER_WORDS:
            return insufficient_response(), []

        cited = [evidence[i] for i in order]
        sources, invalid = resolve_sources([to_source(p) for p in cited], self.retriever.by_id)
        if invalid:
            return insufficient_response(), []
        return {"answer": text, "sources": sources, "insufficient_evidence": False}, cited

    def answer(self, question: str, top_k: int | None = None) -> dict:
        return self.run(question, top_k)[0]


_default: Answerer | None = None


def get_answerer(cfg: dict | None = None) -> Answerer:
    global _default
    if _default is None:
        from src.retrieve import get_retriever

        cfg = cfg or load_config()
        _default = Answerer(get_retriever(cfg), cfg)
    return _default


def answer(question: str, top_k: int | None = None) -> dict:
    """Answer a question from the books, citing the passages used."""
    return get_answerer().answer(question, top_k)


if __name__ == "__main__":
    import json
    import sys

    q = " ".join(sys.argv[1:]) or input("Question: ")
    print(json.dumps(answer(q), indent=2, ensure_ascii=False))
