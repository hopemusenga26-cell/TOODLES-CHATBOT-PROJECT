"""The optional Ollama backend, tested against a small fake Ollama server."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from src.generate import Answerer, OllamaGenerator


@pytest.fixture
def fake_ollama():
    """Serve /api/chat like Ollama does, recording each request."""
    requests = []

    class Handler(BaseHTTPRequestHandler):
        reply = "A cache stores copies of data from main memory [1]."

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if body.get("stream"):  # one JSON object per line, a few words at a time
                words = self.reply.split(" ")
                for n, word in enumerate(words):
                    piece = word + ("" if n == len(words) - 1 else " ")
                    self.wfile.write((json.dumps({"message": {"content": piece}, "done": False}) + "\n").encode())
                self.wfile.write((json.dumps({"message": {"content": ""}, "done": True}) + "\n").encode())
            else:
                self.wfile.write(json.dumps({"message": {"role": "assistant", "content": self.reply}}).encode())

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", requests
    server.shutdown()


def test_ollama_generator_sends_chat_request(fake_ollama):
    url, requests = fake_ollama
    reply = OllamaGenerator("qwen2.5:1.5b", url, max_new_tokens=50)("system text", "user text")
    assert reply == "A cache stores copies of data from main memory [1]."
    path, body = requests[0]
    assert path == "/api/chat"
    assert body["model"] == "qwen2.5:1.5b" and body["stream"] is False
    assert body["options"] == {"temperature": 0, "num_predict": 50}
    assert body["keep_alive"] == "30m"
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


def test_ollama_warm_up_loads_the_model(fake_ollama):
    url, requests = fake_ollama
    OllamaGenerator("qwen2.5:1.5b", url, keep_alive="10m").warm_up()
    assert requests == [("/api/generate", {"model": "qwen2.5:1.5b", "keep_alive": "10m", "options": {}})]


def test_ollama_loads_and_answers_with_the_same_gpu_settings(fake_ollama):
    url, requests = fake_ollama
    generator = OllamaGenerator("qwen2.5:1.5b", url, max_new_tokens=50, num_ctx=2048, num_gpu=99)
    generator.warm_up()
    generator("system", "user")
    load, answer = requests[0][1]["options"], requests[1][1]["options"]
    assert load == {"num_ctx": 2048, "num_gpu": 99}
    assert answer == {"temperature": 0, "num_predict": 50, "num_ctx": 2048, "num_gpu": 99}


def test_ollama_backend_answers_with_checked_citations(fake_ollama, retriever, cfg):
    url, _ = fake_ollama
    cfg["generation"].update(backend="ollama", ollama_model="qwen2.5:1.5b", ollama_url=url)
    response = Answerer(retriever, cfg).answer("What is a cache?")
    assert response["insufficient_evidence"] is False
    assert response["sources"][0]["page"] == 1


def test_ollama_streams_text_as_it_is_written(fake_ollama):
    url, requests = fake_ollama
    drafts = []
    reply = OllamaGenerator("qwen2.5:1.5b", url)("system", "user", on_text=drafts.append)
    assert reply == "A cache stores copies of data from main memory [1]."
    assert requests[0][1]["stream"] is True
    assert len(drafts) > 3 and drafts[0] == "A " and drafts[-1].strip() == reply


def test_streamed_draft_is_replaced_by_checked_answer(fake_ollama, retriever, cfg):
    url, _ = fake_ollama
    cfg["generation"].update(backend="ollama", ollama_model="qwen2.5:1.5b", ollama_url=url)
    drafts = []
    response, cited = Answerer(retriever, cfg).run("What is a cache?", on_draft=drafts.append)
    assert drafts and response["insufficient_evidence"] is False
    assert response["sources"][0]["page"] == 1


def test_missing_ollama_gives_clear_error():
    generator = OllamaGenerator("qwen2.5:1.5b", "http://127.0.0.1:9", timeout=5)
    with pytest.raises(RuntimeError, match="Ollama is not running"):
        generator("system", "user")
