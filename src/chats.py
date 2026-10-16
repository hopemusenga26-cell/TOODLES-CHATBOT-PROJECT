"""Saved conversations for the chat page, one JSON file per chat on this computer.

Chats quote book passages, so the folder is kept out of Git like the books themselves.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.config import read_json, write_json

CHAT_ID = re.compile(r"^[0-9a-f]{12}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def new_chat() -> dict:
    now = _now()
    return {"id": uuid.uuid4().hex[:12], "title": "", "created": now, "updated": now, "turns": []}


def title_from(question: str, limit: int = 48) -> str:
    text = " ".join(question.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _path(folder: str | Path, chat_id: str) -> Path:
    if not CHAT_ID.match(chat_id):  # ids come from file names; never build paths from anything else
        raise ValueError(f"Not a chat id: {chat_id!r}")
    return Path(folder) / f"{chat_id}.json"


def save_chat(chat: dict, folder: str | Path) -> None:
    """Save a chat that has at least one finished answer; empty chats are not kept."""
    if not any("response" in t for t in chat["turns"]):
        return
    chat["updated"] = _now()
    if not chat["title"]:
        chat["title"] = title_from(chat["turns"][0]["question"])
    write_json({**chat, "turns": [t for t in chat["turns"] if "response" in t or "error" in t]},
               _path(folder, chat["id"]))


def load_chat(chat_id: str, folder: str | Path) -> dict | None:
    return read_json(_path(folder, chat_id), default=None)


def delete_chat(chat_id: str, folder: str | Path) -> None:
    _path(folder, chat_id).unlink(missing_ok=True)


def list_chats(folder: str | Path, limit: int = 30) -> list[dict]:
    """Most recently updated first: [{"id", "title", "updated"}]."""
    chats = []
    for path in Path(folder).glob("*.json"):
        if CHAT_ID.match(path.stem):
            data = read_json(path, default=None)
            if data:
                chats.append({"id": data["id"], "title": data.get("title") or "Untitled chat",
                              "updated": data.get("updated", "")})
    return sorted(chats, key=lambda c: c["updated"], reverse=True)[:limit]
