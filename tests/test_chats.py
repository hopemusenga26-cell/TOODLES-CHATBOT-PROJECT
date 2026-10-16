"""Saved conversations: saving, listing, loading and deleting."""
import pytest

from src.chats import delete_chat, list_chats, load_chat, new_chat, save_chat, title_from


def _answered(question, answer="An answer [1]."):
    return {"question": question, "mode": "ollama", "seconds": 1.0,
            "response": {"answer": answer, "sources": [], "insufficient_evidence": False}, "cited": []}


def test_empty_or_unanswered_chats_are_not_saved(tmp_path):
    chat = new_chat()
    save_chat(chat, tmp_path)
    chat["turns"].append({"question": "Still being answered", "pending": True})
    save_chat(chat, tmp_path)
    assert list_chats(tmp_path) == []


def test_saved_chat_round_trips_and_gets_a_title(tmp_path):
    chat = new_chat()
    chat["turns"] = [_answered("What is linear regression and why is it used so widely in statistics?")]
    save_chat(chat, tmp_path)
    loaded = load_chat(chat["id"], tmp_path)
    assert loaded["turns"][0]["response"]["answer"] == "An answer [1]."
    assert loaded["title"] == title_from(chat["turns"][0]["question"])
    assert loaded["title"].endswith("…") and len(loaded["title"]) <= 48


def test_recent_chats_are_listed_newest_first_and_can_be_deleted(tmp_path):
    first, second = new_chat(), new_chat()
    for chat, q in ((first, "First question"), (second, "Second question")):
        chat["turns"] = [_answered(q)]
        save_chat(chat, tmp_path)
    first["turns"].append(_answered("Follow-up"))
    save_chat(first, tmp_path)  # updated last, so listed first
    assert [c["title"] for c in list_chats(tmp_path)] == ["First question", "Second question"]
    delete_chat(first["id"], tmp_path)
    assert [c["id"] for c in list_chats(tmp_path)] == [second["id"]]


def test_chat_ids_cannot_point_outside_the_folder(tmp_path):
    with pytest.raises(ValueError):
        load_chat("../../secret", tmp_path)
