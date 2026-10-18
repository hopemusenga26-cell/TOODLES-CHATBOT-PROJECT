"""Chat interface for the group chatbot.

Layout: a sidebar with new chat, recent chats and book details; an empty chat opens on a
greeting with the message box and suggested questions in the middle; once a conversation
starts, messages fill a centred column and the message box is pinned to the bottom. The book
menu in the sidebar picks where answers come from (all books, or one of them); the mode menu, plain
text under the message box, picks how they are written.

Usage:
    streamlit run src/app.py
"""
from __future__ import annotations

import html
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from src.build_index import META_FILE  # noqa: E402
from src.chats import delete_chat, list_chats, load_chat, new_chat, save_chat, title_from  # noqa: E402
from src.citations import format_citation  # noqa: E402
from src.config import load_config, read_json, resolve  # noqa: E402
from src.logo import mark_svg  # noqa: E402
from src.theme import build_css  # noqa: E402

# Answer modes: (label, what it does and typical time per answer on this laptop). The full
# Transformers model ("hf") is left out of the menu: in testing it was as accurate as Ultra but
# about 12 times slower. It is still used for the reference evaluation.
MODES = {
    "extractive": ("Toodles Instant", "Shows the matching book passages word for word. Instant, but you "
                                      "read and piece together the answer yourself."),
    "ollama": ("Toodles Ultra", "Writes a short answer in its own words, with page citations. About 6 to 9 "
                                "seconds. Recommended."),
}
SUGGESTIONS = [
    "What is linear regression?",
    "Explain the bias-variance trade-off",
    "How does gradient descent train a model?",
    "What does R-squared measure?",
]
SOURCE_FIELDS = ("book_id", "title", "author", "edition", "page", "page_label", "section", "chunk_id", "text")

# ---------- rendering ----------

def render_answer(text: str, draft: bool = False) -> str:
    safe = html.escape(text)
    safe = re.sub(r"\[(\d+(?:, \d+)*)\]",
                  lambda m: "".join(f'<span class="cite">{n.strip()}</span>' for n in m.group(1).split(",")), safe)
    return f'<div class="answer{" draft" if draft else ""}">{safe}</div>'


def short_title(p: dict) -> str:
    title = p.get("title") or p["book_id"]
    return title.split(":")[0]


def page_of(p: dict) -> str:
    return str(p.get("page_label") or p["page"])


def render_chips(cited: list[dict]) -> str:
    chips = "".join(f'<span class="chip"><span class="cite">{n}</span>{html.escape(short_title(p))}, p. '
                    f'{html.escape(page_of(p))}</span>' for n, p in enumerate(cited, start=1))
    return f'<div class="chips">{chips}</div>'


def render_source(n: int, p: dict) -> str:
    label = p.get("page_label")
    pdf_note = f"PDF p. {p['page']}" if label and str(label) != str(p["page"]) else "page"
    sub = ", ".join(html.escape(str(x)) for x in (p.get("author"), p.get("edition")) if x)
    section = f'<div class="sub">{html.escape(p["section"])}</div>' if p.get("section") else ""
    return (
        f'<div class="source"><div class="page"><b>{html.escape(page_of(p))}</b><span>{pdf_note}</span></div>'
        f'<div class="meta"><div class="title"><span class="cite">{n}</span> {html.escape(format_citation(p).split(",")[0])}</div>'
        f'<div class="sub">{sub}</div>{section}</div></div>'
        f'<div class="passage">{html.escape(p["text"])}</div>'
    )


def book_label(book_id: str | None) -> str:
    """Name of a book in the book menu and beside answers; None means all the books."""
    books = st.session_state.books
    if book_id is None:
        return f"All {len(books)} books"
    return books.get(book_id, book_id).split(":")[0]


def show_user(question: str) -> None:
    st.markdown(f'<div class="user-row"><div class="user-bubble">{html.escape(question)}</div></div>',
                unsafe_allow_html=True)


def show_answer(turn: dict) -> None:
    if "error" in turn:
        st.error(turn["error"])
        return
    response, cited = turn["response"], turn.get("cited", [])
    if response["insufficient_evidence"] and turn.get("book"):
        st.markdown(f'<div class="notice"><b>Not in {html.escape(book_label(turn["book"]))}.</b> This book '
                    "doesn't contain enough evidence to answer this. Pick another book, or all books, from the "
                    "Answer from menu in the sidebar.</div>", unsafe_allow_html=True)
    elif response["insufficient_evidence"]:
        st.markdown('<div class="notice"><b>Not in the books.</b> The books don\'t contain enough evidence '
                    "to answer this. Try rephrasing, or ask about a topic they cover.</div>", unsafe_allow_html=True)
    else:
        st.markdown(render_answer(response["answer"]), unsafe_allow_html=True)
        st.markdown(render_chips(cited), unsafe_allow_html=True)
        with st.expander(f"Show {'the passage' if len(cited) == 1 else f'{len(cited)} passages'}"):
            for n, p in enumerate(cited, start=1):
                st.markdown(render_source(n, p), unsafe_allow_html=True)
    label = MODES.get(turn.get("mode"), ("Toodles",))[0]
    source = f"searched {html.escape(book_label(turn['book']))}, " if turn.get("book") else ""
    state = "nomatch" if response["insufficient_evidence"] else "answered"
    st.markdown(f'{mark_svg(22, state)} <span class="turn-meta">{label}, {source}{turn.get("seconds", 0):.1f} s'
                '</span>', unsafe_allow_html=True)


# ---------- state ----------

def chats_dir(cfg: dict) -> Path:
    return resolve(cfg["paths"].get("chats", "data/chats"))


def start_new_chat() -> None:
    st.session_state.chat = new_chat()


def open_chat(chat_id: str) -> None:
    chat = load_chat(chat_id, chats_dir(load_config()))
    if chat:
        st.session_state.chat = chat


def delete_current_chat() -> None:
    delete_chat(st.session_state.chat["id"], chats_dir(load_config()))
    st.session_state.chat = new_chat()


def submit(question: str) -> None:
    question = question.strip()
    if not question:
        return
    chat = st.session_state.chat
    if not chat["turns"]:
        chat["title"] = title_from(question)
    chat["turns"].append({"question": question, "mode": st.session_state.mode,
                          "book": st.session_state.book, "pending": True})


def pick_suggestion() -> None:
    choice = st.session_state.get("suggestion")
    if choice:
        submit(choice)
        st.session_state.suggestion = None


@st.cache_resource(show_spinner=False)
def load_answerer(backend: str):
    from src.generate import Answerer
    from src.retrieve import get_retriever

    cfg = load_config()
    cfg["generation"]["backend"] = backend
    answerer = Answerer(get_retriever(cfg), cfg)
    if backend == "hf":
        _ = answerer.generator  # load the model now, under the loading message, not during the first answer
    elif backend == "ollama":
        try:
            answerer.generator.warm_up()
        except RuntimeError:
            pass  # reported when the question is answered
    return answerer


def answer_pending(turn: dict) -> None:
    """Answer a newly asked question in place, streaming the draft, then store the checked result."""
    status = st.empty()

    def thinking(text: str) -> None:
        status.markdown(f'<div class="thinking">{mark_svg(40, "thinking")}<span>{text}</span></div>',
                        unsafe_allow_html=True)

    thinking("TOODLES is getting the books ready…")
    try:
        answerer = load_answerer(turn["mode"])
    except ValueError as e:  # index built with a different embedding model, or out of date
        turn["error"] = (f"The book index can't be used yet: {e} If a rebuild is running, wait for it to "
                         "finish, then refresh this page.")
        return
    draft = st.empty()
    last_update = [0.0]

    def show_draft(text: str) -> None:
        # Unchecked model text, shown while it is written and replaced by the checked answer.
        if time.perf_counter() - last_update[0] > 0.15:
            draft.markdown(render_answer(text, draft=True) +
                           '<div class="checking">Writing… citations are checked when it finishes.</div>',
                           unsafe_allow_html=True)
            last_update[0] = time.perf_counter()

    thinking("TOODLES is thinking…")
    start = time.perf_counter()
    try:
        books = [turn["book"]] if turn.get("book") else None
        response, cited = answerer.run(turn["question"], on_draft=show_draft, books=books)
    except RuntimeError as e:  # e.g. Ollama not installed or not running
        turn["error"] = f"{e} Or pick another mode from the menu next to the message box."
        return
    finally:
        status.empty()
        draft.empty()
    turn.update(response=response, cited=[{f: p.get(f) for f in SOURCE_FIELDS} for p in cited],
                seconds=round(time.perf_counter() - start, 1))


# ---------- page ----------

def composer(name: str):
    """Message box with the answer-mode menu as plain text under it; returns the submitted text, if any."""
    state = st.session_state
    with st.container():  # keeps the message box in place instead of pinning it to the bottom
        question = st.chat_input(f"Ask {name}…", key="composer")
    with st.container(horizontal=True, horizontal_alignment="right", key="composer_tools"):
        # Values are passed in and stored explicitly: keyed widgets lose their value when the
        # message box moves from the middle of the page to the bottom.
        with st.popover(MODES[state.mode][0], type="tertiary"):
            state.mode = st.radio("Answer mode", list(MODES), index=list(MODES).index(state.mode),
                                  format_func=lambda m: MODES[m][0], captions=[c for _, c in MODES.values()])
    return question


def greeting() -> str:
    hour = datetime.now().hour
    return "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"


def main() -> None:
    cfg = load_config()
    name = cfg.get("app", {}).get("name", f"{cfg['group_id']} book chatbot")
    st.set_page_config(page_title=name, page_icon=str(resolve("assets/toodles_icon.png")), layout="centered", initial_sidebar_state="auto")
    meta = read_json(resolve(cfg["paths"]["index_dir"]) / META_FILE, default=None)
    folder = chats_dir(cfg)

    state = st.session_state
    state.setdefault("chat", new_chat())
    default_mode = cfg.get("app", {}).get("default_answer_writer", cfg["generation"]["backend"])
    state.setdefault("mode", default_mode if default_mode in MODES else "extractive")
    state.setdefault("books", {b["book_id"]: b.get("title") or b["book_id"]
                               for b in read_json(resolve(cfg["paths"]["manifest"]), default=[])})
    state.setdefault("book", None)
    state.setdefault("dark", cfg.get("app", {}).get("theme", "dark") == "dark")
    st.markdown(build_css("dark" if state.dark else "light"), unsafe_allow_html=True)
    chat = state.chat

    with st.sidebar:
        st.markdown(f'<div class="brand">{mark_svg(32)}{html.escape(name)}</div>',
                    unsafe_allow_html=True)
        st.button("New chat", icon=":material/add:", on_click=start_new_chat, type="secondary")
        st.markdown('<div class="side-label">Answer from</div>', unsafe_allow_html=True)
        options = [None, *state.books]
        state.book = st.selectbox("Answer from", options, index=options.index(state.book), format_func=book_label,
                                  label_visibility="collapsed",
                                  help="Where answers come from: all the books, or only the one you pick.")
        recents = list_chats(folder)
        if recents:
            st.markdown('<div class="side-label">Recents</div>', unsafe_allow_html=True)
            for c in recents:
                st.button(c["title"], key=f"open_{c['id']}", on_click=open_chat, args=(c["id"],),
                          type="secondary" if c["id"] == chat["id"] else "tertiary")
        if any("response" in t for t in chat["turns"]):
            st.button("Delete this chat", icon=":material/delete:", on_click=delete_current_chat, type="tertiary")
        st.toggle("Dark mode", key="dark")
        books = f"{meta['books']} books, {meta['passages']:,} passages" if meta else "No book index yet"
        st.markdown(f'<div class="side-foot">{books}<br>Group {html.escape(cfg["group_id"])}, '
                    f'{html.escape(cfg["subject"])}</div>', unsafe_allow_html=True)

    if not meta:
        st.info("No book index yet. Put the PDFs in `data/books/`, list them in `data/book_manifest.json`, "
                "then run `python -m src.pipeline`.")
        return

    if not chat["turns"]:
        count = f"{meta['books']} data science and machine learning books"
        st.markdown(f'<div class="greet"><div class="greet-mark">{mark_svg(66)}</div><h1>{greeting()}</h1>'
                    f'<p>{html.escape(name)} AI answers from {count}, with the page to prove it.</p></div>',
                    unsafe_allow_html=True)
        question = composer(name)
        st.pills("Suggestions", SUGGESTIONS, key="suggestion", on_change=pick_suggestion,
                 label_visibility="collapsed")
    else:
        st.markdown(f'<div class="chat-title">{html.escape(chat["title"])}</div>', unsafe_allow_html=True)
        pending = None
        for turn in chat["turns"]:
            show_user(turn["question"])
            if turn.get("pending"):
                pending = turn, st.container()  # filled in once the rest of the page is drawn
            else:
                show_answer(turn)
        with st.bottom:
            question = composer(name)
        if pending:
            # Answer last: Streamlit keeps showing the previous page's leftovers (such as the suggested
            # questions) until it reaches them, so the whole new page is drawn before the slow part.
            turn, slot = pending
            del turn["pending"]
            with slot:
                answer_pending(turn)
            save_chat(chat, folder)
            st.rerun()

    if question:
        submit(question)
        st.rerun()


main()
