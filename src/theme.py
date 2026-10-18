"""Look of the chat page: yellow, black, white and grey, in a dark and a light mode.

Raised controls use a clay style (soft, rounded, with inner highlights); panels that sit over the
page use a glass style (translucent and blurred). Only the colour values differ between modes.
"""
from __future__ import annotations

from src.logo import CSS as LOGO_CSS

THEMES = {
    "dark": {
        "bg": "#111114", "glow": "rgba(255,194,26,.10)", "glow2": "rgba(160,160,170,.08)",
        "clay": "#1D1D22", "clay-hover": "#25252B", "glass": "rgba(29,29,34,.55)", "glass-strong": "rgba(29,29,34,.94)",
        "line": "rgba(255,255,255,.09)", "text": "#F4F3EE", "muted": "#A2A1A8",
        "accent": "#FFC21A", "accent-ink": "#0A0A0D", "accent-text": "#FFC21A",
        "clay-shadow": "10px 12px 24px rgba(0,0,0,.50), inset 3px 3px 7px rgba(255,255,255,.06), "
                       "inset -4px -5px 9px rgba(0,0,0,.40)",
        "gold-shadow": "8px 10px 22px rgba(0,0,0,.45), inset 3px 3px 7px rgba(255,255,255,.45), "
                       "inset -4px -5px 9px rgba(160,100,0,.40)",
        "notice": "rgba(255,255,255,.05)",
    },
    "light": {
        "bg": "#ECECE8", "glow": "rgba(255,194,26,.18)", "glow2": "rgba(120,120,130,.10)",
        "clay": "#F6F6F3", "clay-hover": "#FFFFFF", "glass": "rgba(255,255,255,.55)", "glass-strong": "rgba(250,250,247,.94)",
        "line": "rgba(10,10,13,.09)", "text": "#141416", "muted": "#5C5C63",
        "accent": "#FFC21A", "accent-ink": "#0A0A0D", "accent-text": "#7A5600",
        "clay-shadow": "8px 10px 22px rgba(120,120,110,.28), inset 3px 3px 7px rgba(255,255,255,.95), "
                       "inset -4px -5px 9px rgba(0,0,0,.06)",
        "gold-shadow": "8px 10px 20px rgba(150,110,0,.25), inset 3px 3px 7px rgba(255,255,255,.55), "
                       "inset -4px -5px 9px rgba(160,100,0,.30)",
        "notice": "rgba(10,10,13,.05)",
    },
}

BASE = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap');
.stApp { background: radial-gradient(60rem 40rem at 12% -10%, var(--glow), transparent 60%),
                     radial-gradient(50rem 40rem at 100% 110%, var(--glow2), transparent 60%), var(--bg);
         color: var(--text); }
.stApp, .stApp p, .stApp label, .stApp li, .stApp summary, .stApp input, .stApp textarea, .stApp button {
    font-family: 'IBM Plex Sans', sans-serif; color: var(--text); }
.stApp [data-testid="stCaptionContainer"], .stApp small { color: var(--muted); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] { max-width: 46rem; padding: 2rem 1rem 3rem; }
[data-testid="stBottom"] > div { background: linear-gradient(to top, var(--bg) 72%, transparent); }
[data-testid="stBottomBlockContainer"] { max-width: 46rem; padding: .75rem 1rem 1.25rem; background: transparent; }
:focus-visible { outline: 2px solid var(--accent) !important; outline-offset: 2px; }

/* Glass: sidebar, passages panel, mode menu */
section[data-testid="stSidebar"] { background: var(--glass); backdrop-filter: blur(18px) saturate(140%);
    -webkit-backdrop-filter: blur(18px) saturate(140%); border-right: 1px solid var(--line); }
[data-testid="stExpander"] details { background: var(--glass); backdrop-filter: blur(14px);
    border: 1px solid var(--line); border-radius: 18px; }
[data-testid="stPopoverBody"] { background: var(--glass-strong) !important; backdrop-filter: blur(18px);
    border: 1px solid var(--line); border-radius: 18px; box-shadow: var(--clay-shadow); }
[data-testid="stPopoverBody"] label, [data-testid="stPopoverBody"] p,
[data-testid="stPopoverBody"] [data-testid="stSliderThumbValue"] { color: var(--text) !important;
    font-family: 'IBM Plex Sans', sans-serif; }
[data-testid="stPopoverBody"] [data-testid="stCaptionContainer"] p,
[data-testid="stPopoverBody"] [data-testid="stWidgetLabel"] p { color: var(--muted) !important; }

/* Clay: buttons, chips, message box */
.stApp .stButton button, [data-testid="stPopover"] button, [data-testid="stButtonGroup"] button {
    background: var(--clay); border: 1px solid var(--line); border-radius: 14px; box-shadow: var(--clay-shadow);
    font-family: 'Space Grotesk', sans-serif; font-weight: 500; transition: transform .15s ease, background .15s ease; }
.stApp .stButton button:hover, [data-testid="stPopover"] button:hover, [data-testid="stButtonGroup"] button:hover {
    background: var(--clay-hover); transform: translateY(-1px); border-color: var(--accent); }
.stApp .stButton button[kind="tertiary"] { background: transparent; box-shadow: none; border-color: transparent; }
[data-testid="stButtonGroup"] > div { justify-content: center; gap: .55rem; }
[data-testid="stChatInput"] { background: var(--clay); border: 1px solid var(--line); border-radius: 22px;
    box-shadow: var(--clay-shadow); }
[data-testid="stChatInput"] > div, [data-testid="stChatInput"] textarea { background: transparent !important; }
/* Only the outer rounded box shows a border: the inner box and the text area draw none. */
[data-testid="stChatInput"] div { border: none !important; box-shadow: none !important; }
[data-testid="stChatInput"] textarea, [data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInput"] textarea:focus-visible { outline: none !important; box-shadow: none !important; }
[data-testid="stChatInput"] textarea::placeholder { color: var(--muted); }
[data-testid="stChatInput"]:focus-within { border-color: var(--accent); }
[data-testid="stChatInputSubmitButton"] { background: var(--accent) !important; border-radius: 12px; }

/* Mode menu under the message box: plain text, no border or shadow, aligned right */
.st-key-composer_tools { margin-top: -.35rem; padding-right: .35rem; }
.st-key-composer_tools [data-testid="stPopover"] button, .st-key-composer_tools [data-testid="stPopover"] button:hover {
    background: transparent; border: none; box-shadow: none; transform: none; padding: .15rem .35rem; min-height: 0; }
.st-key-composer_tools [data-testid="stPopover"] button p { font: 500 .9rem 'IBM Plex Sans', sans-serif; color: var(--text); }
.st-key-composer_tools [data-testid="stPopover"] button span[class*="gray"],
.st-key-composer_tools [data-testid="stPopover"] button p span { color: var(--muted) !important; }
.st-key-composer_tools [data-testid="stPopover"] button:hover p { color: var(--accent-text); }

/* Book menu in the sidebar: a clay control whose open list is glass, like the mode menu */
[data-testid="stSelectbox"] [role="group"] { background: var(--clay); border: 1px solid var(--line);
    border-radius: 14px; box-shadow: var(--clay-shadow); cursor: pointer; transition: background .15s ease; }
[data-testid="stSelectbox"] [role="group"]:hover { background: var(--clay-hover); border-color: var(--accent); }
[data-testid="stSelectbox"] [role="group"]:focus-within { border-color: var(--accent); }
[data-testid="stSelectbox"] [role="group"] input { color: var(--text); font-family: 'Space Grotesk', sans-serif;
    font-weight: 500; cursor: pointer; }
[data-testid="stSelectbox"] [role="group"] svg { color: var(--text); }
[data-testid="stSelectboxVirtualDropdown"] { min-width: min(22rem, 90vw); background: var(--glass-strong); backdrop-filter: blur(18px);
    border: 1px solid var(--line); border-radius: 14px; box-shadow: var(--clay-shadow); }
[data-testid="stSelectboxVirtualDropdown"] [role="option"] { color: var(--text); font-family: 'IBM Plex Sans', sans-serif;
    border-radius: 10px; }
[data-testid="stSelectboxVirtualDropdown"] [role="option"][data-focused="true"] { background: var(--clay-hover); }
[data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"] { color: var(--accent-text); font-weight: 600; }
[data-testid="stChatInputSubmitButton"] svg { fill: var(--accent-ink); color: var(--accent-ink); }

/* Sidebar content */
section[data-testid="stSidebar"] .stButton button { justify-content: flex-start; text-align: left; width: 100%;
    min-height: 2.3rem; }
section[data-testid="stSidebar"] .stButton button p { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.brand { display: flex; align-items: center; gap: .6rem; font: 700 1.25rem 'Space Grotesk', sans-serif;
         letter-spacing: .04em; color: var(--text); margin: .2rem 0 1.1rem; }
.side-label { font: 500 .75rem 'Space Grotesk', sans-serif; letter-spacing: .04em; color: var(--muted);
              margin: 1.2rem 0 .3rem .25rem; }
.side-foot { font-size: .78rem; color: var(--muted); line-height: 1.5; margin-top: 1.2rem; }

/* Empty chat */
.greet { text-align: center; margin: 13vh 0 1.7rem; }
.greet-mark { display: flex; justify-content: center; margin-bottom: 1rem; }
.greet h1 { font: 600 2.6rem/1.1 'Space Grotesk', sans-serif; letter-spacing: -.02em; color: var(--text); margin: 0; }
.greet p { color: var(--muted); margin: .8rem 0 0; }

/* Conversation */
.chat-title { text-align: center; color: var(--muted); font: 500 .85rem 'Space Grotesk', sans-serif; margin: -.75rem 0 1rem; }
.user-row { display: flex; justify-content: flex-end; margin: 1.6rem 0 1rem; }
.user-bubble { background: var(--accent); color: var(--accent-ink); border-radius: 20px 20px 6px 20px;
               padding: .65rem 1.05rem; max-width: 82%; white-space: pre-wrap; line-height: 1.55; font-weight: 500;
               box-shadow: var(--gold-shadow); }
.answer { font-size: 1.04rem; line-height: 1.75; color: var(--text); white-space: pre-line; }
.answer.draft { opacity: .55; }
.checking { font-size: .8rem; color: var(--muted); margin-top: .3rem; }
.cite { display: inline-block; min-width: 1.4em; padding: 0 .32em; margin-left: .15em; border-radius: 6px;
        background: var(--accent); color: var(--accent-ink); font: 700 .72rem/1.55 'Space Grotesk', sans-serif;
        text-align: center; vertical-align: .2em; }
.chips { display: flex; flex-wrap: wrap; gap: .5rem; margin: .8rem 0 .3rem; }
.chip { display: inline-flex; align-items: center; gap: .4rem; background: var(--clay); border: 1px solid var(--line);
        border-radius: 999px; padding: .2rem .75rem .2rem .3rem; font-size: .8rem; color: var(--text);
        box-shadow: var(--clay-shadow); }
.chip .cite { margin: 0; vertical-align: 0; }
.turn-meta { font-size: .75rem; color: var(--muted); }
.notice { background: var(--notice); border: 1px solid var(--line); border-radius: 16px; padding: .8rem 1rem; }
.notice b { color: var(--text); }
.thinking { display: flex; align-items: center; gap: .75rem; color: var(--muted); font-size: .92rem; margin: .4rem 0; }
.source { display: grid; grid-template-columns: 4.5rem 1fr; background: var(--glass); border: 1px solid var(--line);
          border-radius: 14px; margin: .5rem 0 .25rem; overflow: hidden; }
.source .page { background: var(--accent); color: var(--accent-ink); padding: .55rem .4rem; text-align: center; }
.source .page b { display: block; font: 700 1.4rem/1.1 'Space Grotesk', sans-serif; }
.source .page span { font-size: .7rem; }
.source .meta { padding: .55rem .8rem; font-size: .85rem; }
.source .meta .title { font-weight: 600; }
.source .meta .sub { color: var(--muted); }
.passage { font-size: .85rem; color: var(--muted); line-height: 1.6; margin: .25rem 0 .9rem; white-space: pre-line; }
@media (max-width: 640px) { .greet { margin-top: 7vh; } .greet h1 { font-size: 2rem; } .user-bubble { max-width: 92%; } }
@media (prefers-reduced-motion: reduce) { .stApp .stButton button,
    [data-testid="stSelectbox"] [role="group"] { transition: none; } }
"""


def build_css(mode: str) -> str:
    """The full stylesheet for "dark" or "light" mode."""
    values = THEMES.get(mode, THEMES["dark"])
    variables = "".join(f"--{name}:{value};" for name, value in values.items())
    return f"<style>:root{{{variables}}}{BASE}{LOGO_CSS}</style>"
