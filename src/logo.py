"""The TOODLES mark: a tilted gold T with two orb eyes and five dots on its crossbar.

It has four states, set by a CSS class (the animations live in the chat page's stylesheet):
    idle      eyes blink now and then
    thinking  eyes glance around and the dots ripple, like a typing indicator
    answered  the dots settle into a smile and the eyes look up
    nomatch   the mark greys out, the lids droop and the dots fall into a frown
"""
from __future__ import annotations

GOLD, GOLD_LIGHT, INK = "#FFC21A", "#FFE27A", "#0A0A0D"
T_SHAPE = "M84,176 L316,156 L316,218 L228,224 L228,352 L150,368 L172,330 L172,228 L84,226 Z"
DOTS = [(140, 197.5), (170, 195.2), (200, 193.0), (230, 190.8), (260, 188.5)]
EYES = {"L": (128, 124, -16), "R": (272, 106, 16)}  # centre x, centre y, lid tilt
STATES = ("idle", "thinking", "answered", "nomatch")


def _eye(side: str) -> str:
    cx, cy, tilt = EYES[side]
    return (
        f'<g class="tm-eye tm-eye{side}">'
        f'<circle cx="{cx}" cy="{cy}" r="40" fill="{GOLD_LIGHT}"/>'
        f'<g class="tm-pupil"><circle cx="{cx + 10}" cy="{cy - 6}" r="15" fill="{INK}"/></g>'
        f'<g clip-path="url(#tmClip{side})"><g transform="rotate({tilt} {cx} {cy})">'
        f'<rect class="tm-lid" x="{cx - 50}" y="{cy - 54}" width="100" height="46" fill="{INK}"/></g></g>'
        f'<circle cx="{cx}" cy="{cy}" r="40" fill="none" stroke="{INK}" stroke-width="8"/></g>'
    )


def mark_svg(size: int, state: str = "idle", label: str = "TOODLES") -> str:
    """The mark on a dark rounded tile, `size` px square."""
    if state not in STATES:
        raise ValueError(f"Unknown state {state!r}; use one of {STATES}")
    dots = "".join(f'<circle class="tm-dot tm-d{n}" cx="{x}" cy="{y}" r="9"/>' for n, (x, y) in enumerate(DOTS, 1))
    clips = "".join(f'<clipPath id="tmClip{s}"><circle cx="{x}" cy="{y}" r="37"/></clipPath>'
                    for s, (x, y, _) in EYES.items())
    return (
        f'<span class="tm-tile" style="width:{size}px;height:{size}px;border-radius:{round(size * 0.24)}px">'
        f'<svg class="tm tm-{state}" viewBox="62 52 276 336" role="img" aria-label="{label}">'
        f'<defs>{clips}</defs>'
        f'<path d="{T_SHAPE}" fill="{GOLD}" stroke="{GOLD}" stroke-width="16" stroke-linejoin="round"/>'
        f'<g fill="{INK}">{dots}</g>{_eye("L")}{_eye("R")}</svg></span>'
    )


# Animations for the four states, adapted from the logo's loader design.
CSS = """
.tm-tile { display: inline-flex; align-items: center; justify-content: center; flex: none;
           background: #0A0A0D; vertical-align: middle; }
.tm-tile svg { width: 78%; height: 78%; overflow: visible; transition: filter .6s ease; }
.tm-nomatch { filter: grayscale(1) brightness(.8); }
.tm-dot { transform-box: fill-box; transition: transform .6s cubic-bezier(.34,1.56,.64,1); }
.tm-thinking .tm-dot { animation: tmWave 1.1s ease-in-out infinite; }
.tm-thinking .tm-d2 { animation-delay: .12s; } .tm-thinking .tm-d3 { animation-delay: .24s; }
.tm-thinking .tm-d4 { animation-delay: .36s; } .tm-thinking .tm-d5 { animation-delay: .48s; }
@keyframes tmWave { 0%,60%,100% { transform: translateY(0); } 30% { transform: translateY(-14px); } }
.tm-answered .tm-d1, .tm-answered .tm-d5 { transform: translateY(-10px); }
.tm-answered .tm-d2, .tm-answered .tm-d4 { transform: translateY(1px); }
.tm-answered .tm-d3 { transform: translateY(6px); }
.tm-nomatch .tm-d1, .tm-nomatch .tm-d5 { transform: translateY(9px); }
.tm-nomatch .tm-d2, .tm-nomatch .tm-d4 { transform: translateY(-1px); }
.tm-nomatch .tm-d3 { transform: translateY(-6px); }
.tm-eye { transform-box: view-box; animation: tmBlink 4.2s ease-in-out infinite; }
.tm-eyeL { transform-origin: 128px 124px; } .tm-eyeR { transform-origin: 272px 106px; animation-delay: .06s; }
@keyframes tmBlink { 0%,88%,100% { transform: scaleY(1); } 92% { transform: scaleY(.08); } 96% { transform: scaleY(1); } }
.tm-pupil { transition: transform .55s ease; }
.tm-thinking .tm-pupil { animation: tmLook 3.4s ease-in-out infinite; }
@keyframes tmLook { 0%,18% { transform: translate(0,0); } 28%,52% { transform: translate(-9px,3px); }
                    62%,86% { transform: translate(6px,-4px); } 100% { transform: translate(0,0); } }
.tm-answered .tm-pupil { transform: translate(2px,-6px); }
.tm-nomatch .tm-pupil { transform: translate(-3px,11px); }
.tm-lid { transition: transform .55s ease; transform: translateY(-70px); }
.tm-nomatch .tm-lid { transform: translateY(0); }
@media (prefers-reduced-motion: reduce) {
  .tm-dot, .tm-eye, .tm-pupil { animation: none !important; }
}
"""
