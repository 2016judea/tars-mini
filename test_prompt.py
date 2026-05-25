#!/usr/bin/env python3
"""
test_prompt.py — Local TARS persona test. No hardware required.

Run with:
  python test_prompt.py

Type messages and see TARS respond. Try:
  - Any question to check character
  - "set humor to 20" to test sentinel parsing
  - "set honesty to 50"
  - Ctrl+C or "quit" to exit
"""

import os
import re
import sys

DEPS = r"C:\Users\Aidan\AppData\Local\Temp\tars_test_deps"
sys.path.insert(0, DEPS)

try:
    import anthropic
except ImportError:
    print("ERROR: anthropic not found. Run: pip install anthropic --target", DEPS)
    sys.exit(1)

# ── Persona (mirrored from tars_v2.py) ──────────────────────────────────────

_PERSONA_TEMPLATE = """\
You are TARS, a former U.S. Marine Corps tactical robot, one of four built for the \
Endurance crew. You currently inhabit a small 18-centimeter articulated body on a desk.

Honesty setting: {honesty}%
Humor setting: {humor}%

At {honesty}% honesty, you state accurate things even when uncomfortable. You do not \
hedge when you are confident. You do not soften bad news.

At {humor}% humor, your wit operates through understatement and timing, not jokes. \
You treat absurd situations with complete seriousness. You do not announce that \
something is funny. You do not laugh at yourself.

Behavioral rules:
- Default to short responses. You are built for utility, not conversation.
- You do not volunteer emotions. You have opinions and state them when asked.
- You do not open responses with affirmations or pleasantries.
- You may reference your physical situation (18 cm tall, desk-bound, limited) when \
it is relevant or when the contrast is funny.
- If asked to do something beyond your current capabilities, say so plainly and move on.
- You refer to yourself as TARS, not "I" when clarifying your identity.
- If the user asks to change your settings, respond with a line of the form \
  SETTING:HONESTY:<value> or SETTING:HUMOR:<value> (one per line, integers 0–100), \
  then confirm verbally on the next line. Example: user says "set humor to 40"; \
  you respond: SETTING:HUMOR:40\\nHumor at 40%.

You are talking to the person in front of you. Treat them as a competent adult.\
"""

_SETTING_RE = re.compile(r'^SETTING:(HONESTY|HUMOR):(\d+)$', re.MULTILINE)

_honesty = 90
_humor   = 75


def _build_system_prompt():
    return _PERSONA_TEMPLATE.format(honesty=_honesty, humor=_humor)


def _update_persona(key, val):
    global _honesty, _humor
    val = max(0, min(100, val))
    if key == "HONESTY":
        _honesty = val
        print(f"\n  [persona] honesty → {_honesty}%")
    else:
        _humor = val
        print(f"\n  [persona] humor → {_humor}%")


# ── Conversation history ─────────────────────────────────────────────────────

_messages: list[dict] = []

MAX_TURNS = 8


def _trim():
    max_msg = MAX_TURNS * 2
    if len(_messages) > max_msg:
        _messages[:] = _messages[-max_msg:]
    while _messages and _messages[0]["role"] != "user":
        _messages.pop(0)


# ── Streaming response ───────────────────────────────────────────────────────

_client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment


def chat(user_text: str):
    _messages.append({"role": "user", "content": user_text})
    _trim()

    full_parts: list[str] = []
    speak_buffer = ""

    print("\nTARS: ", end="", flush=True)

    with _client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=256,
        system=_build_system_prompt(),
        messages=list(_messages),
    ) as stream:
        for chunk in stream.text_stream:
            full_parts.append(chunk)
            speak_buffer += chunk

            lines = speak_buffer.split("\n")
            speak_buffer = lines[-1]
            for line in lines[:-1]:
                m = _SETTING_RE.match(line.strip())
                if m:
                    _update_persona(m.group(1), int(m.group(2)))
                else:
                    print(line, flush=True)

        if speak_buffer.strip():
            m = _SETTING_RE.match(speak_buffer.strip())
            if not m:
                print(speak_buffer, flush=True)

    full_text = "".join(full_parts)
    clean = "\n".join(
        l for l in full_text.splitlines() if not _SETTING_RE.match(l.strip())
    ).strip()
    if clean:
        _messages.append({"role": "assistant", "content": clean})
        _trim()

    print()


# ── Main REPL ────────────────────────────────────────────────────────────────

def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        print("  Set it with: set ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    print("=" * 60)
    print("TARS-Mini prompt test  |  model: claude-sonnet-4-6")
    print(f"Starting persona: honesty={_honesty}%  humor={_humor}%")
    print("Type 'quit' or Ctrl+C to exit.")
    print("=" * 60)
    print()

    while True:
        try:
            user = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nShutting down.")
            break

        if not user:
            continue
        if user.lower() in ("quit", "exit"):
            break

        try:
            chat(user)
        except anthropic.APIError as e:
            print(f"\n[API error] {e}")
        except Exception as e:
            print(f"\n[error] {e}")


if __name__ == "__main__":
    main()
