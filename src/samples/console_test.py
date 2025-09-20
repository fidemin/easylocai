import asyncio
from datetime import datetime
from typing import List, Dict

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()


def build_session() -> PromptSession:
    return PromptSession(
        multiline=True,
        prompt_continuation=lambda width, line_no, is_soft_wrap: "... ",
    )


def build_keybindings() -> KeyBindings:
    kb = KeyBindings()

    # Newline without submitting
    @kb.add("escape", Keys.Enter)  # Shift+Enter
    def _(event):
        event.current_buffer.insert_text("\n")

    # Submit on Enter
    @kb.add("enter")
    def _(event):
        event.current_buffer.validate_and_handle()

    return kb


def bottom_toolbar_ansi() -> ANSI:
    with console.capture() as cap:
        console.print(
            "[b]Enter[/b]=submit   [b]Alt/Shift+Enter[/b]=newline   [b]Ctrl+C[/b]=cancel   [b]Ctrl+D[/b]=exit",
            style="cyan",
        )
    return ANSI(cap.get())


def render_chat(messages: List[Dict[str, str]]) -> None:
    """Clear screen and render the whole conversation as Rich panels."""
    console.clear()
    console.print(Panel.fit("CLI Chat (Rich × prompt_toolkit)", border_style="magenta"))
    for msg in messages:
        who = msg["role"]
        text = msg["content"]
        border = "green" if who == "user" else "cyan"
        title = "You" if who == "user" else "Assistant"
        console.print(
            Panel(Markdown(text), title=title, border_style=border, padding=(1, 2))
        )


async def cli_chat():
    session = build_session()
    kb = build_keybindings()
    messages: List[Dict[str, str]] = []

    while True:
        render_chat(messages)
        try:
            with patch_stdout():  # keeps Rich output tidy while PTK is active
                user_text = await session.prompt_async(
                    "> ",
                    key_bindings=kb,
                    bottom_toolbar=bottom_toolbar_ansi,
                )
        except EOFError:
            console.print("\n[dim]Goodbye.[/dim]")
            break
        except KeyboardInterrupt:
            # Ctrl+C clears current input line; continue the loop
            continue

        user_text = user_text.rstrip("\n")
        if not user_text.strip():
            continue

        # Add user message and re-render so it appears immediately
        messages.append({"role": "user", "content": user_text})
        render_chat(messages)

        # --- Your model call goes here -------------------------------------
        # Replace this stub with your real LLM call (sync or async).
        # For streaming, you can accumulate chunks then append once done.
        await asyncio.sleep(0.05)
        reply = f"Echo {datetime.now().strftime('%H:%M:%S')}\n\n{user_text}"
        # --------------------------------------------------------------------

        messages.append({"role": "assistant", "content": reply})
        # Print only the new assistant panel instead of redrawing everything:
        console.print(
            Panel(
                Markdown(reply), title="Assistant", border_style="cyan", padding=(1, 2)
            )
        )


if __name__ == "__main__":
    asyncio.run(cli_chat())
