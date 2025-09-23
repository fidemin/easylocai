import threading
import time

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.live import Live
from rich.panel import Panel


def build_session():
    return PromptSession(
        multiline=True,
        prompt_continuation=lambda width, line_no, is_soft_wrap: "... ",
    )


def build_keybindings():
    kb = KeyBindings()

    @kb.add("escape", Keys.Enter)
    def _(event):
        buf = event.current_buffer
        buf.insert_text("\n")

    @kb.add(Keys.Enter)
    def _(event):
        event.current_buffer.validate_and_handle()

    return kb


async def multiline_input(prompt_text: str = "> "):
    session = build_session()
    kb = build_keybindings()
    # Ensure stdout is safe while PTK runs inside asyncio
    with patch_stdout():
        return await session.prompt_async(prompt_text, key_bindings=kb)


def render_chat(console: Console, messages: list[dict[str, str]]) -> None:
    """Clear screen and render the whole conversation as Rich panels."""
    console.clear()
    for msg in messages:
        who = msg["role"]
        text = msg["content"]
        border = "green" if who == "user" else "cyan"
        title = "You" if who == "user" else "Assistant"
        console.print(Panel(text, title=title, border_style=border, padding=(1, 2)))


def spinner_task(
    stop_event,
    console: Console,
    prefix: str,
):
    with Live(console=console, refresh_per_second=4) as live:
        i = 0
        while not stop_event.is_set():
            i %= 4
            spinner = ["|", "/", "-", "\\"][i]
            live.update(f"{prefix}... {spinner}")
            time.sleep(0.1)
            i += 1

        # Clear loading line
        live.update("")


class ConsoleSpinner:
    def __init__(self, console: Console):
        self._console = console
        self._stop_event = threading.Event()
        self._prefix = "Thinking"
        self._thread = threading.Thread(target=self._live_spinner, args=())

    def __enter__(self):
        self._stop_event.clear()
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop_event.set()
        self._thread.join()

    def _live_spinner(self):
        with Live(console=self._console, refresh_per_second=4) as live:
            i = 0
            while not self._stop_event.is_set():
                i %= 4
                spinner = ["|", "/", "-", "\\"][i]
                live.update(f"{self._prefix}... {spinner}")
                time.sleep(0.1)
                i += 1

            # Clear loading line
            live.update("")

    def set_prefix(self, prefix: str):
        self._prefix = prefix
