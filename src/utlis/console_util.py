# console_util.py
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.patch_stdout import patch_stdout


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
