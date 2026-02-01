import json
import logging
from contextlib import AsyncExitStack

from ollama import AsyncClient

from easylocai.search_engines.advanced_search_engine import AdvancedSearchEngine
from rich import get_console

from easylocai.config import user_config_path
from easylocai.schemas.common import UserConversation
from easylocai.utlis.console_util import multiline_input, render_chat, ConsoleSpinner
from easylocai.workflow import EasylocaiWorkflow

logger = logging.getLogger(__name__)


async def run_agent_workflow_main():
    console = get_console()

    ollama_client = AsyncClient(host="http://localhost:11434")
    search_engine = AdvancedSearchEngine()

    config_path = user_config_path()

    with open(config_path) as f:
        config_dict = json.load(f)

    workflow = EasylocaiWorkflow(
        config_dict=config_dict,
        search_engine=search_engine,
        ollama_client=ollama_client,
    )

    messages = []
    user_conversations: list[UserConversation] = []

    stack = AsyncExitStack()
    async with stack:
        await workflow.initialize(stack)

        while True:
            render_chat(console, messages)
            user_input = await multiline_input("> ")
            if user_input.strip().lower() in {"exit", "quit"}:
                break

            messages.append({"role": "user", "content": user_input})
            render_chat(console, messages)

            async_generator = workflow.run(
                user_input,
                user_conversations=user_conversations,
            )

            answer = None
            with ConsoleSpinner(console) as spinner:
                async for output in async_generator:
                    if output.type == "status":
                        spinner.set_prefix(output.message)
                        continue

                    if output.type == "result":
                        answer = output.message
                        messages.append({"role": "assistant", "content": answer})
                        user_conversations.append(
                            UserConversation(
                                user_query=user_input, assistant_answer=answer
                            )
                        )


workflow_registry = {
    "main": run_agent_workflow_main,
}


async def run_agent_workflow(flag: str | None = None):
    if flag is None:
        flag = "main"

    workflow_function = workflow_registry.get(flag)
    if workflow_function is None:
        raise ValueError(f"Unknown workflow flag: {flag}")

    await workflow_function()
