import asyncio
import csv
import logging
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass

from chromadb import Client
from tabulate import tabulate

from easylocai.core.search_engine import SearchEngineCollection, Record
from easylocai.core.tool_manager import ServerManager
from easylocai.search_engines.advanced_search_engine import AdvancedSearchEngine
from easylocai.search_engines.keyword_search_engine import KeywordSearchEngine
from easylocai.search_engines.semantic_search_engine import SemanticSearchEngine

# supress mcp stdio client logs (too noisy)
logging.getLogger("mcp.client.stdio").setLevel(logging.CRITICAL)

mcp_servers = {
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
    },
    "notion_api": {
        "command": "docker",
        "args": ["run", "--rm", "-i", "-e", "NOTION_TOKEN", "mcp/notion"],
        "env": {
            "NOTION_TOKEN": "token_goes_here",
        },
    },
    "web-search": {
        "args": ["open-websearch@latest"],
        "command": "npx",
        "env": {
            "MODE": "stdio",
            "DEFAULT_SEARCH_ENGINE": "duckduckgo",
            "ALLOWED_SEARCH_ENGINES": "duckduckgo,bing,exa",
        },
    },
    "git": {
        "command": "docker",
        "args": [
            "run",
            "--rm",
            "-i",
            "--mount",
            f"type=bind,src={os.environ["HOME"]}/git_test,dst={os.environ['HOME']}/git_test",
            "mcp/git",
        ],
    },
    "kubernetes": {"command": "npx", "args": ["-y", "kubectl-mcp-server"]},
}

inputs = [
    {
        "task": "Read the contents of the file 'data/info.txt'.",
        "expected_tool": "filesystem:read_file",
    },
    {
        "task": "Read the contents of the README.md.",
        "expected_tool": "filesystem:read_file",
    },
    {
        "task": "Write the contents to summary.txt file",
        "expected_tool": "filesystem:write_file",
    },
    {
        "task": "View the directory structure of the current directory",
        "expected_tool": "filesystem:directory_tree",
    },
    {
        "task": "List all files in 'src' directory",
        "expected_tool": "filesystem:list_directory",
    },
    {
        "task": "Rename 'config.old.json' file to 'config.json' file",
        "expected_tool": "filesystem:move_file",
    },
    {
        "task": "Search README.md file in /docs directory",
        "expected_tool": "filesystem:search_files",
    },
    {
        "task": "Create a new directory named 'logs/error'",
        "expected_tool": "filesystem:create_directory",
    },
    {
        "task": "Retrieve the metadata and permissions for 'private.key'",
        "expected_tool": "filesystem:get_file_info",
    },
    {
        "task": "Show me the directory structure of the 'backend' folder",
        "expected_tool": "filesystem:directory_tree",
    },
    {
        "task": "Retrieve the metadata and creation time for 'database.sqlite' file",
        "expected_tool": "filesystem:get_file_info",
    },
    {
        "task": "Read the contents of the image file 'logo.png'",
        "expected_tool": "filesystem:read_media_file",
    },
    {
        "task": "Read the source code from 'main.py' and 'utils.py' files",
        "expected_tool": "filesystem:read_multiple_files",
    },
    {
        "task": "Check which directories I am allowed to access",
        "expected_tool": "filesystem:list_allowed_directories",
    },
    {
        "task": "Search redis documents in notion",
        "expected_tool": "notion_api:API-post-search",
    },
    {
        "task": "Get a list of all users in the Notion workspace",
        "expected_tool": "notion_api:API-get-users",
    },
    {
        "task": "Archive the block with ID 'block-123'",
        "expected_tool": "notion_api:API-delete-a-block",
    },
    {
        "task": "Retrieve all comments from the document",
        "expected_tool": "notion_api:API-retrieve-a-comment",
    },
    {
        "task": "Create a new page in the 'Project Roadmap' database",
        "expected_tool": "notion_api:API-post-page",
    },
    {
        "task": "Get the details of the current Notion user",
        "expected_tool": "notion_api:API-get-self",
    },
    {
        "task": "Query the 'Engineering Tasks' database for open bugs",
        "expected_tool": "notion_api:API-post-database-query",
    },
    {
        "task": "Update the content of the text block 'block-888'",
        "expected_tool": "notion_api:API-update-a-block",
    },
    {
        "task": "Fetch the child blocks of the 'Meeting Notes' page",
        "expected_tool": "notion_api:API-get-block-children",
    },
    {
        "task": "Add a new comment to the project specification page",
        "expected_tool": "notion_api:API-create-a-comment",
    },
    {
        "task": "Retrieve the 'Status' property value for page 'page-456'",
        "expected_tool": "notion_api:API-retrieve-a-page-property",
    },
    {
        "task": "Search for latest news about AI advancements",
        "expected_tool": "web-search:search",
    },
    {
        "task": "Read the README.md from the official React repo on GitHub",
        "expected_tool": "web-search:fetchGithubReadme",
    },
    {
        "task": "Search for Python performance tips",
        "expected_tool": "web-search:search",
    },
    {
        "task": "Search for tutorials on CSDN regarding nginx configuration",
        "expected_tool": "web-search:fetchCsdnArticle",
    },
    {
        "task": "Fetch the documentation for the 'Axios' library from GitHub",
        "expected_tool": "web-search:fetchGithubReadme",
    },
    {
        "task": "Find articles about React Server Components",
        "expected_tool": "web-search:search",
    },
    {
        "task": "fetch the git logs of current branch",
        "expected_tool": "git:git_log",
    },
    {
        "task": "get the git status of the repository",
        "expected_tool": "git:git_status",
    },
    {
        "task": "Stage all modified files for git commit",
        "expected_tool": "git:git_add",
    },
    {
        "task": "Switch the repository to the 'develop' git branch",
        "expected_tool": "git:git_checkout",
    },
    {
        "task": "Create a new git branch called 'bugfix-login-error'",
        "expected_tool": "git:git_create_branch",
    },
    {
        "task": "Show the detailed changes in git commit '7a2b3c4'",
        "expected_tool": "git:git_show",
    },
    {
        "task": "See the differences between staged changes and the last commit",
        "expected_tool": "git:git_diff_staged",
    },
    {
        "task": "See the unstaged changes in the current working directory",
        "expected_tool": "git:git_diff_unstaged",
    },
    {
        "task": "Commit the staged changes with the message 'fix: resolve race condition'",
        "expected_tool": "git:git_commit",
    },
    {
        "task": "Initialize a new git repository in the current folder",
        "expected_tool": "git:git_init",
    },
    {
        "task": "Reset the current branch head to the previous git commit",
        "expected_tool": "git:git_reset",
    },
    {
        "task": "Show the differences between the 'main' and 'feature-api' branches",
        "expected_tool": "git:git_diff",
    },
    {
        "task": "Check which files are currently being tracked or modified with git",
        "expected_tool": "git:git_status",
    },
    {
        "task": "Look up the most recent git commit history",
        "expected_tool": "git:git_log",
    },
    {
        "task": "List all pods in the default namespace",
        "expected_tool": "kubernetes:get_pods",
    },
    {
        "task": "Fetch the kubernetes nodes status",
        "expected_tool": "kubernetes:get_nodes",
    },
    {
        "task": "Fetch the k8s nodes status",
        "expected_tool": "kubernetes:get_nodes",
    },
]


@dataclass
class TestResult:
    task: str
    expected_tool: str
    min_n_results: int | None
    found: bool


def save_results_to_csv(
    exp_ids: list[str],
    exp_results: list[list[TestResult]],
    filename: str = "temp_results.csv",
):
    """Saves the combined test results to a single CSV file."""
    if not exp_results or not exp_results[0]:
        return

    fieldnames = [
        "task",
        "expected_tool",
        *[f"min_n_results_{exp_id}" for exp_id in exp_ids],
        *[f"found_{exp_id}" for exp_id in exp_ids],
    ]

    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(exp_results[0])):
            row = {
                "task": exp_results[0][i].task,
                "expected_tool": exp_results[0][i].expected_tool,
            }
            for exp_id, results in zip(exp_ids, exp_results):
                row[f"min_n_results_{exp_id}"] = results[i].min_n_results
                row[f"found_{exp_id}"] = results[i].found
            writer.writerow(row)

    print(f"\n[System] Results saved to: {filename}")


def calculate_hitrate_by_n(results: list[TestResult], max_n: int) -> dict[int, float]:
    """Calculate hit rate for each n value from 1 to max_n.

    Hit rate at n = (tests found with min_n_results <= n) / total_tests
    """
    total = len(results)
    if total == 0:
        return {}

    hitrate_by_n = {}
    for n in range(1, max_n + 1):
        hits = sum(1 for r in results if r.found and r.min_n_results <= n)
        hitrate_by_n[n] = hits / total
    return hitrate_by_n


def print_hitrate_table(
    exp_ids: list[str], list_of_hitrate_by_n: list[dict[int, float]]
):
    """Print a combined hit rate table comparing all collections side by side."""
    if not list_of_hitrate_by_n:
        return

    headers = ["n", *[f"Hit Rate ({exp_id})" for exp_id in exp_ids]]
    rows = []
    all_ns = list_of_hitrate_by_n[0].keys()
    for n in all_ns:
        row = [n]
        for hitrate_by_n in list_of_hitrate_by_n:
            rate = hitrate_by_n.get(n, 0.0)
            row.append(f"{rate:.1%}")
        rows.append(row)

    print("\nHit Rate by n_results:")
    print(tabulate(rows, headers=headers, tablefmt="grid"))


def save_hitrate_to_csv(
    exp_ids: list[str],
    list_of_hitrate_by_n: list[dict[int, float]],
    filename: str = "temp_hitrate.csv",
):
    """Saves the combined hit rate data to a CSV file."""
    if not list_of_hitrate_by_n:
        return

    fieldnames = ["n", *[f"hitrate_{exp_id}" for exp_id in exp_ids]]
    all_ns = list_of_hitrate_by_n[0].keys()

    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for n in all_ns:
            row = {"n": n}
            for exp_id, hitrate_by_n in zip(exp_ids, list_of_hitrate_by_n):
                row[f"hitrate_{exp_id}"] = hitrate_by_n.get(n, 0.0)
            writer.writerow(row)

    print(f"[System] Hit rate saved to: {filename}")


async def find_min_n_results(
    search_engine_collection: SearchEngineCollection,
    *,
    task: str,
    expected_tool: str,
    max_n: int,
) -> int | None:
    """Find the minimum n_results needed to find the expected tool.

    Returns None if the tool is not found even with max_n results.
    """
    for n in range(1, max_n + 1):
        list_of_record = await search_engine_collection.query([task], top_k=n)
        tool_names = []
        for records in list_of_record:
            for record in records:
                server_name = record.metadata.get("server_name", "unknown_server")
                tool_name = record.metadata.get("tool_name", "unknown_tool")
                tool_names.append(f"{server_name}:{tool_name}")
        if expected_tool in tool_names:
            return n
    return None


async def run_test():
    chroma_db_client = Client()

    semantic_search_engine = SemanticSearchEngine(chroma_db_client)
    keyword_search_engine = KeywordSearchEngine()
    advanced_search_engine = AdvancedSearchEngine(chroma_db_client)

    semantic_collection = await semantic_search_engine.get_or_create_collection(
        "semantic_tools",
    )
    keyword_collection = await keyword_search_engine.get_or_create_collection(
        "keyword_tools", min_ngram=3, max_ngram=5
    )
    hybrid_collection = await advanced_search_engine.get_or_create_collection(
        "hybrid_tools", min_ngram=3, max_ngram=5
    )

    exp_ids = ["semantic", "keyword", "hybrid"]
    exp_collections = [semantic_collection, keyword_collection, hybrid_collection]

    exp_results = [[] for _ in range(len(exp_ids))]

    server_manager = ServerManager()
    server_manager.add_servers_from_dict(mcp_servers)

    records = []
    async with AsyncExitStack() as stack:
        await server_manager.initialize_servers(stack)

        total_tools = 0
        for server in server_manager.list_servers():
            this_tools = await server.list_tools()
            print(f"Server '{server.name}': {this_tools}")
            total_tools += len(this_tools)

            for tool in this_tools:
                id_ = f"{server.name}:{tool.name}"
                metadata = {
                    "server_name": server.name,
                    "tool_name": tool.name,
                }
                records.append(
                    Record(
                        id=id_,
                        document=tool.description,
                        metadata=metadata,
                    )
                )
        print(f"Total tools indexed: {total_tools}\n")

    for collection in exp_collections:
        await collection.add(records)

    for input_ in inputs:
        task = input_["task"]
        expected_tool = input_["expected_tool"]

        for collection, results in zip(exp_collections, exp_results):
            min_n = await find_min_n_results(
                collection,
                task=task,
                expected_tool=expected_tool,
                max_n=20,
            )
            results.append(
                TestResult(
                    task=task,
                    expected_tool=expected_tool,
                    min_n_results=min_n,
                    found=min_n is not None,
                )
            )

    print("=" * 70)
    print("EXPERIMENT RESULTS")
    print("=" * 70)

    header = ["id", "task", "expected_tool", *[f"min n ({id_})" for id_ in exp_ids]]

    rows = []

    for i in range(len(inputs)):
        row = [i + 1, inputs[i]["task"], inputs[i]["expected_tool"]]
        for results in exp_results:
            result = results[i]
            status = f"n={result.min_n_results}" if result.found else "NOT FOUND"
            row.append(status)
        rows.append(row)

    print(tabulate(rows, headers=header, tablefmt="grid"))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"Total tests: {len(exp_results[0])}")

    list_of_found_results = []
    list_of_not_found_results = []

    for exp_id, results in zip(exp_ids, exp_results):
        this_found_results = [r for r in results if r.found]
        list_of_found_results.append(this_found_results)
        this_not_found_results = [r for r in results if not r.found]
        list_of_not_found_results.append(this_not_found_results)

    found_row = ["Found", *[len(r) for r in list_of_found_results]]
    not_found_count_row = ["Not found", *[len(r) for r in list_of_not_found_results]]

    max_n_row = ["Max n_results needed"]
    for found_results in list_of_found_results:
        if found_results:
            max_n_row.append(max(r.min_n_results for r in found_results))
        else:
            max_n_row.append("N/A")

    not_found_tools_row = ["Not found tools"]
    for not_found_results in list_of_not_found_results:
        tools = [r.expected_tool for r in not_found_results]
        not_found_tools_row.append(", ".join(tools) if tools else "None")

    print(
        tabulate(
            [found_row, not_found_count_row, max_n_row, not_found_tools_row],
            headers=["", *exp_ids],
            tablefmt="grid",
        )
    )

    list_of_hitrate_by_n = [
        calculate_hitrate_by_n(results, max_n=20) for results in exp_results
    ]
    print_hitrate_table(exp_ids, list_of_hitrate_by_n)

    save_results_to_csv(exp_ids, exp_results)
    save_hitrate_to_csv(exp_ids, list_of_hitrate_by_n)


if __name__ == "__main__":
    asyncio.run(run_test())
