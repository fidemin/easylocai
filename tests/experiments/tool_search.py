import asyncio
import csv
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass, asdict

from chromadb import Client

from easylocai.core.tool_manager import ToolManager

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
    "kubernetes": {"command": "npx", "args": ["-y", "kubectl-mcp-server"]},
    "web-search": {
        "args": ["open-websearch@latest"],
        "command": "npx",
        "env": {
            "MODE": "stdio",
            "DEFAULT_SEARCH_ENGINE": "duckduckgo",
            "ALLOWED_SEARCH_ENGINES": "duckduckgo,bing,exa",
        },
    },
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
        "task": "Search README.md in /docs directory",
        "expected_tool": "filesystem:search_files",
    },
    {
        "task": "Search redis documents in notion",
        "expected_tool": "notion_api:API-post-search",
    },
    {
        "task": "List all pods in the default namespace",
        "expected_tool": "kubernetes:get_pods",
    },
    {
        "task": "Search for latest news about AI advancements",
        "expected_tool": "web-search:search",
    },
    {
        "task": "Fetch the k8s node status",
        "expected_tool": "kubernetes:get_nodes",
    },
]


@dataclass
class TestResult:
    task: str
    expected_tool: str
    min_n_results: int | None
    found: bool


def save_results_to_csv(results: list[TestResult], filename: str = "temp_results.csv"):
    """Saves the test results list to a CSV file."""
    if not results:
        return

    # Use dataclass fields as CSV headers
    headers = list(asdict(results[0]).keys())

    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

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


def print_hitrate_table(hitrate_by_n: dict[int, float]):
    """Print hit rate table."""
    print("\nHit Rate by n_results:")
    print("-" * 30)
    print(f"{'n':>5} | {'Hit Rate':>10} | {'Bar':<10}")
    print("-" * 30)
    for n, rate in hitrate_by_n.items():
        bar = "█" * int(rate * 10)
        print(f"{n:>5} | {rate:>9.1%} | {bar}")
    print("-" * 30)


def save_hitrate_to_csv(
    hitrate_by_n: dict[int, float], filename: str = "temp_hitrate.csv"
):
    """Saves the hit rate data to a CSV file."""
    if not hitrate_by_n:
        return

    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "hitrate"])
        writer.writeheader()
        for n, rate in hitrate_by_n.items():
            writer.writerow({"n": n, "hitrate": rate})

    print(f"[System] Hit rate saved to: {filename}")


def find_min_n_results(
    tool_manager: ToolManager,
    *,
    task: str,
    expected_tool: str,
    max_n: int,
) -> int | None:
    """Find the minimum n_results needed to find the expected tool.

    Returns None if the tool is not found even with max_n results.
    """
    for n in range(1, max_n + 1):
        tools = tool_manager.search_tools([task], n_results=n)
        tool_names = [f"{tool.server_name}:{tool.name}" for tool in tools]
        if expected_tool in tool_names:
            return n
    return None


async def run_test():
    chromadb_client = Client()
    tool_manager = ToolManager(chromadb_client, mpc_servers=mcp_servers)

    async with AsyncExitStack() as stack:
        await tool_manager.initialize(stack)
        total_tools = 0
        for server in tool_manager._server_manager.list_servers():
            this_tools = await server.list_tools()
            print(f"Server '{server.name}': {this_tools}")
            total_tools += len(this_tools)
        print(f"Total tools indexed: {total_tools}\n")

    results: list[TestResult] = []

    for input_ in inputs:
        task = input_["task"]
        expected_tool = input_["expected_tool"]

        min_n = find_min_n_results(
            tool_manager,
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

    for i, result in enumerate(results, 1):
        status = f"n={result.min_n_results}" if result.found else "NOT FOUND"
        print(f"\n[Test {i}] {status}")
        print(f"  Task: {result.task}")
        print(f"  Expected: {result.expected_tool}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    found_results = [r for r in results if r.found]
    not_found_results = [r for r in results if not r.found]

    print(f"Total tests: {len(results)}")
    print(f"Found: {len(found_results)}")
    print(f"Not found: {len(not_found_results)}")

    if found_results:
        max_n = max(r.min_n_results for r in found_results)
        print(f"\nMaximum n_results needed: {max_n}")
        print(f"Recommended n_results setting: {max_n}")

    if not_found_results:
        print("\nTools not found (may need larger n_results or different query):")
        for r in not_found_results:
            print(f"  - {r.expected_tool}")

    hitrate_by_n = calculate_hitrate_by_n(results, max_n=20)
    print_hitrate_table(hitrate_by_n)

    save_results_to_csv(results)
    save_hitrate_to_csv(hitrate_by_n)


if __name__ == "__main__":
    asyncio.run(run_test())
