from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Custom")


# Add an addition tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers. e.g. a=10, b=30 -> 10 + 30 = 40"""
    return a + b


if __name__ == "__main__":
    mcp.run(transport="stdio")
