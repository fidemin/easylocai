# easylocai

Fully autonomous agentic workflows running locally—no APIs, no data leaks

![run_sample.gif](./docs/assets/images/run_sample.gif)

## Overview

Easylocai is an On-Device Autonomous Agent designed for secure, offline task execution. Unlike cloud-dependent assistants, it leverages the gpt-oss:20b model to perform complex reasoning and actions entirely on your local machine.

By implementing a sophisticated Plan-Execute-Replan orchestration, Easylocai can decompose ambiguous goals into actionable steps, execute them using Model Context Protocol (MCP) tools, and autonomously refine its strategy based on real-time feedback.


## Features
- **Privacy-First Autonomy**: 100% local execution using gpt-oss:20b via Ollama. Your code and data never leave your machine.

- **Agentic Orchestration**: A robust multi-agent loop (Plan → Execute → Replan) that ensures high success rates for long-horizon tasks.

- **MCP Tool Integration**: Seamlessly connects with Model Context Protocol (MCP) servers to interact with your local file system, terminal, and APIs.


## Requirements
To ensure stable performance of the autonomous agent, your system must meet the following criteria:

### System Requirements
- Minimum 16GB RAM (32GB or more recommended for optimal performance)
- Sufficient disk space for model storage and operation

### OS
- **OS:** macOS (Strictly supported)

### Software Requirements
- **Runtime:** Python 3.12. It is recommended to use [pyenv](https://github.com/pyenv/pyenv).
- **LLM Engine:** [Ollama](https://ollama.com/) must be installed and running.
  - **Model:** `gpt-oss:20b` (Make sure to run `ollama pull gpt-oss:20b` before starting).

# Install & Execution 
## Installation
```bash
bash install.sh
````

## Configuration

### MCP server configuration
- file_name: `~/.config/easylocai/config.json`
- example
    ```json
    {
      "mcpServers": {
        "filesystem": {
          "command": "npx",
          "args": [
            "-y",
            "@modelcontextprotocol/server-filesystem",
            "."
          ]
        },
        "kubernetes": {
          "command": "python",
          "args": [
            "-m",
            "kubectl_mcp_tool.mcp_server"
          ],
          "cwd": "~/Programming/kubectl-mcp-server",
          "env": {
            "KUBECONFIG": "~/.kube/config",
            "KUBECTL_MCP_LOG_LEVEL": "ERROR",
            "PYTHONUNBUFFERED": "1"
          }
        },
        "notion_api": {
          "command": "docker",
          "args": [
            "run",
            "--rm",
            "-i",
            "-e", "NOTION_TOKEN",
            "mcp/notion"
          ],
          "env": {
            "NOTION_TOKEN": "<token>"
          }
        }
      }
    }
    ```

## Initialization
```bash
easylocai init
```

If you want to force re-initialization, use `--force` flag:
```bash
easylocai init --force
```

## Run
```bash
easylocai
```


# Development
- refer to [DEVELOPMENT.md](docs/DEVELOPMENT.md)
