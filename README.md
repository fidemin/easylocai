# easylocai-cli
- very simple automatic AI agent for your local machine 

# Requirements
- Only working for MacOS
- Python 3.10+
- Ollama and gpt-oss:20b model should be installed  

# Installation
```bash
bash install.sh
````

# Configuration

### MCP server configuration

- file_name: `mcp_server_config.json`
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


# Run
```bash
./easylocai
```
