# easylocai-cli
- very simple automatic AI agent for your local machine 

# Requirements
- Only working for MacOS
- Python 3.12+
- Ollama and gpt-oss:20b model should be installed  

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


# Architecture

The system is built on a **Modular Agentic Workflow** following a "Plan-Execute-Replan" cycle. Unlike static chain-of-thought systems, this architecture allows the AI to dynamically pivot its strategy based on real-time tool outputs and execution results.

## Core Components

| Component | Responsibility | Key Input |
| :--- | :--- | :--- |
| **PlanAgent** | Analyzes the user query and generates a high-level roadmap of tasks. | User Query, History |
| **SingleTaskAgent** | The "Worker" agent. Executes a specific task via LLM reasoning or iterative tool-calling. | Single Task, Tools |
| **ReplanAgent** | The "Supervisor." Evaluates task results to decide if the goal is met or if the plan needs a pivot. | Task Results, Previous Plan |
| **ServerManager** | Manages the lifecycle and execution of external tools (MCP/Servers). | Config File, Tool Schema |
| **ChromaDB** | Acts as a Vector Store for Tool Discovery (RAG for tools). | Task Descriptions |

---

## System Workflow

The coordination between agents is managed in a dynamic loop that ensures high reliability and adaptive problem-solving.

### Agentic Workflow Diagram

```mermaid
graph TD
%% Entry Point
Start((User Input)) --> PlanAgent[PlanAgent]
%% Initial Phase
subgraph Initialization ["1. Initial Planning"]
    PlanAgent -->|PlanAgent output| IsDirectAnswer{direct response?}
end

%% Execution Loop
subgraph ExecutionLoop ["2. Iterative Task Execution"]
    direction TB
    IsDirectAnswer -->|No: initial tasks| SingleTaskAgent[SingleTaskAgent]

    SingleTaskAgent -->|Task Result| ReplanAgent[ReplanAgent]

    ReplanAgent -->|Replanned Tasks| CheckCompletion{Is Goal Met?}

    CheckCompletion -->|No: Replanned Tasks| SingleTaskAgent
end

%% Exit Paths
IsDirectAnswer -->|Yes| FinalResponse[Display Answer]
CheckCompletion -->|Yes: Final Answer| FinalResponse

FinalResponse --> End((Ready for next user input))
```

### Workflow in Agents
#### PlanAgent

```mermaid
graph TD
    %% Input Node
    Start((Start)) --> Input

    %% Internal Processing Sequence
    subgraph PlanAgent ["PlanAgent.run()"]
        direction LR
        Input[PlanAgentInput]
        
        Input --> Normalizer[QueryNormalizer]
        
        Normalizer --> |Refined Query + User Context| Planner[Planner]
    
        Planner --> |Inital Plan| ToolSearch[(Tool Candidates Search)]
        
        Replanner[Replanner]
        
        Planner --> |Inital Plan| Replanner
        ToolSearch --> |Tool candidates| Replanner
        Normalizer --> |User Context| Replanner
    
        Replanner --> |Final Tasks & Response| FinalOutput[PlanAgentOutput]
    end


    %% Output
    FinalOutput --> End((End))
```

#### SingleTaskAgent

```mermaid
graph TD
    Start((Start)) --> Input[SingleTaskAgentInput]

    subgraph SingleTaskAgent ["SingleTaskAgent.run()"]
        direction TB
        
        Input --> TypeCheck{Check task type}
        
        %% LLM Branch
        TypeCheck -->|task type 'llm'| Reasoning[ReasoningAgent]
        Reasoning -->|reasoning output| Filter[TaskResultFilter]
        
        %% Tool Branch
        TypeCheck -->|task type 'tool'| Search[(Tool Search)]
        Search -->|tool candidates| LoopStart

        subgraph ToolLoop ["Execution Loop (while True)"]
            direction TB
            LoopStart[ToolSelector] --> Finished{finished?}
            
            Finished -->|finished == False| Call[Call Tools]
            Call -->|tool_results| LoopStart
        end
        
        Finished -->|finished == True| Filter
        
        %% Final Synthesis
        Filter -->|Cleaned Result String| Out[SingleTaskAgentOutput]
    end

    Out --> End((End))
```

#### ReplanAgent

```mermaid
graph TD
    Start((Start)) --> Input[ReplanAgentInput]

    subgraph ReplanAgent ["ReplanAgent.run()"]
        direction TB
       
        Input --> |previous plan| Search[(Tool Search)]
        
        Search -->|tool candidates| Replan[Replanner]
        
        %% Mapping Multiple Inputs to Replanner
        Input -.-> |user_query + user_context| Replan
        Input -.-> |task_results| Replan
        Input -.-> |previous plan| Replan

        Replan --> Out[ReplanAgentOutput]
    end

    Out --> End((End))
```
