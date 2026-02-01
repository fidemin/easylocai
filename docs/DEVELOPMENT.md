# Architecture

The system is built on a **Modular Agentic Workflow** following a "Plan-Execute-Replan" cycle. Unlike static chain-of-thought systems, this architecture allows the AI to dynamically pivot its strategy based on real-time tool outputs and execution results.

## Core Components

| Component           | Responsibility                                                                                      | Key Input |
|:--------------------|:----------------------------------------------------------------------------------------------------| :--- |
| **PlanAgent**       | Analyzes the user query and generates a high-level roadmap of tasks.                                | User Query, History |
| **SingleTaskAgent** | The "Worker" agent. Executes a specific task via LLM reasoning or iterative tool-calling.           | Single Task, Tools |
| **ReplanAgent**     | The "Supervisor." Evaluates task results to decide if the goal is met or if the plan needs a pivot. | Task Results, Previous Plan |
| **ToolManager**     | Manages the lifecycle and execution of external mcp tools.                                          | Config File, Tool Schema |
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
    PlanAgent -->|Generate Initial Tasks| TaskInit[Initialize Task List]
end

%% Execution Loop
subgraph ExecutionLoop ["2. Iterative Task Execution"]
    direction TB
    TaskInit --> SingleTaskAgent[SingleTaskAgent]
    
    SingleTaskAgent -->|Task Result| ReplanAgent[ReplanAgent]
    
    ReplanAgent -->|Check response| IsGoalMet{Is Goal met?}
    
    IsGoalMet -->|Updated Tasks| SingleTaskAgent
end

%% Exit Paths
IsGoalMet -->|Yes: Final Answer| UpdateContext[Update User Conversation]
UpdateContext --> FinalResponse[Display Answer]

FinalResponse --> End((Ready for next user input))
```

### Workflow in Agents
#### PlanAgent

```mermaid
graph TD
    %% Input Node
    Start((Start)) --> Input

    %% Internal Processing Sequence
    subgraph PlanAgent ["PlanAgent._run()"]
        direction LR
        Input[PlanAgentInput]
        
        Input --> Reformatter[QueryReformatter]
        
        Reformatter --> |Reformed Query + User Context| Planner[Planner]
    
        Planner --> |Initial Tasks| OutputPrep[Prepare PlanAgentOutput]
    
        OutputPrep --> FinalOutput[PlanAgentOutput]
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
        
        Input --> Search[(Tool Candidate Search)]
        
        Search --> LoopStart[TaskRouter]

        subgraph IterationLoop ["Execution Loop (while True)"]
            direction TB
            LoopStart --> Finished{task finished?}
            
            Finished -->|No| Route{Subtask Type?}
            
            Route -->|tool| ToolExec[Execute Tool Subtask]
            Route -->|reasoning| ReasonExec[Execute Reasoning Subtask]
            
            ToolExec --> Append[Append to Iteration Results]
            ReasonExec --> Append
            
            Append --> LoopStart
        end
        
        Finished -->|Yes| Filter[TaskResultFilter]
        
        %% Final Synthesis
        Filter -->|Filtered Result| Out[SingleTaskAgentOutput]
    end

    Out --> End((End))
```

#### ReplanAgent

```mermaid
graph TD
    %% Input Node
    Start((Start)) --> Input[ReplanAgentInput]

    %% Internal Processing Sequence
    subgraph ReplanAgent ["ReplanAgent._run()"]
        direction TB
       
        Input --> Prep[Prepare ReplannerInput]
        
        %% Core Logic
        Prep --> ReplannerCall[Replanner LLM Call]
        
        %% Output Mapping
        ReplannerCall --> MapOutput[Map to ReplanAgentOutput]
        
        MapOutput --> Out[ReplanAgentOutput]
    end

    %% Final Output
    Out --> End((End))
```
