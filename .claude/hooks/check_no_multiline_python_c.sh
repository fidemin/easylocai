#!/bin/bash
# Blocks multiline python -c commands (newline inside the quoted -c argument).
cmd=$(jq -r '.tool_input.command' 2>/dev/null)

if printf '%s' "$cmd" | perl -0777 -ne 'exit 0 if /python\s+-c\s+[\x27"][^\x27"\n]*\n/; exit 1' 2>/dev/null; then
  printf '{"decision":"block","reason":"Multiline python -c commands are not allowed. Write a script file instead."}\n'
fi