# Production

## Installation

### First-time install

```bash
pipx install easylocai
```

### Reinstall or upgrade

```bash
pipx upgrade easylocai
```

## Configuration

See [`docs/CONFIGURATION.md`](CONFIGURATION.md) for MCP server configuration details.

## Initialization

### First-time initialization
```bash
easylocai init
```

Config file is created at `~/.config/easylocai/config.json` with an empty `mcpServers` dict. If the file already exists, this is a no-op.

### force re-initialization
```bash
easylocai init --force
```

> **Warning:** `--force` overwrites the existing config with `{"mcpServers": {}}`, permanently deleting all MCP server entries you have configured. Back up `~/.config/easylocai/config.json` before using this flag.

## Run

```bash
easylocai
```

### With a workflow flag

```bash
easylocai --flag=<flag>
```
