# PromptEvalWorkflow Output 개선 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** PromptEvalWorkflow 출력을 Markdown 테이블로 변경하고, 설정 메타데이터 요약과 파일 저장(text/json) 기능을 추가한다.

**Architecture:** `PromptEvalWorkflow.run()`에 `output_file`/`output_format` 파라미터를 추가하고, `_print()` → `_print_table()` + `_print_summary()`로 교체한다. `run.py`에 `--output`/`--format` CLI 인자를 추가해 워크플로우로 전달한다.

**Tech Stack:** Python 3.12, pytest-asyncio (asyncio_mode=auto)

---

## File Map

| File | Action | Role |
|------|--------|------|
| `prompt_eval/prompt_eval_workflow.py` | Modify | 테이블 출력, 요약, 파일 저장 로직 |
| `prompt_eval/run.py` | Modify | `--output`, `--format` CLI 인자 파싱 |
| `tests/unittests/prompt_eval/test_prompt_eval_workflow.py` | Create | 출력/저장 로직 단위 테스트 |

---

### Task 1: 테스트 파일 생성 및 `_print_table` 테스트 작성

**Files:**
- Create: `tests/unittests/prompt_eval/__init__.py`
- Create: `tests/unittests/prompt_eval/test_prompt_eval_workflow.py`

- [ ] **Step 1: 테스트 디렉토리 생성**

```bash
mkdir -p tests/unittests/prompt_eval
touch tests/unittests/prompt_eval/__init__.py
```

- [ ] **Step 2: `_print_table` 테스트 작성**

`tests/unittests/prompt_eval/test_prompt_eval_workflow.py`:

```python
import io
import json
import os
import tempfile
from unittest.mock import MagicMock

import pytest

from prompt_eval.prompt_eval_workflow import PromptEvalWorkflow


def make_workflow(tmp_input_file):
    return PromptEvalWorkflow(
        prompt_path_info={"system": "dummy_system.jinja2"},
        input_file_path=tmp_input_file,
        model_info={"host": "http://localhost:11434", "model": "gpt-oss:20b"},
        output_model=None,
    )


SAMPLE_RESULTS = [
    {
        "id": "1",
        "response": '{"tasks": ["Do A"]}',
        "thinking": "I thought about it",
        "expected": {"tasks": ["Do A"]},
        "scoring_criteria": "Single task expected",
    },
    {
        "id": "2",
        "response": '{"tasks": ["Do B", "Do C"]}',
        "thinking": None,
        "expected": {"tasks": ["Do B", "Do C"]},
        "scoring_criteria": None,
    },
]


def test_print_table_contains_header(tmp_path, capsys):
    input_file = tmp_path / "input.json"
    input_file.write_text("[]")
    wf = make_workflow(str(input_file))
    wf._print_table(SAMPLE_RESULTS)
    out = capsys.readouterr().out
    assert "| ID |" in out
    assert "| Response |" in out
    assert "| Expected |" in out
    assert "| Thinking |" in out


def test_print_table_contains_row_data(tmp_path, capsys):
    input_file = tmp_path / "input.json"
    input_file.write_text("[]")
    wf = make_workflow(str(input_file))
    wf._print_table(SAMPLE_RESULTS)
    out = capsys.readouterr().out
    assert "1" in out
    assert "2" in out


def test_print_table_thinking_none_shows_empty(tmp_path, capsys):
    input_file = tmp_path / "input.json"
    input_file.write_text("[]")
    wf = make_workflow(str(input_file))
    wf._print_table(SAMPLE_RESULTS)
    out = capsys.readouterr().out
    # Row 2 has thinking=None, should not crash and should render empty
    assert "| 2 |" in out
```

- [ ] **Step 3: 테스트 실행 (실패 확인)**

```bash
pytest tests/unittests/prompt_eval/test_prompt_eval_workflow.py -v
```

Expected: `AttributeError: 'PromptEvalWorkflow' object has no attribute '_print_table'`

---

### Task 2: `_print_table` + `_print_summary` 구현

**Files:**
- Modify: `prompt_eval/prompt_eval_workflow.py`

- [ ] **Step 1: `_print()` 를 `_print_table()` 로 교체**

`prompt_eval/prompt_eval_workflow.py`에서 `_print` 메서드를 아래로 교체:

```python
def _print_table(self, results: list[dict]):
    headers = ["ID", "Response", "Expected", "Thinking"]
    col_width = 40

    def truncate(val: str, width: int) -> str:
        val = str(val) if val is not None else ""
        val = val.replace("\n", " ")
        return val[:width] + "..." if len(val) > width else val

    header_row = " | ".join(f"{h:<{col_width}}" for h in headers)
    separator = "-+-".join("-" * col_width for _ in headers)
    print(f"| {header_row} |")
    print(f"|-{separator}-|")

    for result in results:
        row = " | ".join([
            f"{truncate(result.get('id', ''), col_width):<{col_width}}",
            f"{truncate(result.get('response', ''), col_width):<{col_width}}",
            f"{truncate(str(result.get('expected', '')), col_width):<{col_width}}",
            f"{truncate(result.get('thinking', ''), col_width):<{col_width}}",
        ])
        print(f"| {row} |")
    print()
```

- [ ] **Step 2: `_print_summary()` 추가**

`_print_table` 아래에 추가:

```python
def _print_summary(self, config_path: str | None = None):
    print("=== Summary ===")
    if config_path:
        print(f"Config:        {config_path}")
    print(f"System prompt: {self._prompt_path_info.get('system', '-')}")
    print(f"User prompt:   {self._prompt_path_info.get('user', '-')}")
    print(f"Input file:    {self._input_file_path}")
    print(f"Output model:  {self._output_model.__module__ + '.' + self._output_model.__name__ if self._output_model else '-'}")
    print(f"Model:         {self._model_info.get('model', '-')}")
```

- [ ] **Step 3: `run()` 시그니처 변경 및 호출부 교체**

기존 `run()` 메서드를 아래로 교체:

```python
async def run(self, config_path: str | None = None, output_file: str | None = None, output_format: str = "text"):
    results = await self.run_and_collect()
    self._print_table(results)
    self._print_summary(config_path=config_path)
    if output_file:
        self._save(results, output_file, output_format, config_path=config_path)
```

- [ ] **Step 4: 테스트 실행 (통과 확인)**

```bash
pytest tests/unittests/prompt_eval/test_prompt_eval_workflow.py::test_print_table_contains_header tests/unittests/prompt_eval/test_prompt_eval_workflow.py::test_print_table_contains_row_data tests/unittests/prompt_eval/test_prompt_eval_workflow.py::test_print_table_thinking_none_shows_empty -v
```

Expected: 3 PASSED

---

### Task 3: `_save()` 구현 및 테스트

**Files:**
- Modify: `prompt_eval/prompt_eval_workflow.py`
- Modify: `tests/unittests/prompt_eval/test_prompt_eval_workflow.py`

- [ ] **Step 1: 파일 저장 테스트 추가**

`tests/unittests/prompt_eval/test_prompt_eval_workflow.py` 끝에 추가:

```python
def test_save_text_creates_markdown_file(tmp_path):
    input_file = tmp_path / "input.json"
    input_file.write_text("[]")
    wf = make_workflow(str(input_file))
    output_file = tmp_path / "results.md"
    wf._save(SAMPLE_RESULTS, str(output_file), "text", config_path="configs/test.json")
    content = output_file.read_text()
    assert "## Eval Results" in content
    assert "| ID |" in content
    assert "## Summary" in content
    assert "configs/test.json" in content


def test_save_json_creates_valid_json(tmp_path):
    input_file = tmp_path / "input.json"
    input_file.write_text("[]")
    wf = make_workflow(str(input_file))
    output_file = tmp_path / "results.json"
    wf._save(SAMPLE_RESULTS, str(output_file), "json", config_path="configs/test.json")
    data = json.loads(output_file.read_text())
    assert "metadata" in data
    assert "results" in data
    assert "summary" in data
    assert data["metadata"]["config"] == "configs/test.json"
    assert len(data["results"]) == 2
    assert data["summary"]["total"] == 2


def test_save_json_result_fields(tmp_path):
    input_file = tmp_path / "input.json"
    input_file.write_text("[]")
    wf = make_workflow(str(input_file))
    output_file = tmp_path / "results.json"
    wf._save(SAMPLE_RESULTS, str(output_file), "json", config_path=None)
    data = json.loads(output_file.read_text())
    first = data["results"][0]
    assert "id" in first
    assert "response" in first
    assert "expected" in first
    assert "thinking" in first
```

- [ ] **Step 2: 테스트 실행 (실패 확인)**

```bash
pytest tests/unittests/prompt_eval/test_prompt_eval_workflow.py::test_save_text_creates_markdown_file tests/unittests/prompt_eval/test_prompt_eval_workflow.py::test_save_json_creates_valid_json -v
```

Expected: `AttributeError: 'PromptEvalWorkflow' object has no attribute '_save'`

- [ ] **Step 3: `_save()` 구현**

`prompt_eval/prompt_eval_workflow.py`에 추가:

```python
def _save(self, results: list[dict], output_file: str, output_format: str, config_path: str | None = None):
    if output_format == "json":
        self._save_json(results, output_file, config_path)
    else:
        self._save_text(results, output_file, config_path)
    print(f"Results saved to: {output_file}")

def _save_text(self, results: list[dict], output_file: str, config_path: str | None):
    from datetime import date
    lines = []
    config_name = os.path.basename(config_path) if config_path else "unknown"
    lines.append(f"## Eval Results — {config_name} — {date.today().isoformat()}\n")

    headers = ["ID", "Response", "Expected", "Thinking"]
    col_width = 60

    def truncate(val, width):
        val = str(val) if val is not None else ""
        val = val.replace("\n", " ")
        return val[:width] + "..." if len(val) > width else val

    header_row = " | ".join(f"{h:<{col_width}}" for h in headers)
    separator = "-+-".join("-" * col_width for _ in headers)
    lines.append(f"| {header_row} |")
    lines.append(f"|-{separator}-|")
    for result in results:
        row = " | ".join([
            f"{truncate(result.get('id', ''), col_width):<{col_width}}",
            f"{truncate(result.get('response', ''), col_width):<{col_width}}",
            f"{truncate(str(result.get('expected', '')), col_width):<{col_width}}",
            f"{truncate(result.get('thinking', ''), col_width):<{col_width}}",
        ])
        lines.append(f"| {row} |")

    lines.append("\n## Summary")
    if config_path:
        lines.append(f"Config:        {config_path}")
    lines.append(f"System prompt: {self._prompt_path_info.get('system', '-')}")
    lines.append(f"User prompt:   {self._prompt_path_info.get('user', '-')}")
    lines.append(f"Input file:    {self._input_file_path}")
    lines.append(f"Output model:  {self._output_model.__module__ + '.' + self._output_model.__name__ if self._output_model else '-'}")
    lines.append(f"Model:         {self._model_info.get('model', '-')}")
    lines.append(f"Total:         {len(results)} cases")

    with open(output_file, "w") as f:
        f.write("\n".join(lines))

def _save_json(self, results: list[dict], output_file: str, config_path: str | None):
    from datetime import date
    data = {
        "metadata": {
            "config": config_path,
            "system_prompt": self._prompt_path_info.get("system"),
            "user_prompt": self._prompt_path_info.get("user"),
            "input_file": self._input_file_path,
            "output_model": (
                self._output_model.__module__ + "." + self._output_model.__name__
                if self._output_model else None
            ),
            "model": self._model_info.get("model"),
            "date": date.today().isoformat(),
        },
        "results": [
            {
                "id": r.get("id"),
                "response": r.get("response"),
                "expected": r.get("expected"),
                "thinking": r.get("thinking"),
            }
            for r in results
        ],
        "summary": {"total": len(results)},
    }
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
```

`import os` 와 `import json` 이 파일 상단에 있는지 확인 (이미 `json`은 있음, `os` 추가 필요):

```python
import json
import os  # 추가
```

- [ ] **Step 4: 테스트 실행 (통과 확인)**

```bash
pytest tests/unittests/prompt_eval/test_prompt_eval_workflow.py -v
```

Expected: 모든 테스트 PASSED

---

### Task 4: `run.py` CLI 인자 추가

**Files:**
- Modify: `prompt_eval/run.py`

- [ ] **Step 1: `--output`, `--format` 인자 파싱 추가**

`prompt_eval/run.py`의 `run()` 함수와 `__main__` 블록을 아래로 교체:

```python
async def run(config_path: str, output_file: str | None = None, output_format: str = "text"):
    with open(config_path) as f:
        config = json.load(f)

    output_model = _load_class(config["output_model"]) if config.get("output_model") else None
    model_info = {**_DEFAULT_MODEL_INFO, **config.get("model_info", {})}

    workflow = PromptEvalWorkflow(
        input_file_path=config["input_file"],
        prompt_path_info=config["prompt_info"],
        model_info=model_info,
        output_model=output_model,
    )
    await workflow.run(config_path=config_path, output_file=output_file, output_format=output_format)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m prompt_eval.run <config_file> [--output <path>] [--format text|json]")
        print("Example: python -m prompt_eval.run resources/prompt_eval/configs/plan_prompt_v2_config.json")
        print("Example: python -m prompt_eval.run resources/prompt_eval/configs/plan_prompt_v2_config.json --output results.md")
        print("Example: python -m prompt_eval.run resources/prompt_eval/configs/plan_prompt_v2_config.json --output results.json --format json")
        sys.exit(1)

    _output_file = None
    _output_format = "text"

    args = sys.argv[1:]
    config_arg = args[0]

    if "--output" in args:
        idx = args.index("--output")
        if idx + 1 < len(args):
            _output_file = args[idx + 1]

    if "--format" in args:
        idx = args.index("--format")
        if idx + 1 < len(args):
            fmt = args[idx + 1]
            if fmt not in ("text", "json"):
                print(f"Error: --format must be 'text' or 'json', got '{fmt}'")
                sys.exit(1)
            _output_format = fmt

    asyncio.run(run(config_arg, output_file=_output_file, output_format=_output_format))
```

- [ ] **Step 2: 전체 테스트 실행**

```bash
pytest tests/unittests/prompt_eval/ -v
```

Expected: 모든 테스트 PASSED

- [ ] **Step 3: 실제 실행으로 stdout 출력 확인**

```bash
python -m prompt_eval.run resources/prompt_eval/configs/plan_prompt_v2_config.json
```

Expected: Markdown 테이블 + Summary 출력

- [ ] **Step 4: 파일 저장 확인 (text)**

```bash
python -m prompt_eval.run resources/prompt_eval/configs/plan_prompt_v2_config.json --output /tmp/eval_results.md
cat /tmp/eval_results.md
```

Expected: Markdown 파일 생성, `## Eval Results`, `## Summary` 섹션 포함

- [ ] **Step 5: 파일 저장 확인 (json)**

```bash
python -m prompt_eval.run resources/prompt_eval/configs/plan_prompt_v2_config.json --output /tmp/eval_results.json --format json
cat /tmp/eval_results.json
```

Expected: JSON 파일 생성, `metadata`, `results`, `summary` 키 포함
