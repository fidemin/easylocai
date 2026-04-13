import json

import pytest

from prompt_eval.prompt_eval_workflow import (
    EvalMetadata,
    EvalOutput,
    EvalResultItem,
    PromptEvalWorkflow,
)


def make_workflow(tmp_input_file):
    return PromptEvalWorkflow(
        prompt_path_info={"system": "dummy_system.jinja2"},
        input_file_path=tmp_input_file,
        model_info={"host": "http://localhost:11434", "model": "gpt-oss:20b"},
        output_model=None,
    )


def make_output(tmp_input_file) -> EvalOutput:
    metadata = EvalMetadata(
        config="configs/test.json",
        system_prompt="resources/prompts/system.jinja2",
        user_prompt=None,
        input_file=tmp_input_file,
        output_model=None,
        model="gpt-oss:20b",
        date="2026-04-13",
        total=2,
    )
    results = [
        EvalResultItem(id="1", response='{"tasks": ["Do A"]}', expected={"tasks": ["Do A"]}, thinking="I thought about it"),
        EvalResultItem(id="2", response='{"tasks": ["Do B", "Do C"]}', expected={"tasks": ["Do B", "Do C"]}, thinking=None),
    ]
    return EvalOutput(metadata=metadata, results=results)


# --- EvalOutput.to_text() ---

def test_to_text_contains_header(tmp_path):
    output = make_output(str(tmp_path / "input.json"))
    text = output.to_text()
    assert "| ID" in text
    assert "Response" in text
    assert "Expected" in text
    assert "Thinking" in text


def test_to_text_contains_rows(tmp_path):
    output = make_output(str(tmp_path / "input.json"))
    text = output.to_text()
    assert "| 1 " in text
    assert "| 2 " in text


def test_to_text_thinking_none_renders_empty(tmp_path):
    output = make_output(str(tmp_path / "input.json"))
    text = output.to_text()
    # Row 2 has thinking=None — should render without crashing
    assert "| 2 " in text


def test_to_text_contains_summary(tmp_path):
    output = make_output(str(tmp_path / "input.json"))
    text = output.to_text()
    assert "## Summary" in text
    assert "configs/test.json" in text
    assert "Total:         2 cases" in text


# --- EvalOutput.to_json() ---

def test_to_json_structure(tmp_path):
    output = make_output(str(tmp_path / "input.json"))
    data = output.to_json()
    assert "metadata" in data
    assert "results" in data
    assert "summary" in data
    assert data["metadata"]["config"] == "configs/test.json"
    assert len(data["results"]) == 2
    assert data["summary"]["total"] == 2


def test_to_json_result_fields(tmp_path):
    output = make_output(str(tmp_path / "input.json"))
    first = output.to_json()["results"][0]
    assert first["id"] == "1"
    assert "response" in first
    assert "expected" in first
    assert "thinking" in first


# --- PromptEvalWorkflow._build_output() ---

def test_build_output_metadata(tmp_path):
    input_file = tmp_path / "input.json"
    input_file.write_text("[]")
    wf = make_workflow(str(input_file))
    raw = [
        {"id": "1", "response": "r1", "thinking": None, "expected": {"tasks": ["A"]}},
    ]
    output = wf._build_output(raw, config_path="configs/test.json")
    assert output.metadata.config == "configs/test.json"
    assert output.metadata.total == 1
    assert output.metadata.output_model is None


def test_build_output_result_items(tmp_path):
    input_file = tmp_path / "input.json"
    input_file.write_text("[]")
    wf = make_workflow(str(input_file))
    raw = [
        {"id": "1", "response": "r1", "thinking": "t1", "expected": {"tasks": ["A"]}},
        {"id": "2", "response": "r2", "thinking": None, "expected": None},
    ]
    output = wf._build_output(raw, config_path=None)
    assert len(output.results) == 2
    assert output.results[0].id == "1"
    assert output.results[1].thinking is None
