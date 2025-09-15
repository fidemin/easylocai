import textwrap


def print_prompt(title: str, prompt: str):
    print(pretty_prompt_text(title, prompt))


def pretty_prompt_text(title: str, prompt: str):
    width = 100
    lines = [textwrap.wrap(title, width=width - 4)[0], ""]
    for line in prompt.strip().splitlines():
        lines.extend(textwrap.wrap(line, width=width - 4) or [""])

    results = [""]
    results.append("+" + "-" * (width - 2) + "+")
    for line in lines:
        results.append("| " + line.ljust(width - 4) + " |")

    results.append("+" + "-" * (width - 2) + "+")

    return "\n".join(results)
