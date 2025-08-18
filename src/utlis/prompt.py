import textwrap


def print_prompt(title: str, prompt: str):
    width = 100
    lines = [textwrap.wrap(title, width=width - 4)[0], ""]
    for line in prompt.strip().splitlines():
        lines.extend(textwrap.wrap(line, width=width - 4) or [""])

    print("+" + "-" * (width - 2) + "+")
    for line in lines:
        print("| " + line.ljust(width - 4) + " |")
    print("+" + "-" * (width - 2) + "+")
