"""Run RefEdit-Bench inference from the command line."""

from __future__ import annotations

import sys

from scripts import refedit_bench_inference_mixed, refedit_bench_inference_text


def main() -> None:
    prompt_mode = "mixed"
    for i, arg in enumerate(sys.argv):
        if arg == "--prompt-mode" and i + 1 < len(sys.argv):
            prompt_mode = sys.argv[i + 1]
            del sys.argv[i : i + 2]
            break
        elif arg.startswith("--prompt-mode="):
            prompt_mode = arg.split("=", 1)[1]
            del sys.argv[i]
            break

    if prompt_mode == "mixed":
        refedit_bench_inference_mixed.main()
    elif prompt_mode == "text":
        refedit_bench_inference_text.main()
    else:
        print(
            f"Error: Unsupported --prompt-mode {prompt_mode!r}. Choose 'mixed' or 'text'.",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
