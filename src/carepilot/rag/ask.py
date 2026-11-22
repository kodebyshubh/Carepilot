"""Command-line question answering over the indexed documents.

Usage: python -m carepilot.rag.ask "What are the symptoms of diabetes?" [--provider groq]
"""

from __future__ import annotations

import argparse
import logging

from carepilot.llm.providers import get_llm
from carepilot.rag.answer import answer_question
from carepilot.rag.retrieve import Retriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--provider", default=None, help="auto | groq | ollama | extractive")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    llm = get_llm(args.provider)
    ans = answer_question(args.question, Retriever.load(), llm)
    print(f"[{ans.provider}] {'ABSTAINED' if ans.abstained else 'ANSWER'} ({ans.reason})\n")
    print(ans.text, "\n")
    for s in ans.sources:
        where = f"{s.title} > {s.section}" if s.section else s.title
        print(f"[{s.n}] {s.publisher}: {where}\n    {s.url}")


if __name__ == "__main__":
    main()
