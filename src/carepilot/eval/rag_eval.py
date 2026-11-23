"""RAG evaluation harness.

Three stages, all on data/eval/rag_golden.jsonl (30 answerable + 10 unanswerable questions):

1. Retrieval ablation (no LLM): hit@1/3/5, MRR@10, doc recall@5 and latency for
   dense, bm25, hybrid (RRF) and hybrid + cross-encoder rerank.
2. Abstention calibration: top reranker score of answerable vs unanswerable questions;
   choose the retrieval-gate threshold, write it to index/rag_config.json.
3. End-to-end answers for one provider: answer rate, key-fact coverage, citation precision
   (cited passages come from a relevant document), sentence citation coverage, abstention
   rate on unanswerable questions and which gate caught them.

Usage:
  python -m carepilot.eval.rag_eval --stage retrieval
  python -m carepilot.eval.rag_eval --stage e2e --provider ollama
  python -m carepilot.eval.rag_eval --stage report     # rebuild markdown from saved json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from carepilot.config import EVAL_DIR, INDEX_DIR, REPORTS_DIR
from carepilot.core.verify import supported_rate, verify_claims
from carepilot.llm.providers import get_llm
from carepilot.rag.answer import answer_question
from carepilot.rag.retrieve import MODES, Retriever

log = logging.getLogger(__name__)
GOLDEN = EVAL_DIR / "rag_golden.jsonl"
RETRIEVAL_JSON = REPORTS_DIR / "rag_eval_retrieval.json"


def load_golden(path: Path = GOLDEN) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


# ---------- stage 1: retrieval ----------


def first_relevant_rank(doc_ids: list[str], relevant: set[str]) -> int | None:
    return next((i for i, d in enumerate(doc_ids, start=1) if d in relevant), None)


def retrieval_metrics(retriever: Retriever, golden: list[dict]) -> dict[str, dict[str, float]]:
    answerable = [g for g in golden if g["answerable"]]
    out: dict[str, dict[str, float]] = {}
    for mode in MODES:
        ranks, recalls, latencies = [], [], []
        for g in answerable:
            t0 = time.perf_counter()
            hits = retriever.search(g["question"], k=10, mode=mode)
            latencies.append(time.perf_counter() - t0)
            docs = [h.chunk.doc_id for h in hits]
            relevant = set(g["relevant_docs"])
            ranks.append(first_relevant_rank(docs, relevant))
            recalls.append(len(relevant & set(docs[:5])) / len(relevant))
        out[mode] = {
            "hit@1": mean(r is not None and r <= 1 for r in ranks),
            "hit@3": mean(r is not None and r <= 3 for r in ranks),
            "hit@5": mean(r is not None and r <= 5 for r in ranks),
            "mrr@10": mean(1 / r if r else 0.0 for r in ranks),
            "doc_recall@5": mean(recalls),
            "latency_ms_median": float(np.median(latencies) * 1000),
        }
        log.info("%-14s %s", mode, {k: round(v, 3) for k, v in out[mode].items()})
    return out


# ---------- stage 2: abstention threshold ----------


def calibrate_threshold(pos: list[float], neg: list[float]) -> dict[str, Any]:
    """Pick the retrieval-gate threshold from top-1 rerank scores.

    Maximize balanced accuracy (mean of answerable-kept rate and unanswerable-rejected
    rate). Among equally good thresholds take the midpoint of the widest gap, which leaves
    the most margin on both sides. Ties are common with 40 questions, so the margin matters.
    """
    candidates = sorted(set(pos) | set(neg))
    grid = [c - 1e-6 for c in candidates] + [candidates[-1] + 1e-6]

    def bal_acc(t: float) -> float:
        return 0.5 * (mean(p >= t for p in pos) + mean(n < t for n in neg))

    best = max(bal_acc(t) for t in grid)
    best_ts = [t for t in grid if abs(bal_acc(t) - best) < 1e-12]
    lo, hi = min(best_ts), max(best_ts)
    below = [s for s in candidates if s < lo]
    above = [s for s in candidates if s >= hi]
    low_edge = max(below) if below else lo
    high_edge = min(above) if above else hi
    threshold = round((low_edge + high_edge) / 2, 2)
    return {
        "abstain_threshold": threshold,
        "balanced_accuracy": round(bal_acc(threshold), 3),
        "answerable_kept": round(mean(p >= threshold for p in pos), 3),
        "unanswerable_rejected": round(mean(n < threshold for n in neg), 3),
        "answerable_min_score": round(min(pos), 2),
        "unanswerable_max_score": round(max(neg), 2),
    }


def leave_one_out(rows: list[dict]) -> dict[str, float]:
    """Honest estimate: calibrate on n-1 questions, judge the held-out one, repeat.

    Calibrating and scoring on the same questions is optimistic; this shows how the gate
    would do on a question it was not tuned on.
    """
    correct_pos, correct_neg = [], []
    for i, held in enumerate(rows):
        rest = rows[:i] + rows[i + 1 :]
        t = calibrate_threshold(
            [r["top_score"] for r in rest if r["answerable"]],
            [r["top_score"] for r in rest if not r["answerable"]],
        )["abstain_threshold"]
        kept = held["top_score"] >= t
        (correct_pos if held["answerable"] else correct_neg).append(
            kept if held["answerable"] else not kept
        )
    return {
        "loo_answerable_kept": round(mean(correct_pos), 3),
        "loo_unanswerable_rejected": round(mean(correct_neg), 3),
    }


def abstention_scores(retriever: Retriever, golden: list[dict]) -> tuple[list, list, list]:
    rows = []
    for g in golden:
        hits = retriever.search(g["question"], k=1)
        rows.append({"id": g["id"], "answerable": g["answerable"], "top_score": hits[0].score})
    pos = [r["top_score"] for r in rows if r["answerable"]]
    neg = [r["top_score"] for r in rows if not r["answerable"]]
    return pos, neg, rows


def run_retrieval_stage() -> None:
    golden = load_golden()
    retriever = Retriever.load()
    metrics = retrieval_metrics(retriever, golden)
    pos, neg, rows = abstention_scores(retriever, golden)
    calib = calibrate_threshold(pos, neg) | leave_one_out(rows)
    (INDEX_DIR / "rag_config.json").write_text(
        json.dumps({"abstain_threshold": calib["abstain_threshold"]}, indent=2), encoding="utf-8"
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    RETRIEVAL_JSON.write_text(
        json.dumps({"retrieval": metrics, "calibration": calib, "scores": rows}, indent=2),
        encoding="utf-8",
    )
    log.info("calibration %s", calib)


# ---------- stage 3: end to end ----------

SENTENCE = re.compile(r"(?<=[.!?])\s+(?!\[)|\n+")  # keep "text. [1]" together


def key_fact_coverage(text: str, facts: list[list[str]]) -> float:
    low = text.lower()
    return mean(any(alt in low for alt in group) for group in facts) if facts else 1.0


def sentence_citation_coverage(text: str) -> float:
    sents = [s for s in SENTENCE.split(text) if len(s.strip()) > 20 and not s.strip().endswith(":")]
    return mean(bool(re.search(r"\[\d", s)) for s in sents) if sents else 0.0


def run_e2e_stage(provider: str | None) -> dict[str, Any]:
    golden = load_golden()
    retriever = Retriever.load()
    llm = get_llm(provider)
    per_q = []
    for g in golden:
        t0 = time.perf_counter()
        ans = answer_question(g["question"], retriever, llm)
        elapsed = time.perf_counter() - t0
        doc_of = {s.n: h.chunk.doc_id for s in ans.sources for h in [ans.retrieved[s.n - 1]]}
        relevant = set(g["relevant_docs"])
        per_q.append(
            {
                "id": g["id"],
                "answerable": g["answerable"],
                "abstained": ans.abstained,
                "reason": ans.reason,
                "fact_coverage": key_fact_coverage(ans.text, g["key_facts"])
                if not ans.abstained
                else 0.0,
                "citation_precision": (
                    mean(doc_of[n] in relevant for n in doc_of) if doc_of and relevant else None
                ),
                "sentence_citation_coverage": sentence_citation_coverage(ans.text)
                if not ans.abstained
                else None,
                "nli_supported": (
                    None if ans.abstained else supported_rate(verify_claims(ans.text, ans.sources))
                ),
                "latency_s": round(elapsed, 2),
                "answer": ans.text,
                # Gate-2 stress test: the unanswerable question again with the retrieval gate
                # off, so the LLM sees the closest (irrelevant) passages and must refuse alone.
                "llm_only_abstained": (
                    None
                    if g["answerable"] or llm.name == "extractive"
                    else answer_question(g["question"], retriever, llm, threshold=-1e9).abstained
                ),
            }
        )
        log.info("%s abstained=%s %.1fs %s", g["id"], ans.abstained, elapsed, ans.reason[:60])

    ans_rows = [r for r in per_q if r["answerable"]]
    un_rows = [r for r in per_q if not r["answerable"]]
    answered = [r for r in ans_rows if not r["abstained"]]
    cps = [r["citation_precision"] for r in answered if r["citation_precision"] is not None]
    summary = {
        "provider": getattr(llm, "label", llm.name),
        "answer_rate": mean(not r["abstained"] for r in ans_rows),
        "fact_coverage": mean(r["fact_coverage"] for r in ans_rows),
        "citation_precision": mean(cps) if cps else None,
        "sentence_citation_coverage": mean(r["sentence_citation_coverage"] for r in answered)
        if answered
        else None,
        "nli_supported": mean(nli)
        if (nli := [r["nli_supported"] for r in answered if r["nli_supported"] is not None])
        else None,
        "abstention_rate_unanswerable": mean(r["abstained"] for r in un_rows),
        "caught_by_retrieval_gate": sum("retrieval gate" in r["reason"] for r in un_rows),
        "caught_by_generation_gate": sum("generation gate" in r["reason"] for r in un_rows),
        "median_latency_s": float(np.median([r["latency_s"] for r in per_q])),
        "llm_only_abstention_rate": (
            None if llm.name == "extractive" else mean(r["llm_only_abstained"] for r in un_rows)
        ),
    }
    name = llm.name
    (REPORTS_DIR / f"rag_eval_e2e_{name}.json").write_text(
        json.dumps({"summary": summary, "questions": per_q}, indent=2), encoding="utf-8"
    )
    log.info("summary %s", summary)
    return summary


# ---------- report ----------


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.0%}"


def write_report() -> Path:
    data = json.loads(RETRIEVAL_JSON.read_text(encoding="utf-8"))
    golden = load_golden()
    n_ans = sum(g["answerable"] for g in golden)
    lines = [
        "# RAG evaluation report",
        "",
        "_Generated by `python -m carepilot.eval.rag_eval`. Do not edit by hand._",
        "",
        f"Golden set: `data/eval/rag_golden.jsonl`, {n_ans} answerable questions with the "
        f"documents that answer them and key facts, plus {len(golden) - n_ans} unanswerable "
        "questions (in-domain traps such as dosing and drug interactions, adjacent medical "
        "topics, and off-topic requests). Small hand-built set: treat differences of a few "
        "points as noise.",
        "",
        "## 1. Retrieval ablation (answerable questions, document-level relevance)",
        "",
        "| Retriever | Hit@1 | Hit@3 | Hit@5 | MRR@10 | Doc recall@5 | Median latency |",
        "|---|---|---|---|---|---|---|",
    ]
    labels = {
        "dense": "Dense (bge-small + FAISS)",
        "bm25": "BM25",
        "hybrid": "Hybrid (RRF)",
        "hybrid_rerank": "Hybrid + cross-encoder rerank",
    }
    for mode, m in data["retrieval"].items():
        lines.append(
            f"| {labels[mode]} | {m['hit@1']:.2f} | {m['hit@3']:.2f} | {m['hit@5']:.2f} | "
            f"{m['mrr@10']:.3f} | {m['doc_recall@5']:.2f} | {m['latency_ms_median']:.0f} ms |"
        )
    c = data["calibration"]
    lines += [
        "",
        "## 2. Retrieval gate calibration (layer 1 abstention)",
        "",
        f"Top-1 cross-encoder score: answerable questions min {c['answerable_min_score']}, "
        f"unanswerable max {c['unanswerable_max_score']}. Chosen threshold "
        f"**{c['abstain_threshold']}** (max balanced accuracy, midpoint of the margin): keeps "
        f"{_pct(c['answerable_kept'])} of answerable questions and rejects "
        f"{_pct(c['unanswerable_rejected'])} of unanswerable ones before any LLM call. "
        "Because the threshold is tuned on these same questions, a leave-one-out estimate is "
        f"the honest number: {_pct(c['loo_answerable_kept'])} kept / "
        f"{_pct(c['loo_unanswerable_rejected'])} rejected on questions not used for tuning. "
        "Unanswerable questions that look on-topic (for example a dosing question about a drug "
        "the documents mention) can pass this gate; layer 2 (the LLM sentinel) exists for them.",
        "",
        "| Question | Answerable | Top-1 score |",
        "|---|---|---|",
    ]
    q_text = {g["id"]: g["question"] for g in golden}
    for r in sorted(data["scores"], key=lambda r: r["top_score"]):
        if not r["answerable"] or r["top_score"] < c["abstain_threshold"] + 3:
            lines.append(
                f"| {q_text[r['id']]} | {'yes' if r['answerable'] else 'no'} | {r['top_score']:.2f} |"
            )
    lines += ["", "_(answerable questions far above the threshold omitted)_", ""]

    e2e = sorted(REPORTS_DIR.glob("rag_eval_e2e_*.json"))
    lines += [
        "## 3. End-to-end answers",
        "",
        "| Provider | Answer rate (answerable) | Key-fact coverage | Citation precision | "
        "Sentences with citation | Sentences entailed by citation (NLI) | "
        "Abstained on unanswerable | Caught by gate 1 / gate 2 | "
        "LLM alone refuses (gate 1 off) | Median latency |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for path in e2e:
        s = json.loads(path.read_text(encoding="utf-8"))["summary"]
        lines.append(
            f"| {s['provider']} | {_pct(s['answer_rate'])} | {_pct(s['fact_coverage'])} | "
            f"{_pct(s['citation_precision'])} | {_pct(s['sentence_citation_coverage'])} | "
            f"{_pct(s.get('nli_supported'))} | "
            f"{_pct(s['abstention_rate_unanswerable'])} | {s['caught_by_retrieval_gate']} / "
            f"{s['caught_by_generation_gate']} | {_pct(s.get('llm_only_abstention_rate'))} | "
            f"{s['median_latency_s']:.1f} s |"
        )
    lines += [
        "",
        "Metric definitions: *key-fact coverage* is the share of expected fact groups found in "
        "the answer text (string match on any accepted phrasing); *citation precision* is the "
        "share of cited passages that come from a document marked relevant; *sentences with "
        "citation* is the share of answer sentences carrying a [n] marker; *LLM alone "
        "refuses* re-asks the unanswerable questions with the retrieval gate disabled to see "
        "whether the model refuses by itself (defense in depth); *sentences entailed by "
        "citation* runs an NLI model (DeBERTa-v3-xsmall) on each sentence against its cited "
        "passage, a strict check that can flag faithful paraphrases. String matching "
        "undercounts paraphrases, so coverage is a lower bound.",
        "",
    ]
    out = REPORTS_DIR / "rag_eval.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["retrieval", "e2e", "report", "all"], default="all")
    parser.add_argument("--provider", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("httpx", "sentence_transformers", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if args.stage in ("retrieval", "all"):
        run_retrieval_stage()
    if args.stage in ("e2e", "all"):
        run_e2e_stage(args.provider)
    write_report()


if __name__ == "__main__":
    main()
