"""Offline cutoff sweeps and score-blind original-passage review packets.

No models, cloud jobs, training or database access. Precision/coverage are
conditional on definite manual labels. Unknown or uncertain labels never become
negative examples. A frozen policy requires complete selected-development review;
held-out sweeps require that frozen policy and never search the threshold grid.

Examples:
  python scripts/analyze-reranker-cutoff.py packet --run-dir data/cutoff-run --config questions.json --output data/review-dev
  python scripts/analyze-reranker-cutoff.py sweep --run-dir data/cutoff-run --config questions.json --labels labels.json --output reports/cutoff-dev
  python scripts/analyze-reranker-cutoff.py packet --run-dir data/cutoff-run --config questions.json --policy-id absolute:-4:cap25 --complete-selection --only-unreviewed --labels labels.json --output data/review-complete
  python scripts/analyze-reranker-cutoff.py freeze --run-dir data/cutoff-run --config questions.json --labels labels.json --policy-id absolute:-4:cap25 --reason 'Reviewed tradeoff' --output data/frozen-policy.json
  python scripts/analyze-reranker-cutoff.py sweep --split heldout --run-dir data/cutoff-run --config questions.json --policy data/frozen-policy.json --labels heldout-labels.json --output reports/cutoff-heldout
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def write_csv(path, records):
    if not records:
        Path(path).write_text("")
        return
    fields = list(dict.fromkeys(k for row in records for k in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    k: json.dumps(v, ensure_ascii=False)
                    if isinstance(v, (list, dict))
                    else v
                    for k, v in row.items()
                }
            )


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def policies(config):
    output = []
    for kind, values in [
        ("absolute", config["absolute_logit_cutoffs"]),
        ("relative_gap", config["relative_gap_cutoffs"]),
    ]:
        for value in values:
            if value is not None and not math.isfinite(float(value)):
                raise ValueError("Policy values must be finite")
            for cap in config["selection_caps"]:
                tag = "none" if value is None else format(float(value), "g")
                output.append(
                    {
                        "policy_id": f"{kind}:{tag}:cap{cap}",
                        "kind": kind,
                        "value": value,
                        "cap": int(cap),
                    }
                )
    return output


def overlap(a, b):
    if (a["lang"], a["page_id"], a["rev_id"]) != (b["lang"], b["page_id"], b["rev_id"]):
        return False
    length = min(len(a["text"]), len(b["text"]))
    return (
        length > 0
        and max(
            0, min(a["end_char"], b["end_char"]) - max(a["start_char"], b["start_char"])
        )
        / length
        > 0.5
    )


def select(ranked, policy):
    if not ranked:
        return []
    value = policy["value"]
    threshold = (
        (-math.inf if value is None else float(value))
        if policy["kind"] == "absolute"
        else float(ranked[0]["rerank_score"]) - float(value)
    )
    output = []
    for passage in ranked:
        if passage["rerank_score"] < threshold or any(
            overlap(passage, p) for p in output
        ):
            continue
        output.append(passage)
        if len(output) >= policy["cap"]:
            break
    return output


def item_id(configuration_sha, question, passage):
    identity = {
        "configuration_sha256": configuration_sha,
        "query": question,
        "source": {
            k: passage[k]
            for k in ["lang", "page_id", "rev_id", "start_char", "end_char", "text"]
        },
    }
    return "R-" + digest(identity)[:24]


def load_run(run_dir, config_path):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    config_sha = file_sha(config_path)
    queries = {}
    registry = {}
    legacy = {}
    missing = []
    identities = []
    ranker = None
    for question in config["questions"]:
        qid = question["id"]
        directory = Path(run_dir) / qid
        path = directory / "ranking.json"
        if not path.exists():
            missing.append(qid)
            continue
        ranking = json.loads(path.read_text())
        ranked = ranking["ranked"]
        if any(not math.isfinite(float(p["rerank_score"])) for p in ranked):
            raise ValueError(f"Nonfinite score in {qid}")
        if any(
            a["rerank_score"] < b["rerank_score"] for a, b in zip(ranked, ranked[1:])
        ):
            raise ValueError(f"Ranking is not descending: {qid}")
        if len({p["id"] for p in ranked}) != len(ranked):
            raise ValueError(f"Duplicate chunk IDs in {qid}")
        identity_path = directory / "identity.json"
        identity = (
            json.loads(identity_path.read_text()) if identity_path.exists() else None
        )
        if identity:
            if (
                identity.get("configuration_sha256") != config_sha
                or identity.get("query") != question
            ):
                raise ValueError(f"Frozen configuration/query mismatch: {qid}")
            identities.append(
                {
                    k: identity.get(k)
                    for k in ["configuration_sha256", "code_hashes", "database"]
                }
            )
        diagnostic = ranking.get("diagnostics", {})
        current = {
            "model": diagnostic.get("model"),
            "revision": diagnostic.get("revision"),
        }
        if ranker is not None and current != ranker:
            raise ValueError("Reranker identity differs between queries")
        ranker = current
        queries[qid] = {
            "question": question,
            "ranked": ranked,
            "ranking_sha256": file_sha(path),
        }
        for rank, passage in enumerate(ranked, 1):
            opaque = item_id(config_sha, question, passage)
            if opaque in registry:
                raise ValueError(
                    "Duplicate source identity within query; inspect chunking"
                )
            registry[opaque] = {
                "query_id": qid,
                "question": question,
                "passage": passage,
                "rank": rank,
            }
            legacy[(qid, str(passage["id"]))] = opaque
    if identities and any(i != identities[0] for i in identities[1:]):
        raise ValueError("Corpus/code identity differs between query checkpoints")
    calibration = {
        "configuration_sha256": config_sha,
        "reranker": ranker,
        "corpus_and_code_identity_sha256": digest(identities[0])
        if identities
        else None,
        "max_pair_tokens": config["max_pair_tokens"],
        "source_language": config.get("source_language", "fi"),
    }
    return {
        "config": config,
        "configuration_sha256": config_sha,
        "queries": queries,
        "registry": registry,
        "legacy": legacy,
        "missing": missing,
        "calibration_identity": calibration,
    }


def boolean(value):
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in (
        "true",
        "false",
        "1",
        "0",
        "yes",
        "no",
    ):
        return value.lower() in ("true", "1", "yes")
    raise ValueError(f"Invalid uncertain flag: {value!r}")


def load_labels(paths, run):
    labels = {}
    sources = []
    for path in paths or []:
        path = Path(path)
        outer_query = None
        if path.suffix.lower() == ".csv":
            with path.open(newline="") as stream:
                records = list(csv.DictReader(stream))
        else:
            data = json.loads(path.read_text())
            if isinstance(data, list):
                records = data
            else:
                if data.get("configuration_sha256") not in (
                    None,
                    run["configuration_sha256"],
                ):
                    raise ValueError(f"Label configuration mismatch: {path}")
                records = data.get("records", data.get("labels"))
                outer_query = data.get("query", {}).get("id")
            if not isinstance(records, list):
                raise ValueError(f"Expected records/labels list: {path}")
        sources.append({"path": str(path), "sha256": file_sha(path)})
        for row in records:
            opaque = row.get("item_id")
            if not opaque:
                opaque = run["legacy"].get(
                    (row.get("query_id", outer_query), str(row.get("id")))
                )
            if opaque not in run["registry"]:
                raise ValueError(
                    f"Unknown label item in {path}: {row.get('item_id') or row.get('id')}"
                )
            grade = row.get("grade")
            if grade in (None, ""):
                continue
            if isinstance(grade, bool) or str(grade) not in ("0", "1", "2", "3"):
                raise ValueError(f"Grade must be integer0–3 or blank: {opaque}")
            label = {
                "grade": int(grade),
                "uncertain": boolean(row.get("uncertain")) is True,
                "reason": row.get("reason", ""),
                "reviewer": row.get("reviewer", ""),
            }
            if opaque in labels and labels[opaque] != label:
                raise ValueError(
                    f"Conflicting duplicate labels for {opaque}; resolve explicitly"
                )
            labels[opaque] = label
    return labels, sources


def query_subset(run, split, allow_partial=False):
    expected = [
        q["id"]
        for q in run["config"]["questions"]
        if split == "all" or q["split"] == split
    ]
    missing = [qid for qid in expected if qid not in run["queries"]]
    if missing and not allow_partial:
        raise ValueError("Missing scored queries: " + ", ".join(missing))
    queries = [run["queries"][qid] for qid in expected if qid in run["queries"]]
    if not queries:
        raise ValueError("No scored queries for this split")
    return queries, missing


def resolve_policy(run, policy_id=None, path=None, frozen=False):
    if path:
        data = json.loads(Path(path).read_text())
        if frozen and data.get("status") != "frozen_from_development_review":
            raise ValueError("Held-out evaluation requires a development-frozen policy")
        if data.get("calibration_identity") != run["calibration_identity"]:
            raise ValueError("Policy model/corpus/config identity mismatch")
        policy = data["policy"]
    else:
        policy = next(
            (p for p in policies(run["config"]) if p["policy_id"] == policy_id), None
        )
    if policy is None or policy not in policies(run["config"]):
        raise ValueError(
            "Unknown configured policy; inspect sweep results for policy IDs"
        )
    return policy


def sampled_passages(ranked, config, size, seed):
    if size < 3:
        raise ValueError(
            "Sample size must be at least3 to cover rank/boundary/tail strata"
        )
    chosen = []

    def add(p):
        if p["id"] in {x["id"] for x in chosen} or any(overlap(p, x) for x in chosen):
            return False
        chosen.append(p)
        return True

    top_count = min(8, max(1, size // 3))
    tail_count = min(4, max(1, size // 6))
    for p in ranked:
        add(p)
        if len(chosen) >= top_count:
            break
    tail_added = 0
    for p in reversed(ranked):
        if add(p):
            tail_added += 1
        if tail_added >= tail_count:
            break
    thresholds = [float(t) for t in config["absolute_logit_cutoffs"] if t is not None]
    if ranked:
        thresholds += [
            ranked[0]["rerank_score"] - float(g) for g in config["relative_gap_cutoffs"]
        ]
    rng = random.Random(seed)
    rng.shuffle(thresholds)
    for cutoff in thresholds:
        for above in (True, False):
            for p in sorted(
                (p for p in ranked if (p["rerank_score"] >= cutoff) == above),
                key=lambda p: abs(p["rerank_score"] - cutoff),
            ):
                if add(p):
                    break
            if len(chosen) >= size:
                return chosen[:size]
    remaining = list(ranked)
    rng.shuffle(remaining)
    for p in remaining:
        add(p)
        if len(chosen) >= size:
            break
    return chosen[:size]


def fence(text):
    runs = [len(m[0]) for m in re.finditer(r"`+", text)]
    return "`" * max(3, 1 + max(runs, default=0))


def create_packet(args, run, labels, label_sources):
    queries, missing = query_subset(run, args.split, args.allow_partial)
    policy = (
        resolve_policy(run, args.policy_id, args.policy)
        if args.policy_id or args.policy
        else None
    )
    if args.complete_selection and policy is None:
        raise ValueError("--complete-selection requires --policy-id or --policy")
    records = []
    key = {}
    for entry in queries:
        q = entry["question"]
        seed = run["config"].get("seed", 42) + int(digest(q["id"])[:8], 16)
        selected = (
            select(entry["ranked"], policy)
            if args.complete_selection
            else sampled_passages(
                entry["ranked"], run["config"], args.sample_size, seed
            )
        )
        for passage in selected:
            opaque = item_id(run["configuration_sha256"], q, passage)
            if (
                args.only_unreviewed
                and opaque in labels
                and not labels[opaque]["uncertain"]
            ):
                continue
            record = {
                "item_id": opaque,
                "query_id": q["id"],
                "question": q["question"],
                "search_focus": q["search_focus"],
                "intent": q.get("intent", ""),
                "query_language": q["language"],
                **{
                    k: passage.get(k)
                    for k in [
                        "title",
                        "lang",
                        "page_id",
                        "rev_id",
                        "start_char",
                        "end_char",
                        "heading_path",
                        "revision_url",
                        "text",
                    ]
                },
                "grade": None,
                "uncertain": None,
                "reason": "",
                "reviewer": "",
            }
            records.append(record)
            origin = run["registry"][opaque]
            key[opaque] = {
                "query_id": q["id"],
                "chunk_id": passage["id"],
                "score": passage["rerank_score"],
                "rank": origin["rank"],
            }
    random.Random(run["config"].get("seed", 42)).shuffle(records)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    packet = {
        "created_at_utc": utc_now(),
        "configuration_sha256": run["configuration_sha256"],
        "split": args.split,
        "label_protocol": run["config"]["label_protocol"],
        "records": records,
        "missing_queries": missing,
        "blinding": "Score/rank/original chunk IDs omitted. Item IDs identify query-specific original source slices.",
        "sampling_scope": (
            "Every retained item of the requested policy (or its still-unreviewed subset), not a population sample. Scores/ranks are hidden; empty grade means unreviewed."
            if args.complete_selection
            else "Purposive top/boundary/low-tail sampling, not a random population sample. Empty grade means unreviewed."
        ),
    }
    write_json(out / "review-packet.json", packet)
    write_csv(out / "review-packet.csv", records)
    write_json(
        out / "review-score-key.json",
        {
            "configuration_sha256": run["configuration_sha256"],
            "items": key,
            "instruction": "Keep separate from the reviewer until labels are recorded.",
        },
    )
    lines = [
        "# Score-blind relevance review",
        "",
        "Grade0 wrong subject/unrelated;1 background only;2 useful specific answer evidence;3 direct substantial answer evidence.",
        "Review only the source text for the focused query. Exact quotations and factual answer support are separate checks. Scores and ranks are hidden.",
        "",
    ]
    for r in records:
        boundary = fence(r["text"])
        lines += [
            "## " + r["item_id"],
            "",
            "**Question:** " + html.escape(r["question"]),
            "**Context:** " + html.escape(r["search_focus"]),
            "**Intent:** " + html.escape(r["intent"]),
            "**Query/source language:** "
            + r["query_language"]
            + " / "
            + str(r["lang"]),
            "**Source:** " + html.escape(str(r["title"])),
            "**Revision:** " + str(r["revision_url"]),
            "**Original offsets:** " + str(r["start_char"]) + ":" + str(r["end_char"]),
            "",
            boundary + "text",
            r["text"],
            boundary,
            "",
            "Grade: ____ · uncertain: ____ · reason: ____________________ · reviewer: ____________________",
            "",
        ]
    (out / "review-packet.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(records)} score-blind items to {out}")


def query_metrics(run, entry, policy, labels):
    q = entry["question"]
    selected = select(entry["ranked"], policy)

    def lookup(p):
        return labels.get(item_id(run["configuration_sha256"], q, p))

    known = [
        p
        for p in entry["ranked"]
        if (lookup(p) is not None and not lookup(p)["uncertain"])
    ]
    positives = [p for p in known if lookup(p)["grade"] >= 2]
    reviewed = [
        p for p in selected if (lookup(p) is not None and not lookup(p)["uncertain"])
    ]
    positive_selected = [p for p in reviewed if lookup(p)["grade"] >= 2]
    uncertain_selected = [
        p for p in selected if lookup(p) is not None and lookup(p)["uncertain"]
    ]
    nr, np, ns = len(reviewed), len(positive_selected), len(selected)
    return {
        "query_id": q["id"],
        "topic_group": q["topic_group"],
        "query_language": q["language"],
        "candidate_chunks": len(entry["ranked"]),
        "selected_count": ns,
        "selected_characters": sum(len(p["text"]) for p in selected),
        "reviewed_candidate_count": len(known),
        "reviewed_positive_pool_count": len(positives),
        "selected_reviewed_count": nr,
        "selected_unreviewed_count": ns - nr,
        "selected_uncertain_count": len(uncertain_selected),
        "selected_positive_count": np,
        "selected_negative_count": nr - np,
        "conditional_precision": np / nr if nr else None,
        "reviewed_pool_positive_coverage": np / len(positives) if positives else None,
        "fully_reviewed_selection_precision": np / ns if ns and nr == ns else None,
        "precision_lower_bound": np / ns if ns else None,
        "precision_upper_bound": (np + ns - nr) / ns if ns else None,
        "selected_review_fraction": nr / ns if ns else None,
        "empty_selection": ns == 0,
        "selected_item_ids": [
            item_id(run["configuration_sha256"], q, p) for p in selected
        ],
    }


def aggregate(rows):
    total = lambda key: sum(r[key] for r in rows)
    ns, nr, np, pool = (
        total(k)
        for k in [
            "selected_count",
            "selected_reviewed_count",
            "selected_positive_count",
            "reviewed_positive_pool_count",
        ]
    )
    nonempty_precisions = [
        r["conditional_precision"]
        for r in rows
        if r["conditional_precision"] is not None
    ]
    answerable_reviewed = [r for r in rows if r["reviewed_positive_pool_count"] > 0]
    return {
        "queries": len(rows),
        "selected_count": ns,
        "selected_reviewed_count": nr,
        "selected_unreviewed_count": ns - nr,
        "selected_uncertain_count": total("selected_uncertain_count"),
        "selected_positive_count": np,
        "reviewed_positive_pool_count": pool,
        "micro_conditional_precision": np / nr if nr else None,
        "macro_conditional_precision": sum(nonempty_precisions)
        / len(nonempty_precisions)
        if nonempty_precisions
        else None,
        "precision_defined_queries": len(nonempty_precisions),
        "reviewed_pool_positive_coverage": np / pool if pool else None,
        "fully_reviewed_selection_precision": np / ns if ns and ns == nr else None,
        "selected_review_fraction": nr / ns if ns else None,
        "queries_with_reviewed_positive": len(answerable_reviewed),
        "queries_retaining_reviewed_positive": sum(
            r["selected_positive_count"] > 0 for r in answerable_reviewed
        ),
        "empty_queries": sum(r["empty_selection"] for r in rows),
    }


def evaluate(run, queries, policy, labels):
    rows = [query_metrics(run, q, policy, labels) for q in queries]
    breakdown = {}
    for field in ["query_language", "topic_group"]:
        groups = defaultdict(list)
        for row in rows:
            groups[row[field]].append(row)
        breakdown[field] = {key: aggregate(values) for key, values in groups.items()}
    return {
        "policy": policy,
        "summary": aggregate(rows),
        "by_query": rows,
        "breakdown": breakdown,
    }


def percent(value):
    return "—" if value is None else f"{100 * value:.1f}%"


def write_sweep(args, run, labels, label_sources):
    queries, missing = query_subset(run, args.split, args.allow_partial)
    if args.split == "heldout":
        if not args.policy:
            raise ValueError(
                "Held-out evaluation requires --policy; threshold-grid exploration is development-only"
            )
        candidates = [resolve_policy(run, path=args.policy, frozen=True)]
    elif args.policy:
        candidates = [resolve_policy(run, path=args.policy)]
    else:
        candidates = policies(run["config"])
    results = [evaluate(run, queries, p, labels) for p in candidates]
    relevant_labels = {
        key: value
        for key, value in labels.items()
        if run["registry"][key]["query_id"] in {e["question"]["id"] for e in queries}
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    analysis = {
        "created_at_utc": utc_now(),
        "configuration_sha256": run["configuration_sha256"],
        "calibration_identity": run["calibration_identity"],
        "split": args.split,
        "missing_queries": missing,
        "label_sources": label_sources,
        "labels_used": len(relevant_labels),
        "uncertain_labels": sum(x["uncertain"] for x in relevant_labels.values()),
        "results": results,
        "automatic_policy_selection": False,
        "definitions": {
            "positive": "Definite manual grade>=2; uncertain labels treated as unreviewed in main metrics.",
            "conditional_precision": "Selected definite positives / selected definite reviewed items.",
            "reviewed_pool_positive_coverage": "Selected definite positives / all definite positives in the reviewed candidate pool; not corpus recall.",
            "unreviewed": "Missing/blank grade or uncertain flag true. Never converted to negative.",
            "selection": "Descending logit cutoff then >50% overlap deduplication and cap. No per-article/context packing/Qwen focus filter here.",
            "sample_bias": "Purposive review sample is not representative. Compare unknown counts; fully review a proposed policy before freezing.",
            "empty": "Empty selections have undefined precision, not perfect precision.",
        },
    }
    write_json(out / "analysis.json", analysis)
    summaries = [
        {"policy_id": x["policy"]["policy_id"], **x["summary"]} for x in results
    ]
    write_csv(out / "policy-summary.csv", summaries)
    per_query = [
        {"policy_id": x["policy"]["policy_id"], **row}
        for x in results
        for row in x["by_query"]
    ]
    write_csv(out / "per-query.csv", per_query)
    lines = [
        "# Reranker cutoff analysis",
        "",
        "Split: **" + args.split + "**. No automatic policy selection.",
        "Precision and positive coverage are conditional on definite reviewed passages. Unreviewed/uncertain entries remain unknown. These results are not corpus recall or answer faithfulness.",
        "",
        "| Policy | Selected | Reviewed | Unreviewed | Conditional precision | Reviewed-positive coverage | Fully reviewed precision | Empty queries |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        s = result["summary"]
        lines.append(
            "| "
            + " | ".join(
                [
                    result["policy"]["policy_id"],
                    str(s["selected_count"]),
                    str(s["selected_reviewed_count"]),
                    str(s["selected_unreviewed_count"]),
                    percent(s["micro_conditional_precision"]),
                    percent(s["reviewed_pool_positive_coverage"]),
                    percent(s["fully_reviewed_selection_precision"]),
                    str(s["empty_queries"]),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "No generation context limit or per-article budget is applied here. A policy must also fit the live pipeline token budget. Full details, query/language/topic breakdowns and selected opaque IDs are in analysis.json.",
        "Review all items retained by a proposed development policy with packet --policy-id ... --complete-selection --only-unreviewed. Then freeze with an explicit policy ID and reason. Held-out evaluation accepts only that frozen policy.",
    ]
    (out / "analysis.md").write_text("\n".join(lines) + "\n")
    print(
        f"Wrote {len(results)} policy evaluations to {out}; labels used={len(relevant_labels)}"
    )


def freeze_policy(args, run, labels, label_sources):
    if Path(args.output).exists():
        raise FileExistsError(
            "Frozen policy output already exists; choose a new filename rather than silently overwriting it"
        )
    queries, _ = query_subset(run, "development", allow_partial=False)
    if not run["calibration_identity"]["corpus_and_code_identity_sha256"]:
        raise ValueError("Cannot freeze without runner identity.json provenance")
    policy = resolve_policy(run, policy_id=args.policy_id)
    result = evaluate(run, queries, policy, labels)
    summary = result["summary"]
    if summary["selected_unreviewed_count"]:
        raise ValueError(
            f"Cannot freeze: {summary['selected_unreviewed_count']} selected development items are unreviewed/uncertain"
        )
    if any(row["reviewed_candidate_count"] == 0 for row in result["by_query"]):
        raise ValueError(
            "Cannot freeze: every development query needs definite reviewed candidates"
        )
    if not summary["selected_positive_count"]:
        raise ValueError(
            "Cannot freeze a policy retaining no reviewed positive development evidence"
        )
    dropped = [
        row["query_id"]
        for row in result["by_query"]
        if row["reviewed_positive_pool_count"] and not row["selected_positive_count"]
    ]
    if dropped:
        raise ValueError(
            "Cannot freeze: policy drops all known positive evidence for "
            + ", ".join(dropped)
        )
    qids = {q["question"]["id"] for q in queries}
    used = {
        key: value
        for key, value in labels.items()
        if run["registry"][key]["query_id"] in qids
    }
    record = {
        "status": "frozen_from_development_review",
        "frozen_at_utc": utc_now(),
        "policy": policy,
        "reason": args.reason,
        "calibration_identity": run["calibration_identity"],
        "development_evaluation": result,
        "development_labels_sha256": digest(used),
        "development_ranking_sha256": {
            q["question"]["id"]: q["ranking_sha256"] for q in queries
        },
        "label_source_files": label_sources,
        "heldout_labels_used": False,
        "limitations": "Selected development passages are fully reviewed; pool coverage remains conditional on reviewed candidates. No calibration probability, corpus recall, answer-support guarantee or generation-context fit is implied.",
    }
    write_json(args.output, record)
    print("Frozen development policy:", policy["policy_id"], "->", args.output)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["packet", "sweep", "freeze"]:
        p = sub.add_parser(name)
        p.add_argument("--run-dir", type=Path, required=True)
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--labels", type=Path, action="append", default=[])
        if name != "freeze":
            p.add_argument(
                "--split",
                choices=["development", "heldout", "all"]
                if name == "packet"
                else ["development", "heldout"],
                default="development",
            )
            p.add_argument(
                "--allow-partial",
                action="store_true",
                help="Exploratory output only; freeze always requires every development query",
            )
            p.add_argument("--policy", type=Path)
        if name == "packet":
            p.add_argument("--policy-id")
            p.add_argument("--sample-size", type=int, default=24)
            p.add_argument("--complete-selection", action="store_true")
            p.add_argument("--only-unreviewed", action="store_true")
        if name == "freeze":
            p.add_argument("--policy-id", required=True)
            p.add_argument("--reason", required=True)
    args = parser.parse_args()
    run = load_run(args.run_dir, args.config)
    labels, sources = load_labels(args.labels, run)
    if args.command == "packet":
        create_packet(args, run, labels, sources)
    elif args.command == "sweep":
        write_sweep(args, run, labels, sources)
    else:
        freeze_policy(args, run, labels, sources)


if __name__ == "__main__":
    main()
