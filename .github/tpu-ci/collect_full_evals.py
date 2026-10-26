"""Retain accuracy predictions and combine complete MMMU-Pro shards per model."""

import base64
import hashlib
import io
import json
import os
import subprocess
import tarfile
from pathlib import Path

MMMU_PRO_QUESTIONS = 1730


def combine_shards(shards, total=MMMU_PRO_QUESTIONS):
    count = shards[0]["shards"] if shards else 0
    if (
        count < 1
        or len(shards) != count
        or {s["shard"] for s in shards} != set(range(count))
        or any(s["shards"] != count for s in shards)
    ):
        raise ValueError(f"Expected all {count} MMMU-Pro shards, got {len(shards)}")
    ids = [qid for shard in shards for qid in shard["ids"]]
    if len(ids) != total or len(set(ids)) != total:
        raise ValueError("MMMU-Pro coverage is incomplete or contains duplicates")
    for shard in shards:
        expected = len(range(shard["shard"], total, count))
        if shard["questions"] != expected or len(shard["ids"]) != expected:
            raise ValueError(f"shard {shard['shard']} has {shard['questions']} rows")
        if not 0 <= shard["correct"] <= expected:
            raise ValueError(f"Invalid correct count for shard {shard['shard']}")
        if abs(shard["score"] - shard["correct"] / expected) > 1e-6:
            raise ValueError(f"Score/count mismatch for shard {shard['shard']}")
    correct = sum(s["correct"] for s in shards)
    return {"questions": total, "correct": correct, "score": correct / total}


def add_summary(groups, row):
    model = row.get("model", "legacy")
    aggregate = row.get("aggregate")
    if aggregate and aggregate.get("model") != model:
        raise ValueError("Aggregate model does not match its shard")
    key = (model, aggregate["id"] if aggregate else "")
    group = groups.setdefault(
        key, {"model": model, "aggregate": aggregate, "shards": {}}
    )
    if group["aggregate"] != aggregate:
        group["invalid"] = True
        raise ValueError(f"Conflicting aggregate metadata for {model}")
    previous = group["shards"].get(row["shard"])
    if previous is not None and previous != row:
        group["invalid"] = True
        raise ValueError(f"Conflicting shard summaries for {model}")
    group["shards"][row["shard"]] = row


def is_shard_job(job):
    return "mmmu_shard" in job["name"] or "mmmupro-shard" in job["name"]


def latest_jobs(jobs):
    """Keep earlier successful legs when GitHub reruns only part of a matrix."""
    selected = {}
    for job in jobs:
        previous = selected.get(job["name"])
        rank = (job.get("run_attempt", 1), job["id"])
        if previous is None or rank > (previous.get("run_attempt", 1), previous["id"]):
            selected[job["name"]] = job
    return list(selected.values())


def main():
    repo, run_id = os.environ["GH_REPO"], os.environ["RUN_ID"]
    data = subprocess.check_output(
        [
            "gh",
            "api",
            "--paginate",
            "--slurp",
            f"repos/{repo}/actions/runs/{run_id}/jobs?filter=all&per_page=100",
        ],
        text=True,
    )
    jobs = [
        job
        for job in latest_jobs(
            [job for page in json.loads(data) for job in page["jobs"]]
        )
        if job.get("conclusion") != "skipped"
    ]
    groups, errors = {}, []
    output = Path("full-eval-results")
    for job in jobs:
        # MMLU-Pro can opt into archives too. Build/plan logs have no predictions.
        if not is_shard_job(job) and "accuracy" not in job["name"]:
            continue
        reported = False
        try:
            log = subprocess.check_output(
                [
                    "gh",
                    "api",
                    f"repos/{repo}/actions/jobs/{job['id']}/logs",
                    "--allow-escape-sequences",
                ],
                text=True,
            )
            chunks, archive_index = None, 0
            for line in log.splitlines():
                if "MMMU_SHARD_RESULT " in line:
                    row = json.loads(line.split("MMMU_SHARD_RESULT ", 1)[1])
                    add_summary(groups, row)
                    reported = True
                if "FULL_EVAL_ARCHIVE_BEGIN" in line:
                    chunks = []
                elif "FULL_EVAL_ARCHIVE_CHUNK " in line and chunks is not None:
                    chunks.append(line.split("FULL_EVAL_ARCHIVE_CHUNK ", 1)[1].strip())
                elif "FULL_EVAL_ARCHIVE_END" in line and chunks:
                    directory = output / str(job["id"])
                    if archive_index:
                        directory /= f"archive-{archive_index}"
                    directory.mkdir(parents=True, exist_ok=True)
                    with tarfile.open(
                        fileobj=io.BytesIO(
                            base64.b64decode("".join(chunks), validate=True)
                        ),
                        mode="r:gz",
                    ) as archive:
                        archive.extractall(directory, filter="data")
                    chunks = None
                    archive_index += 1
        except (subprocess.CalledProcessError, ValueError, tarfile.TarError) as exc:
            errors.append(f"{job['name']}: {exc}")
        if is_shard_job(job) and not reported:
            errors.append(f"Missing MMMU-Pro shard result from {job['name']}")

    completed = []
    for key, group in groups.items():
        if group.get("invalid"):
            continue
        try:
            result = combine_shards(list(group["shards"].values()))
        except ValueError as exc:
            errors.append(f"{group['model']}: {exc}")
            continue
        output.mkdir(exist_ok=True)
        # Preserve the existing single-model artifact path. Multiple models get
        # separate files, including when they share the same question IDs.
        digest = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:12]
        filename = (
            "mmmu-full-summary.json" if len(groups) == 1 else f"mmmu-full-{digest}.json"
        )
        (output / filename).write_text(json.dumps(result, indent=2))
        completed.append({"model": group["model"], **result})
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
            summary.write(
                f"{group['model']} MMMU-Pro full set: "
                f"**{result['correct']}/1730 = {result['score']:.2%}**\n"
            )
        aggregate = group["aggregate"]
        if aggregate:
            labels = {
                "kind": "accuracy",
                "model": aggregate["model"],
                "id": aggregate["id"],
                "display_name": aggregate["display_name"],
            }
            print(
                f"[METRIC] mmmu_pro={result['score']} labels={json.dumps(labels)}",
                flush=True,
            )
            floor = aggregate["baseline"] - aggregate["atol"]
            if result["score"] < floor:
                errors.append(
                    f"{aggregate['id']}: {result['score']:.4f} < floor {floor:.4f}"
                )
    if completed:
        (output / "mmmu-model-summaries.json").write_text(
            json.dumps(completed, indent=2)
        )
    if errors:
        raise ValueError("\n".join(errors))


if __name__ == "__main__":
    main()
