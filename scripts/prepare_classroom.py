import json
import unicodedata
from collections import Counter
from hashlib import file_digest
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow.parquet as pq
from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/allenai/tulu-3-sft-mixture"
BENCHMARKS = ROOT / "data/benchmarks"
OUTPUT = ROOT / "data/sft"
SEED = 42
TRAIN_SIZE = 3000
VALIDATION_SIZE = 300


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def benchmark_prompts():
    questions = set()
    for directory in ("cais/mmlu/all", "openai/gsm8k/main"):
        test_file = BENCHMARKS / directory / "test-00000-of-00001.parquet"
        for text in pq.read_table(test_file, columns=["question"])["question"].to_pylist():
            questions.add(normalize(text))
    with (BENCHMARKS / "google/IFEval/ifeval_input_data.jsonl").open(encoding="utf-8") as file:
        questions.update(normalize(json.loads(line)["prompt"]) for line in file if line.strip())
    return questions


def valid_dialogue(messages):
    if not messages or any(not isinstance(m["content"], str) or not m["content"].strip() for m in messages):
        return False
    turns = messages[1:] if messages[0]["role"] == "system" else messages
    return len(turns) >= 2 and len(turns) % 2 == 0 and all(
        m["role"] == ("user" if i % 2 == 0 else "assistant") for i, m in enumerate(turns)
    )


def select_dialogues(count, excluded=(), seed=SEED):
    manifest = json.loads((RAW / ".snapshot-manifest.json").read_text(encoding="utf-8"))
    shards = sorted(str(RAW / f["path"]) for f in manifest["files"] if f["path"].endswith(".parquet"))
    dataset = load_dataset(
        "parquet", data_files=shards, split="train", cache_dir=str(ROOT / ".cache/huggingface/datasets")
    )
    questions = benchmark_prompts()
    selected = []
    seen_ids = {row["id"] for row in excluded}
    seen_dialogues = {
        tuple(normalize(m["content"]) for m in row["messages"] if m["role"] == "user") for row in excluded
    }
    skipped = Counter()
    for row in dataset.shuffle(seed=seed):
        messages = row["messages"]
        if not valid_dialogue(messages):
            skipped["invalid_dialogue"] += 1
            continue
        prompts = tuple(normalize(m["content"]) for m in messages if m["role"] == "user")
        if any(text in questions or text.removeprefix("question: ") in questions for text in prompts):
            skipped["exact_benchmark_prompt_match"] += 1
            continue
        if row["id"] in seen_ids or prompts in seen_dialogues:
            skipped["duplicate_id_or_dialogue_prompt"] += 1
            continue

        selected.append(row)
        seen_ids.add(row["id"])
        seen_dialogues.add(prompts)
        if len(selected) == count:
            break

    if len(selected) != count:
        raise RuntimeError(f"Expected {count} valid samples, found {len(selected)}.")
    return selected, skipped, manifest


def prepare_samples():
    selected, skipped, manifest = select_dialogues(TRAIN_SIZE + VALIDATION_SIZE)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": manifest["repo_id"],
        "dataset_revision": manifest["revision"],
        "seed": SEED,
        "sampling": "Shuffle all six shards together, then select valid unique dialogues in shuffled order.",
        "benchmark_filter": "Normalized exact user-message match against local MMLU, IFEval and GSM8K test prompts.",
        "skipped_during_sampling": dict(skipped),
        "splits": {},
    }
    with TemporaryDirectory(prefix=".classroom-", dir=OUTPUT) as staging:
        staging = Path(staging)
        for name, rows in (
            ("tulu3_train_3k.jsonl", selected[:TRAIN_SIZE]),
            ("tulu3_val.jsonl", selected[TRAIN_SIZE:]),
        ):
            path = staging / name
            with path.open("w", encoding="utf-8") as file:
                for row in rows:
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")
            with path.open("rb") as file:
                sha256 = file_digest(file, "sha256").hexdigest()
            report["splits"][name] = {
                "rows": len(rows),
                "sha256": sha256,
                "source_counts": dict(sorted(Counter(row["source"] for row in rows).items())),
            }

        report_name = "tulu3_classroom_manifest.json"
        (staging / report_name).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        for name in (*report["splits"], report_name):
            (staging / name).replace(OUTPUT / name)


if __name__ == "__main__":
    prepare_samples()
    print(f"Wrote {TRAIN_SIZE} training and {VALIDATION_SIZE} validation samples to {OUTPUT}")
