import json
from collections import Counter
from hashlib import file_digest
from pathlib import Path
from tempfile import TemporaryDirectory

from prepare_classroom import OUTPUT, select_dialogues


def main():
    excluded_files = ('tulu3_train_3k.jsonl', 'tulu3_train_10k.jsonl', 'tulu3_train_30k.jsonl', 'tulu3_val.jsonl')
    excluded = []
    for name in excluded_files:
        with (OUTPUT / name).open(encoding='utf-8') as file:
            excluded.extend(json.loads(line) for line in file)

    seed = 43
    rows, skipped, upstream = select_dialogues(1000, excluded=excluded, seed=seed)
    with TemporaryDirectory(prefix='.advanced-val-', dir=OUTPUT) as temporary:
        staging = Path(temporary)
        data = staging / 'tulu3_advanced_val_1k.jsonl'
        with data.open('w', encoding='utf-8') as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + '\n')
        with data.open('rb') as file:
            sha256 = file_digest(file, 'sha256').hexdigest()

        report = staging / 'tulu3_advanced_validation_manifest.json'
        report.write_text(json.dumps({
            'dataset': upstream['repo_id'], 'dataset_revision': upstream['revision'], 'seed': seed,
            'file': data.name, 'rows': len(rows), 'sha256': sha256,
            'excluded_files': excluded_files, 'skipped_during_sampling': dict(skipped),
            'benchmark_filter': 'Same normalized exact benchmark prompt check as the classroom split.',
            'source_counts': dict(sorted(Counter(row['source'] for row in rows).items())),
        }, indent=2) + '\n', encoding='utf-8')
        for path in (data, report):
            path.replace(OUTPUT / path.name)
    print(f'Prepared {len(rows):,} shared validation dialogues.')


if __name__ == '__main__':
    main()
