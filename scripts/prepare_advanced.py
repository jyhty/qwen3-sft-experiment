import json
import random
from bisect import bisect_right
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from hashlib import file_digest
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.hparams.parser import _parse_train_args
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from prepare_classroom import OUTPUT, ROOT, SEED, select_dialogues


def write_rows(path, rows):
    with path.open('w', encoding='utf-8') as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + '\n')


def token_lengths(directory, *, expected_rows=30_000, workers=8):
    config = OmegaConf.to_container(OmegaConf.load(ROOT / 'configs/qwen3_4b_base_sft_3k_bf16_lora.yaml'))
    config.update(
        bf16=False, use_cpu=True, do_train=False, do_eval=False, eval_strategy='no', eval_dataset=None,
        dataset='profile', dataset_dir=str(directory), output_dir=str(directory / 'unused'),
        model_name_or_path=str(ROOT / config['model_name_or_path']), cache_dir=str(directory / 'cache'),
        preprocessing_num_workers=workers,
    )
    model_args, data_args, training_args, _, _ = _parse_train_args(config)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, local_files_only=True)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    with redirect_stdout(StringIO()):
        data = get_dataset(template, model_args, data_args, training_args, stage='sft', tokenizer=tokenizer)
    lengths = [(len(row['input_ids']), sum(label != -100 for label in row['labels'])) for row in data['train_dataset']]
    if len(lengths) != expected_rows or any(answer == 0 for _, answer in lengths):
        raise RuntimeError(f'Expected {expected_rows:,} valid profiles with supervised answer tokens.')
    return lengths


def main():
    validation = OUTPUT / 'tulu3_val.jsonl'
    with validation.open(encoding='utf-8') as file:
        excluded = [json.loads(line) for line in file]
    rows, skipped, upstream = select_dialogues(30_000, excluded)
    print('Selected 30,000 dialogues. Measuring lengths with the training template.', flush=True)

    with TemporaryDirectory(prefix='.advanced-', dir=OUTPUT) as temporary:
        staging = Path(temporary)
        full = staging / 'tulu3_train_30k.jsonl'
        write_rows(full, rows)
        registry = json.loads((OUTPUT / 'dataset_info.json').read_text(encoding='utf-8'))
        (staging / 'dataset_info.json').write_text(json.dumps({
            'profile': {**registry['tulu3_3k'], 'file_name': full.name}
        }), encoding='utf-8')
        lengths = token_lengths(staging)

        # Match source, input length, and supervised answer length proportions.
        strata = defaultdict(list)
        for index, (row, (total, answer)) in enumerate(zip(rows, lengths, strict=True)):
            key = (row['source'], bisect_right((256, 512, 1024, 2048), total), bisect_right((128, 512), answer))
            strata[key].append(index)
        quotas = {key: len(indexes) // 3 for key, indexes in strata.items()}
        remainders = sorted(strata, key=lambda key: (-(len(strata[key]) % 3), key))
        for key in remainders[:10_000 - sum(quotas.values())]:
            quotas[key] += 1
        rng = random.Random(SEED)
        subset = sorted(i for key in sorted(strata) for i in rng.sample(strata[key], quotas[key]))
        assert len(subset) == 10_000 and len(set(subset)) == 10_000
        small = staging / 'tulu3_train_10k.jsonl'
        write_rows(small, (rows[i] for i in subset))

        with validation.open('rb') as file:
            validation_sha256 = file_digest(file, 'sha256').hexdigest()
        manifest = {
            'dataset': upstream['repo_id'], 'dataset_revision': upstream['revision'], 'seed': SEED,
            'classroom_validation_excluded': {'file': validation.name, 'rows': len(excluded), 'sha256': validation_sha256},
            'benchmark_filter': 'Same normalized exact benchmark prompt check as the classroom split.',
            'skipped_during_30k_sampling': dict(skipped),
            'subset_strata': {'columns': ['source', 'input_tokens', 'supervised_tokens'],
                              'input_boundaries': [256, 512, 1024, 2048], 'supervised_boundaries': [128, 512]},
            'splits': {},
        }
        for path, indices, epochs in ((small, subset, 3), (full, range(len(rows)), 1)):
            with path.open('rb') as file:
                sha256 = file_digest(file, 'sha256').hexdigest()
            manifest['splits'][path.name] = {
                'rows': len(indices), 'sha256': sha256, 'epochs': epochs,
                'source_counts': dict(sorted(Counter(rows[i]['source'] for i in indices).items())),
                'length_cutoff_rows': sum(lengths[i][0] == 2048 for i in indices),
                'input_tokens': sum(lengths[i][0] for i in indices),
                'supervised_tokens': sum(lengths[i][1] for i in indices),
            }
        report = staging / 'tulu3_advanced_manifest.json'
        report.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        for path in (small, full, report):
            path.replace(OUTPUT / path.name)
        for name, stats in manifest['splits'].items():
            print(name, {key: stats[key] for key in ('rows', 'input_tokens', 'supervised_tokens')})


if __name__ == '__main__':
    main()
