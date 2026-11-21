import argparse
import fcntl
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory

from omegaconf import OmegaConf
import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'models/Qwen3-4B-Base'
OUTPUTS = ROOT / 'outputs'
DEFAULT_GROUPS = ('3k', '10k', '30k')
GROUPS = {'3k': 1125, '10k': 3750, '30k': 3750, 'deita': 3750}
TASK_BATCH_SIZES = {'mmlu': 4, 'gsm8k': 32, 'ifeval': 32}
TASK_SAMPLE_COUNTS = {'mmlu': 14042, 'gsm8k': 1319, 'ifeval': 541}


def group_config(group):
    path = ROOT / f'configs/qwen3_4b_base_sft_{group}_bf16_lora.yaml'
    config = OmegaConf.load(path)
    if (config.per_device_train_batch_size, config.gradient_accumulation_steps) != (8, 1):
        raise ValueError(f'{path.name} must use training batch 8 × 1.')
    return path, ROOT / config.output_dir


def complete(directory, steps):
    required = ('adapter_model.safetensors', 'adapter_config.json', 'trainer_state.json', 'eval_results.json')
    if not all((directory / name).is_file() and (directory / name).stat().st_size for name in required):
        return False
    try:
        state = json.loads((directory / 'trainer_state.json').read_text(encoding='utf-8'))
        evaluation = json.loads((directory / 'eval_results.json').read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return False
    loss = evaluation.get('eval_loss')
    return state.get('global_step') == steps and isinstance(loss, (int, float)) and math.isfinite(loss)


def train(groups=DEFAULT_GROUPS):
    for group in groups:
        steps = GROUPS[group]
        config, directory = group_config(group)
        if complete(directory, steps):
            print(f'{group}: final training artifacts already complete.', flush=True)
            continue
        print(f'{group}: starting {config.name}', flush=True)
        subprocess.run(
            [sys.executable, '-m', 'llamafactory.cli', 'train', str(config)],
            cwd=ROOT, check=True,
        )
        if not complete(directory, steps):
            raise RuntimeError(f'{group}: expected step {steps}, final adapter, and validation results in {directory}.')
        print(f'{group}: complete at step {steps}.', flush=True)


def write_local_tasks(directory):
    package = Path(find_spec('lm_eval').origin).parent / 'tasks'
    data_root = ROOT / 'data/benchmarks'

    def write(task, original, builder, files, **fields):
        for path in files.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        config = {
            'task': task, 'include': str(original), 'dataset_path': builder, 'dataset_name': None,
            'dataset_kwargs': {'data_files': {name: str(path) for name, path in files.items()}}, **fields,
        }
        (directory / f'{task}.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')

    mmlu = package / 'mmlu/default'
    for group in sorted(mmlu.glob('_mmlu*.yaml')):
        (directory / group.name).write_bytes(group.read_bytes())
    for original in sorted(mmlu.glob('mmlu_*.yaml')):
        definition = yaml.safe_load(original.read_text(encoding='utf-8'))
        source = data_root / 'cais/mmlu' / definition['dataset_name']
        files = {split: source / f'{split}-00000-of-00001.parquet' for split in ('dev', 'test', 'validation')}
        write(definition['task'], original, 'parquet', files, tag=definition['tag'], num_fewshot=5)

    ifeval = package / 'ifeval/ifeval.yaml'
    # Keep the official !function scoring hooks and use the same Qwen turn stop for every model.
    from lm_eval.tasks._yaml_loader import load_yaml

    ifeval_generation = load_yaml(ifeval, resolve_func=False)['generation_kwargs']
    write('ifeval', ifeval, 'json', {
        'train': data_root / 'google/IFEval/ifeval_input_data.jsonl',
    }, generation_kwargs={**ifeval_generation, 'until': ['<|im_end|>']})
    gsm = data_root / 'openai/gsm8k/main'
    gsm_config = package / 'gsm8k/gsm8k.yaml'
    write('gsm8k', gsm_config, 'parquet', {
        split: gsm / f'{split}-00000-of-00001.parquet' for split in ('train', 'test')
    })


def check_evaluation(directory):
    from lm_eval.tasks import TaskManager

    tasks = TaskManager(include_path=directory, include_defaults=False).load(list(TASK_BATCH_SIZES))['tasks']
    mmlu = [task for name, task in tasks.items() if name.startswith('mmlu_')]
    if len(mmlu) != 57 or sum(len(task.test_docs()) for task in mmlu) != 14042:
        raise RuntimeError('MMLU subject or sample counts differ from the local snapshot.')
    if len(tasks['ifeval'].test_docs()) != 541 or len(tasks['gsm8k'].test_docs()) != 1319:
        raise RuntimeError('IFEval or GSM8K sample counts differ from the local snapshots.')
    print('Offline task loading verified: MMLU 14,042 (57 subjects), IFEval 541, GSM8K 1,319.', flush=True)


def score_command(directory, output):
    return [sys.executable, '-u', '-B', str(Path(__file__).resolve()), '_score', str(directory), str(output)]


def model_metadata(adapter, smoke=False):
    return {'base': str(BASE), 'adapter': str(adapter) if adapter else None, 'smoke': smoke}


def read_task_result(output, task, adapter):
    try:
        result = json.loads((output / 'results.json').read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if (
        result.get('local_model') == model_metadata(adapter)
        and result.get('benchmark') == {'task': task, 'num_samples': TASK_SAMPLE_COUNTS[task]}
        and result.get('config', {}).get('task_batch_sizes') == {task: TASK_BATCH_SIZES[task]}
        and task in result.get('results', {})
        and len(list(output.glob('samples_*.jsonl'))) == (57 if task == 'mmlu' else 1)
    ):
        return result
    return None


def evaluate_task(directory, destination, group, task, adapter):
    output = destination / group / task
    existing = read_task_result(output, task, adapter)
    if existing is not None:
        print(f'{group}/{task}: reusing completed result.', flush=True)
        return existing
    output.mkdir(parents=True, exist_ok=True)
    cmd = score_command(directory, output) + ['--task', task]
    if adapter:
        cmd += ['--adapter', str(adapter)]
    log_path = output / 'run.log'
    print(f'{group}/{task}: starting with batch {TASK_BATCH_SIZES[task]}. Log: {log_path}', flush=True)
    try:
        with log_path.open('a', encoding='utf-8') as log:
            subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f'{group}/{task}: evaluator exited with status {error.returncode}. See {log_path}.') from error
    result = read_task_result(output, task, adapter)
    if result is None:
        raise RuntimeError(f'{group}/{task}: no complete evaluation result. See {log_path}.')
    print(f'{group}/{task}: completed.', flush=True)
    return result


def summarize_model(output, task_results, adapter):
    summary = {
        'local_model': model_metadata(adapter),
        'num_samples': sum(TASK_SAMPLE_COUNTS.values()),
        'config': {'task_batch_sizes': TASK_BATCH_SIZES},
        'results': {},
        'groups': {},
        'tasks': {},
    }
    for task, result in task_results.items():
        summary['results'].update(result['results'])
        summary['groups'].update(result.get('groups', {}))
        summary['tasks'][task] = {
            'num_samples': TASK_SAMPLE_COUNTS[task],
            'results_file': f'{task}/results.json',
        }
    with TemporaryDirectory(prefix='.summary-', dir=output) as tmp:
        staged = Path(tmp) / 'results.json'
        staged.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        staged.replace(output / 'results.json')
    print(f'{output.name}: all tasks complete. Summary: {output / "results.json"}', flush=True)


def evaluate(directory, groups, destination, *, serial=False):
    adapters = {}
    for group in groups:
        if group == 'base':
            continue
        _, output = group_config(group)
        if not complete(output, GROUPS[group]):
            raise RuntimeError(f'{group} has no complete final adapter at {output}.')
        adapters[group] = output

    for group in groups:
        adapter = adapters.get(group)
        if serial:
            task_results = {
                task: evaluate_task(directory, destination, group, task, adapter)
                for task in TASK_BATCH_SIZES
            }
            summarize_model(destination / group, task_results, adapter)
            continue
        with ThreadPoolExecutor(max_workers=2) as executor:
            workers = {
                task: executor.submit(evaluate_task, directory, destination, group, task, adapter)
                for task in ('mmlu', 'gsm8k')
            }
            task_results = {task: worker.result() for task, worker in workers.items()}
        task_results['ifeval'] = evaluate_task(directory, destination, group, 'ifeval', adapter)
        summarize_model(destination / group, task_results, adapter)


def score(directory, output, adapter, task, smoke):
    import torch
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager
    from lm_eval.utils import handle_non_serializable
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, local_files_only=True, dtype=torch.bfloat16, device_map={'': 'cuda:0'}, attn_implementation='sdpa',
    )
    vocab_rows = model.get_input_embeddings().num_embeddings
    # HFLM's path-based PEFT loader would resize the padded Qwen embedding matrix only for adapters.
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
    elif smoke:
        with TemporaryDirectory(prefix='sft-smoke-lora-') as temporary:
            probe = get_peft_model(model, LoraConfig(
                r=2, lora_alpha=4, target_modules=['model.layers.0.self_attn.q_proj'], task_type='CAUSAL_LM',
            ))
            probe.save_pretrained(temporary, save_embedding_layers=False)
            model = probe.unload()
            vars(model).pop('peft_config', None)
            model = PeftModel.from_pretrained(model, temporary, local_files_only=True)
    if model.get_input_embeddings().num_embeddings != vocab_rows:
        raise RuntimeError('The Base embedding matrix changed size during adapter loading.')
    # Allow the harness's per-task generation limit to override Qwen's saved default.
    model.generation_config.max_new_tokens = None
    batch_size = 4 if smoke else TASK_BATCH_SIZES[task]
    lm = HFLM(pretrained=model.eval(), tokenizer=tokenizer, batch_size=batch_size, max_length=4096)
    tasks = ['mmlu_abstract_algebra', 'ifeval', 'gsm8k'] if smoke else [task]
    result = simple_evaluate(
        model=lm, tasks=tasks, task_manager=TaskManager(include_path=directory, include_defaults=False),
        apply_chat_template=False, log_samples=True, bootstrap_iters=0 if smoke else 1000,
        random_seed=42, numpy_random_seed=42, torch_random_seed=42, fewshot_random_seed=42,
        limit=1 if smoke else None, gen_kwargs={'max_gen_toks': 16} if smoke else None,
    )
    if not result or not result.get('samples'):
        raise RuntimeError('Official evaluator did not return per-sample results.')
    samples = result.pop('samples')
    total = sum(len({row['doc_id'] for row in rows}) for rows in samples.values())
    if total != (3 if smoke else TASK_SAMPLE_COUNTS[task]):
        raise RuntimeError(f'Unexpected number of benchmark samples: {total}.')
    result['local_model'] = model_metadata(adapter, smoke)
    result['config']['task_batch_sizes'] = dict.fromkeys(tasks, batch_size)
    if not smoke:
        result['benchmark'] = {'task': task, 'num_samples': total}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f'.{output.name}-', dir=output.parent) as tmp:
        staging = Path(tmp)
        for task, rows in samples.items():
            with (staging / f'samples_{task}.jsonl').open('w', encoding='utf-8') as file:
                for row in rows:
                    file.write(json.dumps(row, ensure_ascii=False, default=handle_non_serializable) + '\n')
        (staging / 'results.json').write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=handle_non_serializable) + '\n', encoding='utf-8'
        )
        output.mkdir(exist_ok=True)
        for path in sorted(staging.iterdir(), key=lambda path: path.name == 'results.json'):
            path.replace(output / path.name)
    print(f'Evaluated {total:,} examples. Results saved under {output}.', flush=True)


def configure_environment():
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    os.environ.update(HF_HOME=str(ROOT / '.cache/huggingface'),
                      HF_DATASETS_CACHE=str(ROOT / '.cache/huggingface/datasets'),
                      NLTK_DATA=str(ROOT / '.cache/nltk_data'), HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')


def main():
    parser = argparse.ArgumentParser(description='Run the three SFT groups in order, with optional offline evaluation.')
    commands = parser.add_subparsers(dest='command', required=True)
    train_args = commands.add_parser('train')
    train_args.add_argument('--group', choices=GROUPS, help='Train one group instead of the default three.')
    train_args.add_argument('--evaluate', action='store_true', help='Evaluate the selected training run, or Base and the three defaults.')
    eval_args = commands.add_parser('evaluate')
    eval_args.add_argument('group', nargs='?', choices=['all', 'base', *GROUPS], default='all')
    eval_args.add_argument('--serial', action='store_true', help='Run all evaluation tasks in sequence.')
    commands.add_parser('check', help='Read official tasks using local files, without loading a model.')
    commands.add_parser('smoke-eval', help='Check one example per benchmark with a temporary, initialized adapter.')
    worker = commands.add_parser('_score')
    worker.add_argument('directory', type=Path)
    worker.add_argument('output', type=Path)
    worker.add_argument('--adapter', type=Path)
    worker_mode = worker.add_mutually_exclusive_group(required=True)
    worker_mode.add_argument('--task', choices=TASK_BATCH_SIZES)
    worker_mode.add_argument('--smoke', action='store_true')
    args = parser.parse_args()

    configure_environment()
    if args.command == '_score':
        score(args.directory, args.output, args.adapter, args.task, args.smoke)
        return

    with TemporaryDirectory(prefix='sft-local-eval-') as tmp:
        task_dir = Path(tmp) / 'tasks'
        if args.command != 'train' or args.evaluate:
            task_dir.mkdir()
            write_local_tasks(task_dir)
        if args.command == 'check':
            check_evaluation(task_dir)
            return

        devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        if len(devices) != 1 or not devices[0].strip() or devices[0].strip() == '-1':
            raise ValueError('Select one CUDA device for the single-GPU 8 × 1 pipeline.')
        lock_path = ROOT / '.cache/setup/sft-pipeline.lock'
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError('Another project training or evaluation pipeline is active.') from error
            if args.command == 'smoke-eval':
                subprocess.run(score_command(task_dir, Path(tmp) / 'smoke') + ['--smoke'], cwd=ROOT, check=True)
                return
            if args.command == 'train':
                train([args.group] if args.group else DEFAULT_GROUPS)
                if not args.evaluate:
                    return
            requested = args.group
            groups = ['base', *DEFAULT_GROUPS] if requested in (None, 'all') else [requested]
            evaluate(task_dir, groups, OUTPUTS / 'evaluations', serial=getattr(args, 'serial', False))


if __name__ == '__main__':
    main()
