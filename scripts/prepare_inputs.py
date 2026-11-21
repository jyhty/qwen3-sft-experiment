"""Prepare the pinned Hugging Face inputs and tokenizer resources."""

import argparse
import json
from pathlib import Path

import nltk
from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
NLTK_DIR = ROOT / '.cache/nltk_data'


def missing_files(directory, files):
    missing = []
    for entry in files:
        path = directory / entry['path']
        if not path.is_file() or path.stat().st_size != entry['size']:
            missing.append(entry['path'])
    return missing


def prepare_source(source, verify_only):
    directory = ROOT / source['directory']
    manifest = directory / '.snapshot-manifest.json'
    missing = missing_files(directory, source['files'])
    if missing and not verify_only:
        snapshot_download(
            repo_id=source['repo_id'],
            repo_type=source['repo_type'],
            revision=source['revision'],
            local_dir=directory,
            allow_patterns=missing,
        )
        missing = missing_files(directory, source['files'])
    if missing:
        raise RuntimeError(f"{source['repo_id']}: missing or incorrect size: {', '.join(missing)}")
    if not manifest.is_file() or json.loads(manifest.read_text(encoding='utf-8')) != source:
        if verify_only:
            raise RuntimeError(f'{manifest}: missing or different from configs/inputs.json')
        manifest.write_text(json.dumps(source, indent=2) + '\n', encoding='utf-8')
    print(f"{source['repo_id']} @ {source['revision'][:12]} verified.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true', help='Check local files without downloading.')
    args = parser.parse_args()
    sources = json.loads((ROOT / 'configs/inputs.json').read_text(encoding='utf-8'))
    for source in sources:
        prepare_source(source, args.verify_only)
    if not args.verify_only:
        nltk.download('punkt_tab', download_dir=str(NLTK_DIR), quiet=True, raise_on_error=True)
    nltk.data.find('tokenizers/punkt_tab/english/', paths=[str(NLTK_DIR)])


if __name__ == '__main__':
    main()
