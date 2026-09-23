"""
Example (run from the TSCG_TSM project root)::

    python tools/cross_validate.py configs/TSCG-TSM.py \
        --train-ann tools/data/trainlist.txt \
        --test-ann tools/data/testlist.txt \
        --work-dir work_dirs/TSCG-TSM-5fold
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run stratified five-fold CV and one independent test.')
    parser.add_argument('config', help='base MMEngine config file')
    parser.add_argument('--train-ann', default='tools/data/trainlist.txt',
                        help='development/training annotation')
    parser.add_argument('--test-ann', default='tools/data/testlist.txt',
                        help='independent test annotation')
    parser.add_argument('--work-dir', default='work_dirs/TSCG-TSM-5fold',
                        help='directory for splits, logs and checkpoints')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--skip-final-train', action='store_true',
                        help='only run CV; do not train/test the final model')
    parser.add_argument('--resume', action='store_true',
                        help='reuse completed fold/final runs')
    return parser.parse_args()


def read_annotation(path: Path) -> Tuple[List[str], np.ndarray]:
    lines = [line.strip() for line in path.read_text().splitlines()
             if line.strip()]
    if not lines:
        raise ValueError(f'Annotation file is empty: {path}')
    labels = []
    for line in lines:
        fields = line.split()
        if len(fields) < 2:
            raise ValueError(f'Expected "video label", got: {line!r}')
        try:
            labels.append(int(fields[-1]))
        except ValueError as exc:
            raise ValueError(f'Invalid label in: {line!r}') from exc
    return lines, np.asarray(labels, dtype=np.int64)


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n')


def make_config(base_config: Path, output: Path, train_ann: Path,
                val_ann: Path, test_ann: Path, work_dir: Path,
                max_epochs: Optional[int] = None,
                disable_validation: bool = False) -> None:
    """Write a child config with explicit paths for all three loaders."""
    base = str(base_config.resolve()).replace('\\', '/')
    train = str(train_ann.resolve()).replace('\\', '/')
    val = str(val_ann.resolve()).replace('\\', '/')
    test = str(test_ann.resolve()).replace('\\', '/')
    work = str(work_dir.resolve()).replace('\\', '/')
    lines = [
        f'_base_ = {base!r}',
        f'ann_file_train = {train!r}',
        f'ann_file_val = {val!r}',
        f'ann_file_test = {test!r}',
        f'work_dir = {work!r}',
        'train_dataloader = dict(dataset=dict(ann_file=ann_file_train))',
        'val_dataloader = dict(dataset=dict(ann_file=ann_file_val))',
        'test_dataloader = dict(dataset=dict(ann_file=ann_file_test))',
    ]
    if max_epochs is not None:
        lines.append(f'train_cfg = dict(max_epochs={int(max_epochs)})')
    if disable_validation:
        lines.extend([
            'val_cfg = None',
            'val_dataloader = None',
            'val_evaluator = None',
            # Save the selected final epoch even when it is not a multiple of
            # the base checkpoint interval.
            'default_hooks = dict(checkpoint=dict(interval=1, max_keep_ckpts=1, save_best=None))',
        ])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(lines) + '\n')


def run(command: Sequence[str], cwd: Path) -> None:
    print('+', ' '.join(map(str, command)), flush=True)
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(
        [str(cwd), env.get('PYTHONPATH', '')]).rstrip(os.pathsep)
    subprocess.run(list(map(str, command)), cwd=str(cwd), env=env, check=True)


def find_checkpoint(work_dir: Path) -> Path:
    candidates = list(work_dir.glob('best*.pth'))
    if not candidates:
        candidates = list(work_dir.glob('epoch_*.pth'))
    if not candidates:
        candidates = list(work_dir.rglob('*.pth'))
    if not candidates:
        raise FileNotFoundError(f'No checkpoint found under {work_dir}')

    def checkpoint_key(path: Path) -> Tuple[int, float]:
        match = re.search(r'epoch[_-](\d+)', path.name)
        return (int(match.group(1)) if match else -1, path.stat().st_mtime)

    return sorted(candidates, key=checkpoint_key)[-1]


def read_history(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f'Validation history was not written: {path}')
    rows = json.loads(path.read_text())
    rows = [row for row in rows if 'epoch' in row and 'val_accuracy' in row]
    if not rows:
        raise ValueError(f'No validation accuracy was found in {path}')
    return rows


def prediction_value(sample: Any, name: str) -> Any:
    if isinstance(sample, dict):
        return sample.get(name)
    try:
        return getattr(sample, name)
    except AttributeError:
        return None


def evaluate_dump(path: Path) -> Dict[str, float]:
    """Compute top-1 and mean class accuracy from MMEngine DumpResults."""
    with path.open('rb') as file:
        samples = pickle.load(file)
    labels: List[int] = []
    predictions: List[int] = []
    for sample in samples:
        scores = prediction_value(sample, 'pred_score')
        label = prediction_value(sample, 'gt_label')
        if scores is None or label is None:
            if isinstance(sample, dict) and isinstance(sample.get('_data'), dict):
                scores = sample['_data'].get('pred_score', scores)
                label = sample['_data'].get('gt_label', label)
        if scores is None or label is None:
            raise ValueError(f'Unsupported dumped prediction format in {path}')
        if hasattr(scores, 'detach'):
            scores = scores.detach().cpu().numpy()
        if hasattr(label, 'detach'):
            label = label.detach().cpu().numpy()
        labels.append(int(np.asarray(label).reshape(-1)[0]))
        predictions.append(int(np.asarray(scores).reshape(-1).argmax()))
    if not labels:
        raise ValueError(f'No predictions found in {path}')
    labels_array = np.asarray(labels)
    pred_array = np.asarray(predictions)
    per_class = []
    for class_id in np.unique(labels_array):
        mask = labels_array == class_id
        per_class.append(float(np.mean(pred_array[mask] == labels_array[mask])))
    return {
        'top1': float(np.mean(pred_array == labels_array)),
        'mean_class_accuracy': float(np.mean(per_class)),
        'num_samples': float(len(labels)),
    }


def aggregate_curves(histories: Sequence[Sequence[Dict[str, Any]]]) -> Tuple[int, float]:
    by_epoch: Dict[int, List[float]] = {}
    for history in histories:
        for row in history:
            by_epoch.setdefault(int(row['epoch']), []).append(
                float(row['val_accuracy']))
    means = {epoch: mean(values) for epoch, values in by_epoch.items()
             if len(values) == len(histories)}
    if not means:
        raise ValueError('The folds do not share any validation epochs')
    return max(means.items(), key=lambda item: item[1])


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError('--folds must be at least 2')
    project_root = Path(__file__).resolve().parents[1]
    base_config = Path(args.config)
    train_ann = Path(args.train_ann)
    test_ann = Path(args.test_ann)
    work_dir = Path(args.work_dir)
    if not base_config.is_absolute():
        base_config = (project_root / base_config).resolve()
    if not train_ann.is_absolute():
        train_ann = (project_root / train_ann).resolve()
    if not test_ann.is_absolute():
        test_ann = (project_root / test_ann).resolve()
    if not work_dir.is_absolute():
        work_dir = (project_root / work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    lines, labels = read_annotation(train_ann)
    test_lines, _ = read_annotation(test_ann)
    overlap = set(lines).intersection(test_lines)
    if overlap:
        raise ValueError(
            f'Training and independent test annotations overlap: {len(overlap)} samples')
    _, counts = np.unique(labels, return_counts=True)
    if counts.min() < args.folds:
        raise ValueError('Every class must have at least --folds samples')

    splitter = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    fold_summaries: List[Dict[str, Any]] = []
    histories: List[List[Dict[str, Any]]] = []
    for fold_id, (train_idx, val_idx) in enumerate(
            splitter.split(np.zeros(len(lines)), labels), start=1):
        fold_dir = work_dir / f'fold_{fold_id}'
        fold_train = fold_dir / 'splits' / 'train.txt'
        fold_val = fold_dir / 'splits' / 'val.txt'
        write_lines(fold_train, (lines[index] for index in train_idx))
        write_lines(fold_val, (lines[index] for index in val_idx))
        fold_config = fold_dir / 'config.py'
        make_config(base_config, fold_config, fold_train, fold_val, test_ann,
                    fold_dir)
        history_path = fold_dir / 'validation_history.json'
        checkpoint = None
        if args.resume:
            try:
                checkpoint = find_checkpoint(fold_dir)
                read_history(history_path)
            except (FileNotFoundError, ValueError):
                checkpoint = None
        if checkpoint is None:
            command = [sys.executable, 'tools/train.py', str(fold_config),
                       '--work-dir', str(fold_dir), '--seed',
                       str(args.seed + fold_id - 1)]
            if args.deterministic:
                command.append('--deterministic')
            run(command, project_root)
            checkpoint = find_checkpoint(fold_dir)
        history = read_history(history_path)
        histories.append(history)
        best_row = max(history, key=lambda row: float(row['val_accuracy']))
        fold_summaries.append({
            'fold': fold_id,
            'num_train': len(train_idx),
            'num_val': len(val_idx),
            'best_epoch': int(best_row['epoch']),
            'best_val_accuracy': float(best_row['val_accuracy']),
            'checkpoint': str(checkpoint),
        })

    selected_epoch, mean_curve_accuracy = aggregate_curves(histories)
    cv_values = [row['best_val_accuracy'] for row in fold_summaries]
    results: Dict[str, Any] = {
        'protocol': {
            'folds': args.folds,
            'seed': args.seed,
            'train_annotation': str(train_ann),
            'independent_test_annotation': str(test_ann),
            'test_used_for_fold_selection': False,
        },
        'folds': fold_summaries,
        'cv_mean_best_val_accuracy': float(mean(cv_values)),
        'cv_std_best_val_accuracy': float(pstdev(cv_values)),
        'cv_best_epoch_from_mean_curve': int(selected_epoch),
        'cv_mean_accuracy_at_selected_epoch': float(mean_curve_accuracy),
    }

    if not args.skip_final_train:
        final_dir = work_dir / 'final_model'
        final_config = final_dir / 'config.py'
        make_config(base_config, final_config, train_ann, train_ann, test_ann,
                    final_dir, max_epochs=selected_epoch,
                    disable_validation=True)
        final_checkpoint = None
        if args.resume:
            try:
                final_checkpoint = find_checkpoint(final_dir)
            except FileNotFoundError:
                pass
        if final_checkpoint is None:
            run([sys.executable, 'tools/train.py', str(final_config),
                 '--work-dir', str(final_dir), '--no-validate', '--seed',
                 str(args.seed)], project_root)
            final_checkpoint = find_checkpoint(final_dir)

        test_dump = final_dir / 'independent_test_predictions.pkl'
        test_config = final_dir / 'test_config.py'
        make_config(base_config, test_config, train_ann, train_ann, test_ann,
                    final_dir)
        run([sys.executable, 'tools/test.py', str(test_config),
             str(final_checkpoint), '--work-dir', str(final_dir), '--dump',
             str(test_dump)], project_root)
        results['final_model'] = {
            'checkpoint': str(final_checkpoint),
            'selected_epochs': int(selected_epoch),
            'independent_test': evaluate_dump(test_dump),
        }

    result_path = work_dir / 'cv_results.json'
    result_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f'Cross-validation summary written to {result_path}')


if __name__ == '__main__':
    main()
