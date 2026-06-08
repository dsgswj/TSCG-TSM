"""Validation Monitor Hook for tracking validation metrics.

This hook monitors validation accuracy and loss during training,
saves them to local JSON file for visualization.
"""

import json
import os
from datetime import datetime
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class ValidationMonitorHook(Hook):
    """Monitor validation metrics (accuracy and loss).

    Saves statistics to validation_history.json in the work directory.

    Args:
        save_to_file (bool): Whether to save statistics to JSON file. Default: True.
    """

    def __init__(self, save_to_file=True):
        self.save_to_file = save_to_file
        self.history = []
        self.save_path = None

    def before_train(self, runner):
        """Initialize save path before training."""
        if self.save_to_file:
            work_dir = runner.work_dir if hasattr(runner, 'work_dir') else './work_dirs'
            self.save_path = os.path.join(work_dir, 'validation_history.json')
            runner.logger.info(f"Validation metrics will be saved to: {self.save_path}")

    def after_val_epoch(self, runner, metrics=None):
        """Log validation metrics after each validation epoch."""
        if metrics is None:
            metrics = {}

        # Extract common metrics
        val_data = {
            'epoch': runner.epoch,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        # Try to extract accuracy and loss from metrics
        # MMAction2 uses different metric names, try common ones
        for key in ['accuracy/top1', 'acc', 'top1_acc', 'acc_metric']:
            if key in metrics:
                val_data['val_accuracy'] = float(metrics[key])
                break

        for key in ['loss', 'val_loss', 'validation_loss']:
            if key in metrics:
                val_data['val_loss'] = float(metrics[key])
                break

        # If not found in standard keys, try to find any numeric value
        if 'val_accuracy' not in val_data and 'val_loss' not in val_data:
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and not k.startswith('_'):
                    if 'acc' in k.lower():
                        val_data['val_accuracy'] = float(v)
                    elif 'loss' in k.lower():
                        val_data['val_loss'] = float(v)

        # Add to history
        if val_data:
            self.history.append(val_data)

            # Save to file
            if self.save_to_file and self.save_path:
                self._save_to_file()

            # Print summary
            msg = f"Epoch {runner.epoch}: "
            if 'val_accuracy' in val_data:
                msg += f"Val Acc = {val_data['val_accuracy']:.4f}"
            if 'val_loss' in val_data:
                if 'val_accuracy' in val_data:
                    msg += ", "
                msg += f"Val Loss = {val_data['val_loss']:.4f}"
            runner.logger.info(msg)

    def _save_to_file(self):
        """Save history to JSON file."""
        try:
            with open(self.save_path, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"Failed to save validation history: {e}")

    def after_train(self, runner):
        """Final save after training completes."""
        if self.save_to_file and self.save_path:
            self._save_to_file()
            runner.logger.info(f"Validation metrics saved to: {self.save_path}")


def build_validation_monitor_hook(save_to_file=True):
    """Factory function to build ValidationMonitorHook.

    Args:
        save_to_file (bool): Whether to save statistics to JSON file.

    Returns:
        ValidationMonitorHook instance.
    """
    return ValidationMonitorHook(save_to_file=save_to_file)
