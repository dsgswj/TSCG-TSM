# Copyright (c) OpenMMLab. All rights reserved.
"""Hook for logging best validation results to local file."""

import os
import json
from datetime import datetime
from mmengine.hooks import Hook
from mmengine.logging import print_log

# Logging level constants
INFO = 20
WARNING = 30
ERROR = 40
DEBUG = 10


class BestResultLoggerHook(Hook):
    """Log best validation results to local file.
    
    This hook monitors validation metrics during training and saves the best
    epoch's validation accuracy and loss to a local file when training completes.
    
    Args:
        log_file (str): Path to save the results file. If None, will use
            work_dir/best_results.json. Default: None.
        metric_key (str): The key of the metric to monitor. Default: 'acc_top1'.
        loss_key (str): The key of the loss to monitor. Default: 'loss'.
        save_mode (str): When to save results. Options:
            - 'end': Save at the end of training (default)
            - 'best': Save whenever a new best is found
            - 'both': Save at both end and when new best is found
        overwrite (bool): Whether to overwrite existing file. Default: True.
    """
    
    def __init__(self,
                 log_file=None,
                 metric_key='acc_top1',
                 loss_key='loss',
                 save_mode='end',
                 overwrite=True):
        super().__init__()
        self.log_file = log_file
        self.metric_key = metric_key
        self.loss_key = loss_key
        self.save_mode = save_mode
        self.overwrite = overwrite
        
        # Initialize tracking variables
        self.best_metric = -float('inf')
        self.best_epoch = 0
        self.best_metric_loss = None
        self.best_results = {}
        
    def before_train(self, runner) -> None:
        """Initialize log file path before training."""
        if self.log_file is None:
            self.log_file = os.path.join(
                runner.work_dir,
                'best_results.json'
            )
        print_log(f"BestResultLoggerHook: Will save results to {self.log_file}",
                 logger='current', level=INFO)
    
    def after_val_epoch(self, runner, metrics) -> None:
        """Check if current epoch is the best and save if needed."""
        # Get current metric value
        if self.metric_key in metrics:
            current_metric = metrics[self.metric_key]
        else:
            # Try to find accuracy metric
            for key in metrics.keys():
                if 'acc' in key.lower() or 'accuracy' in key.lower():
                    self.metric_key = key
                    current_metric = metrics[key]
                    break
            else:
                print_log(f"BestResultLoggerHook: Metric '{self.metric_key}' not found in metrics",
                         logger='current', level=WARNING)
                return
        
        # Get current loss value
        if self.loss_key in metrics:
            current_loss = metrics[self.loss_key]
        else:
            # Try to find loss metric
            for key in metrics.keys():
                if 'loss' in key.lower():
                    current_loss = metrics[key]
                    break
            else:
                current_loss = None

        current_epoch = runner.epoch

        # Check if this is the best metric so far
        is_best = current_metric > self.best_metric

        if is_best:
            self.best_metric = current_metric
            self.best_epoch = current_epoch
            self.best_metric_loss = current_loss

            # Store best results
            self.best_results = {
                'best_epoch': int(current_epoch),
                f'best_{self.metric_key}': float(current_metric),
                f'{self.loss_key}_at_best': float(current_loss) if current_loss is not None else None,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'work_dir': runner.work_dir,
                'config': runner.cfg.pretty_text if hasattr(runner, 'cfg') else None
            }

            print_log(
                f"BestResultLoggerHook: New best {self.metric_key}: {current_metric:.4f} "
                f"at epoch {current_epoch}, loss: {current_loss if current_loss is not None else 'N/A'}",
                logger='current', level=INFO
            )

            # Save if mode is 'best' or 'both'
            if self.save_mode in ['best', 'both']:
                self._save_results(runner)
    
    def after_train(self, runner) -> None:
        """Save final results after training completes."""
        # Save final best results if mode is 'end' or 'both'
        if self.save_mode in ['end', 'both']:
            self._save_results(runner, final=True)

        # Print summary
        print_log(
            f"\n{'='*80}\n"
            f"BestResultLoggerHook: Training Summary\n"
            f"{'='*80}\n"
            f"Best {self.metric_key}: {self.best_metric:.4f}\n"
            f"Best Epoch: {self.best_epoch}\n"
            f"Loss at Best: {self.best_metric_loss if self.best_metric_loss is not None else 'N/A'}\n"
            f"Results saved to: {self.log_file}\n"
            f"{'='*80}\n",
            logger='current', level=INFO
        )
    
    def _save_results(self, runner, final=False) -> None:
        """Save results to file."""
        # Prepare data to save
        save_data = {
            'best_epoch': int(self.best_epoch),
            f'best_{self.metric_key}': float(self.best_metric),
            f'{self.loss_key}_at_best': float(self.best_metric_loss) if self.best_metric_loss is not None else None,
            'is_final': final,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'work_dir': runner.work_dir,
        }
        
        # Add additional metrics if available
        if hasattr(runner, 'message_hub'):
            try:
                # Try to get all validation metrics
                metrics = runner.message_hub.get_log_scalars()
                if metrics:
                    save_data['all_metrics'] = {}
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            save_data['all_metrics'][key] = float(value)
            except Exception as e:
                print_log(f"BestResultLoggerHook: Could not get additional metrics: {e}",
                         logger='current', level=DEBUG)
        
        # Determine file mode
        file_mode = 'w' if self.overwrite else 'a'
        
        # Save as JSON
        try:
            with open(self.log_file, file_mode) as f:
                if file_mode == 'a' and not final:
                    # Append mode: save as a list entry
                    import json
                    try:
                        # Read existing data
                        f.seek(0)
                        existing = json.load(f)
                        if isinstance(existing, list):
                            existing.append(save_data)
                            json.dump(existing, f, indent=2)
                        else:
                            # Convert to list
                            json.dump([existing, save_data], f, indent=2)
                    except (json.JSONDecodeError, ValueError):
                        # File is empty or invalid, create new list
                        json.dump([save_data], f, indent=2)
                else:
                    # Write or overwrite mode
                    import json
                    json.dump(save_data, f, indent=2)
            
            print_log(
                f"BestResultLoggerHook: Results saved to {self.log_file}",
                logger='current',
                level=INFO
            )
            
            # Also save as human-readable text
            txt_file = self.log_file.replace('.json', '.txt')
            with open(txt_file, 'w' if self.overwrite else 'a') as f:
                f.write("="*80 + "\n")
                f.write("BEST VALIDATION RESULTS\n")
                f.write("="*80 + "\n")
                f.write(f"Best Epoch: {save_data['best_epoch']}\n")
                f.write(f"Best {self.metric_key}: {save_data[f'best_{self.metric_key}']:.4f}\n")
                f.write(f"Loss at Best: {save_data[f'{self.loss_key}_at_best']}\n")
                f.write(f"Timestamp: {save_data['timestamp']}\n")
                f.write(f"Work Dir: {save_data['work_dir']}\n")
                if final:
                    f.write("\n[FINAL RESULTS - Training Completed]\n")
                f.write("="*80 + "\n\n")
            
            print_log(
                f"BestResultLoggerHook: Text results saved to {txt_file}",
                logger='current',
                level=INFO
            )
            
        except Exception as e:
            print_log(
                f"BestResultLoggerHook: Failed to save results: {e}",
                logger='current',
                level=ERROR
            )
