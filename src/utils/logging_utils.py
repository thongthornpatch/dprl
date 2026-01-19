"""Logging utilities."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    verbose: bool = True
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.

    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level (default: INFO)
        verbose: Whether to print to console

    Returns:
        Configured logger

    Example:
        >>> logger = setup_logger('my_experiment', 'logs/experiment.log')
        >>> logger.info('Starting experiment...')
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers = []

    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler (if verbose)
    if verbose:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler (if log_file specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class MetricsTracker:
    """
    Metrics tracker for storing step-by-step training metrics.

    Stores complete history and can save to JSON.
    """

    def __init__(self):
        self.history = {}

    def log(self, step: int, metrics: dict):
        """Log all metrics for a given step."""
        self.history[step] = metrics

    def save(self, filepath: str):
        """Save metrics history to JSON file."""
        import json
        from pathlib import Path

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)

    def load(self, filepath: str):
        """Load metrics history from JSON file."""
        import json
        with open(filepath) as f:
            self.history = json.load(f)


class MetricsLogger:
    """
    Simple metrics logger for tracking training progress.

    Example:
        >>> metrics = MetricsLogger()
        >>> metrics.log('train_loss', 0.5, step=100)
        >>> metrics.log('eval_reward', 0.8, step=100)
        >>> metrics.get_summary()
    """

    def __init__(self):
        self.metrics = {}

    def log(self, metric_name: str, value: float, step: int):
        """Log a metric value at a given step."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append((step, value))

    def get_latest(self, metric_name: str) -> Optional[float]:
        """Get the latest value for a metric."""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None
        return self.metrics[metric_name][-1][1]

    def get_summary(self) -> dict:
        """Get summary statistics for all metrics."""
        summary = {}
        for metric_name, values in self.metrics.items():
            if values:
                vals = [v for _, v in values]
                summary[metric_name] = {
                    'latest': vals[-1],
                    'mean': sum(vals) / len(vals),
                    'min': min(vals),
                    'max': max(vals),
                    'count': len(vals),
                }
        return summary

    def print_summary(self):
        """Print summary statistics."""
        summary = self.get_summary()
        print("\n" + "=" * 80)
        print("METRICS SUMMARY")
        print("=" * 80)
        for metric_name, stats in summary.items():
            print(f"\n{metric_name}:")
            for stat_name, value in stats.items():
                if stat_name != 'count':
                    print(f"  {stat_name}: {value:.4f}")
                else:
                    print(f"  {stat_name}: {value}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    # Test the logger
    logger = setup_logger('test_logger', 'logs/test.log')
    logger.info('Test info message')
    logger.warning('Test warning message')
    logger.error('Test error message')

    # Test metrics logger
    metrics = MetricsLogger()
    for i in range(10):
        metrics.log('loss', 1.0 / (i + 1), step=i * 10)
        metrics.log('reward', 0.1 * i, step=i * 10)

    metrics.print_summary()
