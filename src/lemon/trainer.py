"""
trainer - Training utilities

This library provides training utilities for neural network training.
It includes training loops, progress logging, history tracking, and checkpoint management.

Key Features
------------
- High-level Trainer API for simplified training
- Progress logging with automatic file output
- Training history tracking
- Checkpoint save/load functionality
- Integration with nnlib models, optimizers, and metrics

Note: Metrics have been moved to lemon.nnlib.metrics
"""

import lemon.numlib as nm
import lemon.nnlib as nl
import lemon.nnlib.data as dl
import lemon.checkpoint as ck
import numpy as np
from datetime import datetime
import os
import sys
import json
import csv
import copy

# Import metrics from nnlib for backward compatibility
from lemon.nnlib.metrics import (
    Metric,
    Accuracy,
    BinaryAccuracy,
    TopKAccuracy,
    MAE,
    MSE,
    RMSE,
)

# ==============================
# Progress Logger
# ==============================

# ANSI Color Codes for Modern Style
class ModernColors:
    """Modern-themed ANSI color codes"""
    # Neon colors
    CYAN = '\033[96m'      # Bright cyan (ネオンシアン)
    MAGENTA = '\033[95m'   # Bright magenta (ネオンマゼンタ)
    GREEN = '\033[92m'     # Bright green (ネオングリーン)
    YELLOW = '\033[93m'    # Bright yellow (ネオンイエロー)
    RED = '\033[91m'       # Bright red (ネオンレッド)
    BLUE = '\033[94m'      # Bright blue (ネオンブルー)

    # Special effects
    BOLD = '\033[1m'       # Bold
    DIM = '\033[2m'        # Dim
    UNDERLINE = '\033[4m'  # Underline
    BLINK = '\033[5m'      # Blink (注意: 一部の端末では動作しない)
    REVERSE = '\033[7m'    # Reverse colors

    # Reset
    RESET = '\033[0m'      # Reset all formatting

    @staticmethod
    def is_enabled():
        """Check if ANSI colors should be enabled"""
        # Force enable if FORCE_COLOR is set
        if os.environ.get('FORCE_COLOR'):
            return True
        # Disable colors if NO_COLOR env var is set
        if os.environ.get('NO_COLOR'):
            return False
        # Enable if running in a terminal or if TERM is set
        if os.environ.get('TERM'):
            return True
        # Fallback: check if stdout is a tty
        return hasattr(sys, 'stdout') and hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()


class ProgressLogger:
    """
    Training progress logger with automatic file output and anomaly detection

    Features
    --------
    - Timestamped console output
    - Automatic CSV/JSON logging
    - Metadata recording (architecture, hyperparameters)
    - Anomaly detection (NaN, gradient explosion, overfitting)
    - Verbose level control (0=silent, 1=epoch, 2=detailed)

    Parameters
    ----------
    experiment_dir : str or None, optional
        Directory to save all experiment files. If None, file logging is disabled (default: None)
    verbose : int, optional
        Verbosity level: 0=silent, 1=epoch summary, 2=detailed (default: 1)
    style : str, optional
        Display style: 'default' or 'modern' (default: 'modern')
    """

    def __init__(self, experiment_dir=None, verbose=1, metrics=None, style='modern'):
        self.experiment_dir = experiment_dir
        self.verbose = verbose
        self.logging_enabled = experiment_dir is not None
        self.metrics = metrics or []  # Store metric objects
        self.style = style  # Display style
        self.use_colors = style == 'modern' and ModernColors.is_enabled()

        # File paths (only if logging enabled)
        if self.logging_enabled:
            # Create experiment directory
            os.makedirs(experiment_dir, exist_ok=True)
            self.csv_path = os.path.join(experiment_dir, "training.csv")
            self.json_path = os.path.join(experiment_dir, "metadata.json")
        else:
            self.csv_path = None
            self.json_path = None

        # State tracking
        self.epoch_data = []
        self.metadata = {}
        self.start_time = None
        self.epoch_start_time = None
        self.prev_val_loss = None
        self.val_loss_increases = 0

        # CSV file setup
        self.csv_file = None
        self.csv_writer = None

    def start_training(self, model, optimizer, epochs, batch_size, seed=None, **kwargs):
        """Record training start and metadata"""
        self.start_time = datetime.now()

        # Build model architecture string
        arch_str = self._build_architecture_string(model)
        total_params = sum(p.data.size for p in model.parameters())

        # Collect metadata
        self.metadata = {
            "experiment_dir": self.experiment_dir,
            "timestamp": self.start_time.isoformat(),
            "model_architecture": arch_str,
            "total_params": int(total_params),
            "optimizer": optimizer.__class__.__name__,
            "learning_rate": getattr(optimizer, "lr", "N/A"),
            "batch_size": batch_size,
            "epochs": epochs,
            "device": "cuda" if nm.is_gpu_enabled() else "cpu",
            **kwargs,  # Additional hyperparameters
        }

        # Add seed if specified
        if seed is not None:
            self.metadata["seed"] = seed

        # Save metadata (only if logging enabled)
        if self.logging_enabled:
            with open(self.json_path, "w") as f:
                json.dump(self.metadata, f, indent=2)

        if self.verbose > 0:
            if self.style == 'modern' and self.use_colors:
                # Modern style initialization
                C = ModernColors
                print(f"\n{C.CYAN}{'═' * 50}{C.RESET}")
                print(f"{C.CYAN}◆{C.RESET} {C.MAGENTA}{C.BOLD}NEURAL NETWORK INITIALIZATION{C.RESET}")
                print(f"{C.CYAN}{'═' * 50}{C.RESET}")
                print(f"{C.GREEN}▸{C.RESET} {C.DIM}Parameters:{C.RESET} {C.YELLOW}{total_params:,}{C.RESET}")

                # Device info with CUDA memory
                device_str = self.metadata['device'].upper()
                if self.metadata['device'] == 'cuda':
                    try:
                        # Get CUDA memory info
                        import cupy as cp
                        mempool = cp.get_default_memory_pool()
                        total_bytes = mempool.total_bytes()
                        used_bytes = mempool.used_bytes()
                        total_gb = total_bytes / (1024 ** 3)
                        used_gb = used_bytes / (1024 ** 3)
                        print(f"{C.GREEN}▸{C.RESET} {C.DIM}Device:{C.RESET} {C.CYAN}{C.BOLD}⚡ {device_str}{C.RESET}")
                        print(f"{C.GREEN}▸{C.RESET} {C.DIM}GPU Memory:{C.RESET} {C.YELLOW}{used_gb:.2f}GB{C.RESET} / {C.DIM}{total_gb:.2f}GB{C.RESET}")
                    except:
                        print(f"{C.GREEN}▸{C.RESET} {C.DIM}Device:{C.RESET} {C.CYAN}{C.BOLD}⚡ {device_str}{C.RESET}")
                else:
                    print(f"{C.GREEN}▸{C.RESET} {C.DIM}Device:{C.RESET} {C.BLUE}{device_str}{C.RESET}")

                if self.logging_enabled:
                    print(f"{C.GREEN}▸{C.RESET} {C.DIM}Experiment:{C.RESET} {self.experiment_dir}")
                print(f"{C.CYAN}{'═' * 50}{C.RESET}\n")
            else:
                # Default style
                print(f"[{self._timestamp()}] Training started")
                print(f"  Parameters: {total_params:,}")
                print(f"  Device: {self.metadata['device']}")
                if self.logging_enabled:
                    print(f"  Experiment: {self.experiment_dir}")
                print()

    def start_epoch(self, epoch):
        """Mark epoch start"""
        self.epoch_start_time = datetime.now()

    def log_epoch(
        self,
        epoch,
        total_epochs,
        train_metrics,
        val_metrics=None,
        is_best=False,
        lr=None,
        grad_norm=None,
    ):
        """Log epoch results"""
        duration = (datetime.now() - self.epoch_start_time).total_seconds()

        # Prepare data
        epoch_info = {
            "timestamp": self._timestamp(),
            "epoch": epoch + 1,
            "duration": duration,
            **{f"train_{k}": v for k, v in train_metrics.items()},
        }

        if val_metrics:
            epoch_info.update({f"val_{k}": v for k, v in val_metrics.items()})

        if lr is not None:
            epoch_info["lr"] = lr
        if grad_norm is not None:
            epoch_info["grad_norm"] = grad_norm

        # Save to CSV (only if logging enabled)
        if self.logging_enabled:
            self._write_csv(epoch_info)

        # Console output
        if self.verbose >= 1:
            self._print_epoch(
                epoch,
                total_epochs,
                train_metrics,
                val_metrics,
                duration,
                is_best,
                lr,
                grad_norm,
            )

        # Anomaly detection
        if self.verbose >= 1:
            self._check_anomalies(train_metrics, val_metrics)

        # Track validation loss
        if val_metrics and "loss" in val_metrics:
            if self.prev_val_loss is not None:
                if val_metrics["loss"] > self.prev_val_loss:
                    self.val_loss_increases += 1
                else:
                    self.val_loss_increases = 0
            self.prev_val_loss = val_metrics["loss"]

    def end_training(self, best_epoch=None, best_metrics=None):
        """Log training completion"""
        total_duration = (datetime.now() - self.start_time).total_seconds()

        if self.csv_file:
            self.csv_file.close()

        if self.verbose > 0:
            print()
            print(f"[{self._timestamp()}] Training complete")
            print(f"  Duration: {self._format_duration(total_duration)}")
            if best_epoch is not None and best_metrics:
                print(
                    f"  Best: epoch {best_epoch + 1}, {self._format_metrics(best_metrics)}"
                )
            if self.logging_enabled:
                print(f"  Results saved to: {self.experiment_dir}")

    def _build_architecture_string(self, model):
        """Build human-readable architecture string"""
        if hasattr(model, "modules") and hasattr(model.modules, "__iter__"):
            # Sequential model
            layers = []
            for module in model.modules:
                layer_str = self._layer_to_string(module)
                if layer_str:
                    layers.append(layer_str)
            return f"Sequential([{', '.join(layers)}])"
        else:
            # Custom model
            return repr(model).split("\n")[0]  # First line of repr

    def _layer_to_string(self, layer):
        """Convert layer to string"""
        name = layer.__class__.__name__

        # Linear layer
        if hasattr(layer, "in_features") and hasattr(layer, "out_features"):
            return f"Linear({layer.in_features}->{layer.out_features})"

        # Conv2D layer
        if hasattr(layer, "in_channels") and hasattr(layer, "out_channels"):
            return f"Conv2D({layer.in_channels}->{layer.out_channels})"

        # Dropout
        if hasattr(layer, "p"):
            return f"{name}(p={layer.p})"

        # Generic
        return name

    def _write_csv(self, data):
        """Write data to CSV file"""
        if self.csv_file is None:
            self.csv_file = open(self.csv_path, "w", newline="")
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=data.keys())
            self.csv_writer.writeheader()

        self.csv_writer.writerow(data)
        self.csv_file.flush()

    def _print_epoch(
        self,
        epoch,
        total_epochs,
        train_metrics,
        val_metrics,
        duration,
        is_best,
        lr,
        grad_norm,
    ):
        """Print epoch summary to console"""
        if self.style == 'modern' and self.use_colors:
            self._print_epoch_modern(epoch, total_epochs, train_metrics, val_metrics, duration, is_best, lr, grad_norm)
        else:
            self._print_epoch_default(epoch, total_epochs, train_metrics, val_metrics, duration, is_best, lr, grad_norm)

    def _print_epoch_default(
        self,
        epoch,
        total_epochs,
        train_metrics,
        val_metrics,
        duration,
        is_best,
        lr,
        grad_norm,
    ):
        """Print epoch summary in default style"""
        # Epoch header
        epoch_str = f"Epoch {epoch + 1:3d}/{total_epochs}"
        print(f"[{self._timestamp()}] {epoch_str}")

        # Training metrics
        train_str = self._format_metrics_modern(train_metrics)
        print(f"  train │ {train_str} │ {duration:.1f}s")

        # Validation metrics
        if val_metrics:
            val_str = self._format_metrics_modern(val_metrics)
            best_marker = " ★" if is_best else ""
            print(f"  val   │ {val_str}{best_marker}")

        # Additional info (verbose=2)
        if self.verbose >= 2:
            extras = []
            if lr is not None:
                extras.append(f"lr={lr:.2e}")
            if grad_norm is not None:
                extras.append(f"grad_norm={grad_norm:.2e}")
            if extras:
                print(f"        │ {' '.join(extras)}")

        print()  # Blank line for readability

    def _print_epoch_modern(
        self,
        epoch,
        total_epochs,
        train_metrics,
        val_metrics,
        duration,
        is_best,
        lr,
        grad_norm,
    ):
        """Print epoch summary in modern style - no box, free flowing neon"""
        C = ModernColors

        # Epoch header
        epoch_str = f"{epoch + 1:03d}/{total_epochs:03d}"
        timestamp = self._timestamp()[11:]  # 時刻のみ (HH:MM:SS)

        # Training metrics
        train_str = self._format_metrics_modern(train_metrics)
        print(f"{C.CYAN}»{C.RESET} {C.MAGENTA}{timestamp}{C.RESET} {C.CYAN}{C.BOLD}EP {epoch_str}{C.RESET} {C.GREEN}▸{C.RESET} {C.GREEN}{train_str}{C.RESET} {C.YELLOW}[{duration:.1f}s]{C.RESET}")

        # Validation metrics
        if val_metrics:
            val_str = self._format_metrics_modern(val_metrics)
            best_marker = f" {C.YELLOW}★{C.RESET}" if is_best else ""
            print(f"{C.CYAN}»{C.RESET} {C.MAGENTA}VAL{C.RESET} {C.GREEN}▸{C.RESET} {C.MAGENTA}{val_str}{C.RESET}{best_marker}")

        # Additional info (verbose=2)
        if self.verbose >= 2 and (lr is not None or grad_norm is not None):
            extras = []
            if lr is not None:
                extras.append(f"{C.BLUE}lr{C.RESET}={C.DIM}{lr:.2e}{C.RESET}")
            if grad_norm is not None:
                extras.append(f"{C.BLUE}grad{C.RESET}={C.DIM}{grad_norm:.2e}{C.RESET}")
            if extras:
                print(f"{C.CYAN}»{C.RESET} {' '.join(extras)}")

    def _check_anomalies(self, train_metrics, val_metrics):
        """Check for training anomalies"""
        warnings = []

        # Check for NaN
        for k, v in train_metrics.items():
            if np.isnan(v) or np.isinf(v):
                warnings.append(f"NaN/Inf detected in train_{k}")

        if val_metrics:
            for k, v in val_metrics.items():
                if np.isnan(v) or np.isinf(v):
                    warnings.append(f"NaN/Inf detected in val_{k}")

        # Check overfitting
        if val_metrics and "loss" in train_metrics and "loss" in val_metrics:
            gap = val_metrics["loss"] - train_metrics["loss"]
            if gap > 0.2:  # Significant gap
                warnings.append(
                    f"Large train-val loss gap ({gap:.3f}) - possible overfitting"
                )

        # Check validation degradation
        if self.val_loss_increases >= 3:
            warnings.append(
                f"Validation loss increasing for {self.val_loss_increases} consecutive epochs"
            )

        # Print warnings
        for warning in warnings:
            print(f"  WARNING: {warning}")

    def _format_metrics(self, metrics):
        """Format metrics dictionary to string"""
        return " ".join([f"{k}={v:.4f}" for k, v in metrics.items()])

    def _format_metrics_modern(self, metrics):
        """Format metrics with appropriate precision for modern display"""
        parts = []
        for k, v in metrics.items():
            if k == "loss":
                # Loss is always formatted with 4 decimal places
                parts.append(f"loss={v:.4f}")
            else:
                # Try to find the metric object and use its format() method
                metric_obj = self._find_metric_by_name(k)
                if metric_obj:
                    parts.append(metric_obj.format(v))
                else:
                    # Fallback: if metric not found, use old logic
                    if "accuracy" in k:
                        parts.append(f"{k}={v:5.2f}%")
                    else:
                        parts.append(f"{k}={v:.4f}")
        return " ".join(parts)

    def _find_metric_by_name(self, name):
        """Find metric object by name"""
        for metric in self.metrics:
            if metric.name() == name:
                return metric
        return None

    def _format_duration(self, seconds):
        """Format duration in human-readable format"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"

    def _timestamp(self):
        """Get current timestamp string"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ==============================
# History
# ==============================


class History:
    """
    Training history container

    Stores metrics during training and provides easy access.

    Examples
    --------
    >>> history = History()
    >>> history.add(loss=0.5, accuracy=0.8)
    >>> history.add(loss=0.4, accuracy=0.85)
    >>> print(history.loss)  # [0.5, 0.4]
    >>> print(history.accuracy)  # [0.8, 0.85]
    """

    def __init__(self):
        self.history = {}

    def add(self, **metrics):
        """Add metrics for current epoch"""
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

    def __getattr__(self, name):
        """Allow attribute-style access to history"""
        if name == "history":
            return object.__getattribute__(self, "history")
        if name in self.history:
            return self.history[name]
        raise AttributeError(f"'{name}' not found in history")

    def save(self, path):
        """Save history to JSON file"""
        import json

        with open(path, "w") as f:
            json.dump(self.history, f)

    @classmethod
    def load(cls, path):
        """Load history from JSON file"""
        import json

        with open(path, "r") as f:
            data = json.load(f)
        history = cls()
        history.history = data
        return history


# ==============================
# Trainer
# ==============================


class Trainer:
    """
    High-level training container

    Simplifies the training loop by managing model, dataset, loss,
    optimizer, and metrics in one place.

    Parameters
    ----------
    model : Module
        Neural network model
    dataset : Dataset
        Training dataset
    loss : Module, optional
        Loss function (default: CrossEntropyLoss())
    optimizer : Optimizer, optional
        Optimizer (default: Adam with lr=0.001)
    metrics : List[Metric], optional
        List of metrics to track (default: [Accuracy()])
    batch_size : int, optional
        Batch size (default: 32)
    validation_split : float, optional
        Fraction of data to use for validation (default: 0.0)
    schedulers : List[Scheduler] or None, optional
        List of parameter schedulers (default: None)
    experiment_dir : str or None, optional
        Directory to save all experiment files (logs, models, etc.).
        If None, no files are saved (default: None)
    save_best : bool or str, optional
        Save best model during training.
        - False: Don't save (default)
        - True: Save as "best_model.onnx" in experiment_dir
        - str: Custom filename (saved in experiment_dir)
        Requires experiment_dir to be set.
    restore_best_weights : bool, optional
        Whether to restore best weights after training (default: False)
    verbose : int, optional
        Verbosity level: 0=silent, 1=epoch summary, 2=detailed (default: 1)
    seed : int or None, optional
        Random seed for reproducibility. If specified, automatically sets
        the random seed and records it in metadata (default: None)
    style : str, optional
        Display style: 'default' or 'modern' (default: 'modern')

    Examples
    --------
    >>> model = Sequential([Linear(784, 10)])
    >>> dataset = MNIST(train=True)
    >>> trainer = Trainer(
    ...     model=model,
    ...     dataset=dataset,
    ...     experiment_dir="./experiments/mnist_run_001",
    ...     save_best=True,
    ... )
    >>> history = trainer.fit(epochs=10)
    >>> print(history.loss)
    """

    def __init__(
        self,
        model,
        dataset,
        loss=None,
        optimizer=None,
        metrics=None,
        batch_size=32,
        validation_split=0.0,
        schedulers=None,
        experiment_dir=None,
        save_best=False,
        restore_best_weights=False,
        verbose=1,
        seed=None,
        style='modern',
    ):
        # Validation
        if not isinstance(model, nl.Module):
            raise TypeError("model must be a Module")
        if not isinstance(dataset, dl.Dataset):
            raise TypeError("dataset must be a Dataset")
        if validation_split < 0 or validation_split >= 1:
            raise ValueError("validation_split must be in [0, 1)")

        # Set defaults
        if loss is None:
            loss = nl.CrossEntropyLoss()
        if optimizer is None:
            optimizer = nl.Adam(model.parameters(), lr=0.001)
        if metrics is None:
            metrics = [Accuracy()]

        # Set random seed if specified
        if seed is not None:
            nm.seed(seed)

        # Validate schedulers
        if schedulers is not None:
            if not isinstance(schedulers, list):
                raise TypeError("schedulers must be a list of Scheduler objects")
            self.schedulers = schedulers
        else:
            self.schedulers = []

        # Validate save_best
        if save_best and experiment_dir is None:
            raise ValueError("experiment_dir must be set when save_best is enabled")

        # Store configuration
        self.model = model
        self.dataset = dataset
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.experiment_dir = experiment_dir
        self.save_best = save_best
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.seed = seed
        self.style = style

        # Internal state
        self.history = None
        self.best_metric = float("inf")
        self.best_weights = None

    def fit(self, epochs):
        """
        Fit the model to the data

        Parameters
        ----------
        epochs : int
            Number of epochs to train

        Returns
        -------
        History
            Training history containing loss and metrics

        Examples
        --------
        >>> history = learner.fit(epochs=10)
        >>> print(f"Final loss: {history.loss[-1]:.4f}")
        """
        # Data splitting
        if self.validation_split > 0:
            train_size = int(len(self.dataset) * (1 - self.validation_split))
            val_size = len(self.dataset) - train_size
            train_dataset, val_dataset = dl.random_split(
                self.dataset, [train_size, val_size]
            )
        else:
            train_dataset = self.dataset
            val_dataset = None

        # Create DataLoaders
        train_loader = dl.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        if val_dataset:
            val_loader = dl.DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )

        # Initialize history and logger
        self.history = History()
        logger = ProgressLogger(
            experiment_dir=self.experiment_dir, verbose=self.verbose, metrics=self.metrics, style=self.style
        )

        # Start training
        logger.start_training(
            model=self.model,
            optimizer=self.optimizer,
            epochs=epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            seed=self.seed,
        )

        # Track best model
        best_epoch = None
        best_val_metrics = None

        # Training loop
        for epoch in range(epochs):
            logger.start_epoch(epoch)

            # Training phase
            nl.train.enable()
            epoch_losses = []
            epoch_metrics = {metric.name(): [] for metric in self.metrics}

            total_batches = len(train_loader)
            for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
                # Forward
                y_pred = self.model(batch_x)
                loss = self.loss(y_pred, batch_y)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Record
                epoch_losses.append(loss.item())
                for metric in self.metrics:
                    value = metric(y_pred, batch_y)
                    epoch_metrics[metric.name()].append(value)

                # Show batch progress
                if self.verbose >= 1:
                    elapsed = (datetime.now() - logger.epoch_start_time).total_seconds()
                    progress = (batch_idx + 1) / total_batches
                    current_loss = float(nm.mean(epoch_losses).item())

                    if logger.style == 'modern' and logger.use_colors:
                        # Modern style progress
                        C = ModernColors
                        bar_length = 20
                        filled = int(bar_length * progress)
                        bar = f"{C.CYAN}{'▰' * filled}{C.DIM}{'▱' * (bar_length - filled)}{C.RESET}"
                        print(
                            f"\r{C.CYAN}»{C.RESET} {C.MAGENTA}EP {epoch + 1:03d}{C.RESET} {bar} {C.GREEN}{batch_idx + 1}/{total_batches}{C.RESET} {C.YELLOW}loss={current_loss:.4f}{C.RESET} {C.DIM}[{elapsed:.1f}s]{C.RESET}",
                            end="",
                            flush=True,
                        )
                    else:
                        # Default style progress
                        bar_length = 30
                        filled = int(bar_length * progress)
                        bar = "█" * filled + "░" * (bar_length - filled)
                        print(
                            f"\r  Epoch {epoch + 1}/{epochs} [{bar}] {batch_idx + 1}/{total_batches} - loss: {current_loss:.4f} - {elapsed:.1f}s",
                            end="",
                            flush=True,
                        )

            # Clear progress bar line
            if self.verbose >= 1:
                print("\r" + " " * 100 + "\r", end="")

            # Calculate averages
            train_loss = float(nm.mean(epoch_losses).item())
            train_metrics = {
                "loss": train_loss,
                **{
                    name: float(nm.mean(values).item())
                    for name, values in epoch_metrics.items()
                },
            }

            # Record to history
            self.history.add(**train_metrics)

            # Validation phase
            val_metrics_dict = None
            is_best = False

            if val_dataset:
                nl.train.disable()
                val_losses = []
                val_metrics = {metric.name(): [] for metric in self.metrics}

                for batch_x, batch_y in val_loader:
                    y_pred = self.model(batch_x)
                    loss = self.loss(y_pred, batch_y)

                    val_losses.append(loss.item())
                    for metric in self.metrics:
                        value = metric(y_pred, batch_y)
                        val_metrics[metric.name()].append(value)

                val_loss = float(nm.mean(val_losses).item())
                val_metrics_avg = {
                    name: float(nm.mean(values).item())
                    for name, values in val_metrics.items()
                }

                val_metrics_dict = {"loss": val_loss, **val_metrics_avg}

                # Record to history
                self.history.add(**{f"val_{k}": v for k, v in val_metrics_dict.items()})

                # Check if best model
                if val_loss < self.best_metric:
                    self.best_metric = val_loss
                    is_best = True
                    best_epoch = epoch
                    best_val_metrics = val_metrics_dict.copy()

                    if self.save_best:
                        self._save_model()

                    if self.restore_best_weights:
                        self.best_weights = self._copy_weights()
            else:
                # Save based on train_loss if no validation
                if train_loss < self.best_metric:
                    self.best_metric = train_loss
                    is_best = True
                    best_epoch = epoch
                    best_val_metrics = {"loss": train_loss}

                    if self.save_best:
                        self._save_model()

                    if self.restore_best_weights:
                        self.best_weights = self._copy_weights()

            # Log epoch
            logger.log_epoch(
                epoch=epoch,
                total_epochs=epochs,
                train_metrics=train_metrics,
                val_metrics=val_metrics_dict,
                is_best=is_best,
                lr=getattr(self.optimizer, "lr", None),
            )

            # Scheduler step
            for scheduler in self.schedulers:
                # Check if scheduler requires a metric (plateau-based schedulers)
                # ReduceOnPlateauScheduler and its subclasses need a metric argument
                from lemon.nnlib.scheduler.reduce_on_plateau import (
                    ReduceOnPlateauScheduler,
                )

                if isinstance(scheduler, ReduceOnPlateauScheduler):
                    # Use validation loss if available, otherwise training loss
                    metric = (
                        val_metrics_dict["loss"] if val_metrics_dict else train_loss
                    )
                    scheduler.step(metric)
                else:
                    scheduler.step()

        # End training
        logger.end_training(best_epoch=best_epoch, best_metrics=best_val_metrics)

        # Restore best weights if enabled
        if self.restore_best_weights and self.best_weights is not None:
            self._restore_weights(self.best_weights)
            if self.verbose > 0:
                print(f"  Restored best weights (val_loss={self.best_metric:.4f})")

        return self.history

    def evaluate(self, dataset):
        """
        Evaluate model on dataset

        Parameters
        ----------
        dataset : Dataset
            Dataset to evaluate on

        Returns
        -------
        Dict[str, float]
            Dictionary containing loss and metrics

        Examples
        --------
        >>> test_metrics = learner.evaluate(test_dataset)
        >>> print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
        """
        nl.train.disable()

        loader = dl.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        losses = []
        metrics_values = {metric.name(): [] for metric in self.metrics}

        for batch_x, batch_y in loader:
            y_pred = self.model(batch_x)
            loss = self.loss(y_pred, batch_y)

            losses.append(loss.item())
            for metric in self.metrics:
                value = metric(y_pred, batch_y)
                metrics_values[metric.name()].append(value)

        result = {
            "loss": float(nm.mean(losses).item()),
            **{
                name: float(nm.mean(values).item())
                for name, values in metrics_values.items()
            },
        }

        return result

    def predict(self, x):
        """
        Make predictions

        Parameters
        ----------
        x : Array
            Input data

        Returns
        -------
        Array
            Predictions

        Examples
        --------
        >>> predictions = learner.predict(x_test)
        """
        nl.train.disable()

        if not isinstance(x, nm.NumType):
            x = nm.tensor(x)

        return self.model(x)

    def _save_model(self):
        """Save best model to experiment directory"""
        if not self.experiment_dir:
            return

        # Determine filename
        if isinstance(self.save_best, str):
            filename = self.save_best
        else:
            filename = "best_model.onnx"

        # Ensure .onnx extension
        if not filename.endswith(".onnx"):
            filename += ".onnx"

        path = os.path.join(self.experiment_dir, filename)

        import lemon.onnx_io as ox

        input_shape = ox._infer_input_shape(self.model)
        sample_input = nm.zeros((1,) + input_shape)

        ox.export_model(self.model, path, sample_input=sample_input, verbose=False)

    def save(self, path):
        """
        Save model to ONNX format

        Parameters
        ----------
        path : str
            Path to save model (.onnx)

        Examples
        --------
        >>> learner.save("my_model.onnx")
        """
        import lemon.onnx_io as ox

        input_shape = ox._infer_input_shape(self.model)
        sample_input = nm.zeros((1,) + input_shape)

        ox.export_model(self.model, path, sample_input=sample_input, verbose=False)

    def _copy_weights(self):
        """Copy current model weights"""
        weights = []
        for param in self.model.parameters():
            weights.append(copy.deepcopy(param.data))
        return weights

    def _restore_weights(self, weights):
        """Restore model weights"""
        for param, saved_weight in zip(self.model.parameters(), weights):
            param.data = saved_weight

    def save_weights(self, path):
        """
        Save model weights to file

        Parameters
        ----------
        path : str
            Path to save weights (.npz)

        Examples
        --------
        >>> learner.save_weights("model_weights.npz")
        """
        weights_dict = {}
        for i, param in enumerate(self.model.parameters()):
            weights_dict[f"param_{i}"] = nm.as_numpy(param.data)

        np.savez(path, **weights_dict)

    def load_weights(self, path):
        """
        Load model weights from file

        Parameters
        ----------
        path : str
            Path to load weights from (.npz)

        Examples
        --------
        >>> learner.load_weights("model_weights.npz")
        """
        weights_dict = np.load(path)

        for i, param in enumerate(self.model.parameters()):
            key = f"param_{i}"
            if key in weights_dict:
                xp = nm.get_array_module(param.data._data)
                param.data._data = xp.asarray(weights_dict[key])
            else:
                raise ValueError(f"Missing parameter {key} in file {path}")

    def save_checkpoint(self, path, epoch, loss, metadata=None, verbose=True):
        """
        Save complete training checkpoint including optimizer and scheduler state

        This allows resuming training from the exact point where it was stopped,
        preserving all training state including optimizer momentum and scheduler progress.

        Parameters
        ----------
        path : str
            Path to save checkpoint file (.pkl)
        epoch : int
            Current epoch number
        loss : float
            Current loss value
        metadata : dict, optional
            Additional metadata to save
        verbose : bool, optional
            Print save information (default: True)

        Examples
        --------
        >>> learner.save_checkpoint("checkpoint_epoch_10.pkl", epoch=10, loss=0.5)
        >>> # Resume later:
        >>> info = learner.load_checkpoint("checkpoint_epoch_10.pkl")
        >>> learner.fit(epochs=20)  # Continue training
        """
        ck.save_checkpoint(
            filepath=path,
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            loss=loss,
            schedulers=self.schedulers if len(self.schedulers) > 0 else None,
            metadata=metadata,
            verbose=verbose,
        )

    def load_checkpoint(self, path, verbose=True):
        """
        Load training checkpoint and restore complete training state

        Restores model parameters, optimizer state (momentum, etc.), and scheduler state,
        allowing training to resume from exactly where it left off.

        Parameters
        ----------
        path : str
            Path to checkpoint file (.pkl)
        verbose : bool, optional
            Print loading information (default: True)

        Returns
        -------
        dict
            Dictionary with 'epoch', 'loss', and 'metadata'

        Examples
        --------
        >>> info = learner.load_checkpoint("checkpoint_epoch_10.pkl")
        >>> print(f"Resuming from epoch {info['epoch']}")
        >>> learner.fit(epochs=20)  # Continue training
        """
        info = ck.load_checkpoint(
            filepath=path,
            model=self.model,
            optimizer=self.optimizer,
            schedulers=self.schedulers if len(self.schedulers) > 0 else None,
            verbose=verbose,
        )

        # Update internal state
        self.best_metric = info["loss"]

        return info


__all__ = [
    "ProgressLogger",
    "Metric",
    "Accuracy",
    "BinaryAccuracy",
    "MAE",
    "MSE",
    "RMSE",
    "TopKAccuracy",
    "History",
    "Trainer",
]
