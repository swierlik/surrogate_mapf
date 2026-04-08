"""Dataset management for (omega, throughput) pairs.

Saves everything needed for surrogate model training:
    - Raw solution vectors (pre-normalization)
    - Mean throughput per solution
    - Per-eval throughputs (all n_evals individual sim results)
    - Generation metadata

Files produced:
    {prefix}_solutions.npy  - shape (N, sol_size), appended incrementally
    {prefix}_log.csv        - generation, sol_idx, emitter_id, mean_throughput, eval_0..eval_4
    {prefix}_best.csv       - generation, best_throughput, best_mean_throughput
"""

import csv
import numpy as np
from pathlib import Path


class RunLogger:
    """Logs (solution, throughput) pairs incrementally to disk.

    Supports resume mode: pass resume=True to append to existing files
    instead of overwriting them.
    """

    def __init__(self, log_dir, prefix="cmaes", n_evals=5, resume=False):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.n_evals = n_evals

        self._sol_path = self.log_dir / f"{prefix}_solutions.npy"
        self._solutions_buffer = []
        self._csv_path = self.log_dir / f"{prefix}_log.csv"
        self._best_path = self.log_dir / f"{prefix}_best.csv"

        eval_headers = [f"eval_{i}" for i in range(n_evals)]
        csv_header = ["generation", "sol_idx", "emitter_id", "mean_throughput"] + eval_headers
        best_header = ["generation", "best_throughput", "best_mean_throughput",
                       "gen_wallclock_s", "cumulative_wallclock_s"]

        if resume and self._csv_path.exists():
            # Append mode: don't overwrite existing data
            self._csv_file = open(self._csv_path, "a", newline="")
            self._writer = csv.writer(self._csv_file)
            # best.csv is always opened in append mode per-write, so no action needed
        else:
            # Fresh start
            self._csv_file = open(self._csv_path, "w", newline="")
            self._writer = csv.writer(self._csv_file)
            self._writer.writerow(csv_header)

            with open(self._best_path, "w", newline="") as f:
                csv.writer(f).writerow(best_header)

    def log_generation(self, generation, solutions, mean_throughputs,
                       all_throughputs, emitter_ids=None):
        """Log a full generation of evaluations.

        Args:
            generation: int, generation number.
            solutions: np.ndarray shape (batch_size, sol_size), raw solutions.
            mean_throughputs: np.ndarray shape (batch_size,).
            all_throughputs: np.ndarray shape (batch_size, n_evals).
            emitter_ids: optional np.ndarray shape (batch_size,), which emitter
                         produced each solution.
        """
        batch_size = len(solutions)
        if emitter_ids is None:
            emitter_ids = np.zeros(batch_size, dtype=int)

        for i in range(batch_size):
            self._solutions_buffer.append(solutions[i].astype(np.float32))
            row = [
                generation,
                i,
                int(emitter_ids[i]),
                f"{mean_throughputs[i]:.6f}",
            ]
            row.extend([f"{t:.6f}" for t in all_throughputs[i]])
            self._writer.writerow(row)

        self._csv_file.flush()

    def log_best(self, generation, best_throughput, best_mean_throughput=None,
                 gen_wallclock_s=0.0, cumulative_wallclock_s=0.0):
        """Append best throughput and timing for this generation."""
        if best_mean_throughput is None:
            best_mean_throughput = best_throughput
        with open(self._best_path, "a", newline="") as f:
            csv.writer(f).writerow([
                generation,
                f"{best_throughput:.6f}",
                f"{best_mean_throughput:.6f}",
                f"{gen_wallclock_s:.2f}",
                f"{cumulative_wallclock_s:.2f}",
            ])

    def flush_solutions(self):
        """Write buffered solutions to .npy (incremental append)."""
        if not self._solutions_buffer:
            return

        new_arr = np.stack(self._solutions_buffer, axis=0)

        if self._sol_path.exists():
            existing = np.load(self._sol_path)
            combined = np.concatenate([existing, new_arr], axis=0)
        else:
            combined = new_arr

        np.save(self._sol_path, combined)
        self._solutions_buffer.clear()

    def close(self):
        """Flush everything and close files."""
        self.flush_solutions()
        self._csv_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def load_run_data(log_dir, prefix="cmaes"):
    """Load saved run data for surrogate training.

    Returns:
        solutions: np.ndarray shape (N, sol_size)
        mean_throughputs: np.ndarray shape (N,)
        all_throughputs: np.ndarray shape (N, n_evals) or None if not available
    """
    log_dir = Path(log_dir)
    solutions = np.load(log_dir / f"{prefix}_solutions.npy")

    mean_throughputs = []
    all_throughputs = []

    with open(log_dir / f"{prefix}_log.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mean_throughputs.append(float(row["mean_throughput"]))
            evals = []
            for key in sorted(row.keys()):
                if key.startswith("eval_"):
                    evals.append(float(row[key]))
            all_throughputs.append(evals)

    mean_tp = np.array(mean_throughputs)
    all_tp = np.array(all_throughputs) if all_throughputs[0] else None

    return solutions, mean_tp, all_tp
