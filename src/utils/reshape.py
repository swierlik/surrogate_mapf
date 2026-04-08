"""Convert flat solution vectors to spatial (H, W, C) tensors for CNN input.

The flat vector layout is [wait_costs(948), edge_weights(3126)] for the
warehouse-33x36 map.  Edge ordering matches the C++ simulator and the
reference comp_uncompress_edge_matrix(): for each valid cell in row-major
order, edges are stored as [Right, Up, Left, Down], skipping directions
where the neighbor is out of bounds or an obstacle.

Tensor channels: [Right, Up, Left, Down, Wait, ObstacleMask]
"""

import numpy as np
from src.simulator.evaluate import MapInfo, OBSTACLE_IDX


class SolutionReshaper:
    """Converts flat solution vectors to spatial tensors (and back).

    Precomputes index mapping tables once; subsequent conversions are
    fully vectorized NumPy operations with no Python loops.
    """

    _instance = None

    def __init__(self):
        info = MapInfo.get()
        self.h = info.n_row
        self.w = info.n_col
        self.map_np = info.map_np  # (h, w) int array, OBSTACLE_IDX=1
        self.n_valid = info.n_valid_vertices  # 948
        self.n_edges = info.n_valid_edges     # 3126

        # Obstacle mask: 1.0 where obstacle, 0.0 where valid
        self.obstacle_mask = (self.map_np == OBSTACLE_IDX).astype(np.float32)

        # Valid cells in row-major order
        valid_rows = []
        valid_cols = []
        for r in range(self.h):
            for c in range(self.w):
                if self.map_np[r, c] != OBSTACLE_IDX:
                    valid_rows.append(r)
                    valid_cols.append(c)
        assert len(valid_rows) == self.n_valid, \
            f"Expected {self.n_valid} valid cells, got {len(valid_rows)}"
        self.valid_rows = np.array(valid_rows)
        self.valid_cols = np.array(valid_cols)

        # Edge mapping: direction order [Right(+c), Up(-r), Left(-c), Down(+r)]
        # matching ggo_public comp_uncompress_edge_matrix
        directions = [(0, 1), (-1, 0), (0, -1), (1, 0)]  # R, U, L, D
        edge_rows = []
        edge_cols = []
        edge_chans = []
        for r, c in zip(valid_rows, valid_cols):
            for d_idx, (dr, dc) in enumerate(directions):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.h and 0 <= nc < self.w \
                        and self.map_np[nr, nc] != OBSTACLE_IDX:
                    edge_rows.append(r)
                    edge_cols.append(c)
                    edge_chans.append(d_idx)
        assert len(edge_rows) == self.n_edges, \
            f"Expected {self.n_edges} edges, got {len(edge_rows)}"
        self.edge_rows = np.array(edge_rows)
        self.edge_cols = np.array(edge_cols)
        self.edge_chans = np.array(edge_chans)

    @classmethod
    def get(cls):
        """Return cached singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def flat_to_tensor(self, flat_vec, add_obstacle_mask=True, fill_value=0.0):
        """Convert a single flat (4074,) vector to (H, W, 5 or 6) tensor.

        Channels: [Right, Up, Left, Down, Wait, (ObstacleMask)]
        """
        n_chan = 6 if add_obstacle_mask else 5
        tensor = np.full((self.h, self.w, n_chan), fill_value, dtype=np.float32)

        # Wait costs → channel 4
        tensor[self.valid_rows, self.valid_cols, 4] = flat_vec[:self.n_valid]

        # Edge weights → channels 0-3
        tensor[self.edge_rows, self.edge_cols, self.edge_chans] = \
            flat_vec[self.n_valid:]

        if add_obstacle_mask:
            tensor[:, :, 5] = self.obstacle_mask

        return tensor

    def flat_to_tensor_batch(self, flat_vecs, add_obstacle_mask=True,
                             fill_value=0.0):
        """Batch convert (N, 4074) → (N, H, W, 5 or 6).

        Uses vectorized fancy indexing — no Python loops over the batch.
        """
        N = flat_vecs.shape[0]
        n_chan = 6 if add_obstacle_mask else 5
        tensors = np.full((N, self.h, self.w, n_chan), fill_value,
                          dtype=np.float32)

        # Wait costs: (N, 948) → channel 4
        tensors[:, self.valid_rows, self.valid_cols, 4] = \
            flat_vecs[:, :self.n_valid]

        # Edge weights: (N, 3126) → channels 0-3
        tensors[:, self.edge_rows, self.edge_cols, self.edge_chans] = \
            flat_vecs[:, self.n_valid:]

        if add_obstacle_mask:
            tensors[:, :, :, 5] = self.obstacle_mask[np.newaxis, :, :]

        return tensors
