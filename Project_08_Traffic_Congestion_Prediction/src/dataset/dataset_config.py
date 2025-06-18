import numpy as np
from collections import deque

# === 1) Load dataset ===
dataset = np.load(r'dataset/traffic_dataset_13_smoothen.npy', allow_pickle=True)
T_total, E, C_origin = dataset.shape

day_steps = 480
week_steps = day_steps * 7

# (The dataset was generated using notebooks in the dataset_generation directory)

# === 2) Node and edge definitions (preprocessed) ===
converted_nodes = [{'id': 0, 'type': 'H', 'coords': (10, 10)}, {'id': 1, 'type': 'I', 'coords': (5, 8)}, {'id': 2, 'type': 'S', 'coords': (7, 2)}, {'id': 3, 'type': 'C', 'coords': (6, 0)}, {'id': 4, 'type': 'S', 'coords': (2, 7)}, {'id': 5, 'type': 'C', 'coords': (5, 0)}, {'id': 6, 'type': 'S', 'coords': (2, 2)}, {'id': 7, 'type': 'I', 'coords': (4, 9)}, {'id': 8, 'type': 'O', 'coords': (1, 1)}, {'id': 9, 'type': 'C', 'coords': (5, 1)}, {'id': 10, 'type': 'R', 'coords': (7, 1)}, {'id': 11, 'type': 'R', 'coords': (9, 3)}, {'id': 12, 'type': 'I', 'coords': (4, 8)}, {'id': 13, 'type': 'S', 'coords': (7, 7)}, {'id': 14, 'type': 'C', 'coords': (8, 4)}, {'id': 15, 'type': 'O', 'coords': (2, 1)}, {'id': 16, 'type': 'R', 'coords': (8, 1)}, {'id': 17, 'type': 'R', 'coords': (8, 2)}, {'id': 18, 'type': 'C', 'coords': (1, 3)}, {'id': 19, 'type': 'R', 'coords': (9, 4)}]
converted_edges = [{'start': 0, 'end': 7, 'distance': 12.165525060596439, 'road_type': 'urban'}, {'start': 7, 'end': 0, 'distance': 12.165525060596439, 'road_type': 'urban'}, {'start': 1, 'end': 12, 'distance': 2.0, 'road_type': 'urban'}, {'start': 12, 'end': 1, 'distance': 2.0, 'road_type': 'urban'}, {'start': 2, 'end': 6, 'distance': 10.0, 'road_type': 'highway'}, {'start': 6, 'end': 2, 'distance': 10.0, 'road_type': 'highway'}, {'start': 2, 'end': 13, 'distance': 10.0, 'road_type': 'highway'}, {'start': 13, 'end': 2, 'distance': 10.0, 'road_type': 'highway'}, {'start': 2, 'end': 17, 'distance': 2.0, 'road_type': 'urban'}, {'start': 17, 'end': 2, 'distance': 2.0, 'road_type': 'urban'}, {'start': 3, 'end': 5, 'distance': 2.0, 'road_type': 'urban'}, {'start': 5, 'end': 3, 'distance': 2.0, 'road_type': 'urban'}, {'start': 4, 'end': 6, 'distance': 10.0, 'road_type': 'highway'}, {'start': 6, 'end': 4, 'distance': 10.0, 'road_type': 'highway'}, {'start': 4, 'end': 7, 'distance': 5.656854249492381, 'road_type': 'urban'}, {'start': 7, 'end': 4, 'distance': 5.656854249492381, 'road_type': 'urban'}, {'start': 4, 'end': 13, 'distance': 10.0, 'road_type': 'highway'}, {'start': 13, 'end': 4, 'distance': 10.0, 'road_type': 'highway'}, {'start': 5, 'end': 6, 'distance': 7.211102550927978, 'road_type': 'urban'}, {'start': 6, 'end': 5, 'distance': 7.211102550927978, 'road_type': 'urban'}, {'start': 5, 'end': 9, 'distance': 2.0, 'road_type': 'urban'}, {'start': 9, 'end': 5, 'distance': 2.0, 'road_type': 'urban'}, {'start': 5, 'end': 15, 'distance': 6.324555320336759, 'road_type': 'urban'}, {'start': 15, 'end': 5, 'distance': 6.324555320336759, 'road_type': 'urban'}, {'start': 5, 'end': 17, 'distance': 7.211102550927978, 'road_type': 'urban'}, {'start': 17, 'end': 5, 'distance': 7.211102550927978, 'road_type': 'urban'}, {'start': 6, 'end': 15, 'distance': 2.0, 'road_type': 'urban'}, {'start': 15, 'end': 6, 'distance': 2.0, 'road_type': 'urban'}, {'start': 6, 'end': 18, 'distance': 2.8284271247461903, 'road_type': 'urban'}, {'start': 18, 'end': 6, 'distance': 2.8284271247461903, 'road_type': 'urban'}, {'start': 7, 'end': 12, 'distance': 2.0, 'road_type': 'urban'}, {'start': 12, 'end': 7, 'distance': 2.0, 'road_type': 'urban'}, {'start': 8, 'end': 15, 'distance': 2.0, 'road_type': 'urban'}, {'start': 15, 'end': 8, 'distance': 2.0, 'road_type': 'urban'}, {'start': 10, 'end': 16, 'distance': 2.0, 'road_type': 'urban'}, {'start': 16, 'end': 10, 'distance': 2.0, 'road_type': 'urban'}, {'start': 10, 'end': 17, 'distance': 2.8284271247461903, 'road_type': 'urban'}, {'start': 17, 'end': 10, 'distance': 2.8284271247461903, 'road_type': 'urban'}, {'start': 11, 'end': 19, 'distance': 2.0, 'road_type': 'urban'}, {'start': 19, 'end': 11, 'distance': 2.0, 'road_type': 'urban'}, {'start': 13, 'end': 14, 'distance': 6.324555320336759, 'road_type': 'urban'}, {'start': 14, 'end': 13, 'distance': 6.324555320336759, 'road_type': 'urban'}, {'start': 14, 'end': 17, 'distance': 4.0, 'road_type': 'urban'}, {'start': 17, 'end': 14, 'distance': 4.0, 'road_type': 'urban'}, {'start': 14, 'end': 18, 'distance': 14.142135623730951, 'road_type': 'urban'}, {'start': 18, 'end': 14, 'distance': 14.142135623730951, 'road_type': 'urban'}, {'start': 14, 'end': 19, 'distance': 2.0, 'road_type': 'urban'}, {'start': 19, 'end': 14, 'distance': 2.0, 'road_type': 'urban'}, {'start': 15, 'end': 17, 'distance': 12.165525060596439, 'road_type': 'urban'}, {'start': 17, 'end': 15, 'distance': 12.165525060596439, 'road_type': 'urban'}]

# === 3) edge_idx_map ===
# Map (u, v) tuple to unique edge ID
edge_idx_map = {
    (e['start'], e['end']): idx
    for idx, e in enumerate(converted_edges)
}

# === 4) node_idx_map ===
# Automatically generate mapping: edge ID → (u, v) tuple
node_idx_map = {
    idx: uv
    for uv, idx in edge_idx_map.items()
}

# === 5) edge_adj_mat ===
# Adjacency matrix between edges (E x E)
edge_adj_mat = np.zeros((len(edge_idx_map), len(edge_idx_map)))

for edge_info in edge_idx_map:
    id = edge_idx_map[edge_info]
    u, v = edge_info

    for edge_info_2 in edge_idx_map:
        U, V = edge_info_2
        ID = edge_idx_map[edge_info_2]
        if u==V or v==U:
            edge_adj_mat[id,ID] = 1
            edge_adj_mat[ID,id] = 1
np.fill_diagonal(edge_adj_mat, 0.0)

# === 6) edge_degree_list ===
# Edge degree list (number of neighboring edges)
edge_degree_list = edge_adj_mat.sum(axis=1).astype(int)  # shape (E,)

# === 7) edge_spd: Edge-to-edge shortest path distance ===
def _compute_spd(adj: np.ndarray) -> np.ndarray:
    E = adj.shape[0]
    spd = np.full((E, E), -1, dtype=int)
    for src in range(E):
        dist = -np.ones(E, dtype=int)
        dist[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            # adj[u] == 1 means neighbor
            for v, connected in enumerate(adj[u]):
                if connected and dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        spd[src] = dist
    return spd

edge_spd = _compute_spd(edge_adj_mat)


# === 8) edge_index ===
# shape: (2, E), source and destination node indices for each edge
starts = np.array([e['start'] for e in converted_edges], dtype=np.int64)
ends   = np.array([e['end']   for e in converted_edges], dtype=np.int64)
edge_index = np.stack([starts, ends], axis=0)  # (2, E)

# === 9) edge_attr ===
# shape: (E, F_e), edge features (distance + one-hot encoded road type)
distances = np.array([e['distance'] for e in converted_edges], dtype=np.float32)
road_types = [e['road_type'] for e in converted_edges]
type_to_idx = {t:i for i,t in enumerate(sorted(set(road_types)))}

onehot = np.zeros((len(road_types), len(type_to_idx)), dtype=np.float32)
for i, t in enumerate(road_types):
    onehot[i, type_to_idx[t]] = 1.0

edge_attr = np.concatenate([distances[:,None], onehot], axis=1)  # (E, 1+num_types)

river_info = (np.array([ 0.        ,  2.5       ,  3.33333333,  6.66666667,  7.5       ,
       10.        ]), np.array([6.86811936, 6.86811936, 6.45145269, 6.45145269, 6.03478602,
       6.03478602]), 1.6977288245972708)