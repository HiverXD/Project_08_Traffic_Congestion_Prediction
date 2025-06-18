# src/utils/visualization.py

import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx

from tqdm import tqdm
from collections import defaultdict

def add_tod_dow(raw_data: np.ndarray, week_steps: int, C_origin: int) -> np.ndarray:
    """
    Add time-of-day (tod) and day-of-week (dow) channels to the raw data.

    Args:
        raw_data: np.ndarray, shape (T_total, E, C_orig)
            - T_total: total number of timesteps
            - E: number of edges (sensors)
            - C_orig: original number of feature channels (e.g., volume, density, flow)
        week_steps: int, default=480*7
            - total number of steps per week (e.g., assuming 480 steps per day for 7 days)
        C_origin: int, default=3
            - number of channels in the original data; used to verify shape compatibility

    Returns:
        np.ndarray, shape (T_total, E, C_orig + 2)
        - the last two channels are tod (float, 0–24) and dow (float, 0–6).
    """
    if raw_data.shape[2] == C_origin:
        pass
    elif raw_data.shape[2] == (C_origin + 2):
        return raw_data
    else:
        raise Exception('shape error')
    
    T_total, E, C_orig = raw_data.shape
    day_steps = week_steps // 7

    # 1) Generate all timestep indices
    timesteps = np.arange(T_total)

    # 2) Time-of-day feature (0–24)
    tod = (timesteps % day_steps) * (24.0 / day_steps)
    # 3) Day-of-week feature (0–6)
    dow = (timesteps // day_steps) % 7

    # 4) Expand from (T,1,1) to (T,E,1)
    tod_feat = np.tile(tod[:, None, None], (1, E, 1)).astype(np.float32)
    dow_feat = np.tile(dow[:, None, None], (1, E, 1)).astype(np.float32)

    # 5) Concatenate original + tod + dow in order
    return np.concatenate([raw_data, tod_feat, dow_feat], axis=-1)

def visualize_predictions(model, expanded_data, edge_ids, device, edge_index, edge_attr,
                          interval=(0,480), channel=0, pred_offsets=np.array([3, 6, 12]), window=12):
    """
    Visualize predicted vs. actual values for specified edges using the model.

    Parameters
    ----------
    model : torch.nn.Module
        trained prediction model
    expanded_data : np.ndarray
        expanded input time series data of shape (T, E, C+2),
        expanded using add_tod_dow as in the dataloader.
    edge_ids : list[int]
        list of edge indices to visualize
    device : torch.device
        device for model computation
    edge_index : torch.Tensor
        graph edge index information
    edge_attr : torch.Tensor
        graph edge attribute matrix
    interval : tuple[int, int], optional
        visualization interval (start, end)
    channel : int, optional
        target channel to plot
    pred_offsets : np.ndarray, optional
        prediction time offsets
    window : int, optional
        sliding window size
    """
    start, end = interval
    data = expanded_data[start:end]

    model.eval()
    T_total, E, C_all = data.shape

    # temporary storage for predictions
    pred_lists = {e: defaultdict(list) for e in edge_ids}

    with torch.no_grad():
        for t0 in range(window - 1, T_total - int(pred_offsets.max())):
            x_win = data[t0 - window + 1:t0 + 1, :, :]
            x_tensor = torch.from_numpy(x_win[None]).float().to(device)  # [1, T, E, C_in]
            
            preds = model(x_tensor, edge_index, edge_attr)              # [1, n_pred, E, C_out]
            preds = preds.cpu().numpy()[0]                              # [n_pred, E, C_out]

            for i, offset in enumerate(pred_offsets):
                t_pred = t0 + offset
                if t_pred >= T_total:
                    continue
                for e in edge_ids:
                    # e.g., visualize only the specified channel
                    pred_lists[e][t_pred].append(preds[i, e, channel])

    # reconstruct time series by averaging
    for e in edge_ids:
        pred_series = np.full(T_total, np.nan, dtype=float)
        for t, vals in pred_lists[e].items():
            pred_series[t] = np.mean(vals)
        actual_series = data[:T_total, e, channel]  # actual values for the channel

        # plot
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(T_total), actual_series,    label=f'Actual Edge {e}')
        plt.plot(np.arange(T_total), pred_series, '--', label=f'Predicted Edge {e}')
        plt.xlabel('Time Step')
        plt.ylabel('Volume Channel')
        plt.title(f'Edge {e}, channel {channel}: Actual vs. Predicted')
        plt.legend()
        plt.tight_layout()
        plt.show()

def plot_mape_violin(model, loader, device, edge_index, edge_attr, bw_method=0.5):
    model.eval()
    step_mapes = [[], [], []]
    channel_mapes = [[], [], []]
    names_ch = ['volume', 'density', 'flow']
    names_steps = ['+3', '+6', '+12']
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x, edge_index, edge_attr)  # [B,3,E,3]
            # step-wise
            for i in range(3):
                t, p = y[:, i], pred[:, i]
                mask = t.abs() > 1e-3
                step_mapes[i].append(((p[mask] - t[mask]).abs() / t[mask]).mean().item())
            # channel-wise
            for ci in range(3):
                t, p = y[..., ci], pred[..., ci]
                mask = t.abs() > 1e-3
                channel_mapes[ci].append(((p[mask] - t[mask]).abs() / t[mask]).mean().item())

    # Convert to numpy arrays
    step_mapes_np = [np.array(vals) for vals in step_mapes]
    channel_mapes_np = [np.array(vals) for vals in channel_mapes]
    
    # Compute stats
    step_means = np.array([arr.mean() for arr in step_mapes_np])
    step_stds = np.array([arr.std() for arr in step_mapes_np])
    ch_means = np.array([arr.mean() for arr in channel_mapes_np])
    ch_stds = np.array([arr.std() for arr in channel_mapes_np])
    
    # Overall
    all_values = np.concatenate(step_mapes_np + channel_mapes_np)
    overall_mean = all_values.mean()
    overall_std = all_values.std()
    
    # 1) Step-wise Violin Plot
    fig, ax = plt.subplots()
    ax.set_axisbelow(True)
    parts = ax.violinplot(step_mapes_np, positions=[1, 2, 3],
                          showmeans=True, bw_method=bw_method)
    for pc in parts['bodies']:
        pc.set_alpha(0.7)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(names_steps)
    ax.set_ylabel('MAPE'); ax.set_title('Step-wise MAPE')
    ax.grid(True)
    
    # 2) Channel-wise Violin Plot
    fig, ax = plt.subplots()
    ax.set_axisbelow(True)
    parts = ax.violinplot(channel_mapes_np, positions=[1, 2, 3],
                          showmeans=True, bw_method=bw_method)
    for pc in parts['bodies']:
        pc.set_alpha(0.7)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(names_ch)
    ax.set_ylabel('MAPE'); ax.set_title('Channel-wise MAPE')
    ax.grid(True)
    
    # Print statistics
    print("Step-wise MAPE:")
    for step, mean, std in zip(names_steps, step_means, step_stds):
        print(f"  {step}: mean={mean:.4f}, std={std:.4f}")
    print("Channel-wise MAPE:")
    for ch, mean, std in zip(names_ch, ch_means, ch_stds):
        print(f"  {ch}: mean={mean:.4f}, std={std:.4f}")
    print(f"Overall MAPE: mean={overall_mean:.4f}, std={overall_std:.4f}")

def plot_city_edge_mape(converted_nodes, converted_edges, river_info,
                        loader, model, device,
                        edge_index, edge_attr,
                        city_size=10, save=False, output_dir='figures'):
    """
    converted_nodes: [{'id': int, 'coords': (x,y)}, ...]
    converted_edges: [{'start': u, 'end': v}, ...]
    loader: DataLoader yielding (x_batch, y_batch)
    model: trained model returning [B, n_pred, E, C]
    edge_index, edge_attr: graph tensors
    """
    model.eval()
    E = len(converted_edges)
    mape_edges = [[] for _ in range(E)]
    
    # 1) Collect edge-wise APE per batch
    with torch.no_grad():
        for x_batch, y_batch in tqdm(loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch, edge_index, edge_attr)  # [B,n_pred,E,C]
            
            mask = y_batch.abs() > 1e-3
            ape = torch.zeros_like(y_batch)
            ape[mask] = (pred[mask] - y_batch[mask]).abs() / y_batch[mask]
            ape_np  = ape.cpu().numpy()
            mask_np = mask.cpu().numpy()
            
            for e in range(E):
                vals = ape_np[:, :, e, :][mask_np[:, :, e, :]]
                if vals.size > 0:
                    mape_edges[e].append(vals.mean())

    # 2) Compute average MAPE per edge
    mape_avg = np.array([np.mean(lst) if lst else np.nan for lst in mape_edges])
    valid = ~np.isnan(mape_avg)
    mape_valid = mape_avg[valid]
    
    # 3) Map widths: smaller MAPE → thicker (max_w), larger MAPE → thinner (min_w)
    min_w, max_w = 0.5, 5.0
    if mape_valid.size:
        mn, mx = mape_valid.min(), mape_valid.max()
        widths = np.full(E, min_w)
        widths[valid] = max_w - (mape_valid - mn)/(mx - mn)*(max_w - min_w)
    else:
        widths = np.full(E, (min_w + max_w)/2)

    # 4) Create graph
    G = nx.DiGraph()
    pos = {nd['id']: nd['coords'] for nd in converted_nodes}
    for nd in converted_nodes:
        G.add_node(nd['id'])

    edgelist = []
    edge_labels = {}
    for i, e in enumerate(converted_edges):
        u, v = e['start'], e['end']
        edgelist.append((u, v))
        edge_labels[(u, v)] = f"{mape_avg[i]:.2f}" if not np.isnan(mape_avg[i]) else ""

    # 5) Visualization
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_facecolor('white')
    ax.axis('off')

    xs, ys, river_width = river_info
    ax.plot(xs, ys,
            color='cadetblue',
            linewidth=river_width*5,
            alpha=1.0)

    # Nodes
    nx.draw_networkx_nodes(G, pos,
                           node_color='lightgray',
                           node_size=300,
                           edgecolors='black',
                           ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    # Edges
    nx.draw_networkx_edges(G, pos,
                           edgelist=edgelist,
                           width=widths,
                           edge_color='black',
                           arrows=True,
                           arrowstyle='-|>',
                           arrowsize=12,
                           ax=ax)
    nx.draw_networkx_edge_labels(G, pos,
                                 edge_labels=edge_labels,
                                 font_size=7,
                                 label_pos=0.5,
                                 ax=ax)

    ax.set_xlim(-1, city_size+1)
    ax.set_ylim(-1, city_size+1)
    ax.axis('on')
    ax.set_axisbelow(True)
    ax.grid('on')
    plt.title("Edge-wise Average MAPE Visualization", fontsize=14)

    if save:
        import os
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/edge_mape.png",
                    dpi=300, bbox_inches='tight')
    plt.show()

def plot_dow_mape_violin_filtered(loader, model, device,
                                  edge_index, edge_attr,
                                  dow_idx,
                                  clip_percentile=95):
    """
    Plot day-of-week (DOW) step-wise & edge-wise MAPE distributions,
    clipping values above the specified clip_percentile percentile to that percentile.
    """
    model.eval()
    # Extract shapes from the first batch
    x0, y0 = next(iter(loader))
    B0, n_pred, E, C = model(x0.to(device), edge_index, edge_attr).shape

    # Initialize storage by weekday
    step_mapes = {d: [[] for _ in range(n_pred)] for d in range(7)}
    edge_mapes = {d: [] for d in range(7)}

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x, y = x_batch.to(device), y_batch.to(device)
            pred = model(x, edge_index, edge_attr)  # [B,n_pred,E,C]

            mask = y.abs() > 1e-3
            ape = torch.zeros_like(y)
            ape[mask] = (pred[mask] - y[mask]).abs() / y[mask]
            ape_np = ape.cpu().numpy()        # [B,n_pred,E,C]
            # average over channels → [B,n_pred,E]
            mape_se = ape.mean(dim=-1).cpu().numpy()

            # Extract DOW values (assumed same for each batch)
            dow_vals = x[..., dow_idx].cpu().numpy()[:,0,0].astype(int)

            for b, d in enumerate(dow_vals):
                # step-wise
                for s in range(n_pred):
                    step_mapes[d][s].extend(mape_se[b, s, :].tolist())
                # edge-wise (average over steps → per-edge)
                edge_avg = mape_se[b].mean(axis=0)  # (E,)
                edge_mapes[d].extend(edge_avg.tolist())

    days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

    # Helper function: clip list values at given percentile
    def clip_list(data_list, perc):
        if not data_list:
            return data_list
        thresh = np.percentile(data_list, perc)
        return np.minimum(data_list, thresh)

    # 1) Step-wise plot
    fig, axes = plt.subplots(1, n_pred, figsize=(4*n_pred,4), sharey=True)
    for s, ax in enumerate(axes):
        data = [clip_list(step_mapes[d][s], clip_percentile) for d in range(7)]
        parts = ax.violinplot(data, positions=np.arange(1,8), showmeans=True)
        ax.set_title(f"Step +{[3,6,12][s]}")
        ax.set_xticks(np.arange(1,8)); ax.set_xticklabels(days, rotation=45)
        ax.set_ylabel('MAPE' if s==0 else None)
        ax.grid(True); ax.set_axisbelow(True)
    fig.suptitle(f"Step-wise MAPE (clipped at {clip_percentile}‰)", y=1.02)
    plt.tight_layout()
    plt.show()

    # 2) Edge-wise plot
    fig, ax = plt.subplots(figsize=(8,4))
    data = [clip_list(edge_mapes[d], clip_percentile) for d in range(7)]
    parts = ax.violinplot(data, positions=np.arange(1,8), showmeans=True)
    ax.set_xticks(np.arange(1,8)); ax.set_xticklabels(days, rotation=45)
    ax.set_ylabel('MAPE')
    ax.set_title(f"Edge-wise MAPE (clipped at {clip_percentile}‰)")
    ax.grid(True); ax.set_axisbelow(True)
    plt.tight_layout()
    plt.show()
