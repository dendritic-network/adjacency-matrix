# streamlit run dnm_connectivity_app.py --server.runOnSave false

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random
from types import SimpleNamespace
import torch
import networkx as nx

# -----------------------------------------------------------------------------
# --- MODEL AND UTILITY FUNCTIONS ---
# (This entire section is unchanged)
# -----------------------------------------------------------------------------
def _find_closest_free_block(size, ideal_center, occupied_slots, N_in):
    if size <= 0: return []
    valid_start_range_end = N_in - size
    if valid_start_range_end < 0: return []
    ideal_start_pos = ideal_center - size // 2
    clamped_start_pos = max(0, min(ideal_start_pos, valid_start_range_end))
    for offset in range(N_in):
        start_pos_right = clamped_start_pos + offset
        if start_pos_right <= valid_start_range_end:
            candidate_block = set(range(start_pos_right, start_pos_right + size))
            if not candidate_block.intersection(occupied_slots): return list(candidate_block)
        if offset > 0:
            start_pos_left = clamped_start_pos - offset
            if start_pos_left >= 0:
                candidate_block = set(range(start_pos_left, start_pos_left + size))
                if not candidate_block.intersection(occupied_slots): return list(candidate_block)
    return []

def _adjust_samples_probabilistic_local(samples, target_total):
    integer_parts = np.floor(samples).astype(int)
    fractional = samples - integer_parts
    total_integer = np.sum(integer_parts)
    remainder = target_total - total_integer
    if remainder > 0:
        if np.sum(fractional) > 0:
            probs = fractional / np.sum(fractional)
            extra_indices = np.random.choice(len(samples), size=remainder, p=probs, replace=True)
            np.add.at(integer_parts, extra_indices, 1)
        else:
            extra_indices = np.random.choice(len(samples), size=remainder, replace=True)
            np.add.at(integer_parts, extra_indices, 1)
    return integer_parts.tolist()

def _adjust_samples_clipping_local(samples, target_total):
    samples = np.array(samples)
    target_total = int(target_total)
    samples[samples < 0] = 0
    integer_parts = np.floor(samples).astype(int)
    fractional = samples - integer_parts
    current_total = np.sum(integer_parts)
    diff = target_total - current_total
    if diff > 0:
        probs = fractional / np.sum(fractional) if np.sum(fractional) > 0 else np.ones_like(fractional) / len(fractional)
        extra_indices = np.random.choice(len(samples), size=diff, p=probs, replace=True)
        np.add.at(integer_parts, extra_indices, 1)
    elif diff < 0:
        while diff < 0:
            can_reduce = np.where(integer_parts > 1)[0]
            if not len(can_reduce): break
            idx_to_reduce = np.random.choice(can_reduce)
            integer_parts[idx_to_reduce] -= 1
            diff += 1
    return integer_parts.tolist()

def pick_connections_for_output_node_local(j, N_in, N_out, D_total, M, gamma, args):
    base_center = j * (N_in / N_out)
    center = int(round(base_center))
    base_window = D_total
    window_size = int(round(base_window + gamma * (N_in - base_window)))
    window_size = max(D_total, min(window_size, N_in))
    window_start = center - window_size // 2
    uniform_centers = np.linspace(window_start, window_start + window_size, M, endpoint=False) + window_size/(2*M)
    group_centers = (1 - gamma) * center + gamma * uniform_centers
    group_sizes = []
    if args.synaptic_dist == "fixed":
        base, remainder = D_total // M, D_total % M
        group_sizes = [base + 1] * remainder + [base] * (M - remainder)
        random.shuffle(group_sizes)
    elif args.synaptic_dist == "uniform":
        spread = getattr(args, "uniform_spread", 2)
        mean_connections = D_total / M
        low = max(1, mean_connections * (1 - spread))
        high = mean_connections * (1 + spread)
        random_values = np.random.uniform(low, high, M)
        group_sizes = _adjust_samples_probabilistic_local(random_values, D_total)
    elif args.synaptic_dist in ["spatial_gaussian", "spatial_inversegaussian"]:
        gc_rounded = [int(round(gc)) for gc in group_centers]
        distances = np.abs(np.array(gc_rounded) - center)
        sigma = args.synaptic_std * N_in
        if args.synaptic_dist == "spatial_gaussian": gaussian_vals = np.exp(-0.5 * (distances / sigma)**2)
        else:
            max_dist = np.max(distances) if len(distances) > 0 else 0
            inverse_distances = max_dist - distances
            gaussian_vals = np.exp(-0.5 * (inverse_distances / sigma)**2)
        probs = gaussian_vals / np.sum(gaussian_vals) if np.sum(gaussian_vals) > 0 else np.ones(M) / M
        group_assignments = np.random.choice(M, size=D_total, p=probs)
        group_sizes = np.bincount(group_assignments, minlength=M)
    candidate_groups = [{'ideal_center': int(round(gc)), 'size': g_size} for gc, g_size in zip(group_centers, group_sizes)]
    final_connections = set()
    if j < N_out / 2:
        candidate_groups.sort(key=lambda g: g['ideal_center'])
        last_placed_neuron_idx = -1
        for group in candidate_groups:
            g_size = group['size']
            if g_size <= 0: continue
            ideal_start = group['ideal_center'] - g_size // 2
            actual_start = max(ideal_start, last_placed_neuron_idx + 1)
            if actual_start + g_size > N_in: continue
            new_block = range(actual_start, actual_start + g_size)
            final_connections.update(new_block)
            last_placed_neuron_idx = actual_start + g_size - 1
    else:
        candidate_groups.sort(key=lambda g: g['ideal_center'], reverse=True)
        next_available_slot = N_in
        for group in candidate_groups:
            g_size = group['size']
            if g_size <= 0: continue
            ideal_end = (group['ideal_center'] - g_size // 2) + g_size
            actual_end = min(ideal_end, next_available_slot)
            actual_start = actual_end - g_size
            if actual_start < 0: continue
            new_block = range(actual_start, actual_end)
            final_connections.update(new_block)
            next_available_slot = actual_start
    final_list = list(final_connections)
    if len(final_list) > D_total:
        final_list.sort(key=lambda x: abs(x - center))
        return final_list[:D_total]
    elif len(final_list) < D_total:
        needed = D_total - len(final_list)
        available_slots = list(set(range(N_in)) - set(final_list))
        available_slots.sort(key=lambda x: abs(x - center))
        final_list.extend(available_slots[:needed])
    return final_list

def create_dendritic_sparse_scheduler_local(sparsity, w, args):
    N_in, N_out = w.shape
    base_M, total_target = args.M, int(round((1 - sparsity) * N_in * N_out))
    degree_dist = getattr(args, "degree_dist", "fixed")
    if degree_dist == "fixed":
        D_float = N_in * (1 - sparsity)
        if D_float % 1 == 0: connection_counts = [int(D_float)] * N_out
        else:
            K1, K2 = int(D_float), int(D_float) + 1
            count_K2 = int(round((total_target - K1 * N_out) / (K2 - K1)))
            count_K1 = N_out - count_K2
            connection_counts = [K1] * count_K1 + [K2] * count_K2
    elif degree_dist == "gaussian":
        D_float = N_in * (1 - sparsity)
        samples = np.random.normal(D_float, args.degree_std, N_out)
        samples = np.clip(samples, 1, N_in)
        connection_counts = _adjust_samples_clipping_local(samples, total_target)
    elif degree_dist == "uniform":
        D_float = N_in * (1 - sparsity)
        spread = getattr(args, "degree_spread", 4 * D_float)
        samples = np.random.uniform(D_float-spread, D_float+spread, N_out)
        connection_counts = _adjust_samples_clipping_local(samples, total_target)
    elif degree_dist in ["spatial_gaussian", "spatial_inversegaussian"]:
        center, sigma = (N_out - 1) / 2.0, args.degree_std * N_in * (1 - sparsity)
        distances = np.abs(np.arange(N_out) - center)
        if degree_dist == "spatial_gaussian": weights = np.exp(-0.5 * (distances / sigma)**2)
        else:
            max_weight = np.exp(-0.5 * (0 / sigma)**2)
            weights = max_weight - np.exp(-0.5 * (distances / sigma)**2)
        samples = weights * (total_target / np.sum(weights)) if np.sum(weights) > 0 else np.ones(N_out)*(total_target/N_out)
        connection_counts = _adjust_samples_clipping_local(samples, total_target)
    else: raise ValueError(f"Unknown degree distribution: {degree_dist}")
    if degree_dist not in ["spatial_gaussian", "spatial_inversegaussian"]: random.shuffle(connection_counts)
    connection_counts = np.clip(np.array(connection_counts, dtype=int), 1, N_in)
    diff = total_target - np.sum(connection_counts)
    while diff != 0:
        if diff > 0:
            candidates = np.where(connection_counts < N_in)[0]
            if not len(candidates): break
            idx = np.random.choice(candidates)
            connection_counts[idx] += 1
            diff -= 1
        else:
            candidates = np.where(connection_counts > 1)[0]
            if not len(candidates): break
            idx = np.random.choice(candidates)
            connection_counts[idx] -= 1
            diff += 1
    adj = np.zeros((N_in, N_out), dtype=int)
    for j in range(N_out):
        gamma_j, M_j = _get_neuron_params(j, N_out, args)
        D_total = connection_counts[j]
        M_j = max(1, min(M_j, D_total))
        connections = pick_connections_for_output_node_local(j, N_in, N_out, D_total, M_j, gamma_j, args)
        for i in connections: adj[i, j] = 1
    _apply_rewiring(adj, N_in, N_out, args.random_rewiring)
    return torch.LongTensor(adj).to(w.device)

def _adjust_samples_original(samples, target_total):
    samples, rounded = np.array(samples), np.round(samples).astype(int)
    
    # --- CORRECTED CODE ---
    clipped = np.clip(rounded, 1, None)
    current_total = np.sum(clipped)
    # --- END CORRECTION ---

    diff = target_total - current_total
    if diff != 0:
        fractional = samples - rounded
        if diff > 0:
            indices = np.argsort(-fractional)[:diff]
            clipped[indices] += 1
        else:
            reducible_mask = clipped > 1
            reducible_indices = np.where(reducible_mask)[0]
            if len(reducible_indices) > 0:
                sorted_indices = reducible_indices[np.argsort(fractional[reducible_indices])]
                for i in range(min(-diff, len(sorted_indices))): clipped[sorted_indices[i]] -= 1
    final_diff = target_total - np.sum(clipped)
    if final_diff > 0:
        for i in range(final_diff): clipped[i % len(clipped)] += 1
    elif final_diff < 0:
        removed = 0
        for i in range(len(clipped)):
            if clipped[i] > 1 and removed < -final_diff:
                clipped[i] -= 1
                removed += 1
    return clipped.tolist()

def symmetric_positions_original(center, D_total, window_size, N_in):
    half_window = (window_size - 1) // 2
    positions = np.linspace(center - half_window, center + half_window, D_total, dtype=int)
    positions = [(x % N_in) for x in positions]
    unique_positions = list(set(positions))
    while len(unique_positions) < D_total: unique_positions.extend(unique_positions)
    return unique_positions[:D_total]

def pick_connections_for_output_node_original(j, N_in, N_out, D_total, M, gamma, args):
    base_center = j * (N_in / N_out)
    center = int(round(base_center)) % N_in
    window_size = int(round(D_total + gamma * (N_in - D_total)))
    window_size = max(D_total, min(window_size, N_in))
    window_start = center - window_size // 2
    uniform_centers = np.linspace(window_start, window_start + window_size, M, endpoint=False) + window_size/(2*M)
    group_centers = (1 - gamma) * center + gamma * uniform_centers
    group_centers = [int(round(gc)) % N_in for gc in group_centers]
    connections = []
    if args.synaptic_dist == "fixed":
        if gamma == 0: return symmetric_positions_original(center, D_total, window_size, N_in)
        base, remainder = D_total // M, D_total % M
        group_sizes = [base + 1] * remainder + [base] * (M - remainder)
        random.shuffle(group_sizes)
        for gc, g_size in zip(group_centers, group_sizes):
            half = g_size // 2
            connections.extend([(gc - half + k) % N_in for k in range(g_size)])
    elif args.synaptic_dist in ["spatial_gaussian", "spatial_inversegaussian"]:
        distances = [min(abs(gc - base_center), N_in - abs(gc - base_center)) for gc in group_centers]
        sigma = args.synaptic_std * N_in
        if args.synaptic_dist == "spatial_gaussian": gaussian_vals = np.exp(-0.5 * (np.array(distances)/sigma)**2)
        else:
            max_dist = np.max(distances) if distances else 0
            inverse_distances = max_dist - np.array(distances)
            gaussian_vals = np.exp(-0.5 * (inverse_distances/sigma)**2)
        probs = gaussian_vals / np.sum(gaussian_vals) if np.sum(gaussian_vals) > 0 else np.ones(M)/M
        group_assignments = np.random.choice(M, size=D_total, p=probs)
        group_sizes = np.bincount(group_assignments, minlength=M)
        for gc, size in zip(group_centers, group_sizes):
            connections.extend(get_closest_nodes_centered(gc, N_in, size))
    else:
        st.warning(f"Synaptic distribution '{args.synaptic_dist}' is not fully defined for the 'Original' model and will use fixed distribution.")
        base, remainder = D_total // M, D_total % M
        group_sizes = [base + 1] * remainder + [base] * (M - remainder)
        random.shuffle(group_sizes)
        for gc, g_size in zip(group_centers, group_sizes):
            half = g_size // 2
            connections.extend([(gc - half + k) % N_in for k in range(g_size)])
    unique_connections = list(set(connections))
    if len(unique_connections) < D_total:
        needed = D_total - len(unique_connections)
        available = list(set(range(N_in)) - set(unique_connections))
        unique_connections.extend(random.sample(available, needed))
    return unique_connections[:D_total]

def create_dendritic_sparse_scheduler_original(sparsity, w, args):
    N_in, N_out, base_M = w.shape[0], w.shape[1], args.M
    total_target = int(round((1 - sparsity) * N_in * N_out))
    degree_dist = getattr(args, "degree_dist", "fixed")
    if degree_dist == "fixed":
        D_float = N_in * (1 - sparsity)
        K1, K2 = int(D_float), int(D_float) + 1
        count_K2 = int(round((total_target - K1 * N_out) / (K2 - K1))) if (K2 - K1) != 0 else 0
        count_K1 = N_out - count_K2
        connection_counts = [K1] * count_K1 + [K2] * count_K2
    elif degree_dist in ["gaussian", "uniform"]:
        D_float = N_in * (1 - sparsity)
        if degree_dist == "gaussian":
            std = getattr(args, "degree_std", 2 * D_float)
            samples = np.random.normal(D_float, std, N_out)
        else:
            spread = getattr(args, "degree_spread", 4 * D_float)
            samples = np.random.uniform(D_float - spread, D_float + spread, N_out)
        samples = np.clip(samples, 1, N_in)
        connection_counts = _adjust_samples_original(samples, total_target)
    elif degree_dist in ["spatial_gaussian", "spatial_inversegaussian"]:
        center, sigma = (N_out - 1) / 2.0, args.degree_std * N_in * (1 - sparsity)
        distances = np.abs(np.arange(N_out) - center)
        if degree_dist == "spatial_gaussian": weights = np.exp(-0.5 * (distances / sigma)**2)
        else: weights = 1 - np.exp(-0.5 * (distances / sigma)**2)
        samples = weights * (total_target / np.sum(weights)) if np.sum(weights) > 0 else np.ones(N_out)*(total_target/N_out)
        connection_counts = _adjust_samples_original(samples, total_target)
    else: raise ValueError(f"Unknown degree distribution: {degree_dist}")
    if degree_dist not in ["spatial_gaussian", "spatial_inversegaussian"]: random.shuffle(connection_counts)
    adj = np.zeros((N_in, N_out), dtype=int)
    for j in range(N_out):
        gamma_j, M_j = _get_neuron_params(j, N_out, args)
        D_total = connection_counts[j]
        M_j = max(1, min(M_j, D_total))
        connections = pick_connections_for_output_node_original(j, N_in, N_out, D_total, M_j, gamma_j, args)
        for i in connections: adj[i, j] = 1
    _apply_rewiring(adj, N_in, N_out, args.random_rewiring)
    return torch.LongTensor(adj).to(w.device)

def get_closest_nodes_centered(i, N, count):
    indices = [i]
    d = 1
    while len(indices) < count:
        indices.append((i - d) % N)
        if len(indices) < count: indices.append((i + d) % N)
        d += 1
    return indices[:count]

def _get_neuron_params(j, N_out, args):
    if args.gamma_dist == "fixed": gamma_j = args.gamma
    elif args.gamma_dist == "gaussian": gamma_j = np.random.normal(args.gamma, args.gamma_std)
    elif args.gamma_dist == "uniform":
        spread = 0.25
        gamma_j = np.random.uniform(args.gamma - spread, args.gamma + spread)
    elif args.gamma_dist in ["spatial_gaussian", "spatial_inversegaussian"]:
        center_out, max_dist = (N_out - 1) / 2.0, (N_out - 1) / 2.0
        distance = abs(j - center_out)
        normalized_dist = distance / max_dist if max_dist > 0 else 0
        mean_gamma_j = (1.0 - normalized_dist) if args.gamma_dist == "spatial_gaussian" else (args.gamma * normalized_dist)
        gamma_j = np.random.normal(mean_gamma_j, args.gamma_std)
    gamma_j = np.clip(gamma_j, 0.0, 1.0)
    D_total_avg = args.N_in * (1 - args.sparsity)
    if args.M_dist == "fixed": M_j = args.M
    elif args.M_dist == "gaussian": M_j = np.random.normal(args.M, args.M_std)
    elif args.M_dist == "uniform":
        spread = getattr(args, "M_spread", args.M)
        M_j = np.random.uniform(args.M - spread, args.M + spread)
    elif args.M_dist in ["spatial_gaussian", "spatial_inversegaussian"]:
        center_out, sigma = (N_out - 1) / 2.0, args.M_std * N_out
        distance = abs(j - center_out)
        if args.M_dist == "spatial_gaussian": weight = np.exp(-0.5 * (distance / sigma)**2)
        else: weight = 1.0 - 0.9 * np.exp(-0.5 * (distance / sigma)**2)
        M_j = weight * args.M * 2
    M_j = int(np.round(np.clip(M_j, 1, D_total_avg)))
    return gamma_j, M_j

def _apply_rewiring(adj, N_in, N_out, rewire_prob):
    if rewire_prob == 0: return
    total_edges = int(np.sum(adj))
    to_rewire_mask = np.random.binomial(1, p=rewire_prob, size=total_edges)
    current_edge, removed_count = 0, 0
    for i in range(N_in):
        for j in range(N_out):
            if adj[i, j] == 1:
                if to_rewire_mask[current_edge] == 1:
                    adj[i, j] = 0
                    removed_count += 1
                current_edge += 1
    added_count = 0
    while added_count < removed_count:
        i_rand, j_rand = np.random.randint(0, N_in), np.random.randint(0, N_out)
        if adj[i_rand, j_rand] == 0:
            adj[i_rand, j_rand] = 1
            added_count += 1

def create_dnm_connectivity(model_type, num_inputs, num_outputs, sparsity, num_dendrites,
                            dendrite_dist, gamma, gamma_dist, synaptic_dist, degree_dist, **kwargs):
    w = torch.zeros((num_inputs, num_outputs))
    args = SimpleNamespace(M=num_dendrites, M_dist=dendrite_dist, degree_dist=degree_dist,
        synaptic_dist=synaptic_dist, gamma=gamma, gamma_dist=gamma_dist, random_rewiring=0,
        degree_std=2.0, M_std=num_dendrites/20, gamma_std=0.2, synaptic_std=0.1,
        N_in=num_inputs, sparsity=sparsity, **kwargs)
    if model_type == "Bounded": mask = create_dendritic_sparse_scheduler_local(sparsity, w, args)
    elif model_type == "Wrap-around": mask = create_dendritic_sparse_scheduler_original(sparsity, w, args)
    else: raise ValueError(f"Unknown model type: {model_type}")
    return mask.cpu().numpy()

def create_network_graph(masks, layer_sizes):
    G = nx.DiGraph()
    node_offsets = np.cumsum([0] + layer_sizes)
    for layer_idx, size in enumerate(layer_sizes):
        for node_idx in range(int(node_offsets[layer_idx]), int(node_offsets[layer_idx] + size)):
            G.add_node(node_idx, layer=layer_idx)
    for layer_idx, mask in enumerate(masks):
        sources, targets = np.where(mask == 1)
        for src, tgt in zip(sources, targets):
            G.add_edge(int(node_offsets[layer_idx] + src), int(node_offsets[layer_idx+1] + tgt))
    return G

def plot_network_graph(G, layer_sizes, ax):
    pos = {}
    for layer_idx, size in enumerate(layer_sizes):
        y_positions = np.linspace(1, 0, size)
        for i in range(size):
            node_idx = int(np.sum(layer_sizes[:layer_idx])) + i
            pos[node_idx] = (layer_idx, y_positions[i])
    for edge in G.edges():
        ax.plot([pos[edge[0]][0], pos[edge[1]][0]], [pos[edge[0]][1], pos[edge[1]][1]], 'gray', alpha=0.1, lw=0.5)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for layer_idx in range(len(layer_sizes)):
        nodes = range(int(np.sum(layer_sizes[:layer_idx])), int(np.sum(layer_sizes[:layer_idx+1])))
        nodes_in_pos = [n for n in nodes if n in pos]
        if nodes_in_pos:
            ax.scatter([pos[n][0] for n in nodes_in_pos], [pos[n][1] for n in nodes_in_pos], s=10, color=colors[layer_idx], label=f'Layer {layer_idx}')
    ax.axis('off')
    ax.set_title('Network Structure')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(layer_sizes))


# -----------------------------------------------------------------------------
# --- STREAMLIT UI ---
# (This section contains all the new changes)
# -----------------------------------------------------------------------------
st.title("Dendritic Network Model Visualization")

base_layer_sizes = [784, 1568, 1568, 1568]
# Define the fixed 25% scale for the MLP visualization
viz_layer_sizes = [int(s * 0.25) for s in base_layer_sizes]

# --- Sidebar parameters for the model ---
st.sidebar.header("Model Configuration")
model_type = st.sidebar.radio("Bounded or Wrapped?", ["Wrap-around", "Bounded"], help="**Local**: Boundary-aware... **Original**: Modulo-wrapping...")
sparsity = st.sidebar.slider("Sparsity", 0.5, 0.99, 0.9, 0.01)
num_dendrites = st.sidebar.slider("Avg. Dendrites (M)", 1, 11, 3, 2)
gamma = st.sidebar.slider("Avg. Receptive Field (α)", 0.0, 1.0, 1.0, 0.05)
st.sidebar.subheader("Parameter Distributions")
dendrite_dist = st.sidebar.selectbox("Dendritic Distribution", ["fixed", "gaussian", "uniform", "spatial_gaussian", "spatial_inversegaussian"])
gamma_dist = st.sidebar.selectbox("Receptive Field Width Distribution", ["fixed", "gaussian", "uniform", "spatial_gaussian", "spatial_inversegaussian"])
degree_dist = st.sidebar.selectbox("Degree Distribution", ["fixed", "uniform", "gaussian", "spatial_gaussian", "spatial_inversegaussian"])
synaptic_dist = st.sidebar.selectbox("Synaptic Distribution", ["fixed", "uniform", "spatial_gaussian", "spatial_inversegaussian"])

st.sidebar.subheader("Additional Visualizations")
# --- NEW: Checkbox to control MLP visualization ---
show_mlp_viz = st.sidebar.checkbox("Show Scaled-Down MLP Visualization")
if show_mlp_viz:
    st.sidebar.warning("⚠️ Enabling the MLP visualization runs a second, separate simulation and may take a few moments to generate.")

# --- Main app logic ---
if st.button("Generate Network"):
    # --- SIMULATION 1: FULL-SIZED NETWORK (100%) ---
    with st.spinner("Creating full-size network for adjacency matrix..."):
        full_masks = []
        for i in range(len(base_layer_sizes)-1):
            conn_matrix = create_dnm_connectivity(
                model_type=model_type, num_inputs=base_layer_sizes[i], num_outputs=base_layer_sizes[i+1],
                sparsity=sparsity, num_dendrites=num_dendrites, dendrite_dist=dendrite_dist,
                gamma=gamma, gamma_dist=gamma_dist, synaptic_dist=synaptic_dist, degree_dist=degree_dist
            )
            full_masks.append(conn_matrix)

    actual_sparsity = 1.0 - (np.sum(full_masks[0]) / full_masks[0].size)
    st.info(f"**Full Network Sparsity (Layer 0 -> 1):** `{actual_sparsity:.4f}`")

    # Plot Adjacency Matrix from the full-sized network
    st.subheader(f"Adjacency Matrix (Full Size: {base_layer_sizes[0]}x{base_layer_sizes[1]})")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(full_masks[0], aspect='auto', cmap='Blues')
    ax.set_xlabel(f"Output Neurons (Layer 1: {base_layer_sizes[1]})")
    ax.set_ylabel(f"Input Neurons (Layer 0: {base_layer_sizes[0]})")
    st.pyplot(fig)

    # --- NEW: Conditional execution of SIMULATION 2 ---
    if show_mlp_viz:
        # --- SIMULATION 2: SCALED-DOWN NETWORK (25%) ---
        with st.spinner("Creating scaled-down network for MLP visualization..."):
            viz_masks = []
            for i in range(len(viz_layer_sizes)-1):
                viz_conn_matrix = create_dnm_connectivity(
                    model_type=model_type, num_inputs=viz_layer_sizes[i], num_outputs=viz_layer_sizes[i+1],
                    sparsity=sparsity, num_dendrites=num_dendrites, dendrite_dist=dendrite_dist,
                    gamma=gamma, gamma_dist=gamma_dist, synaptic_dist=synaptic_dist, degree_dist=degree_dist
                )
                viz_masks.append(viz_conn_matrix)

        # Plot the MLP graph from the scaled-down network
        st.subheader(f"MLP Visualization (Scaled Down: {viz_layer_sizes[0]}x{viz_layer_sizes[1]})")
        if sum(m.sum() for m in viz_masks) > 0:
            G_viz = create_network_graph(viz_masks, viz_layer_sizes)
            fig_graph, ax_graph = plt.subplots(figsize=(8, 5))
            plot_network_graph(G_viz, viz_layer_sizes, ax_graph)
            st.pyplot(fig_graph)
        else:
            st.warning("No connections were generated in the scaled-down network to visualize.")

