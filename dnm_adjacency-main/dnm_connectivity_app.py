# streamlit run dnm_connectivity_app.py --server.runOnSave false

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random
from types import SimpleNamespace
import torch
import networkx as nx
from matplotlib.collections import LineCollection
import powerlaw
from networkx.algorithms.community import greedy_modularity_communities

def _find_closest_free_block(size, ideal_center, occupied_slots, N_in):
    """
    Searches for a contiguous block of 'size' that is free of 'occupied_slots'.
    The search starts at the closest valid position to 'ideal_center' and expands outwards.
    """
    if size <= 0:
        return []

    # 1. Determine the valid range for the STARTING position of a block of 'size'.
    # A block of size 20 in a layer of 784 can start from 0 up to 764 (784-20).
    valid_start_range_end = N_in - size
    if valid_start_range_end < 0:
        # The block is larger than the entire layer, which shouldn't happen with enough free space.
        print(f"Warning: Block size {size} is larger than the layer {N_in}. Cannot place.")
        return []

    # 2. Calculate the ideal STARTING position and clamp it to the valid range.
    # This is the core fix. We clamp the desired start position, not the center.
    ideal_start_pos = ideal_center - size // 2
    clamped_start_pos = max(0, min(ideal_start_pos, valid_start_range_end))

    # 3. Search outwards from the clamped starting position.
    # The loop will check clamped_start_pos, then clamped_start_pos+1, clamped_start_pos-1, etc.
    for offset in range(N_in):
        # Candidate to the right
        start_pos_right = clamped_start_pos + offset
        if start_pos_right <= valid_start_range_end:
            candidate_block = set(range(start_pos_right, start_pos_right + size))
            if not candidate_block.intersection(occupied_slots):
                return list(candidate_block)

        # Candidate to the left (avoid double-checking the starting point)
        if offset > 0:
            start_pos_left = clamped_start_pos - offset
            if start_pos_left >= 0:
                candidate_block = set(range(start_pos_left, start_pos_left + size))
                if not candidate_block.intersection(occupied_slots):
                    return list(candidate_block)

    # This should not be reached if there is enough total space in the layer.
    print("Warning: Could not place a dendrite block. Returning empty.")
    return []


def _adjust_samples(samples, target_total):
    """Convert float samples to integers using probabilistic rounding"""
    # Calculate integer parts and fractional remainders
    integer_parts = np.floor(samples).astype(int)
    fractional = samples - integer_parts

    # Calculate how many connections we need to add
    total_integer = np.sum(integer_parts)
    remainder = target_total - total_integer

    # Probabilistically distribute remaining connections
    if remainder > 0:
        # Get probabilities from fractional parts
        if np.sum(fractional) > 0:
            probs = fractional / np.sum(fractional)
            # Randomly choose which indices get extra connection
            extra_indices = np.random.choice(len(samples), size=remainder, p=probs, replace=True)
            np.add.at(integer_parts, extra_indices, 1)
        else:
            # If all fractional parts are zero, distribute remainder evenly
            extra_indices = np.random.choice(len(samples), size=remainder, replace=True)
            np.add.at(integer_parts, extra_indices, 1)

    return integer_parts.tolist()


def symmetric_positions(center, D_total, window_size, N_in):
    """
    Compute D_total positions evenly spaced within a non-wrapping window
    symmetric about the given center.
    """
    # 1. Calculate and clamp the window boundaries.
    window_start = center - window_size // 2
    if window_start < 0:
        window_start = 0

    window_end = window_start + window_size
    if window_end > N_in:
        window_end = N_in
        window_start = window_end - window_size

    # 2. Generate D_total equidistant positions within the clamped window.
    # The endpoint is window_end - 1 to stay within bounds.
    if D_total > 1:
        positions = np.linspace(window_start, window_end - 1, D_total, dtype=int).tolist()
    elif D_total == 1:
        positions = [center]
    else:
        positions = []

    # 3. Ensure uniqueness and pad if necessary (unlikely with linspace but robust).
    unique_positions = list(set(positions))
    needed = D_total - len(unique_positions)

    if needed > 0:
        available = list(set(range(window_start, window_end)) - set(unique_positions))
        to_add = min(needed, len(available))
        additional = random.sample(available, to_add)
        unique_positions.extend(additional)

    return unique_positions[:D_total]


def get_closest_nodes_centered(i, N, count):
    """
    Returns a list of 'count' indices from a layer of size N,
    ordered as: i, i-1, i+1, i-2, i+2, … (with modulo wrapping)
    """
    indices = [i]
    d = 1
    while len(indices) < count:
        indices.append((i - d) % N)
        if len(indices) < count:
            indices.append((i + d) % N)
        d += 1
    return indices[:count]


def pick_connections_for_output_node(j, N_in, N_out, sparsity, D_total, M, gamma, args):
    """
    Determines connections for an output neuron using a symmetric tiling strategy
    that handles left and right boundaries correctly.
    """
    # 1. Determine ideal center and calculate group centers and sizes (Unchanged)
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
        base = D_total // M
        remainder = D_total % M
        group_sizes = [base + 1] * remainder + [base] * (M - remainder)
        random.shuffle(group_sizes)
    elif args.synaptic_dist == "uniform":
        # Note: _adjust_samples should be defined elsewhere in your script
        spread = getattr(args, "uniform_spread", 2)
        mean_connections = D_total / M
        low = max(1, mean_connections * (1 - spread))
        high = mean_connections * (1 + spread)
        random_values = np.random.uniform(low, high, M)
        group_sizes = _adjust_samples(random_values, D_total)
    # ... (include other elif blocks for 'gaussian', etc. as in your original code) ...
    elif args.synaptic_dist in ["spatial_gaussian", "spatial_inversegaussian"]:
        gc_rounded = [int(round(gc)) for gc in group_centers]
        distances = np.abs(np.array(gc_rounded) - center)
        sigma = args.synaptic_std * N_in
        if args.synaptic_dist == "spatial_gaussian":
            gaussian_vals = np.exp(-0.5 * (distances / sigma)**2)
        else:
            max_dist = np.max(distances) if len(distances) > 0 else 0
            inverse_distances = max_dist - distances
            gaussian_vals = np.exp(-0.5 * (inverse_distances / sigma)**2)
        probs = gaussian_vals / np.sum(gaussian_vals) if np.sum(gaussian_vals) > 0 else np.ones(M) / M
        group_assignments = np.random.choice(M, size=D_total, p=probs)
        group_sizes = np.bincount(group_assignments, minlength=M)


    # 2. --- CORE LOGIC CHANGE: Implement SYMMETRIC tiling ---
    candidate_groups = []
    for gc, g_size in zip(group_centers, group_sizes):
        candidate_groups.append({'ideal_center': int(round(gc)), 'size': g_size})

    final_connections = set()

    # Decide strategy based on the output neuron's position
    if j < N_out / 2:
        # --- STRATEGY 1: Left-to-Right Tiling for left-side neurons ---
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
        # --- STRATEGY 2: Right-to-Left Tiling for right-side neurons ---
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

    # 3. Final check to ensure exact D_total connections (Unchanged)
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

def _adjust_samples(samples, target_total):
    """
    Converts float samples to integers using probabilistic rounding while ensuring
    the sum is exactly target_total and each value is at least 1 (if possible).
    """
    samples = np.array(samples)
    target_total = int(target_total)

    # Ensure samples are non-negative
    samples[samples < 0] = 0

    # First pass: integer parts and fractional remainders
    integer_parts = np.floor(samples).astype(int)
    fractional = samples - integer_parts

    # Adjust to meet the target total
    current_total = np.sum(integer_parts)
    diff = target_total - current_total

    if diff > 0:
        # Probabilistically add 1s to meet the target
        probs = fractional / np.sum(fractional) if np.sum(fractional) > 0 else np.ones_like(fractional) / len(fractional)
        extra_indices = np.random.choice(len(samples), size=diff, p=probs, replace=True)
        np.add.at(integer_parts, extra_indices, 1)
    elif diff < 0:
        # Deterministically remove 1s from the largest integer parts
        # while trying to keep at least 1 connection per group
        while diff < 0:
            can_reduce = np.where(integer_parts > 1)[0]
            if not len(can_reduce): break # Cannot reduce further

            # Choose an index to reduce
            idx_to_reduce = np.random.choice(can_reduce)
            integer_parts[idx_to_reduce] -= 1
            diff += 1

    return integer_parts.tolist()


def create_dendritic_sparse_scheduler(sparsity, w, args):
    N_in = min(w.shape[0], w.shape[1])
    N_out = max(w.shape[0], w.shape[1])
    base_M = args.M 

    total_target = int(round((1 - sparsity) * N_in * N_out))
    degree_dist = getattr(args, "degree_dist", "fixed")  

    if degree_dist == "fixed":
        D_float = N_in * (1 - sparsity)
        if D_float % 1 == 0:
             connection_counts = [int(D_float)]*N_out
        else:
            K1 = int(D_float)
            K2 = K1 + 1
            count_K1 = int(round(abs(total_target - K2 * N_out) / abs(K1 - K2)))
            count_K2 = N_out - count_K1
            connection_counts = [K1] * count_K1 + [K2] * count_K2
    elif degree_dist == "gaussian":
        D_float = N_in * (1 - sparsity)
        connection_std = getattr(args, "degree_std", 2)
        samples = np.random.normal(D_float, connection_std, N_out)
        samples = np.clip(samples, 1, N_in)  
        connection_counts = _adjust_samples(samples, total_target)
    elif degree_dist == "uniform":
        D_float = N_in * (1 - sparsity)
        spread = getattr(args, "degree_spread", 4 * D_float)
        samples = np.random.uniform(D_float-spread, D_float+spread, N_out)
        connection_counts = _adjust_samples(samples, total_target)
    elif degree_dist in ["spatial_gaussian", "spatial_inversegaussian"]:
        center = (N_out - 1) / 2.0
        D_float = N_in * (1 - sparsity)
        sigma = args.degree_std * D_float
        distances = np.abs(np.arange(N_out) - center)

        if degree_dist == "spatial_gaussian":
            weights = np.exp(-0.5 * (distances / sigma)**2)
        else:
            max_weight = np.exp(-0.5 * (0 / sigma)**2)
            weights = max_weight - np.exp(-0.5 * (distances / sigma)**2)
            weights = np.clip(weights, 0, None)

        samples = weights * (total_target / np.sum(weights)) if np.sum(weights) > 0 else np.ones(N_out)*(total_target/N_out)
        connection_counts = _adjust_samples(samples, total_target)
    else:
        raise ValueError(f"Unknown degree distribution: {degree_dist}")

    if degree_dist not in ["spatial_gaussian", "spatial_inversegaussian"]:
        random.shuffle(connection_counts)
    connection_counts = np.array(connection_counts)

    # Convert to numpy array for vector operations
    connection_counts = np.array(connection_counts, dtype=int)

    # 1. Initial clipping to valid range
    connection_counts = np.clip(connection_counts, 1, N_in)

    # 2. Gradual adjustment to reach exact total
    current_total = np.sum(connection_counts)
    diff = total_target - current_total

    # Create adjustment sequence based on difference
    if diff != 0:
        # Calculate how many nodes we can adjust
        adjustable_inc = (connection_counts < N_in).sum()
        adjustable_dec = (connection_counts > 1).sum()

        # Calculate maximum possible adjustment
        max_inc = adjustable_inc * (N_in - 1)
        max_dec = adjustable_dec * (N_in - 1)

        if diff > 0 and diff > max_inc:
            raise ValueError(f"Cannot add {diff} connections (max possible: {max_inc})")
        if diff < 0 and -diff > max_dec:
            raise ValueError(f"Cannot remove {-diff} connections (max possible: {max_dec})")

        # Create probability distribution for adjustments
        probabilities = np.ones(N_out) / N_out  # Uniform distribution

        while diff != 0:
            if diff > 0:
                # Find all nodes that can be increased
                candidates = np.where(connection_counts < N_in)[0]
                if len(candidates) == 0:
                    break
                # Select random candidate weighted by probability
                if probabilities[candidates].sum() > 0:
                    idx = np.random.choice(candidates, p=probabilities[candidates]/probabilities[candidates].sum())
                    connection_counts[idx] += 1
                    diff -= 1
                else:
                    break
            else:
                # Find all nodes that can be decreased
                candidates = np.where(connection_counts > 1)[0]
                if len(candidates) == 0:
                    break
                # Select random candidate weighted by probability
                if probabilities[candidates].sum() > 0:
                    idx = np.random.choice(candidates, p=probabilities[candidates]/probabilities[candidates].sum())
                    connection_counts[idx] -= 1
                    diff += 1
                else:
                    break


    # Final validation
    final_total = np.sum(connection_counts)
    if final_total != total_target:
        raise RuntimeError(f"Connection count mismatch: {final_total} vs {total_target} (Δ={final_total-total_target})")

    if sum(connection_counts) != total_target:
        # Force correct total by adjusting first elements
        diff = total_target - sum(connection_counts)
        for i in range(abs(diff)):
            idx = i % N_out
            if diff > 0:
                connection_counts[idx] += 1
            else:
                connection_counts[idx] = max(1, connection_counts[idx] - 1)
    connection_counts = [max(1, min(c, N_in)) for c in connection_counts]
    # Initialize adjacency matrix
    adj = np.zeros((N_in, N_out), dtype=int)

    gamma = args.gamma if args.gamma is not None else 0.5
    gamma_dist = args.gamma_dist if args.gamma_dist is not None else "fixed"

    M_dist = getattr(args, "M_dist")
    M_vals = None
    gammas = []

    if M_dist in ["spatial_gaussian", "spatial_inversegaussian"]:
        center_out = (N_out - 1) / 2.0
        sigma = args.M_std * N_out
        distances = np.abs(np.arange(N_out) - center_out)

        if M_dist == "spatial_gaussian":
            weights = np.exp(-0.5 * (distances / sigma)**2)
        else:
            max_weight = 1.0
            min_weight = 0.1
            weights = max_weight - (max_weight - min_weight) * np.exp(-0.5 * (distances / sigma)**2)
            weights = np.clip(weights, min_weight, max_weight)

        sum_weights = np.sum(weights)
        if sum_weights == 0:
            weights = np.ones(N_out)  
        scaling_factor = (N_out * base_M) / sum_weights
        weights *= scaling_factor

        M_vals_float = weights.copy()
        M_vals = np.floor(M_vals_float).astype(int) 
        fractional = M_vals_float - M_vals

        remaining = (N_out * base_M) - np.sum(M_vals)
        if remaining > 0 and np.sum(fractional) > 0:
            probs = fractional / np.sum(fractional)
            extra_indices = np.random.choice(N_out, size=remaining, p=probs, replace=True)
            np.add.at(M_vals, extra_indices, 1)

        M_vals = np.clip(M_vals, 1, None)

    for j in range(N_out):
        if gamma_dist == "fixed":
            gamma_j = gamma
        elif gamma_dist == "gaussian":
            mean_gamma = gamma
            gamma_std = getattr(args, "gamma_std", gamma*0.1)
            gamma_j = np.clip(np.random.normal(mean_gamma, gamma_std), 0, 1)
            gammas.append(gamma_j)
        elif gamma_dist=="uniform":
            spread = 0.25
            gamma_j = np.random.uniform(gamma - spread, gamma + spread)
        elif gamma_dist == "spatial_gaussian":
            center_out = (N_out - 1) / 2.0
            distance = abs(j - center_out)
            max_distance = center_out
            if max_distance > 0:
                normalized_dist = distance / max_distance
                mean_gamma_j = 1.0 - normalized_dist  
            else:
                 mean_gamma_j = 1.0
            gamma_std = getattr(args, "gamma_std", gamma*0.05)
            gamma_j = np.random.normal(mean_gamma_j, gamma_std)
            gamma_j = np.clip(gamma_j, 0.0, 1.0)
        elif gamma_dist == "spatial_inversegaussian":
            center_out = (N_out - 1) / 2.0
            distance = abs(j - center_out)
            max_distance = center_out
            if max_distance > 0:
                mean_gamma_j = gamma * (distance / max_distance)
            else:
                 mean_gamma_j = 0.0
            gamma_std = getattr(args, "gamma_std", gamma*0.05)
            gamma_j = np.random.normal(mean_gamma_j, gamma_std)
            gamma_j = np.clip(gamma_j, 0.0, 1.0)
        else:
            raise ValueError("Unknown gamma distribution: {}".format(gamma_dist))

        D_total = connection_counts[j]
        if M_dist == "fixed":
            M_j = base_M
        elif M_dist == "gaussian":
            M_std = getattr(args, "M_std", base_M /4)
            M_j = np.random.normal(base_M, M_std / 2)
            M_j = int(np.round(np.clip(M_j, 1, D_total)))
        elif M_dist == "uniform":
            spread = getattr(args, "M_spread", base_M)
            M_j = np.random.uniform(base_M - spread, base_M + spread)
            M_j = int(np.round(np.clip(M_j, 1, D_total)))
        elif M_dist in ["spatial_gaussian", "spatial_inversegaussian"]:
            M_j = M_vals[j]
            M_j = max(1, min(M_j, D_total)) 
        else:
            raise ValueError(f"Unknown M distribution: {M_dist}")
        base_window = D_total
        window_size = int(round(base_window + gamma_j * (N_in - base_window)))
        window_size = max(D_total, min(window_size, N_in))

        connections = pick_connections_for_output_node(
            j, N_in, N_out, sparsity, D_total, M_j, gamma_j, args
        )
        for i in connections:
            adj[i, j] = 1
    degrees = adj.sum(axis=0)
    # print("Degree stats:", np.min(degrees), np.max(degrees), np.mean(degrees))
    # --- REWIRING ---
    if args.random_rewiring != 0:
        total_edges = int(np.sum(adj))
        randomness = np.random.binomial(1, p=args.random_rewiring, size=total_edges)
        count = 0
        for i in range(N_in):
            for j in range(N_out):
                if adj[i, j] == 1:
                    if randomness[count] == 1:
                        adj[i, j] = 0 
                    count += 1

        removed_edges = total_edges - int(np.sum(adj))
        nrAdd = 0
        while nrAdd < removed_edges:
            i_rand = np.random.randint(0, N_in)
            j_rand = np.random.randint(0, N_out)
            if adj[i_rand, j_rand] == 0:
                adj[i_rand, j_rand] = 1
                nrAdd += 1
        # print("After rewiring, total edges:", np.sum(adj), "removed:", removed_edges)

    if w.shape[0] != N_in:
        return torch.LongTensor(adj).to(w.device).t()
    return torch.LongTensor(adj).to(w.device)



def create_dnm_connectivity(
    num_inputs,
    num_outputs,
    sparsity,
    num_dendrites,
    dendrite_dist,
    gamma,
    gamma_dist,
    synaptic_dist,
    degree_dist,
    M=4,
    show_sparsity=False,  # New parameter to control sparsity display
    **kwargs
):
    # Dummy weight matrix just for dimension
    w = np.zeros((num_inputs, num_outputs))
    # Build args namespace for compatibility
    args = SimpleNamespace(
        M=M,
        M_dist=dendrite_dist,
        degree_dist=degree_dist,
        synaptic_dist=synaptic_dist,
        gamma=gamma,
        gamma_dist=gamma_dist,
        random_rewiring=0,
        degree_std=2.0,  # Add this default value
        M_std = 0.14,
        gamma_std = 0.05,
        synaptic_std = 0.1
    )
    # This calls your scheduler and returns numpy mask
    mask = create_dendritic_sparse_scheduler(sparsity, w, args)
    adj_np = mask.cpu().numpy() if hasattr(mask, "cpu") else mask

    # Only show sparsity if requested
    if show_sparsity:
        actual_sparsity = 1.0 - (adj_np.sum() / adj_np.size)
        st.info(f"**Exact sparsity:** {actual_sparsity:.6f}")

    # Ensure output is numpy
    if hasattr(mask, "cpu"):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = np.array(adj_np)
    return mask_np

def create_network_graph(masks, layer_sizes):
    """Create a network graph from the connectivity matrices"""
    G = nx.DiGraph()
    node_offsets = np.cumsum([0] + layer_sizes)

    # Add nodes with layer attribute
    for layer_idx, size in enumerate(layer_sizes):
        for node_idx in range(node_offsets[layer_idx], node_offsets[layer_idx] + size):
            G.add_node(node_idx, layer=layer_idx)

    # Add edges between layers
    for layer_idx, mask in enumerate(masks):
        sources, targets = np.where(mask == 1)
        for src, tgt in zip(sources, targets):
            source_node = node_offsets[layer_idx] + src
            target_node = node_offsets[layer_idx+1] + tgt
            G.add_edge(source_node, target_node)

    return G

def plot_network_graph(G, layer_sizes, ax):
    """Plot the network graph with layers arranged horizontally"""
    # Use a layered layout
    pos = {}
    layer_gap = 1.0
    node_gap = 0.05

    for layer_idx, size in enumerate(layer_sizes):
        y_positions = np.linspace(0, 1, size)
        for i in range(size):
            node_idx = np.sum(layer_sizes[:layer_idx]) + i
            pos[node_idx] = (layer_idx * layer_gap, y_positions[i])

    # Draw edges
    for edge in G.edges():
        source, target = edge
        xs, ys = pos[source]
        xt, yt = pos[target]
        ax.plot([xs, xt], [ys, yt], 'gray', alpha=0.1, linewidth=0.5)

    # Draw nodes
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Different colors per layer
    for layer_idx in range(len(layer_sizes)):
        start = np.sum(layer_sizes[:layer_idx])
        end = start + layer_sizes[layer_idx]
        layer_nodes = range(start, end)
        x = [pos[node][0] for node in layer_nodes]
        y = [pos[node][1] for node in layer_nodes]
        ax.scatter(x, y, s=10, color=colors[layer_idx], label=f'Layer {layer_idx}')

    ax.axis('off')
    ax.set_title('Network Structure')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(layer_sizes))

# --- Streamlit UI ---
st.title("Dendritic Network Model Visualization")

# Define fixed layer sizes
layer_sizes = [784, 1568, 1568, 1568]

# Sidebar parameters
st.sidebar.header("Network Parameters")
sparsity = st.sidebar.slider("Sparsity", 0.5, 0.99, 0.9)
num_dendrites = st.sidebar.slider("Avg Dendrites (M)", 1, 31, 15, step=2)  # FIX 3
gamma = st.sidebar.slider("Avg Window Size (gamma)", 0.0, 1.0, 0.5)  # FIX 4

st.sidebar.subheader("Distributions")
dendrite_dist = st.sidebar.selectbox("Dendrite Distribution", [
    "fixed", "gaussian", "uniform", "spatial_gaussian", "spatial_inversegaussian"
])
gamma_dist = st.sidebar.selectbox("Window Size Distribution", [
    "fixed", "gaussian", "uniform", "spatial_gaussian", "spatial_inversegaussian"
])
degree_dist = st.sidebar.selectbox("Degree Distribution", [
    "fixed", "uniform", "gaussian", "spatial_gaussian", "spatial_inversegaussian"
])
synaptic_dist = st.sidebar.selectbox("Synaptic Distribution", [
    "fixed", "uniform", "gaussian", "spatial_gaussian", "spatial_inversegaussian"
])

if st.button("Generate Network"):
    with st.spinner("Creating network..."):
        # Generate masks for all layers
        masks = []
        sparsity_shown = False  # Track if sparsity has been shown
        for i in range(len(layer_sizes)-1):
            conn_matrix = create_dnm_connectivity(
                layer_sizes[i], layer_sizes[i+1], sparsity, num_dendrites,
                dendrite_dist, gamma, gamma_dist,
                synaptic_dist, degree_dist, num_dendrites
            )
            masks.append(conn_matrix)

            # Only show sparsity once
            if not sparsity_shown:
                actual_sparsity = 1.0 - (np.sum(conn_matrix) / (conn_matrix.shape[0] * conn_matrix.shape[1]))
                st.info(f"**Exact sparsity:** {actual_sparsity:.6f}")
                sparsity_shown = True

        # Plot as heatmap (first layer only)
        st.subheader("Adjacency Matrix") 
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(masks[0], aspect='auto', cmap='Blues')
        ax.set_xlabel(f"Output Neurons (Layer 1: {layer_sizes[1]})")
        ax.set_ylabel(f"Input Neurons (Layer 0: {layer_sizes[0]})")
        st.pyplot(fig)

        # Create network graph
        G = create_network_graph(masks, layer_sizes)
