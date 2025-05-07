import torch
import numpy as np

class GradientSmoother:

    def __init__(self):
        pass

    def __call__(self, x):
        pass


# TODO: Combine the two functions into a single one
def compute_neighbor_averaged_gradients_accumulate(residual_blocks, k_neighbors, device, gamma=0.5, alpha=1, mult=-1, equal_avg=False, direction='both',
                                                   backwards=False, same_nb_weight=False):
    """Compute convex combination of gradients based on k neighboring blocks with decaying weights."""
    num_blocks = len(residual_blocks)
    if backwards:
        block_enum = reversed(list(enumerate(residual_blocks)))
    else:
        block_enum = enumerate(residual_blocks)
    for i, block in block_enum:
        if direction == 'left':
            neighbor_indices = [j for j in range(max(0, i - k_neighbors), i+1)]
        elif direction == 'right':
            neighbor_indices = [j for j in range(i, min(num_blocks, i + k_neighbors + 1))]
        else:
            neighbor_indices = [j for j in range(max(0, i - k_neighbors), min(num_blocks, i + k_neighbors + 1))]

        #weights = torch.tensor([(-1) ** (i-j) * 1.0 / (abs(i - j) + 1) for j in neighbor_indices], device=device) if not equal_avg else torch.ones(len(neighbor_indices), device=device)
        if equal_avg:
            # weights = torch.ones(len(neighbor_indices), device=device)
            # weights /= weights.sum()
            decay_values = torch.tensor([
                  0.0 if j == i else 1.0
                  for j in neighbor_indices
              ], device=device)
            total_decay = decay_values.sum()

            # weights = torch.tensor([
            #     gamma if j == i else mult * (1 - gamma) * np.exp(-alpha * abs(i - j)) / total_decay
            #     for j in neighbor_indices
            # ], device=device)
            if same_nb_weight:
                weights = torch.tensor([
                    gamma if j == i else mult * gamma / total_decay
                    for j in neighbor_indices
                ], device=device)
            else:
                weights = torch.tensor([
                    gamma if j == i else mult * (1 - gamma) / total_decay
                    for j in neighbor_indices
                ], device=device)
        else:
            # decay_values = torch.tensor([
            #       0.0 if j == i else np.exp(-alpha * abs(i - j))
            #       for j in neighbor_indices
            #   ], device=device)
            
            decay_values = torch.tensor([
                  0.0 if j == i else 2 ** (-abs(i - j))
                  for j in neighbor_indices
              ], device=device)

            total_decay = decay_values.sum()

            # weights = torch.tensor([
            #     gamma if j == i else mult * (1 - gamma) * np.exp(-alpha * abs(i - j)) / total_decay
            #     for j in neighbor_indices
            # ], device=device)
            if same_nb_weight:
                weights = torch.tensor([
                    gamma if j == i else mult * gamma * (2 ** (-abs(i - j))) / total_decay
                    for j in neighbor_indices
                ], device=device)
            else:
                weights = torch.tensor([
                    gamma if j == i else mult * (1 - gamma) * (2 ** (-abs(i - j))) / total_decay
                    for j in neighbor_indices
                ], device=device)
            # weights = torch.tensor([
            #     gamma if j == i else mult * (1 - gamma) * 0.5
            #     for j in neighbor_indices
            # ], device=device)
            #print(weights)
            # weights = torch.tensor([1.0 / (abs(i - j) + 1) for j in neighbor_indices], device=device)
            # weights /= weights.sum()  # Normalize to ensure convex combination

        for param_idx, param in enumerate(block.parameters()):
            if param.grad is not None:
                avg_grad = torch.zeros_like(param.grad)

                for j, weight in zip(neighbor_indices, weights):
                    neighbor_params = list(residual_blocks[j].parameters())
                    if param_idx < len(neighbor_params):
                        neighbor_param = neighbor_params[param_idx]
                        if neighbor_param.grad is not None:
                            avg_grad += weight * neighbor_param.grad

                param.grad.copy_(avg_grad)

def compute_neighbor_averaged_gradients(residual_blocks, k_neighbors, device, gamma=0.5, alpha=1, mult=-1, equal_avg=False, direction='both', same_nb_weight=False):
    """Compute convex combination of gradients based on k neighboring blocks with decaying weights."""
    num_blocks = len(residual_blocks)
    original_grads = []
    for block in residual_blocks:
        original_grads.append([p.grad.clone() if p.grad is not None else None for p in block.parameters()])
    for i, block in enumerate(residual_blocks):
        if direction == 'left':
            neighbor_indices = [j for j in range(max(0, i - k_neighbors), i+1)]
        elif direction == 'right':
            neighbor_indices = [j for j in range(i, min(num_blocks, i + k_neighbors + 1))]
        else:
            neighbor_indices = [j for j in range(max(0, i - k_neighbors), min(num_blocks, i + k_neighbors + 1))]
        #neighbor_indices = [j for j in range(max(0, i - k_neighbors), min(num_blocks, i + k_neighbors + 1))]

        #weights = torch.tensor([(-1) ** (i-j) * 1.0 / (abs(i - j) + 1) for j in neighbor_indices], device=device) if not equal_avg else torch.ones(len(neighbor_indices), device=device)
        if equal_avg:
            # weights = torch.ones(len(neighbor_indices), device=device)
            # weights /= weights.sum()
            decay_values = torch.tensor([
                  0.0 if j == i else 1.0
                  for j in neighbor_indices
              ], device=device)
            total_decay = decay_values.sum()

            # weights = torch.tensor([
            #     gamma if j == i else mult * (1 - gamma) * np.exp(-alpha * abs(i - j)) / total_decay
            #     for j in neighbor_indices
            # ], device=device)
            if same_nb_weight:
                weights = torch.tensor([
                    gamma if j == i else mult * gamma / total_decay
                    for j in neighbor_indices
                ], device=device)
            else:
                weights = torch.tensor([
                    gamma if j == i else mult * (1 - gamma) / total_decay
                    for j in neighbor_indices
                ], device=device)
        else:
            # decay_values = torch.tensor([
            #       0.0 if j == i else np.exp(-alpha * abs(i - j))
            #       for j in neighbor_indices
            #   ], device=device)
            
            decay_values = torch.tensor([
                  0.0 if j == i else 2 ** (-abs(i - j))
                  for j in neighbor_indices
              ], device=device)

            total_decay = decay_values.sum()

            # weights = torch.tensor([
            #     gamma if j == i else mult * (1 - gamma) * np.exp(-alpha * abs(i - j)) / total_decay
            #     for j in neighbor_indices
            # ], device=device)
            if same_nb_weight:
                weights = torch.tensor([
                    gamma if j == i else mult * gamma * (2 ** (-abs(i - j))) / total_decay
                    for j in neighbor_indices
                ], device=device)
            else:
                weights = torch.tensor([
                    gamma if j == i else mult * (1 - gamma) * (2 ** (-abs(i - j))) / total_decay
                    for j in neighbor_indices
                ], device=device)
            # weights = torch.tensor([
            #     gamma if j == i else mult * (1 - gamma) * 0.5
            #     for j in neighbor_indices
            # ], device=device)
            #print(weights)
            # weights = torch.tensor([1.0 / (abs(i - j) + 1) for j in neighbor_indices], device=device)
            # weights /= weights.sum()  # Normalize to ensure convex combination

        for param_idx, param in enumerate(block.parameters()):
            if param.grad is not None:
                avg_grad = torch.zeros_like(param.grad)

                for j, weight in zip(neighbor_indices, weights):
                    if param_idx < len(original_grads[j]) and original_grads[j][param_idx] is not None:
                        avg_grad += weight * original_grads[j][param_idx]

                param.grad.copy_(avg_grad)

def flatten_gradients(block):
    return torch.cat([p.grad.view(-1) for p in block.parameters() if p.grad is not None])
