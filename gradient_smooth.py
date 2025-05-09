import torch
import numpy as np

import os
import utils

sigmoid = lambda x: 1 / (1 + np.exp(-x))

def get_smoother(args):
    return GradientSmoother(
            gamma=args.smooth_gamma, k=args.smooth_k, 
            alpha=args.smooth_alpha, mult=args.smooth_mult,
            accumulate=args.smooth_accumulate, direction=args.smooth_direction, 
            same_nb_weight=args.smooth_same_nb_weight, rescale_grad=args.smooth_rescale_grad
        )

class GradientSmoother:

    def __init__(self, gamma=1, k=1, alpha=np.log(2), mult=1, equal_avg=False, direction='both', 
                accumulate=True, same_nb_weight=False, alpha_scheduler=None, gamma_scheduler=None, 
                end_alpha=None, alpha_start_epoch=0, alpha_end_epoch=None, 
                end_gamma=None, gamma_start_epoch=0, gamma_end_epoch=None,
                rescale_grad=False):
        """
        alpha_scheduler: interpolates alpha between `alpha` and `end_alpha`, with final being achieved at `end_alpha_epoch`
        - `sigmoid`: use a sigmoid function to interpolate
        """
        self.gamma = gamma
        self.k = k
        self.alpha = alpha
        self.mult = mult
        self.equal_avg = equal_avg
        self.direction = direction 
        self.accumulate = accumulate
        self.backwards = (direction == 'right')
        self.same_nb_weight = same_nb_weight
        self.rescale_grad = rescale_grad

        self.epoch = None # must set epoch later to use scheduler
        self.alpha_scheduler = alpha_scheduler
        self.gamma_scheduler = gamma_scheduler

        if self.alpha_scheduler is not None:
            self.alpha_scheduler = alpha_scheduler
            self.alpha_start_epoch = alpha_start_epoch
            assert alpha_end_epoch is not None, "Must specify end alpha if scheduling"
            self.alpha_end_epoch = alpha_end_epoch
            self.end_alpha = self.alpha if end_alpha is None else end_alpha

        if self.gamma_scheduler is not None:
            self.gamma_scheduler = gamma_scheduler
            self.gamma_start_epoch = gamma_start_epoch
            assert gamma_end_epoch is not None, "Must specify end gamma if scheduling"
            self.gamma_end_epoch = gamma_end_epoch
            self.end_gamma = self.gamma if end_gamma is None else end_gamma
 

    def __call__(self, residual_blocks, device):
        """Compute convex combination of gradients based on k neighboring blocks with decaying weights."""
        k_neighbors = self.k
        num_blocks = len(residual_blocks)

        if self.backwards:
            block_enum = reversed(list(enumerate(residual_blocks)))
        else:
            block_enum = enumerate(residual_blocks)

        for i, block in block_enum:
            if self.direction == 'left':
                neighbor_indices = list(range(max(0, i - k_neighbors), i+1))
            elif self.direction == 'right':
                neighbor_indices = list(range(i, min(num_blocks, i + k_neighbors + 1)))
            else:
                neighbor_indices = list(range(max(0, i - k_neighbors), min(num_blocks, i + k_neighbors + 1)))

            weights = self._get_weights(i, neighbor_indices, device)
            
            for param_idx, param in enumerate(block.parameters()):
                if param.grad is not None:
                    norm_og = torch.norm(param.grad)
                    avg_grad = torch.zeros_like(param.grad)

                    for j, weight in zip(neighbor_indices, weights):
                        neighbor_params = list(residual_blocks[j].parameters())

                        if param_idx < len(neighbor_params):
                            neighbor_param = neighbor_params[param_idx]

                            if neighbor_param.grad is not None:
                                avg_grad += weight * neighbor_param.grad
                    
                    if self.rescale_grad:
                        avg_grad *= norm_og / torch.norm(avg_grad)

                    param.grad.copy_(avg_grad)

    def _get_weights(self, i, neighbor_indices, device):
        mult = self.mult
        gamma = self.gamma
        alpha = self.alpha

        if self.alpha_scheduler is not None:
            assert self.epoch is not None, "Must set epoch with `step()` if using alpha scheduler"
            alpha = self.alpha_scheduler(self.alpha, self.epoch)
        
        if self.equal_avg:
            decay_values = torch.tensor([0.0 if j == i else 1.0 for j in neighbor_indices], device=device)
            total_decay = decay_values.sum()

            if self.same_nb_weight:
                weights = torch.tensor(
                    [gamma if j == i else mult * gamma / total_decay for j in neighbor_indices], 
                    device=device)
            else:
                weights = torch.tensor(
                    [gamma if j == i else mult * (1 - gamma) / total_decay for j in neighbor_indices], 
                    device=device)

        else:
            decay_values = torch.tensor([0.0 if j == i else np.exp(-alpha * abs(i - j)) for j in neighbor_indices], device=device)
            total_decay = decay_values.sum()

            if self.same_nb_weight:
                weights = torch.tensor(
                    [gamma if j == i else mult * gamma * np.exp(-alpha * abs(i - j)) / total_decay for j in neighbor_indices], 
                    device=device)
            else:
                weights = torch.tensor(
                    [gamma if j == i else mult * (1 - gamma) * np.exp(-alpha * abs(i - j)) / total_decay for j in neighbor_indices], 
                    device=device)
        
        return weights

    def set_epoch(self, epoch):
        self.epoch = epoch

    def step(self):
        if self.epoch is not None:
            self.epoch += 1

        self.alpha = self.get_alpha()
        self.gamma = self.get_gamma()

    def get_alpha(self, epoch=None):
        if self.alpha_scheduler is None:
            return self.alpha

        epoch = self.epoch if epoch is None else epoch
        assert epoch is not None, "Cannot get alpha: passed epoch or smoother epoch is None"

        if self.alpha_scheduler == 'sigmoid':
            alpha = self._sigmoid_schedule(epoch, self.alpha_start_epoch, self.alpha_end_epoch, self.alpha, self.end_alpha)
        elif self.alpha_scheduler == 'linear':
            alpha = self._linear_schedule(epoch, self.alpha_start_epoch, self.alpha_end_epoch, self.alpha, self.end_alpha)
        else:
            raise NotImplementedError("scheduler is not implemented")
        return alpha

    def get_gamma(self, epoch=None):
        if self.gamma_scheduler is None:
            return self.gamma

        epoch = self.epoch if epoch is None else epoch
        assert epoch is not None, "Cannot get gamma: passed epoch or smoother epoch is None"
        
        if self.gamma_scheduler == 'sigmoid':
            gamma = self._sigmoid_schedule(epoch, self.gamma_start_epoch, self.gamma_end_epoch, self.gamma, self.end_gamma)
        elif self.gamma_scheduler == 'linear':
            gamma = self._linear_schedule(epoch, self.gamma_start_epoch, self.gamma_end_epoch, self.gamma, self.end_gamma)
        else:
            raise NotImplementedError("scheduler is not implemented")

        return gamma

    def _sigmoid_schedule(self, epoch, start_epoch, end_epoch, start_val, end_val):
        x = epoch - (end_epoch + start_epoch) / 2
        k = 6 / (end_epoch - start_epoch)
        val = start_val + (end_val - start_val) * sigmoid(k*x)
        return val

    def _linear_schedule(self, epoch, start_epoch, end_epoch, start_val, end_val):
        if epoch <= start_epoch:
            val = start_val
        elif epoch >= end_epoch:
            val = end_val
        else:
            slope = (end_val - start_val) / (end_epoch - start_epoch)
            val = start_val + slope * (epoch - start_epoch)
        return val


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
            #print(f"[Rank {utils.get_rank()}] avg before {param.grad.mean().item()}")
            if param.grad is not None:
                avg_grad = torch.zeros_like(param.grad)

                for j, weight in zip(neighbor_indices, weights):
                    neighbor_params = list(residual_blocks[j].parameters())

                    if param_idx < len(neighbor_params):
                        neighbor_param = neighbor_params[param_idx]

                        if neighbor_param.grad is not None:
                            avg_grad += weight * neighbor_param.grad
                
                assert not torch.allclose(param.grad, avg_grad), "Error: param.grad and avg_grad are the same - smoothing not working"
                param.grad.copy_(avg_grad)
            #print(f"[Rank {utils.get_rank()}] avg after {param.grad.mean().item()}")
            

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
