#!/usr/bin/env python3
"""
Native PyTorch PPO implementation optimized for GPU training.
This implementation maximizes GPU utilization for RTX 3090.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from collections import deque
import time

logger = logging.getLogger(__name__)


class PolicyNetwork(nn.Module):
    """Actor-Critic network optimized for GPU training."""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], action_size: int):
        super().__init__()
        
        # Build network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size),  # Layer normalization for stability
                nn.Dropout(0.1)  # Dropout for regularization
            ])
            prev_size = hidden_size
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(prev_size, action_size),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value function)
        self.critic = nn.Linear(prev_size, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=0.01)
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network."""
        shared = self.shared_layers(x)
        action_probs = self.actor(shared)
        value = self.critic(shared)
        return action_probs, value
    
    def get_action(self, state, deterministic=False):
        """Get action from the policy."""
        with torch.no_grad():
            action_probs, value = self.forward(state)
            
            if deterministic:
                action = torch.argmax(action_probs, dim=-1)
            else:
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
            
            log_prob = torch.log(action_probs.gather(1, action.unsqueeze(-1))).squeeze(-1)
            
        return action, log_prob, value.squeeze(-1)


class PPOBuffer:
    """Experience buffer for PPO training."""
    
    def __init__(self, capacity: int, obs_dim: int, device: str):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.device = device
        self.reset()
    
    def reset(self):
        """Reset the buffer."""
        self.states = torch.zeros((self.capacity, self.obs_dim), device=self.device)
        self.actions = torch.zeros(self.capacity, dtype=torch.long, device=self.device)
        self.rewards = torch.zeros(self.capacity, device=self.device)
        self.values = torch.zeros(self.capacity, device=self.device)
        self.log_probs = torch.zeros(self.capacity, device=self.device)
        self.dones = torch.zeros(self.capacity, dtype=torch.bool, device=self.device)
        self.ptr = 0
        self.size = 0
    
    def store(self, state, action, reward, value, log_prob, done):
        """Store a transition."""
        self.states[self.ptr] = torch.tensor(state, device=self.device, dtype=torch.float32)
        self.actions[self.ptr] = torch.tensor(action, device=self.device, dtype=torch.long)
        self.rewards[self.ptr] = torch.tensor(reward, device=self.device, dtype=torch.float32)
        self.values[self.ptr] = torch.tensor(value, device=self.device, dtype=torch.float32)
        self.log_probs[self.ptr] = torch.tensor(log_prob, device=self.device, dtype=torch.float32)
        self.dones[self.ptr] = torch.tensor(done, device=self.device, dtype=torch.bool)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def get_batch(self, batch_size: int):
        """Get a random batch of experiences."""
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.values[indices],
            self.log_probs[indices],
            self.dones[indices]
        )
    
    def compute_advantages(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        """Compute GAE advantages."""
        if self.size == 0:
            return torch.tensor([]), torch.tensor([])

        # Only work with the filled portion of the buffer
        rewards = self.rewards[:self.size]
        values = self.values[:self.size]
        dones = self.dones[:self.size]

        advantages = torch.zeros_like(rewards)
        last_advantage = 0

        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + gamma * next_value * (1 - dones[t].float()) - values[t]
            advantages[t] = delta + gamma * gae_lambda * (1 - dones[t].float()) * last_advantage
            last_advantage = advantages[t]

        returns = advantages + values

        # Normalize advantages
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns


class NativePPOAgent:
    """Native PyTorch PPO agent optimized for GPU training."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: List[int] = [256, 256, 128],
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "cuda",
        use_mixed_precision: bool = True
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.use_mixed_precision = use_mixed_precision
        
        # Create policy network
        self.policy = PolicyNetwork(obs_dim, hidden_sizes, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Mixed precision training
        if use_mixed_precision and device.startswith('cuda'):
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Mixed precision training enabled")
        else:
            self.scaler = None
        
        # Training metrics
        self.training_stats = {
            'policy_loss': deque(maxlen=100),
            'value_loss': deque(maxlen=100),
            'entropy': deque(maxlen=100),
            'kl_divergence': deque(maxlen=100)
        }
    
    def get_action(self, state, deterministic=False):
        """Get action from the policy."""
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, log_prob, value = self.policy.get_action(state_tensor, deterministic)
        return action.item(), log_prob.item(), value.item()
    
    def update(self, buffer: PPOBuffer, n_epochs: int = 10, batch_size: int = 2048):
        """Update the policy using PPO."""
        
        # Compute advantages
        advantages, returns = buffer.compute_advantages(self.gamma, self.gae_lambda)
        
        # Get all data
        states = buffer.states[:buffer.size]
        actions = buffer.actions[:buffer.size]
        old_log_probs = buffer.log_probs[:buffer.size]
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        n_updates = 0
        
        for epoch in range(n_epochs):
            # Shuffle data
            indices = torch.randperm(buffer.size, device=self.device)
            
            for start in range(0, buffer.size, batch_size):
                end = min(start + batch_size, buffer.size)
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Forward pass
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        action_probs, values = self.policy(batch_states)
                        
                        # Calculate losses
                        policy_loss, value_loss, entropy_loss, kl_div = self._calculate_losses(
                            action_probs, values, batch_actions, batch_advantages, 
                            batch_returns, batch_old_log_probs
                        )
                        
                        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
                    
                    # Backward pass with mixed precision
                    self.optimizer.zero_grad()
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    action_probs, values = self.policy(batch_states)
                    
                    # Calculate losses
                    policy_loss, value_loss, entropy_loss, kl_div = self._calculate_losses(
                        action_probs, values, batch_actions, batch_advantages, 
                        batch_returns, batch_old_log_probs
                    )
                    
                    total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
                    
                    # Backward pass
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                # Track statistics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_loss.item()
                total_kl += kl_div.item()
                n_updates += 1
        
        # Update training stats
        self.training_stats['policy_loss'].append(total_policy_loss / n_updates)
        self.training_stats['value_loss'].append(total_value_loss / n_updates)
        self.training_stats['entropy'].append(total_entropy / n_updates)
        self.training_stats['kl_divergence'].append(total_kl / n_updates)
        
        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'kl_divergence': total_kl / n_updates
        }
    
    def _calculate_losses(self, action_probs, values, actions, advantages, returns, old_log_probs):
        """Calculate PPO losses."""
        # Current log probabilities
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(-1))).squeeze(-1)
        
        # Ratio for PPO
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Policy loss with clipping
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(values.squeeze(-1), returns)
        
        # Entropy loss
        entropy_loss = -(action_probs * torch.log(action_probs + 1e-8)).sum(-1).mean()
        
        # KL divergence for monitoring
        kl_div = (old_log_probs - log_probs).mean()
        
        return policy_loss, value_loss, entropy_loss, kl_div
    
    def save(self, path: str):
        """Save the model."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': dict(self.training_stats)
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load the model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'training_stats' in checkpoint:
            self.training_stats = checkpoint['training_stats']
        logger.info(f"Model loaded from {path}")
    
    def get_stats(self):
        """Get training statistics."""
        return {
            'avg_policy_loss': np.mean(self.training_stats['policy_loss']) if self.training_stats['policy_loss'] else 0,
            'avg_value_loss': np.mean(self.training_stats['value_loss']) if self.training_stats['value_loss'] else 0,
            'avg_entropy': np.mean(self.training_stats['entropy']) if self.training_stats['entropy'] else 0,
            'avg_kl_divergence': np.mean(self.training_stats['kl_divergence']) if self.training_stats['kl_divergence'] else 0
        }
