import torch
import json

class InstanceParameterStore:
    def __init__(self, base_sigma=2.0):
        self.base_sigma = base_sigma
        self.sigma_dict = {}      # Current σ for each instance
        self.mean_dict = {}       # Current μ for each instance
        self.sigma_history = {}   # History of σ for each instance
        self.mean_history = {}    # History of μ for each instance

    def initialize(self, face_ids, initial_labels):
        for fid, label in zip(face_ids, initial_labels):
            self.sigma_dict[fid] = self.base_sigma
            self.mean_dict[fid] = label
            self.sigma_history[fid] = [self.base_sigma]  # Initialize history
            self.mean_history[fid] = [label]             # Initialize history

    def update(self, face_ids, new_sigmas, new_means):
        for fid, sigma, mean in zip(face_ids, new_sigmas, new_means):
            self.sigma_dict[fid] = sigma
            self.mean_dict[fid] = mean
            # Append new values to history
            self.sigma_history[fid].append(sigma)
            self.mean_history[fid].append(mean)

    def get_params(self, face_ids):
        sigmas = [self.sigma_dict[fid] for fid in face_ids]
        means = [self.mean_dict[fid] for fid in face_ids]
        return torch.tensor(sigmas), torch.tensor(means)

    def get_history(self, face_ids):
        """Retrieve full history for specific instances."""
        return {
            fid: {
                "sigma": self.sigma_history[fid],
                "mean": self.mean_history[fid]
            } for fid in face_ids
        }
    
    def save_sigma_history(self, filename: str):
        """Save σ history to a JSON file."""
        data = {
            str(fid): [float(x) for x in hist]  # Convert all values to Python floats
            for fid, hist in self.sigma_history.items()
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

    def save_mean_history(self, filename: str):
        """Save μ history to a JSON file."""
        data = {
            str(fid): [float(x) for x in hist]  # Convert all values to Python floats
            for fid, hist in self.mean_history.items()
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)