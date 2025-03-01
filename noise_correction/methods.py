import torch
from torch.nn.functional import softmax


def default_update_function(means, sigmas, ev, indices, config):
    """Default update function for means and sigmas."""
    alpha = config['training']['alpha']
    beta = config['training']['beta']
    best_predicted_labels = torch.tensor(ev['predicted_label']['age'][indices], dtype=torch.float32)
    
    error = torch.abs(best_predicted_labels - means)
    new_sigmas = sigmas + alpha * (error - sigmas)
    new_means = beta * means + (1 - beta) * best_predicted_labels
    return new_means, new_sigmas


def default_update_function_v2(means, sigmas, ev, indices, config):
    """Default update function for means and sigmas."""
    alpha = config['training']['alpha']
    beta = config['training']['beta']
    best_predicted_labels = torch.tensor(ev['predicted_label']['age'][indices], dtype=torch.float32)
    
    error = torch.abs(best_predicted_labels - means)
    new_sigmas = sigmas + (1 - alpha) * (error - sigmas)
    new_means = beta * means + (1 - beta) * best_predicted_labels
    return new_means, new_sigmas


def stable_mean_default_update_function(means, sigmas, ev, indices, config):
    """Default update function for means and sigmas."""
    alpha = config['training']['alpha']
    beta = config['training']['beta']
    best_predicted_labels = torch.tensor(ev['predicted_label']['age'][indices], dtype=torch.float32)
    noisy_labels = torch.tensor(ev['true_label']['age'][indices], dtype=torch.long)

    best_predicted_labels = best_predicted_labels - (torch.mean(best_predicted_labels) - noisy_labels[0])
    best_predicted_labels = torch.clamp(best_predicted_labels, min=0, max=7)
    best_predicted_labels = torch.round(best_predicted_labels)
    
    error = torch.abs(best_predicted_labels - means)
    new_sigmas = sigmas + alpha * (error - sigmas)
    new_means = beta * means + (1 - beta) * best_predicted_labels
    return new_means, new_sigmas


def stable_median_default_update_function(means, sigmas, ev, indices, config):
    """Default update function for means and sigmas."""
    alpha = config['training']['alpha']
    beta = config['training']['beta']
    best_predicted_labels = torch.tensor(ev['predicted_label']['age'][indices], dtype=torch.float32)
    noisy_labels = torch.tensor(ev['true_label']['age'][indices], dtype=torch.long)

    best_predicted_labels = best_predicted_labels - (torch.median(best_predicted_labels) - noisy_labels[0])
    best_predicted_labels = torch.clamp(best_predicted_labels, min=0, max=7)
    best_predicted_labels = torch.round(best_predicted_labels)
    
    error = torch.abs(best_predicted_labels - means)
    new_sigmas = sigmas + alpha * (error - sigmas)
    new_means = beta * means + (1 - beta) * best_predicted_labels
    return new_means, new_sigmas


def confidence_adaptive_update_function(means, sigmas, ev, indices, config):
    """
    Update function for means and sigmas using Confidence-Adaptive Learning Rates (CALR).
    """
    # Extract posterior probabilities and apply softmax
    posterior_probs = torch.tensor(ev['posterior']['age'][indices], dtype=torch.float32)
    predicted_probs = softmax(posterior_probs, dim=1)  # Apply softmax to get probabilities
    
    # Compute predicted labels (class with highest probability)
    best_predicted_labels = torch.tensor(ev['predicted_label']['age'][indices], dtype=torch.float32)
    
    # Compute prediction confidence (1 - entropy)
    epsilon = 1e-10  # Small constant to avoid log(0)
    # entropy = -torch.sum(predicted_probs * torch.log(predicted_probs + epsilon), dim=1)
    # confidence = 1 - entropy
    confidence = torch.max(predicted_probs, dim=1).values
    # Clip confidence to a minimum of 0
    confidence = torch.clamp(confidence, min=0)
    
    # Extract noisy labels
    all_noisy_labels = torch.tensor(ev['true_label']['age'], dtype=torch.long)
    noisy_labels = torch.tensor(ev['true_label']['age'][indices], dtype=torch.long)
    
    # Compute class frequencies
    class_counts = torch.bincount(all_noisy_labels)
    class_frequencies = class_counts[noisy_labels] / len(all_noisy_labels)
    
    # Compute adaptive learning rates
    alpha_base = config['training']['alpha']
    beta_base = config['training']['beta']
    alpha_k = alpha_base * (confidence / (1 - torch.log(class_frequencies + epsilon)))
    beta_k = beta_base * (confidence / (1 - torch.log(class_frequencies + epsilon)))
    
    # Update means and sigmas
    error = torch.abs(best_predicted_labels - means)
    new_sigmas = sigmas + alpha_k * (error - sigmas)
    new_means = (1 - beta_k) * means + beta_k * best_predicted_labels
    
    return new_means, new_sigmas


def mean_confidence_adaptive_update_function(means, sigmas, ev, indices, config):
    """
    Update function for means and sigmas using Confidence-Adaptive Learning Rates (CALR).
    """
    # Extract posterior probabilities and apply softmax
    posterior_probs = torch.tensor(ev['posterior']['age'][indices], dtype=torch.float32)
    predicted_probs = softmax(posterior_probs, dim=1)  # Apply softmax to get probabilities
    
    # Compute predicted labels (class with highest probability)
    best_predicted_labels = torch.tensor(ev['predicted_label']['age'][indices], dtype=torch.float32)
    
    # Compute prediction confidence (1 - entropy)
    epsilon = 1e-10  # Small constant to avoid log(0)
    # entropy = -torch.sum(predicted_probs * torch.log(predicted_probs + epsilon), dim=1)
    # confidence = 1 - entropy
    confidence = torch.max(predicted_probs, dim=1).values
    # Clip confidence to a minimum of 0
    confidence = torch.clamp(confidence, min=0)
    
    # Extract noisy labels
    all_noisy_labels = torch.tensor(ev['true_label']['age'], dtype=torch.long)
    noisy_labels = torch.tensor(ev['true_label']['age'][indices], dtype=torch.long)
    
    # Compute class frequencies
    class_counts = torch.bincount(all_noisy_labels)
    class_frequencies = class_counts[noisy_labels] / len(all_noisy_labels)
    
    # Compute adaptive learning rates
    # alpha_base = config['training']['alpha']
    beta_base = config['training']['beta']
    # alpha_k = alpha_base * (confidence / (1 - torch.log(class_frequencies + epsilon)))
    beta_k = beta_base * (confidence / (1 - torch.log(class_frequencies + epsilon)))
    
    new_sigmas = sigmas.clone()

    # Update means and sigmas
    # error = torch.abs(best_predicted_labels - means)
    # new_sigmas = sigmas + alpha_k * (error - sigmas)
    new_means = (1 - beta_k) * means + beta_k * best_predicted_labels
    
    return new_means, new_sigmas


def sigma_confidence_adaptive_update_function(means, sigmas, ev, indices, config):
    """
    Update function for means and sigmas using Confidence-Adaptive Learning Rates (CALR).
    """
    # Extract posterior probabilities and apply softmax
    posterior_probs = torch.tensor(ev['posterior']['age'][indices], dtype=torch.float32)
    predicted_probs = softmax(posterior_probs, dim=1)  # Apply softmax to get probabilities
    
    # Compute predicted labels (class with highest probability)
    best_predicted_labels = torch.tensor(ev['predicted_label']['age'][indices], dtype=torch.float32)
    
    # Compute prediction confidence (1 - entropy)
    epsilon = 1e-10  # Small constant to avoid log(0)
    # entropy = -torch.sum(predicted_probs * torch.log(predicted_probs + epsilon), dim=1)
    # confidence = 1 - entropy
    confidence = torch.max(predicted_probs, dim=1).values
    # Clip confidence to a minimum of 0
    confidence = torch.clamp(confidence, min=0)
    
    # Extract noisy labels
    all_noisy_labels = torch.tensor(ev['true_label']['age'], dtype=torch.long)
    noisy_labels = torch.tensor(ev['true_label']['age'][indices], dtype=torch.long)
    
    # Compute class frequencies
    class_counts = torch.bincount(all_noisy_labels)
    class_frequencies = class_counts[noisy_labels] / len(all_noisy_labels)
    
    # Compute adaptive learning rates
    alpha_base = config['training']['alpha']
    beta_base = config['training']['beta']
    alpha_k = alpha_base * (confidence / (1 - torch.log(class_frequencies + epsilon)))
    # beta_k = beta_base * (confidence / (1 - torch.log(class_frequencies + epsilon)))

    new_means = means.clone()
    
    # Update means and sigmas
    error = torch.abs(best_predicted_labels - means)
    new_sigmas = sigmas + alpha_k * (error - sigmas)
    # new_means = (1 - beta_k) * means + beta_k * best_predicted_labels
    
    return new_means, new_sigmas


def confidence_adaptive_update_function_v5(means, sigmas, ev, indices, config):
    """
    Update function for means and sigmas using Confidence-Adaptive Learning Rates (CALR).
    """
    # Extract posterior probabilities and apply softmax
    posterior_probs = torch.tensor(ev['posterior']['age'][indices], dtype=torch.float32)
    predicted_probs = softmax(posterior_probs, dim=1)  # Apply softmax to get probabilities
    
    # Compute predicted labels (class with highest probability)
    best_predicted_labels = torch.tensor(ev['predicted_label']['age'][indices], dtype=torch.float32)
    
    # Compute prediction confidence (1 - entropy)
    epsilon = 1e-10  # Small constant to avoid log(0)
    # entropy = -torch.sum(predicted_probs * torch.log(predicted_probs + epsilon), dim=1)
    # confidence = 1 - entropy
    confidence = torch.max(predicted_probs, dim=1).values
    # Clip confidence to a minimum of 0
    confidence = torch.clamp(confidence, min=0)

    # Compute mean confidence
    median_confidence = torch.median(confidence)

    # Filter indices where confidence is higher than mean confidence
    valid_indices = confidence > median_confidence
    
    # Extract noisy labels
    all_noisy_labels = torch.tensor(ev['true_label']['age'], dtype=torch.long)
    noisy_labels = torch.tensor(ev['true_label']['age'][indices], dtype=torch.long)
    
    # Compute class frequencies
    class_counts = torch.bincount(all_noisy_labels)
    class_frequencies = class_counts[noisy_labels] / len(all_noisy_labels)
    
    # Compute adaptive learning rates
    alpha_base = config['training']['alpha']
    beta_base = config['training']['beta']
    alpha_k = alpha_base * (confidence / (1 - torch.log(class_frequencies + epsilon)))
    beta_k = beta_base * (confidence / (1 - torch.log(class_frequencies + epsilon)))

    # Only update means and sigmas for valid indices
    new_means = means.clone()
    new_sigmas = sigmas.clone()
    
    if valid_indices.any():
        # Update means and sigmas
        error = torch.abs(best_predicted_labels - means)
        new_sigmas[valid_indices] = sigmas[valid_indices] + alpha_k[valid_indices] * (error[valid_indices] - sigmas[valid_indices])
        new_means[valid_indices] = (1 - beta_k[valid_indices]) * means[valid_indices] + beta_k[valid_indices] * best_predicted_labels[valid_indices]
    
    return new_means, new_sigmas


def confidence_adaptive_update_function_v4(means, sigmas, ev, indices, config):
    """
    Update function for means and sigmas using Confidence-Adaptive Learning Rates (CALR).
    """
    # Extract posterior probabilities and apply softmax
    posterior_probs = torch.tensor(ev['posterior']['age'][indices], dtype=torch.float32)
    predicted_probs = softmax(posterior_probs, dim=1)  # Apply softmax to get probabilities
    
    # Compute predicted labels (class with highest probability)
    best_predicted_labels = torch.tensor(ev['predicted_label']['age'][indices], dtype=torch.float32)
    
    # Compute prediction confidence (1 - entropy)
    epsilon = 1e-10  # Small constant to avoid log(0)
    # entropy = -torch.sum(predicted_probs * torch.log(predicted_probs + epsilon), dim=1)
    # confidence = 1 - entropy
    confidence = torch.max(predicted_probs, dim=1).values
    # Clip confidence to a minimum of 0
    confidence = torch.clamp(confidence, min=0)

    # Compute mean confidence
    mean_confidence = torch.mean(confidence)

    # Filter indices where confidence is higher than mean confidence
    valid_indices = confidence > mean_confidence
    
    # Extract noisy labels
    all_noisy_labels = torch.tensor(ev['true_label']['age'], dtype=torch.long)
    noisy_labels = torch.tensor(ev['true_label']['age'][indices], dtype=torch.long)
    
    # Compute class frequencies
    class_counts = torch.bincount(all_noisy_labels)
    class_frequencies = class_counts[noisy_labels] / len(all_noisy_labels)
    
    # Compute adaptive learning rates
    alpha_base = config['training']['alpha']
    beta_base = config['training']['beta']
    alpha_k = alpha_base * (confidence / (1 - torch.log(class_frequencies + epsilon)))
    beta_k = beta_base * (confidence / (1 - torch.log(class_frequencies + epsilon)))

    # Only update means and sigmas for valid indices
    new_means = means.clone()
    new_sigmas = sigmas.clone()
    
    if valid_indices.any():
        # Update means and sigmas
        error = torch.abs(best_predicted_labels - means)
        new_sigmas[valid_indices] = sigmas[valid_indices] + alpha_k[valid_indices] * (error[valid_indices] - sigmas[valid_indices])
        new_means[valid_indices] = (1 - beta_k[valid_indices]) * means[valid_indices] + beta_k[valid_indices] * best_predicted_labels[valid_indices]
    
    return new_means, new_sigmas


def confidence_adaptive_update_function_v3(means, sigmas, ev, indices, config):
    """
    Update function for means and sigmas using Confidence-Adaptive Learning Rates (CALR),
    considering only elements with confidence higher than the mean confidence.
    """
    # Extract posterior probabilities and apply softmax
    posterior_probs = torch.tensor(ev['posterior']['age'][indices], dtype=torch.float32)
    predicted_probs = softmax(posterior_probs, dim=1)  # Apply softmax to get probabilities
    
    # Compute predicted labels (class with highest probability)
    best_predicted_labels = torch.tensor(ev['predicted_label']['age'][indices], dtype=torch.float32)
    
    # Compute prediction confidence (max probability per sample)
    confidence = torch.max(predicted_probs, dim=1).values
    confidence = torch.clamp(confidence, min=0)  # Ensure non-negative values
    
    # Compute mean confidence
    mean_confidence = torch.mean(confidence)
    
    # Filter indices where confidence is higher than mean confidence
    valid_indices = confidence > mean_confidence
    
    # Extract noisy labels
    all_noisy_labels = torch.tensor(ev['true_label']['age'], dtype=torch.long)
    noisy_labels = torch.tensor(ev['true_label']['age'][indices], dtype=torch.long)
    
    # Compute class frequencies
    epsilon = 1e-10  # Small constant to avoid log(0)
    class_counts = torch.bincount(all_noisy_labels, minlength=posterior_probs.shape[1])
    class_frequencies = class_counts[noisy_labels] / len(all_noisy_labels)
    
    # Compute adaptive learning rates
    alpha_base = config['training']['alpha']
    beta_base = config['training']['beta']
    alpha_k = alpha_base * (confidence / (1 - torch.log(class_frequencies + epsilon)))
    beta_k = beta_base * (confidence / (1 - torch.log(class_frequencies + epsilon)))
    
    # Only update means and sigmas for valid indices
    new_means = means.clone()
    new_sigmas = sigmas.clone()
    
    if valid_indices.any():
        error = torch.abs(best_predicted_labels - means)
        new_sigmas[valid_indices] = sigmas[valid_indices] + (1 - alpha_k[valid_indices]) * (error[valid_indices] - sigmas[valid_indices])
        new_means[valid_indices] = beta_k[valid_indices] * means[valid_indices] + (1 - beta_k[valid_indices]) * best_predicted_labels[valid_indices]
    
    return new_means, new_sigmas


def confidence_adaptive_update_function_v2(means, sigmas, ev, indices, config):
    """
    Update function for means and sigmas using Confidence-Adaptive Learning Rates (CALR).
    """
    # Extract posterior probabilities and apply softmax
    posterior_probs = torch.tensor(ev['posterior']['age'][indices], dtype=torch.float32)
    predicted_probs = softmax(posterior_probs, dim=1)  # Apply softmax to get probabilities
    
    # Compute predicted labels (class with highest probability)
    best_predicted_labels = torch.tensor(ev['predicted_label']['age'][indices], dtype=torch.float32)
    
    # Compute prediction confidence (1 - entropy)
    epsilon = 1e-10  # Small constant to avoid log(0)
    # entropy = -torch.sum(predicted_probs * torch.log(predicted_probs + epsilon), dim=1)
    # confidence = 1 - entropy
    confidence = torch.max(predicted_probs, dim=1).values
    # Clip confidence to a minimum of 0
    confidence = torch.clamp(confidence, min=0)
    
    # Extract noisy labels
    all_noisy_labels = torch.tensor(ev['true_label']['age'], dtype=torch.long)
    noisy_labels = torch.tensor(ev['true_label']['age'][indices], dtype=torch.long)
    
    # Compute class frequencies
    class_counts = torch.bincount(all_noisy_labels)
    class_frequencies = class_counts[noisy_labels] / len(all_noisy_labels)
    
    # Compute adaptive learning rates
    alpha_base = config['training']['alpha']
    beta_base = config['training']['beta']
    alpha_k = alpha_base * (confidence / (1 - torch.log(class_frequencies + epsilon)))
    beta_k = beta_base * (confidence / (1 - torch.log(class_frequencies + epsilon)))
    
    # Update means and sigmas
    error = torch.abs(best_predicted_labels - means)
    new_sigmas = sigmas + (1 - alpha_k) * (error - sigmas)
    new_means = beta_k * means + (1 - beta_k) * best_predicted_labels
    
    return new_means, new_sigmas


def flexible_confidence_adaptive_update_function(means, sigmas, ev, indices, config, rounded_means):
    """
    Update function for means and sigmas using Confidence-Adaptive Learning Rates (CALR).
    """
    # Extract posterior probabilities and apply softmax
    posterior_probs = torch.tensor(ev['posterior']['age'][indices], dtype=torch.float32)
    predicted_probs = softmax(posterior_probs, dim=1)  # Apply softmax to get probabilities
    
    # Compute predicted labels (class with highest probability)
    best_predicted_labels = torch.tensor(ev['predicted_label']['age'][indices], dtype=torch.float32)
    
    # Compute prediction confidence (1 - entropy)
    epsilon = 1e-10  # Small constant to avoid log(0)
    # entropy = -torch.sum(predicted_probs * torch.log(predicted_probs + epsilon), dim=1)
    # confidence = 1 - entropy
    confidence = torch.max(predicted_probs, dim=1).values
    # Clip confidence to a minimum of 0
    confidence = torch.clamp(confidence, min=0)

    # all_noisy_labels = torch.tensor(ev['true_label']['age'], dtype=torch.long)
    # noisy_labels = torch.tensor(ev['true_label']['age'][indices], dtype=torch.long)
    
    # Compute class frequencies
    class_counts = torch.bincount(rounded_means)
    class_frequencies = class_counts[torch.round(means).to(dtype=torch.int)] / len(rounded_means)
    
    # Compute adaptive learning rates
    alpha_base = config['training']['alpha']
    beta_base = config['training']['beta']
    alpha_k = alpha_base * (confidence / (1 - torch.log(class_frequencies + epsilon)))
    beta_k = beta_base * (confidence / (1 - torch.log(class_frequencies + epsilon)))
    
    # Update means and sigmas
    error = torch.abs(best_predicted_labels - means)
    new_sigmas = sigmas + alpha_k * (error - sigmas)
    new_means = (1 - beta_k) * means + beta_k * best_predicted_labels
    
    return new_means, new_sigmas


def quartile_confidence_adaptive_update_function(means, sigmas, ev, indices, config):
    """
    Update function for means and sigmas using Confidence-Adaptive Learning Rates (CALR).
    """
    # Extract posterior probabilities and apply softmax
    posterior_probs = torch.tensor(ev['posterior']['age'][indices], dtype=torch.float32)
    predicted_probs = softmax(posterior_probs, dim=1)  # Apply softmax to get probabilities
    
    # Compute predicted labels (class with highest probability)
    best_predicted_labels = torch.tensor(ev['predicted_label']['age'][indices], dtype=torch.float32)
    
    # Compute prediction confidence (1 - entropy)
    epsilon = 1e-10  # Small constant to avoid log(0)
    # entropy = -torch.sum(predicted_probs * torch.log(predicted_probs + epsilon), dim=1)
    # confidence = 1 - entropy
    confidence = torch.max(predicted_probs, dim=1).values
    
    # Extract noisy labels
    all_noisy_labels = torch.tensor(ev['true_label']['age'], dtype=torch.long)
    noisy_labels = torch.tensor(ev['true_label']['age'][indices], dtype=torch.long)
    
    # Compute class frequencies
    class_counts = torch.bincount(all_noisy_labels)
    class_frequencies = class_counts[noisy_labels] / len(all_noisy_labels)
    
    # Compute adaptive learning rates
    alpha_base = config['training']['alpha']
    beta_base = config['training']['beta']
    alpha_k = alpha_base * (confidence / (1 - torch.log(class_frequencies + epsilon)))
    beta_k = beta_base * (confidence / (1 - torch.log(class_frequencies + epsilon)))
    
    # Initialize new means and sigmas
    new_means = means.clone()
    new_sigmas = sigmas.clone()
    
    # Mask for data points where predicted_label != true_label
    mismatch_mask = best_predicted_labels != noisy_labels
    
    # Confidence values for mismatched predictions
    mismatch_confidence = confidence[mismatch_mask]
    
    # Compute Q3 (third quartile) of confidence for mismatched predictions
    if len(mismatch_confidence) > 0:  # Ensure there are mismatched predictions
        Q3 = torch.quantile(mismatch_confidence, 0.75)
        
        # Mask for data points where confidence > Q3
        high_confidence_mask = confidence > Q3
        
        # Update means only for high-confidence mismatched predictions
        update_mask = mismatch_mask & high_confidence_mask
        new_means[update_mask] = (1 - beta_k[update_mask]) * means[update_mask] + beta_k[update_mask] * best_predicted_labels[update_mask]

        # Print the number of data points for which the mean is updated
        num_updates = torch.sum(update_mask).item()
        print(f"Number of data points updated for class {noisy_labels[0].item()}: {num_updates}")
    else:
        # If no mismatched predictions, print 0 updates
        print(f"Number of data points updated for class {noisy_labels[0].item()}: 0")

    
    # Update sigmas for all data points in the class
    error = torch.abs(best_predicted_labels - means)
    new_sigmas = sigmas + alpha_k * (error - sigmas)    

    return new_means, new_sigmas


def stable_mean_carl_update_function(means, sigmas, pred_labels, posteriors, config, val_means, label_mask):
    """
    Update function for means and sigmas using Confidence-Adaptive Learning Rates (CALR).
    """
    # Extract posterior probabilities and apply softmax
    posterior_probs = torch.tensor(posteriors, dtype=torch.float32)
    predicted_probs = softmax(posterior_probs, dim=1)  # Apply softmax to get probabilities
    
    # Compute predicted labels (class with highest probability)
    best_predicted_labels = torch.tensor(pred_labels, dtype=torch.float32)
    rounded_means = torch.round(means).to(dtype=torch.int)
    rounded_val_means = torch.round(val_means).to(dtype=torch.int)
    
    # Compute prediction confidence (1 - entropy)
    epsilon = 1e-10  # Small constant to avoid log(0)
    # entropy = -torch.sum(predicted_probs * torch.log(predicted_probs + epsilon), dim=1)
    # confidence = 1 - entropy
    confidence = torch.max(predicted_probs, dim=1).values
    
    best_predicted_labels = best_predicted_labels - (torch.mean(best_predicted_labels) - rounded_means)
    best_predicted_labels = torch.clamp(best_predicted_labels, min=0, max=7)
    best_predicted_labels = torch.round(best_predicted_labels).to(dtype=torch.int)
    
    # Compute class frequencies
    class_counts = torch.bincount(rounded_val_means)
    class_frequencies = class_counts[rounded_means] / len(rounded_val_means)
    
    # Compute adaptive learning rates
    alpha_base = config['training']['alpha']
    beta_base = config['training']['beta']
    alpha_k = alpha_base * (confidence / (1 - torch.log(class_frequencies + epsilon)))
    beta_k = beta_base * (confidence / (1 - torch.log(class_frequencies + epsilon)))
    
    # Update means and sigmas
    error = torch.abs(best_predicted_labels - means)
    new_sigmas = sigmas + alpha_k * (error - sigmas)
    new_means = (1 - beta_k) * means + beta_k * best_predicted_labels
    
    return new_means, new_sigmas


def stable_median_carl_update_function(means, sigmas, pred_labels, posteriors, config, val_means, label_mask):
    """
    Update function for means and sigmas using Confidence-Adaptive Learning Rates (CALR).
    """
    # Extract posterior probabilities and apply softmax
    posterior_probs = torch.tensor(posteriors, dtype=torch.float32)
    predicted_probs = softmax(posterior_probs, dim=1)  # Apply softmax to get probabilities
    
    # Compute predicted labels (class with highest probability)
    best_predicted_labels = torch.tensor(pred_labels, dtype=torch.float32)
    rounded_means = torch.round(means).to(dtype=torch.int)
    rounded_val_means = torch.round(val_means).to(dtype=torch.int)
    
    # Compute prediction confidence (1 - entropy)
    epsilon = 1e-10  # Small constant to avoid log(0)
    # entropy = -torch.sum(predicted_probs * torch.log(predicted_probs + epsilon), dim=1)
    # confidence = 1 - entropy
    confidence = torch.max(predicted_probs, dim=1).values
    
    best_predicted_labels = best_predicted_labels - (torch.median(best_predicted_labels) - rounded_means)
    best_predicted_labels = torch.clamp(best_predicted_labels, min=0, max=7)
    best_predicted_labels = torch.round(best_predicted_labels).to(dtype=torch.int)
    
    # Compute class frequencies
    class_counts = torch.bincount(rounded_val_means)
    class_frequencies = class_counts[rounded_means] / len(rounded_val_means)
    
    # Compute adaptive learning rates
    alpha_base = config['training']['alpha']
    beta_base = config['training']['beta']
    alpha_k = alpha_base * (confidence / (1 - torch.log(class_frequencies + epsilon)))
    beta_k = beta_base * (confidence / (1 - torch.log(class_frequencies + epsilon)))
    
    # Update means and sigmas
    error = torch.abs(best_predicted_labels - means)
    new_sigmas = sigmas + alpha_k * (error - sigmas)
    new_means = (1 - beta_k) * means + beta_k * best_predicted_labels
    
    return new_means, new_sigmas