"""
Implements the training loop and inference on test set.

Functions:
    - :py:meth:`train_model`
    - :py:meth:`eval_model`
    - :py:meth:`dry_training`

"""

import os
import copy
import time
import json
import random
import torch
import logging
import numpy as np
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt

from typing import Dict
from tqdm import tqdm

def eval_model(model: nn.Module,
               config: dict,
               loss_matrix: torch.tensor,
               dataloader: torch.utils.data.DataLoader,
               device: str):
    """
    Evaluates a given prediction model on a Dataloader and computes the error according to the provided loss matrix.

    The function is called after the model is finished training.

    Args:
        model (nn.Module): Prediction model.
        config (dict): Configuration file parsed as a dictionary.
        loss_matrix (torch.tensor): Matrix defining the 'loss' of some metric, i.e., 0/1 loss or mae loss for all combinations of predicted/true labels.
        dataloader (torch.utils.data.DataLoader): Dataloader to evaluate the model on.

    Returns:        
        - Posterior probability of the different classes for all samples (and all prediction heads)
        - Predicted class for each sample (and each prediction head)
        - True class for each sample (and each prediction head)
        - Unique IDs of the samples
        - Assignment of the samples to training (0), validation (1) or test (2) parts
        - Mean error on the training, validation and test parts for each prediction head
    """

    # set model to evaluate mode
    model.eval()

    # prepare containers for results
    true_label = {}
    posterior = {}
    for head in config['heads']:
        true_label[head['tag']] = []
        posterior[head['tag']] = []

    folder = []
    id = []

    # loop over samples from the dataloader
    for inputs, labels, ids, folders, means, sigmas in dataloader:

        # remember ids of the samples and whether they are a training, validation or test sample (folders 0,1,2)
        id.append(ids)
        folder.append(folders)

        inputs = inputs.to(device)

        # get model prediction
        with torch.no_grad():
            heads = model(inputs)

        # for each prediction head, compute and remember the posterior
        for head, head_logits in heads.items():

            # head_labels = labels[head].to(device)
            head_labels = torch.round(means).to(device, dtype=torch.int32)

            posterior[head].append(model.get_head_posterior(head_logits, head))
            true_label[head].append(head_labels)

    # flatten the results
    id = torch.cat(id)
    folder = torch.cat(folder)
    for head in config['heads']:
        posterior[head['tag']] = torch.cat(posterior[head['tag']])
        true_label[head['tag']] = torch.cat(true_label[head['tag']])

    # get predicted label from the posteriors
    predicted_label = {head['tag']: None for head in config['heads']}
    for head in config['heads']:
        _, predicted_label[head['tag']] = torch.min(torch.matmul(
            posterior[head['tag']], loss_matrix[head['tag']]), 1)

    # prepare containers for results on train, validation and test parts
    error_tags = {head['tag']: None for head in config['heads']}
    error = {'trn': error_tags.copy(), 'val': error_tags.copy(),
             'tst': error_tags.copy()}

    # compute the mean error for the different parts
    for i, set in enumerate(['trn', 'val', 'tst']):
        index = torch.squeeze(torch.argwhere(folder == i)).to(device)
        for head in config['heads']:
            set_true_label = torch.index_select(
                true_label[head['tag']], 0, index)
            set_predicted_label = torch.index_select(
                predicted_label[head['tag']], 0, index)
            error[set][head['tag']] = torch.mean(
                loss_matrix[head['tag']][set_true_label, set_predicted_label]).cpu().detach().numpy().tolist()

    # convert results to numpy
    for head in config['heads']:
        true_label[head['tag']] = true_label[head['tag']
                                             ].cpu().detach().numpy()
        predicted_label[head['tag']
                        ] = predicted_label[head['tag']].cpu().detach().numpy()
        posterior[head['tag']] = posterior[head['tag']].cpu().detach().numpy()

    id = id.cpu().detach().numpy()
    folder = folder.cpu().detach().numpy()

    return posterior, predicted_label, true_label, id, folder, error

def update_history(history, ids, parameters):
    for i, parameter in zip(ids, parameters):
        if str(i) in history:
            history[str(i)].append(parameter)
        else:
            history[str(i)] = [parameter]
    
    return history

def save_history(history, filename: str):
    data = {
        fid: [float(x) for x in hist]  # Convert all values to Python floats
        for fid, hist in history.items()
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def train_model(model: nn.Module,
                config: dict,
                dataloaders: Dict[str, torch.utils.data.DataLoader],
                loss_matrix: torch.tensor,
                optimizer,
                device,
                output_dir):
    """
    Trains the model.

    If a checkpoint file for the given configuration can be found, the training is resumed.

    The training loop monitors validation metrics at the end of each epoch and saves the best achieved weights.

    The training logs are saved locally as well as uploaded to Weights & Biases.

    Args:
        model (nn.Module): Prediction model to train.
        config (dict): Configuration parsed as dictionary.
        dataloaders (Dict[str, torch.utils.data.DataLoader]): Dictionary of 'trn', 'val' and 'test' dataloaders.
        loss_matrix (torch.tensor): Matrix defining the 'loss' of some metric, i.e., 0/1 loss or mae loss for all combinations of predicted/true labels.
    """

    since = time.time()
    use_amp = ('use_amp' in config["optimizer"].keys()) and (
        config['optimizer']['use_amp'])

    # visualize validation and training images
    # dry_training(config, dataloaders, output_dir)

    num_epochs = config['optimizer']['num_epochs']
    improve_patience = config['optimizer']['improve_patience']
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # get head names and weights (when multiple heads are trained, the loss is their combination) from config
    head_names = []
    weights = {}
    for head in config['heads']:
        head_names.append(head['tag'])
        weights[head['tag']] = head['weight']

    # find if there is a checkpoint file
    checkpoint_file = ""
    for root, subdirs, files in os.walk(output_dir):
        for filename in files:
            if "checkpoint" in filename:
                checkpoint_file = output_dir + filename

    if os.path.exists(checkpoint_file):
        # resume optimization from the last checkpoint
        checkpoint = torch.load(checkpoint_file)

        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if "scaler_state_dict" in checkpoint.keys():
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        else:
            print(f"Could not find state dictionary for GradScaler.")
            logging.info(f"Could not find state dictionary for GradScaler.")
            
        best_model_wts = copy.deepcopy(checkpoint['best_model_wts'])
        best_model_epoch = checkpoint['best_model_epoch']
        log_history = checkpoint['log_history']
        min_val_error = checkpoint['min_val_error']
        
        with open(output_dir + 'mean_history.json', 'r') as file:
            mean_history = json.load(file)

        with open(output_dir + 'sigma_history.json', 'r') as file:
            sigma_history = json.load(file)

        # resend logs to wandb
        # for log in log_history:
        #     wandb.log(log)
    else:
        # start from scratch
        start_epoch = 0
        best_model_wts = copy.deepcopy(model.state_dict())
        min_val_error = np.Inf
        best_model_epoch = 0
        log_history = []
        mean_history = {}
        sigma_history = {}
    
    # Main optimization loop
    for epoch in range(start_epoch, num_epochs):

        # stop if there is no improvement for more than improve_patience epochs
        if epoch - best_model_epoch > improve_patience:
            print(f"No improvement after {improve_patience} epochs -> halt.")
            logging.info(f"No improvement after {improve_patience} epochs -> halt.")
            break

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 40)
        logging.info('-' * 40)

        # Each epoch has a training and validation phase
        log = {}
        for phase in ['trn', 'val']:
            if phase == 'trn':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = {x: 0.0 for x in head_names}
            running_error = {x: 0.0 for x in head_names}

            # Iterate over data
            with torch.set_grad_enabled(phase == 'trn'):

                n_examples = 0
                # yoyo = time.time()
                for inputs, labels, ids, folders, means, sigmas in tqdm(dataloaders[phase], bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}'):
                    inputs = inputs.to(device)
                    
                    # Retrieve σ and μ for noisy training
                    # if param_store is not None:
                    #     sigmas, means = param_store.get_params(ids.cpu().numpy())
                    # else:
                    #     sigmas = torch.ones_like(labels['age']) * config['training']['base_sigma']
                    #     means = labels['age']

                    sigmas = sigmas.to(device)
                    means = means.to(device)

                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                        # print('before model', time.time() - yoyo)
                        heads = model(inputs)

                        batch_size = inputs.size(0)
                        loss_fce = 0.0
                        # print('before loop', time.time() - yoyo)
                        for head, head_logits in heads.items():

                            head_labels = labels[head].to(device)
                            # corrected_labels = torch.round(means).to(device=device, dtype=torch.int)

                            # evaluate loss function
                            head_loss = model.get_head_loss(
                                head_logits, head_labels, head, sigmas, means)
                            loss_fce += weights[head] * head_loss

                            # compute error metric
                            head_posterior = model.get_head_posterior(
                                head_logits, head)
                            _, predicted_labels = torch.min(
                                torch.matmul(head_posterior, loss_matrix[head]), 1)
                            head_err = torch.mean(
                                loss_matrix[head][head_labels.data, predicted_labels])

                            # update running average
                            running_loss[head] = n_examples*running_loss[head]/(
                                n_examples+batch_size) + batch_size*head_loss/(n_examples+batch_size)
                            running_error[head] = n_examples*running_error[head]/(
                                n_examples+batch_size) + batch_size*head_err/(n_examples+batch_size)

                        n_examples += batch_size

                        # print('after loop', time.time() - yoyo)
                    # backward + optimize only if in training phase
                    # print('before loss', time.time() - yoyo)
                                            # Update σ and μ for noisy training
                    # Evaluate best model to get predicted labels before updating means and sigmas
                    # if config['training']['mode'] == 'noisy' and phase == 'val' and epoch > 9:
                    #     best_model = copy.deepcopy(model)
                    #     best_model.load_state_dict(best_model_wts)
                    #     best_model.eval()

                    #     with torch.no_grad():
                    #         best_heads = best_model(inputs)
                    #         best_head_posterior = best_model.get_head_posterior(
                    #             best_heads[head], head)
                    #         _, best_predicted_labels = torch.min(
                    #             torch.matmul(best_head_posterior, loss_matrix[head]), 1)

                    #     with torch.no_grad():
                    #         error = torch.abs(best_predicted_labels - means)

                    #         # Vectorized computation of new_sigmas and new_means
                    #         alpha = torch.tensor([config['training']['alpha'][lbl] for lbl in labels['age']], device=device)
                    #         beta = torch.tensor([config['training']['beta'][lbl] for lbl in labels['age']], device=device)

                    #         new_sigmas = sigmas + alpha * (error - sigmas)
                    #         new_means = beta * means + (1 - beta) * best_predicted_labels

                    #         dataloaders[phase].dataset.update_parameters(
                    #             ids.cpu().numpy(), new_means.cpu().numpy(), new_sigmas.cpu().numpy())
                    mean_history = update_history(mean_history, ids.cpu().numpy(), means.cpu().numpy())
                    sigma_history = update_history(sigma_history, ids.cpu().numpy(), sigmas.cpu().numpy())
                    
                    if phase == 'trn':
                        optimizer.zero_grad()
                        scaler.scale(loss_fce).backward()
                        # print('after loss', time.time() - yoyo)
                        scaler.step(optimizer)
                        # print('after step', time.time() - yoyo)
                        scaler.update()
                        # print('after update', time.time() - yoyo)

            # compute weighted error and weighted loss
            weighted_error = 0.0
            weighted_loss = 0.0
            for head in head_names:
                running_error[head] = running_error[head].cpu(
                ).detach().numpy()
                running_loss[head] = running_loss[head].cpu(
                ).detach().numpy()
                weighted_error += weights[head]*running_error[head]
                weighted_loss += weights[head]*running_loss[head]

            # if validation error improved deep copy the model
            if phase == 'val' and weighted_error <= min_val_error:
                min_val_error = weighted_error
                best_model_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())

            # log errors and losses
            log[phase+"_loss"] = weighted_loss
            log[phase+"_error"] = weighted_error
            for head in head_names:
                log[phase+"_loss_"+head] = running_loss[head]
                log[phase+"_error_"+head] = running_error[head]

            # print loss and error value for current phase
            loss_msg = f"loss: {weighted_loss:.4f}"
            error_msg = f"error: {weighted_error:.4f}"
            for head in head_names:
                loss_msg += f" {head}_loss:{running_loss[head]:.4f}"
                error_msg += f" {head}_error:{running_error[head]:.4f}"
            print(f"[{phase} phase]")
            logging.info(f"[{phase} phase]")
            print(error_msg)
            logging.info(error_msg)
            print(loss_msg)
            logging.info(loss_msg)
            if phase == 'val':
                print(f"Best Epoch: {best_model_epoch}")
                logging.info(f"Best Epoch: {best_model_epoch}")

            # Update data_splitX.csv files
            dataloaders[phase].dataset.update_csv_file()

        # log elapsed time
        log['elapsed_minutes'] = (time.time() - since)/60

        # update wandb
        # wandb.log(log)

        # append log
        log_history.append(log)

        # remove old checkpoint and save the new one
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            'best_model_wts': best_model_wts,
            'best_model_epoch': best_model_epoch,
            'log_history': log_history,
            'min_val_error': min_val_error
        }
        checkpoint_file = output_dir + f"checkpoint_{epoch}.pth"
        torch.save(checkpoint, checkpoint_file)
        print(f"Checkpoint saved to {checkpoint_file}")
        logging.info(f"Checkpoint saved to {checkpoint_file}")
        
        if config['training']['mode'] == 'noisy':
            save_history(mean_history, output_dir + 'mean_history.json')
            save_history(sigma_history, output_dir + 'sigma_history.json')
        
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best epoch: {:d}'.format(best_model_epoch))
    logging.info('Best epoch: {:d}'.format(best_model_epoch))
    

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, log_history


def dry_training(config,
                 dataloaders,
                 output_dir):
    """
    Samples a batch of training and validation data and visualizes the images.
    This is useful to ensure that data augmentations are working as intended.
    """

    # Each epoch has a training and validation phase
    for phase in ['trn', 'val']:

        # get some random training images
        dataiter = iter(dataloaders[phase])
        images, labels, _, _, _, _ = next(dataiter)

        # create grid of images
        img_grid = np.transpose(
            torchvision.utils.make_grid(images).numpy(), (1, 2, 0))

        for i in [0, 1, 2]:
            nconst = img_grid[:, :, i].max() - img_grid[:, :, i].min()
            if nconst > 0:
                img_grid[:, :, i] = (img_grid[:, :, i] -
                                     img_grid[:, :, i].min())/nconst

        img_fname = os.path.join(output_dir, f"{phase}_inputs.jpeg")

        plt.imsave(img_fname, img_grid)
