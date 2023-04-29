import os
import logging

import numpy as np
import torch
import torch.nn.functional as f
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence, Independent

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        
def get_logger(
    logpath, filepath, package_files=[], displaying=True, saving=True, debug=False
):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="w")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger



def compute_masked_likelihood(mu, data, mask, likelihood_func):
    # Compute the likelihood per patient and per attribute so that we don't priorize patients with more measurements
    n_traj_samples, n_traj, n_timepoints, n_dims = data.size()

    res = []
    for i in range(n_traj_samples):
        for k in range(n_traj):
            for j in range(n_dims):
                data_masked = torch.masked_select(
                    data[i, k, :, j], mask[i, k, :, j].bool()
                )

                # assert(torch.sum(data_masked == 0.) < 10)

                mu_masked = torch.masked_select(mu[i, k, :, j], mask[i, k, :, j].bool())
                log_prob = likelihood_func(mu_masked, data_masked, indices=(i, k, j))
                res.append(log_prob)
    # shape: [n_traj*n_traj_samples, 1]

    res = torch.stack(res, 0).cuda()
    res = res.reshape((n_traj_samples, n_traj, n_dims))
    # Take mean over the number of dimensions
    res = torch.mean(res, -1)  # !!!!!!!!!!! changed from sum to mean
    res = res.transpose(0, 1)
    return res


def mse(mu, data, indices=None):
    n_data_points = mu.size()[-1]

    if n_data_points > 0:
        mse = nn.MSELoss()(mu, data)
    else:
        mse = torch.zeros([1]).cuda().squeeze()
    return mse


def compute_mse(mu, data, mask=None):
    if len(mu.size()) == 3:
        mu = mu.unsqueeze(0)

    if len(data.size()) == 3:
        data = data.unsqueeze(0)

    n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()

    if mask is None:
        mu_flat = mu.reshape(n_traj_samples * n_traj, n_timepoints * n_dims)
        n_traj_samples, n_traj, n_timepoints, n_dims = data.size()
        data_flat = data.reshape(n_traj_samples * n_traj, n_timepoints * n_dims)
        res = mse(mu_flat, data_flat)
    else:
        res = compute_masked_likelihood(mu, data, mask, mse)
    return res


def gaussian_log_likelihood(mu_2d, data_2d, obsrv_std, indices=None):
    n_data_points = mu_2d.size()[-1]

    if n_data_points > 0:
        gaussian = Independent(
            Normal(loc=mu_2d, scale=obsrv_std.repeat(n_data_points)), 1
        )
        log_prob = gaussian.log_prob(data_2d)
        log_prob = log_prob / n_data_points
    else:
        log_prob = torch.zeros([1]).cuda().squeeze()
    return log_prob


def masked_gaussian_log_density(mu, data, obsrv_std, mask=None):
    # these cases are for plotting through plot_estim_density
    if len(mu.size()) == 3:
        # add additional dimension for gp samples
        mu = mu.unsqueeze(0)

    if len(data.size()) == 2:
        # add additional dimension for gp samples and time step
        data = data.unsqueeze(0).unsqueeze(2)
    elif len(data.size()) == 3:
        # add additional dimension for gp samples
        data = data.unsqueeze(0)

    n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()

    assert data.size()[-1] == n_dims

    # Shape after permutation: [n_traj, n_traj_samples, n_timepoints, n_dims]
    if mask is None:
        mu_flat = mu.reshape(n_traj_samples * n_traj, n_timepoints * n_dims)
        n_traj_samples, n_traj, n_timepoints, n_dims = data.size()
        data_flat = data.reshape(n_traj_samples * n_traj, n_timepoints * n_dims)

        res = gaussian_log_likelihood(mu_flat, data_flat, obsrv_std)
        res = res.reshape(n_traj_samples, n_traj).transpose(0, 1)

    else:
        # Compute the likelihood per patient so that we don't priorize patients with more measurements
        func = lambda mu, data, indices: gaussian_log_likelihood(
            mu, data, obsrv_std=obsrv_std, indices=indices
        )
        res = compute_masked_likelihood(mu, data, mask, func)
    return res


def get_gaussian_likelihood(truth, pred_y, mask=None):
    # pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
    # truth shape  [n_traj, n_tp, n_dim]

    obsrv_std = torch.Tensor([0.01]).cuda()
    if mask is not None:
        mask = mask.repeat(pred_y.size(0), 1, 1, 1)

    # Compute likelihood of the data under the predictions
    log_density_data = masked_gaussian_log_density(pred_y, truth, obsrv_std, mask=mask)
    log_density_data = log_density_data.permute(1, 0)

    # Compute the total density
    # Take mean over n_traj_samples
    log_density = torch.mean(log_density_data, 0)

    # shape: [n_traj]
    return log_density


def get_mse(truth, pred_y, mask=None):
    # pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
    # truth shape  [n_traj, n_tp, n_dim]
    if mask is not None:
        mask = mask.repeat(pred_y.size(0), 1, 1, 1)

    # Compute likelihood of the data under the predictions
    log_density_data = compute_mse(pred_y, truth, mask=mask)
    # shape: [1]
    return torch.mean(log_density_data)


