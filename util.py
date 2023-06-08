import os
import json
import statistics
import pickle
import torch
import numpy as np
from typing import Union
import argparse


def save_params(
        config: argparse.Namespace
) -> str:
    """
    Saves args for reproducing results
    """
    dir_ = os.getcwd()
    path = f"{dir_}/runs/{config.dataset}/{config.censor_dist}" \
           f"/{config.model}/{config.timestamp}"

    if not os.path.exists(path):
        os.makedirs(path)

    with open(f'{path}/commandline_args.txt', 'w') as f:
        json.dump(config.__dict__, f, indent=2)

    return path


def print_performance(
        l1_true: list = None,
        l1_unc: list = None,
        l1_hinge: list = None,
        l1_margin: list = None,
        l1_ipcw1: list = None,
        l1_ipcw2: list = None,
        l1_pseudo_obs: list = None,
        l1_pseudo_obs_pop: list = None,
        path: str = None
) -> None:
    """
    Print performance using mean and std. And also save to file.
    """
    l1_true = np.array(l1_true) if l1_true is not None else None
    l1_unc = np.array(l1_unc) if l1_unc is not None else None
    l1_hinge = np.array(l1_hinge) if l1_hinge is not None else None
    l1_margin = np.array(l1_margin) if l1_margin is not None else None
    l1_ipcw1 = np.array(l1_ipcw1) if l1_ipcw1 is not None else None
    l1_ipcw2 = np.array(l1_ipcw2) if l1_ipcw2 is not None else None
    l1_pseudo_obs = np.array(l1_pseudo_obs) if l1_pseudo_obs is not None else None
    l1_pseudo_obs_pop = np.array(l1_pseudo_obs_pop) if l1_pseudo_obs_pop is not None else None
    prf = f""
    prf += f"L1-true: {l1_true.mean():.3f} " \
           f"+/- {l1_true.std():.3f}\n" if l1_true is not None else f""
    prf += f"L1-uncensored: {l1_unc.mean():.3f} " \
           f"+/- {l1_unc.std():.3f}\n" if l1_unc is not None else f""
    prf += f"L1-hinge: {l1_hinge.mean():.3f} " \
           f"+/- {l1_hinge.std():.3f}\n" if l1_hinge is not None else f""
    prf += f"L1-margin: {l1_margin.mean():.3f} " \
           f"+/- {l1_margin.std():.3f}\n" if l1_margin is not None else f""
    prf += f"L1-ipcw-v1: {l1_ipcw1.mean():.3f} " \
           f"+/- {l1_ipcw1.std():.3f}\n" if l1_ipcw1 is not None else f""
    prf += f"L1-ipcw-v2: {l1_ipcw2.mean():.3f} " \
           f"+/- {l1_ipcw2.std():.3f}\n" if l1_ipcw2 is not None else f""
    prf += f"L1-pseudo-obs: {l1_pseudo_obs.mean():.3f} " \
           f"+/- {l1_pseudo_obs.std():.3f}\n" if l1_pseudo_obs is not None else f""
    prf += f"L1-pseudo-obs-pop: {l1_pseudo_obs_pop.mean():.3f} " \
           f"+/- {l1_pseudo_obs_pop.std():.3f}\n" if l1_pseudo_obs_pop is not None else f""
    print(prf)

    prf_for_excel = f"{l1_true.mean():.3f} {l1_true.std():.3f} | {l1_unc.mean(): 3f} {l1_unc.std():.3f} " \
                    f"{l1_hinge.mean():.3f} {l1_hinge.std():.3f} {l1_margin.mean():.3f} {l1_margin.std():.3f} " \
                    f"{l1_ipcw1.mean():.3f} {l1_ipcw1.std():.3f} {l1_ipcw2.mean():.3f} {l1_ipcw2.std():.3f} " \
                    f"{l1_pseudo_obs.mean():.3f} {l1_pseudo_obs.std():.3f} " \
                    f"{l1_pseudo_obs_pop.mean():.3f} {l1_pseudo_obs_pop.std():.3f}\n"

    if path is not None:
        prf_dict = {
            'l1_true': l1_true,
            'l1_unc': l1_unc,
            'l1_hinge': l1_hinge,
            'l1_margin': l1_margin,
            'l1_ipcw1': l1_ipcw1,
            'l1_ipcw2': l1_ipcw2,
            'l1_pseudo_obs': l1_pseudo_obs,
            'l1_pseudo_obs_pop': l1_pseudo_obs_pop
        }
        with open(f"{path}/performance.pkl", 'wb') as f:
            pickle.dump(prf_dict, f)

        with open(f"{path}/performance.txt", 'w') as f:
            f.write(prf)
            f.write(prf_for_excel)


def is_monotonic(
        array: Union[torch.Tensor, np.ndarray, list]
):
    return (all(array[i] <= array[i + 1] for i in range(len(array) - 1)) or
            all(array[i] >= array[i + 1] for i in range(len(array) - 1)))


def make_monotonic(
        array: Union[torch.Tensor, np.ndarray, list]
):
    for i in range(len(array) - 1):
        if not array[i] >= array[i + 1]:
            array[i + 1] = array[i]
    return array
