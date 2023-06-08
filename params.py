import argparse


def generate_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default="SUPPORT",
                        choices=["GBM", "SUPPORT", "Metabric", "MIMIC-IV", "MIMIC-IV_hosp"],
                        help="Dataset name for make semi-synthetic data.")
    parser.add_argument('--censor_dist', type=str, default="GBM",
                        choices=["uniform", "uniform_truc", "exponential", "original_ind", "original_dep", "GBM"],
                        help="Type of synthetic censoring.")
    parser.add_argument('--model', type=str, default="MTLR",
                        choices=["MTLR", "CoxPH"],
                        help="Model name.")

    # General parameters
    parser.add_argument('--num_epochs', type=int, default=50000,
                        help="Number of maximum training epoch.")
    parser.add_argument('--early_stop', type=bool, default=True,
                        help="Whether to use early stop for training.")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for initialization")
    parser.add_argument('--batch_size', type=int, default=512,
                        help="Batch size for training.")
    parser.add_argument('--lr', type=float, default=0.001,  # 0.001 for MTLR, 0.01 for Cox
                        help="Learning rate.")
    parser.add_argument('--c1', type=float, default=0.01,
                        help="Hyperparameter for the regularization term.")
    parser.add_argument('--order', type=str, default='1',
                        help="Regularization type, L1, L2, or L21.")
    parser.add_argument('--hidden_size', type=list, default=[],
                        help="Hidden neurons in neural network.")
    parser.add_argument('--norm', type=bool, default=False,
                        help="Whether to use batch norm in neural network.")
    parser.add_argument('--dropout', type=float, default=0.4,
                        help="Dropout rate.")
    parser.add_argument('--activation', type=str, default='ReLU',
                        help="Activation layer name.")

    args = parser.parse_args()
    return args
