import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import copy
from tqdm import trange

from model.loss import PartialLikelihood, LikelihoodMTLR
from util_survival import baseline_hazard, extract_survival, survival_data_split, reformat_survival


def build_sequential_nn(in_features, dims, norm, activation, dropout):
    """Build a sequential neural network."""
    layers = []
    for i in range(len(dims)):
        if i == 0:
            layers.append(nn.Linear(in_features, dims[i]))
        else:
            layers.append(nn.Linear(dims[i - 1], dims[i]))
        if i < len(dims) - 1:
            if norm:
                layers.append(nn.BatchNorm1d(dims[i]))
            layers.append(getattr(nn, activation)())
            if dropout is not None:
                layers.append(nn.Dropout(dropout))
    return layers


def build_sequential_mtlr(in_features, dims, norm, activation, dropout):
    layers = build_sequential_nn(in_features, dims, norm, activation, dropout)
    layers.pop()
    if len(dims) > 1:
        layers.append(BaseMTLR(dims[-2], dims[-1]))
    else:
        layers.append(BaseMTLR(in_features, dims[-1]))
    return nn.Sequential(*layers)


class BaseMTLR(nn.Module):
    """Multi-task logistic regression for individualised
    survival prediction.

    The MTLR time-logits are computed as:
    `z = sum_k x^T w_k + b_k`,
    where `w_k` and `b_k` are learnable weights and biases for each time
    interval.

    Note that a slightly more efficient reformulation is used here, first
    proposed in [2]_.

    References
    ----------
    ..[1] C.-N. Yu et al., ‘Learning patient-specific cancer survival
    distributions as a sequence of dependent regressors’, in Advances in neural
    information processing systems 24, 2011, pp. 1845–1853.
    ..[2] P. Jin, ‘Using Survival Prediction Techniques to Learn
    Consumer-Specific Reservation Price Distributions’, Master's thesis,
    University of Alberta, Edmonton, AB, 2015.
    """

    def __init__(self, in_features: int, num_time_bins: int):
        """Initialises the module.

        Parameters
        ----------
        in_features
            Number of input features.
        num_time_bins
            The number of bins to divide the time axis into.
        """
        super().__init__()
        if num_time_bins < 1:
            raise ValueError("The number of time bins must be at least 1")
        if in_features < 1:
            raise ValueError("The number of input features must be at least 1")
        self.in_features = in_features
        self.num_time_bins = num_time_bins + 1  # + extra time bin [max_time, inf)

        self.mtlr_weight = nn.Parameter(torch.Tensor(self.num_time_bins - 1, self.in_features))
        self.mtlr_bias = nn.Parameter(torch.Tensor(self.num_time_bins - 1))

        # `G` is the coding matrix from [2]_ used for fast summation.
        # When registered as buffer, it will be automatically
        # moved to the correct device and stored in saved
        # model state.
        self.register_buffer(
            "G",
            torch.tril(
                torch.ones(self.num_time_bins - 1,
                           self.num_time_bins,
                           requires_grad=True)))
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass on a batch of examples.

        Parameters
        ----------
        x : torch.Tensor, shape (num_samples, num_features)
            The input data.

        Returns
        -------
        torch.Tensor, shape (num_samples, num_time_bins - 1)
            The predicted time logits.
        """
        out = F.linear(x, self.mtlr_weight, self.mtlr_bias)
        return torch.matmul(out, self.G)

    def reset_parameters(self):
        """Resets the model parameters."""
        nn.init.xavier_normal_(self.mtlr_weight)
        nn.init.constant_(self.mtlr_bias, 0.)

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}, num_time_bins={self.num_time_bins})"


class MTLR(nn.Module):
    """MTLR model with regularization"""

    def __init__(self, config):
        super().__init__()
        output_size = len(config.time_bins)
        self.in_features = config.n_features
        self.dropout = config.dropout
        self.norm = config.norm
        self.dims = copy.deepcopy(config.hidden_size)
        self.dims.append(output_size)
        self.activation = config.activation
        self.batch_size = config.batch_size
        self.time_bins = config.time_bins
        self.loss = LikelihoodMTLR(config.c1, order=config.order, reduction='mean')

        self.model = self._build_model()

    def _build_model(self):
        return build_sequential_mtlr(self.in_features, self.dims, self.norm, self.activation, self.dropout)

    def forward(self, x):
        return self.model(x)

    def predict_survival(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            assert logits.dim() == 2, "The logits should have dimension with with size (n_data, n_bins)"
            G = torch.tril(torch.ones(logits.shape[1], logits.shape[1])).to(logits.device)
            density = torch.softmax(logits, dim=1)
            return torch.matmul(density, G)

    def train_model(self, dataset, device, num_epochs, optimizer, path=None,
                    # TODO: delete optimizer, and add weight_decay
                    early_stop=True, patience=20, random_state=42, score_type='nll', verbose=True):
        self.to(device)
        self.train()
        self.reset_parameters()

        data_train, _, data_val = survival_data_split(dataset, stratify_colname='both',
                                                      frac_train=0.9, frac_test=0.1,
                                                      random_state=random_state)
        train_size = data_train.shape[0]
        x_train, y_train = reformat_survival(data_train, self.time_bins)
        x_val, y_val = reformat_survival(data_val, self.time_bins)
        t_val, e_val = data_val.time.values, data_val.event.values
        x_val, y_val = x_val.to(device), y_val.to(device)
        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=self.batch_size, shuffle=True)

        best_val_score = float('inf')
        best_ep = -1

        pbar = trange(num_epochs, disable=not verbose)
        for i in pbar:
            total_loss = 0
            for xi, yi in train_loader:
                xi, yi = xi.to(device), yi.to(device)
                optimizer.zero_grad()
                y_pred = self.forward(xi)
                train_loss = self.loss(y_pred, yi, self)

                train_loss.backward()
                optimizer.step()

                total_loss += (train_loss / train_size).item()
            logits_outputs = self.forward(x_val)
            eval_score = self.loss(logits_outputs, y_val, self).item()
            pbar.set_description(f"[epoch {i + 1: 4}/{num_epochs}]")
            pbar.set_postfix_str(f"Train nll = {total_loss:.4f}; "
                                 f"Validation {score_type} = {eval_score:.4f};")

            if early_stop:
                if best_val_score > eval_score:
                    best_val_score = eval_score
                    best_ep = i
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                    }, path)
                if (i - best_ep) > patience:
                    print(f"Validation loss converges at {best_ep + 1}-th epoch.")
                    break
        self.model.load_state_dict(torch.load(path)['model_state_dict'])

    def reset_parameters(self):
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


class CoxPH(nn.Module):
    """CoxPH model with regularization"""

    def __init__(self, config):
        super().__init__()
        self.in_features = config.n_features
        self.dropout = config.dropout
        self.norm = config.norm
        self.dims = copy.deepcopy(config.hidden_size)
        self.dims.append(1)
        self.activation = config.activation

        self.time_bins = None
        self.baseline_hazard = None
        self.cum_baseline_hazard = None
        self.baseline_survival = None
        self.loss = PartialLikelihood(config.c1, order=config.order, reduction='mean')

        self.model = self._build_model()

    def _build_model(self):
        layers = build_sequential_nn(self.in_features, self.dims, self.norm, self.activation, self.dropout)
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def predict_risk(self, x):
        with torch.no_grad():
            return self.forward(x)

    def predict_survival(self, x):
        with torch.no_grad():
            risks = self.predict_risk(x)
            n_data = len(risks)
            risk_score = torch.exp(risks)
            risk_score = risk_score.squeeze()
            survival_curves = torch.empty((n_data, self.baseline_survival.shape[0]), dtype=torch.double).to(
                risks.device)
            for i in range(n_data):
                survival_curve = torch.pow(self.baseline_survival, risk_score[i])
                survival_curves[i] = survival_curve
                # survival_curves[i, :] = make_monotonic(survival_curve)      # TODO: check if we need it
            return survival_curves

    def cal_baseline_survival(self, dataset):
        x, t, e = extract_survival(dataset)
        device = next(self.parameters()).device
        x, t, e = x.to(device), t.to(device), e.to(device)
        outputs = self.forward(x)
        self.time_bins, self.baseline_hazard, self.cum_baseline_hazard, self.baseline_survival = baseline_hazard(
            outputs, t, e)

    def train_model(self, dataset, device, num_epochs, optimizer, path=None,
                    early_stop=True, patience=20, random_state=42, score_type='nll', verbose=True):
        self.to(device)
        self.train()
        self.reset_parameters()

        data_train, _, data_val = survival_data_split(dataset, stratify_colname='both',
                                                      frac_train=0.9, frac_test=0.1,
                                                      random_state=random_state)
        x_train, t_train, e_train = extract_survival(data_train)
        x_val, t_val, e_val = extract_survival(data_val)
        x_val, t_val, e_val = x_val.to(device), t_val.to(device), e_val.to(device)
        train_loader = DataLoader(TensorDataset(x_train, t_train, e_train),
                                  batch_size=data_train.shape[0], shuffle=True)

        best_val_score = float('inf')
        best_ep = -1

        pbar = trange(num_epochs, disable=not verbose)
        for i in pbar:
            nll_loss = 0
            for xi, ti, ei in train_loader:
                xi, ti, ei = xi.to(device), ti.to(device), ei.to(device)
                optimizer.zero_grad()
                y_pred = self.forward(xi)
                nll_loss = self.loss(y_pred, ti, ei, self)

                nll_loss.backward()
                optimizer.step()
            logits_outputs = self.forward(x_val)
            eval_score = self.loss(logits_outputs, t_val, e_val, self).item()
            pbar.set_description(f"[epoch {i + 1: 4}/{num_epochs}]")
            pbar.set_postfix_str(f"Train nll = {nll_loss.item():.4f}; "
                                 f"Validation {score_type} = {eval_score:.4f};")

            if early_stop:
                if best_val_score > eval_score:
                    best_val_score = eval_score
                    best_ep = i
                    torch.save(self, path)
                if (i - best_ep) > patience:
                    print(f"Validation loss converges at {best_ep + 1}-th epoch.")
                    break

    def reset_parameters(self):
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

