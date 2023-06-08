import numpy as np
import torch
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm
from sklearn.neural_network import MLPRegressor
from lifelines import WeibullAFTFitter, KaplanMeierFitter
from pycox.models import DeepHitSingle
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
import torchtuples as tt
import os
from collections import OrderedDict

from data import make_semi_synth_data
from util import print_performance, save_params
from util_survival import make_time_bins, survival_stratified_cv, survival_data_split
from SurvivalEVAL import ScikitSurvivalEvaluator, PycoxEvaluator, LifelinesEvaluator
from SurvivalEVAL.Evaluations.MeanError import mean_error
from params import generate_parser
from SurvivalEVAL.Evaluations.util import predict_median_survival_time

from MDN.data import get_dataloader
from MDN.trainers import MDNTrainer
from MDN.mdn_models import MDNModel


# Survival loss.
def survival_loss(outputs, labels):
    if torch.isnan(torch.abs(outputs["lambda"]).max()):
        exit()
    batch_loss = -labels * torch.log(
        outputs["lambda"].clamp(min=1e-8)) + outputs["Lambda"]
    return torch.mean(batch_loss)


class SurvivalLossMeter(object):
    def __init__(self):
        super(SurvivalLossMeter, self).__init__()
        self.reset()

    def add(self, outputs, labels):
        self.values.append(survival_loss(outputs, labels).item())

    def value(self):
        return [np.mean(self.values)]

    def reset(self):
        self.values = []


def train_MDN(x_train, t_train, e_train, x_val, t_val, e_val, x_test, t_test, e_test, args):
    random_state = np.random.RandomState(seed=0)
    dataloaders = {}
    batch_size = 512
    lr = 0.0018990388726217436
    weight_decay = 1.9486241836466362e-05
    feature_size = x_train.shape[1]
    model_config = OrderedDict([('hidden_size', 15),
                     ('num_components', 15),
                     ('init_type', 'residual')])
    dataloaders["train"] = get_dataloader(t_train, e_train, x_train, batch_size=batch_size,
                                          random_state=random_state, is_eval=False)

    dataloaders["valid"] = get_dataloader(t_val, e_val, x_val, batch_size=x_val.shape[0],
                                          random_state=random_state, is_eval=True)
    dataloaders["test"] = get_dataloader(t_test, e_test, x_test, batch_size=x_test.shape[0],
                                            random_state=random_state, is_eval=True)
    model = MDNModel(model_config=model_config, feature_size=feature_size)
    model.to(args.device)

    # Optimization criterions.
    criterions = {}
    criterions["survival_loss"] = survival_loss
    # Optimizer.
    optimizer = optim.RMSprop(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay)

    # Evaluation metrics.
    metrics = {}
    metrics["survival_loss"] = SurvivalLossMeter()

    trainer = MDNTrainer(
        model=model,
        device=args.device,
        criterions=criterions,
        optimizer=optimizer,
        dataloaders=dataloaders,
        metrics=metrics,
        earlystop_metric_name="survival_loss",
        batch_size=batch_size,
        num_epochs=100,
        patience=10,
        grad_clip=100)

    trainer.train()
    return trainer


if __name__ == "__main__":
    args = generate_parser()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # torch.autograd.set_detect_anomaly(True)

    data = make_semi_synth_data(args.dataset, args.censor_dist)
    assert "time" in data.columns and "event" in data.columns and "true_time" in data.columns, """The (synthetic) event 
    time variable, true event time variable and censor indicator variable is missing or need to be renamed."""
    args.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.feature_names = data.drop(columns=['time', 'event', 'true_time']).columns.tolist()
    args.n_features = data.shape[1] - 3
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset == "GBM":
        rescale = 776.2
    elif args.dataset == "SUPPORT":
        rescale = 405.8
    elif args.dataset == "Metabric":
        rescale = 71
    elif args.dataset == "MIMIC-IV":
        rescale = 880.8
    elif args.dataset == "MIMIC-IV_hosp":
        rescale = 59.2
    else:
        raise NotImplementedError
    reg_l1_true = []
    reg_l1_unc = []
    reg_l1_hinge = []
    reg_l1_margin = []
    reg_l1_ipcw1 = []
    reg_l1_ipcw2 = []
    reg_l1_pseudo_obs = []
    reg_l1_po_pop = []

    km_l1_true = []
    km_l1_unc = []
    km_l1_hinge = []
    km_l1_margin = []
    km_l1_ipcw1 = []
    km_l1_ipcw2 = []
    km_l1_pseudo_obs = []
    km_l1_po_pop = []

    aft_l1_true = []
    aft_l1_unc = []
    aft_l1_hinge = []
    aft_l1_margin = []
    aft_l1_ipcw1 = []
    aft_l1_ipcw2 = []
    aft_l1_pseudo_obs = []
    aft_l1_po_pop = []

    gb_l1_true = []
    gb_l1_unc = []
    gb_l1_hinge = []
    gb_l1_margin = []
    gb_l1_ipcw1 = []
    gb_l1_ipcw2 = []
    gb_l1_pseudo_obs = []
    gb_l1_po_pop = []

    rsf_l1_true = []
    rsf_l1_unc = []
    rsf_l1_hinge = []
    rsf_l1_margin = []
    rsf_l1_ipcw1 = []
    rsf_l1_ipcw2 = []
    rsf_l1_pseudo_obs = []
    rsf_l1_po_pop = []

    deephit_l1_true = []
    deephit_l1_unc = []
    deephit_l1_hinge = []
    deephit_l1_margin = []
    deephit_l1_ipcw1 = []
    deephit_l1_ipcw2 = []
    deephit_l1_pseudo_obs = []
    deephit_l1_po_pop = []

    mdn_l1_true = []
    mdn_l1_unc = []
    mdn_l1_hinge = []
    mdn_l1_margin = []
    mdn_l1_ipcw1 = []
    mdn_l1_ipcw2 = []
    mdn_l1_pseudo_obs = []
    mdn_l1_po_pop = []

    sca_l1_true = []
    sca_l1_unc = []
    sca_l1_hinge = []
    sca_l1_margin = []
    sca_l1_ipcw1 = []
    sca_l1_ipcw2 = []
    sca_l1_pseudo_obs = []
    sca_l1_po_pop = []

    for i, [data_train, data_test] in enumerate(tqdm(survival_stratified_cv(data, data.time.values,
                                                                            data.event.values, number_folds=5))):

        # Process data.
        data_train = data_train.drop(columns=["true_time"])
        true_test_time = data_test.true_time.values
        true_test_event = np.ones(data_test.shape[0])
        data_test = data_test.drop(columns=["true_time"])

        event_times = data_train['time'].values
        event_indicators = data_train['event'].values.astype('bool')

        y_train = np.empty(dtype=[('cens', bool), ('time', np.float64)], shape=event_times.shape[0])
        y_train['cens'] = event_indicators
        y_train['time'] = event_times

        X_train = data_train.drop(["time", "event"], axis=1).values
        X_test = data_test.drop(["time", "event"], axis=1).values
        T_train, T_test = data_train['time'].values, data_test['time'].values
        E_train, E_test = data_train['event'].values.astype('bool'), data_test['event'].values.astype('bool')

        print('*' * 10 + 'Training Regressor', '*' * 10)
        reg = MLPRegressor(hidden_layer_sizes=tuple(args.hidden_size), activation='relu', solver='adam',
                           batch_size=args.batch_size, learning_rate='constant', learning_rate_init=1e-3,
                           max_iter=args.num_epochs, shuffle=True, random_state=args.seed, tol=1e-4,
                           early_stopping=True, validation_fraction=0.1)
        reg.fit(X_train[E_train == 1], T_train[E_train == 1])
        reg_pred = reg.predict(X_test)
        reg_l1_true.append(mean_error(reg_pred, true_test_time, true_test_event, error_type="absolute", method='Uncensored'))
        reg_l1_unc.append(mean_error(reg_pred, T_test, E_test, error_type="absolute", method='Uncensored'))
        reg_l1_hinge.append(mean_error(reg_pred, T_test, E_test, T_train, E_train, error_type="absolute", method='Hinge'))
        reg_l1_margin.append(mean_error(reg_pred, T_test, E_test, T_train, E_train, error_type="absolute", method='Margin'))
        reg_l1_ipcw1.append(mean_error(reg_pred, T_test, E_test, T_train, E_train, error_type="absolute", method="IPCW-v1"))
        reg_l1_ipcw2.append(mean_error(reg_pred, T_test, E_test, T_train, E_train, error_type="absolute", method="IPCW-v2"))
        reg_l1_pseudo_obs.append(mean_error(reg_pred, T_test, E_test, T_train, E_train, error_type="absolute", method="Pseudo_obs"))
        reg_l1_po_pop.append(mean_error(reg_pred, T_test, E_test, T_train, E_train, error_type="absolute", method="Pseudo_obs_pop"))

        print('*' * 10 + 'Training KM' + '*' * 10)
        km = KaplanMeierFitter()
        km.fit(T_train, event_observed=E_train)
        time_bins = km.survival_function_.index.values
        survival_curves = km.survival_function_.KM_estimate.values
        km_pred = predict_median_survival_time(survival_curves, time_bins)
        # km_pred = km.median_survival_time_
        predicted_time = np.ones_like(T_test) * km_pred
        km_l1_true.append(mean_error(predicted_time, true_test_time, true_test_event, error_type="absolute", method='Uncensored'))
        km_l1_unc.append(mean_error(predicted_time, T_test, E_test, error_type="absolute", method="Uncensored"))
        km_l1_hinge.append(mean_error(predicted_time, T_test, E_test, T_train, E_train, error_type="absolute", method="Hinge"))
        km_l1_margin.append(mean_error(predicted_time, T_test, E_test, T_train, E_train, error_type="absolute", method="Margin"))
        km_l1_ipcw1.append(mean_error(predicted_time, T_test, E_test, T_train, E_train, error_type="absolute", method="IPCW-v1"))
        km_l1_ipcw2.append(mean_error(predicted_time, T_test, E_test, T_train, E_train, error_type="absolute", method="IPCW-v2"))
        km_l1_pseudo_obs.append(mean_error(predicted_time, T_test, E_test, T_train, E_train, error_type="absolute", method="Pseudo_obs"))
        km_l1_po_pop.append(mean_error(predicted_time, T_test, E_test, T_train, E_train, error_type="absolute", method="Pseudo_obs_pop"))
        print('*' * 10 + 'Training WeibullAFT', '*' * 10)
        aft_est = WeibullAFTFitter(penalizer=0.01)
        aft_est.fit(data_train, duration_col='time', event_col='event')
        predicted_time = aft_est.predict_expectation(data_test).values
        aft_l1_true.append(mean_error(predicted_time, true_test_time, true_test_event, error_type="absolute", method='Uncensored'))
        aft_l1_unc.append(mean_error(predicted_time, T_test, E_test, method="Uncensored"))
        aft_l1_hinge.append(mean_error(predicted_time, T_test, E_test, T_train, E_train, error_type="absolute", method="Hinge"))
        aft_l1_margin.append(mean_error(predicted_time, T_test, E_test, T_train, E_train, error_type="absolute", method="Margin"))
        aft_l1_ipcw1.append(mean_error(predicted_time, T_test, E_test, T_train, E_train, error_type="absolute", method="IPCW-v1"))
        aft_l1_ipcw2.append(mean_error(predicted_time, T_test, E_test, T_train, E_train, error_type="absolute", method="IPCW-v2"))
        aft_l1_pseudo_obs.append(mean_error(predicted_time, T_test, E_test, T_train, E_train, error_type="absolute", method="Pseudo_obs"))
        aft_l1_po_pop.append(mean_error(predicted_time, T_test, E_test, T_train, E_train, error_type="absolute", method="Pseudo_obs_pop"))
        print('*' * 10 + 'Training GBCM', '*' * 10)
        gbcm_est = GradientBoostingSurvivalAnalysis(loss="coxph").fit(X_train, y_train)
        surv = gbcm_est.predict_survival_function(X_test)
        true_evaler = ScikitSurvivalEvaluator(surv, true_test_time, true_test_event)
        gb_l1_true.append(true_evaler.mae(method="Uncensored"))
        evaler = ScikitSurvivalEvaluator(surv, T_test, E_test, T_train, E_train)
        gb_l1_unc.append(evaler.mae(method="Uncensored"))
        gb_l1_hinge.append(evaler.mae(method="Hinge"))
        gb_l1_margin.append(evaler.mae(method="Margin"))
        gb_l1_ipcw1.append(evaler.mae(method="IPCW-v1"))
        gb_l1_ipcw2.append(evaler.mae(method="IPCW-v2"))
        gb_l1_pseudo_obs.append(evaler.mae(method="Pseudo_obs"))
        gb_l1_po_pop.append(evaler.mae(method="Pseudo_obs_pop"))
        print('*' * 10 + 'Training RSF', '*' * 10)
        rsf_est = RandomSurvivalForest(n_estimators=50, min_samples_leaf=3).fit(X_train, y_train)
        surv = rsf_est.predict_survival_function(X_test)
        true_evaler = ScikitSurvivalEvaluator(surv, true_test_time, true_test_event)
        rsf_l1_true.append(true_evaler.mae(method="Uncensored"))
        evaler = ScikitSurvivalEvaluator(surv, T_test, E_test, T_train, E_train)
        rsf_l1_unc.append(evaler.mae(method="Uncensored"))
        rsf_l1_hinge.append(evaler.mae(method="Hinge"))
        rsf_l1_margin.append(evaler.mae(method="Margin"))
        rsf_l1_ipcw1.append(evaler.mae(method="IPCW-v1"))
        rsf_l1_ipcw2.append(evaler.mae(method="IPCW-v2"))
        rsf_l1_pseudo_obs.append(evaler.mae(method="Pseudo_obs"))
        rsf_l1_po_pop.append(evaler.mae(method="Pseudo_obs_pop"))
        print('*' * 10 + 'Training Deephit', '*' * 10)
        T_train_val = T_train
        E_train_val = E_train
        data_train, _, data_val = survival_data_split(data_train, stratify_colname='both', frac_train=0.9,
                                                      frac_test=0.1, random_state=args.seed)
        X_train = data_train.drop(columns=['event', 'time'], inplace=False).values.astype("float32")
        X_val = data_val.drop(columns=['event', 'time'], inplace=False).values.astype("float32")
        T_train, T_val = data_train['time'].values.astype("float32"), data_val['time'].values.astype("float32")
        E_train, E_val = data_train['event'].values.astype("bool"), data_val['event'].values.astype("bool")

        time_bins = make_time_bins(T_train, event=E_train)

        labtrans = DeepHitSingle.label_transform(len(time_bins))
        y_train = labtrans.fit_transform(*(T_train, E_train))
        y_val = labtrans.transform(*(T_val, E_val))

        train = (X_train, y_train)
        val = (X_val, y_val)

        in_features = X_train.shape[1]
        out_features = labtrans.out_features

        net = tt.practical.MLPVanilla(in_features, args.hidden_size, out_features, args.norm, args.dropout)
        model = DeepHitSingle(net, tt.optim.Adam, alpha=0.2, sigma=0.1, duration_index=labtrans.cuts)
        batch_size = args.batch_size
        # lr_finder = model.lr_finder(X_train, y_train, batch_size, tolerance=3)
        # _ = lr_finder.plot()
        # print(f"Best lr is {lr_finder.get_best_lr()}")

        lr_best = 0.001
        model.optimizer.set_lr(lr_best)

        callbacks = [tt.callbacks.EarlyStopping()]
        log = model.fit(X_train, y_train, args.batch_size, args.num_epochs, callbacks, val_data=val)

        surv = model.predict_surv_df(X_test.astype("float32"))
        surv.iloc[0, :] = 1

        true_evaler = PycoxEvaluator(surv, true_test_time, true_test_event)
        deephit_l1_true.append(true_evaler.mae(method="Uncensored"))
        eval = PycoxEvaluator(surv, T_test, E_test, T_train_val, E_train_val)
        deephit_l1_unc.append(eval.mae(method="Uncensored"))
        deephit_l1_hinge.append(eval.mae(method="Hinge"))
        deephit_l1_margin.append(eval.mae(method="Margin"))
        deephit_l1_ipcw1.append(eval.mae(method="IPCW-v1"))
        deephit_l1_ipcw2.append(eval.mae(method="IPCW-v2"))
        deephit_l1_pseudo_obs.append(eval.mae(method="Pseudo_obs"))
        deephit_l1_po_pop.append(eval.mae(method="Pseudo_obs_pop"))
        print('*' * 10 + 'Training Survival-MDN', '*' * 10)
        trainner = train_MDN(X_train, T_train/rescale, E_train, X_val, T_val/rescale, E_val, X_test, T_test/rescale, E_test, args)
        predicted_time = trainner.model.predict_time(torch.Tensor(X_test).to(args.device)) * rescale
        predicted_time = predicted_time.detach().cpu().numpy()
        mdn_l1_true.append(mean_error(predicted_time, true_test_time, true_test_event, error_type="absolute", method='Uncensored'))
        mdn_l1_unc.append(mean_error(predicted_time, T_test, E_test, T_train_val, E_train_val, error_type="absolute", method='Uncensored'))
        mdn_l1_hinge.append(mean_error(predicted_time, T_test, E_test, T_train_val, E_train_val, error_type="absolute", method='Hinge'))
        mdn_l1_margin.append(mean_error(predicted_time, T_test, E_test, T_train_val, E_train_val, error_type="absolute", method='Margin'))
        mdn_l1_ipcw1.append(mean_error(predicted_time, T_test, E_test, T_train_val, E_train_val, error_type="absolute", method='IPCW-v1'))
        mdn_l1_ipcw2.append(mean_error(predicted_time, T_test, E_test, T_train_val, E_train_val, error_type="absolute", method='IPCW-v2'))
        mdn_l1_pseudo_obs.append(mean_error(predicted_time, T_test, E_test, T_train_val, E_train_val, error_type="absolute", method='Pseudo_obs'))
        mdn_l1_po_pop.append(mean_error(predicted_time, T_test, E_test, T_train_val, E_train_val, error_type="absolute", method='Pseudo_obs_pop'))
        print('*' * 10 + 'Testing SCA', '*' * 10)
        # SCA uses TF 1.8 and python 3.6, which is not compatible with the rest of the code.
        # Therefore, we use a subprocess to run the SCA code. We run the code in run_SCA.py
        # and save the data. There we simply load the predicted time and calculate the l1 loss.
        sca_file_name = 'SCA/results/{}_{}_{}.npy'.format(args.dataset, args.censor_dist, i)
        try:
            predicted_time = np.load(sca_file_name)
            sca_l1_true.append(mean_error(predicted_time, true_test_time, true_test_event, error_type="absolute", method='Uncensored'))
            sca_l1_unc.append(mean_error(predicted_time, T_test, E_test, T_train_val, E_train_val, error_type="absolute", method='Uncensored'))
            sca_l1_hinge.append(mean_error(predicted_time, T_test, E_test, T_train_val, E_train_val, error_type="absolute", method='Hinge'))
            sca_l1_margin.append(mean_error(predicted_time, T_test, E_test, T_train_val, E_train_val, error_type="absolute", method='Margin'))
            sca_l1_ipcw1.append(mean_error(predicted_time, T_test, E_test, T_train_val, E_train_val, error_type="absolute", method='IPCW-v1'))
            sca_l1_ipcw2.append(mean_error(predicted_time, T_test, E_test, T_train_val, E_train_val, error_type="absolute", method='IPCW-v2'))
            sca_l1_pseudo_obs.append(mean_error(predicted_time, T_test, E_test, T_train_val, E_train_val, error_type="absolute",
                                             method='Pseudo_obs'))
            sca_l1_po_pop.append(mean_error(predicted_time, T_test, E_test, T_train_val, E_train_val, error_type="absolute",
                                         method='Pseudo_obs_pop'))
        except FileNotFoundError:
            print("file '{}' not found".format(sca_file_name))

    dir_ = os.getcwd()
    path = f"{dir_}/runs/{args.dataset}/{args.censor_dist}" \
           f"/Regressor/{args.timestamp}"
    if not os.path.exists(path):
        os.makedirs(path)
    print('*' * 10 + 'Regressor results', '*' * 10)
    print_performance(l1_true=reg_l1_true, l1_unc=reg_l1_unc, l1_hinge=reg_l1_hinge, l1_margin=reg_l1_margin,
                      l1_ipcw1=reg_l1_ipcw1, l1_ipcw2=reg_l1_ipcw2, l1_pseudo_obs=reg_l1_pseudo_obs,
                      l1_pseudo_obs_pop=reg_l1_po_pop, path=path)

    path = f"{dir_}/runs/{args.dataset}/{args.censor_dist}" \
           f"/KM/{args.timestamp}"
    if not os.path.exists(path):
        os.makedirs(path)
    print('*' * 10 + 'KM results', '*' * 10)
    print_performance(l1_true=km_l1_true, l1_unc=km_l1_unc, l1_hinge=km_l1_hinge, l1_margin=km_l1_margin,
                      l1_ipcw1=km_l1_ipcw1, l1_ipcw2=km_l1_ipcw2, l1_pseudo_obs=km_l1_pseudo_obs,
                      l1_pseudo_obs_pop=km_l1_po_pop, path=path)

    path = f"{dir_}/runs/{args.dataset}/{args.censor_dist}" \
           f"/WeibullAFT/{args.timestamp}"
    if not os.path.exists(path):
        os.makedirs(path)
    print('*' * 10 + 'AFT results', '*' * 10)
    print_performance(l1_true=aft_l1_true, l1_unc=aft_l1_unc, l1_hinge=aft_l1_hinge, l1_margin=aft_l1_margin,
                      l1_ipcw1=aft_l1_ipcw1, l1_ipcw2=aft_l1_ipcw2, l1_pseudo_obs=aft_l1_pseudo_obs,
                      l1_pseudo_obs_pop=aft_l1_po_pop, path=path)

    path = f"{dir_}/runs/{args.dataset}/{args.censor_dist}" \
           f"/GBCM/{args.timestamp}"
    if not os.path.exists(path):
        os.makedirs(path)
    print('*' * 10 + 'GBCM results', '*' * 10)
    print_performance(l1_true=gb_l1_true, l1_unc=gb_l1_unc, l1_hinge=gb_l1_hinge, l1_margin=gb_l1_margin,
                      l1_ipcw1=gb_l1_ipcw1, l1_ipcw2=gb_l1_ipcw2, l1_pseudo_obs=gb_l1_pseudo_obs,
                      l1_pseudo_obs_pop=gb_l1_po_pop, path=path)

    path = f"{dir_}/runs/{args.dataset}/{args.censor_dist}" \
           f"/RSF/{args.timestamp}"
    if not os.path.exists(path):
        os.makedirs(path)
    print('*' * 10 + 'RSF results', '*' * 10)
    print_performance(l1_true=rsf_l1_true, l1_unc=rsf_l1_unc, l1_hinge=rsf_l1_hinge, l1_margin=rsf_l1_margin,
                      l1_ipcw1=rsf_l1_ipcw1, l1_ipcw2=rsf_l1_ipcw2, l1_pseudo_obs=rsf_l1_pseudo_obs,
                      l1_pseudo_obs_pop=rsf_l1_po_pop, path=path)

    path = f"{dir_}/runs/{args.dataset}/{args.censor_dist}" \
           f"/DeepHit/{args.timestamp}"
    if not os.path.exists(path):
        os.makedirs(path)
    print('*' * 10 + 'deephit results', '*' * 10)
    print_performance(l1_true=deephit_l1_true, l1_unc=deephit_l1_unc, l1_hinge=deephit_l1_hinge,
                      l1_margin=deephit_l1_margin, l1_ipcw1=deephit_l1_ipcw1, l1_ipcw2=deephit_l1_ipcw2,
                      l1_pseudo_obs=deephit_l1_pseudo_obs, l1_pseudo_obs_pop=deephit_l1_po_pop, path=path)

    path = f"{dir_}/runs/{args.dataset}/{args.censor_dist}" \
           f"/Survival_MDN/{args.timestamp}"
    if not os.path.exists(path):
        os.makedirs(path)
    print('*' * 10 + 'Survival_MDN results', '*' * 10)
    print_performance(l1_true=mdn_l1_true, l1_unc=mdn_l1_unc, l1_hinge=mdn_l1_hinge,
                      l1_margin=mdn_l1_margin, l1_ipcw1=mdn_l1_ipcw1, l1_ipcw2=mdn_l1_ipcw2,
                      l1_pseudo_obs=mdn_l1_pseudo_obs, l1_pseudo_obs_pop=mdn_l1_po_pop, path=path)

    path = f"{dir_}/runs/{args.dataset}/{args.censor_dist}" \
           f"/SCA/{args.timestamp}"
    if not os.path.exists(path):
        os.makedirs(path)
    print('*' * 10 + 'SCA results', '*' * 10)
    print_performance(l1_true=sca_l1_true, l1_unc=sca_l1_unc, l1_hinge=sca_l1_hinge,
                      l1_margin=sca_l1_margin, l1_ipcw1=sca_l1_ipcw1, l1_ipcw2=sca_l1_ipcw2,
                      l1_pseudo_obs=sca_l1_pseudo_obs, l1_pseudo_obs_pop=sca_l1_po_pop,path=path)

