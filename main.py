import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm
from lifelines import WeibullAFTFitter
from pycox.models import DeepHitSingle

from data import make_semi_synth_data
from model import MTLR, CoxPH
from util import print_performance, save_params
from util_survival import make_time_bins, survival_stratified_cv
from SurvivalEVAL import SurvivalEvaluator
from params import generate_parser

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
    args.n_features = data.shape[1] - 3
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    path = save_params(args)
    l1_true = []
    l1_unc = []
    l1_hinge = []
    l1_margin = []
    l1_ipcw1 = []
    l1_ipcw2 = []
    l1_pseudo_obs = []
    l1_po_pop = []
    for i, [data_train, data_test] in enumerate(tqdm(survival_stratified_cv(data, data.time.values,
                                                                            data.event.values, number_folds=5))):

        # Process data.
        data_train = data_train.drop(columns=["true_time"])
        true_test_time = data_test.true_time.values
        true_test_event = np.ones(data_test.shape[0])
        data_test = data_test.drop(columns=["true_time"])
        # get time_bins for discrete model, MTLR and DeepHit
        time_bins = make_time_bins(data_train["time"].values, event=data_train["event"].values)

        if args.model == "MTLR":
            args.time_bins = time_bins
            model = MTLR(args)
        elif args.model == "CoxPH":
            model = CoxPH(args)
        elif args.model == "DeepHit":
            model = DeepHitSingle(args)
        elif args.model == "AFT":
            model = WeibullAFTFitter(args)
        else:
            raise ValueError(f"Unknown model name: {args.model}")
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)
        print('*' * 40 + 'Training' + '*' * 40)
        model.train_model(data_train, args.device, args.num_epochs, optim,
                          path=path + f"/exp_{i}.pth", early_stop=args.early_stop, random_state=args.seed)
        # model.train_model(data_train, args.device, args.num_epochs, optim,
        #                   path=None, random_state=args.seed)

        if isinstance(model, MTLR):
            time_bins = time_bins
            time_bins = torch.cat([torch.tensor([0]), time_bins], dim=0).to(args.device)
        elif isinstance(model, CoxPH):
            model.cal_baseline_survival(data_train)
            time_bins = model.time_bins

        model.eval()
        x_test = torch.tensor(data_test.drop(["time", "event"], axis=1).values, dtype=torch.float, device=args.device)
        survival_outputs = model.predict_survival(x_test)

        print('*' * 40 + 'Start Evaluation' + '*' * 40)
        true_evaler = SurvivalEvaluator(survival_outputs, time_bins, true_test_time, true_test_event)
        l1_true.append(true_evaler.mae(method="Uncensored"))
        evaler = SurvivalEvaluator(survival_outputs, time_bins, data_test.time.values, data_test.event.values,
                                   data_train.time.values, data_train.event.values)
        l1_unc.append(evaler.mae(method="Uncensored"))
        l1_hinge.append(evaler.mae(method="Hinge"))
        l1_margin.append(evaler.mae(method="Margin"))
        l1_ipcw1.append(evaler.mae(method="IPCW-v1"))
        l1_ipcw2.append(evaler.mae(method="IPCW-v2"))
        l1_pseudo_obs.append(evaler.mae(method="Pseudo_obs"))
        l1_po_pop.append(evaler.mae(method="Pseudo_obs_pop"))
    print_performance(l1_true=l1_true, l1_unc=l1_unc, l1_hinge=l1_hinge, l1_margin=l1_margin, l1_ipcw1=l1_ipcw1,
                      l1_ipcw2=l1_ipcw2, l1_pseudo_obs=l1_pseudo_obs, l1_pseudo_obs_pop=l1_po_pop, path=path)
