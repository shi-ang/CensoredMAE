import numpy as np
from datetime import datetime
from tqdm import tqdm
import os

from data import make_semi_synth_data
from util_survival import survival_stratified_cv, survival_data_split
from params import generate_parser

from SCA.model.sca import SCA
import tensorflow as tf

if __name__ == "__main__":
    args = generate_parser()

    np.random.seed(args.seed)
    GPUID = '0'
    tf.reset_default_graph()
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUID)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.set_random_seed(args.seed)

    data = make_semi_synth_data(args.dataset, args.censor_dist)
    assert "time" in data.columns and "event" in data.columns and "true_time" in data.columns, """The (synthetic) event 
    time variable, true event time variable and censor indicator variable is missing or need to be renamed."""
    args.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.feature_names = data.drop(columns=['time', 'event', 'true_time']).columns.tolist()
    args.n_features = data.shape[1] - 3

    for i, [data_train, data_test] in enumerate(tqdm(survival_stratified_cv(data, data.time.values,
                                                                            data.event.values, number_folds=5))):

        # Process data.
        data_train = data_train.drop(columns=["true_time"])
        true_test_time = data_test.true_time.values
        true_test_event = np.ones(data_test.shape[0])
        data_test = data_test.drop(columns=["true_time"])

        X_test = data_test.drop(["time", "event"], axis=1).values
        T_test = data_test['time'].values
        E_test = data_test['event'].values.astype('bool')

        data_train, _, data_val = survival_data_split(data_train, stratify_colname='both', frac_train=0.9,
                                                      frac_test=0.1, random_state=args.seed)
        X_train = data_train.drop(columns=['event', 'time'], inplace=False).values.astype("float32")
        X_val = data_val.drop(columns=['event', 'time'], inplace=False).values.astype("float32")
        T_train, T_val = data_train['time'].values.astype("float32"), data_val['time'].values.astype("float32")
        E_train, E_val = data_train['event'].values.astype("bool"), data_val['event'].values.astype("bool")

        print('*' * 10 + 'Training SCA', '*' * 10)
        train_data = {'x': X_train, 't': T_train, 'e': E_train}
        test_data = {'x': X_test, 't': T_test, 'e': E_test}
        val_data = {'x': X_val, 't': T_val, 'e': E_val}
        non_par = SCA(batch_size=args.batch_size, learning_rate=3e-4,
                      beta1=0.9, beta2=0.999, require_improvement=10000,
                      num_iterations=40000, seed=args.seed, l2_reg=0.001, hidden_dim=[50, 50, 50],
                      train_data=train_data, test_data=test_data, valid_data=val_data,
                      input_dim=train_data['x'].shape[1], num_examples=train_data['x'].shape[0], keep_prob=0.8,
                      sample_size=200, max_epochs=400, gamma_0=2, n_clusters=25)
        with non_par.session:
            non_par.train_test()

        predicted_time = np.load('SCA/matrix/Test_predicted_t_median.npy')
        np.save('SCA/results/{}_{}_{}.npy'.format(args.dataset, args.censor_dist, i), predicted_time)
