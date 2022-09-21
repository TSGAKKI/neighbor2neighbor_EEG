import argparse


def get_args():
    parser = argparse.ArgumentParser('Train DCRNN')

    # General args
    parser.add_argument('--cuda', type=str, default= '1')

    parser.add_argument('--save_dir',
                        type=str,
                        default='result',
                        help='Directory to save the outputs and checkpoints.')

    parser.add_argument('--debug', type=str, default= '0') 

    parser.add_argument('--rand_seed',
                        type=int,
                        default=123,
                        help='Random seed.')

    parser.add_argument('--data_dir',
                        type=str,
                        default=None,
                        help='Dir of raw EEG EOG EMG (.py) .')
    # model settings
    parser.add_argument('--dataset', type=str, default= 'EEGDenoiseNet')

    parser.add_argument('--position', type=str, default= 'full') 

    parser.add_argument('--noise_type', type=str, default= 'EOG') 

    parser.add_argument('--subject_independent', type=str, default= '0') 

    # Training/test args
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,

                        help='Training batch size.')

    parser.add_argument('--dropout',
                        type=float,
                        default=0.0,
                        help='Dropout rate for dropout layer before final FC.')

    parser.add_argument('--eval_every',
                        type=int,
                        default=1,
                        help='Evaluate on dev set every x epoch.')

    parser.add_argument('--lr_init',
                        type=float,
                        default=3e-4,
                        help='Initial learning rate.')

    parser.add_argument('--num_epochs',
                        type=int,
                        default=100,
                        help='Number of epochs for training.')
    parser.add_argument("--Lambda1", type=float, default=1.0)
    parser.add_argument("--Lambda2", type=float, default=1.0)   
    parser.add_argument("--increase_ratio", type=float, default=2.0)
    parser.add_argument(
        '--patience',
        type=int,
        default=5,
        help='Number of epochs of patience before early stopping.')

    args = parser.parse_args()

    # must provide load_model_path if testing only
    # if (args.load_model_path is None) and not(args.do_train):
    #     raise ValueError(
    #         'For evaluation only, please provide trained model checkpoint in argument load_model_path.')

    return args
