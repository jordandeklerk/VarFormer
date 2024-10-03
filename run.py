import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Basic Config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='Task name, options: [long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, default=1, help='Status: 1 for training, 0 for testing')
    parser.add_argument('--model_id', type=str, default='test', help='Model ID')
    parser.add_argument('--model', type=str, default='BayesFormer',
                        help='Model name, options: [BayesFormer, Autoformer, Transformer, TimesNet]')

    # Data Loader
    parser.add_argument('--data', type=str, default='ETTm1', help='Dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='Root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='Data file')
    parser.add_argument('--features', type=str, default='M',
                        help='Forecasting task, options: [M, S, MS]; M: multivariate predict multivariate, S: univariate predict univariate, MS: multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='Target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='Frequency for time features encoding, options: [s: secondly, t: minutely, h: hourly, d: daily, b: business days, w: weekly, m: monthly]')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='Location of model checkpoints')
    parser.add_argument('--exp_type', type=str, default='multi', help='Experiment type')

    # Forecasting Setting
    parser.add_argument('--seq_len', type=int, default=96, help='Input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='Start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='Prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='Subset for M4')

    # Model Define (Common Arguments)
    parser.add_argument('--enc_in', type=int, default=1, help='Encoder input size')
    parser.add_argument('--dec_in', type=int, default=1, help='Decoder input size')
    parser.add_argument('--c_in', type=int, default=1, help='Input size')
    parser.add_argument('--embed_type', type=str, default='fixed', help='Embedding type')
    parser.add_argument('--c_out', type=int, default=1, help='Output size')
    parser.add_argument('--d_model', type=int, default=512, help='Dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='Number of encoder layers')
    parser.add_argument('--d_layers', type=int, default=3, help='Number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='Dimension of FCN')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--factor', type=int, default=1, help='Attention factor')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='Time features encoding, options: [timeF, fixed, learned]')
    parser.add_argument('--output_attention', action='store_true', help='Whether to output attention in encoder')

    # BayesFormer Specific Arguments
    parser.add_argument('--inter_dim', type=int, default=128, help='Intermediate dimension for BayesFormer')
    parser.add_argument('--latent_dim', type=int, default=64, help='Latent dimension for BayesFormer')
    parser.add_argument('--input_dim', type=int, default=1, help='Input dimension for BayesFormer')
    parser.add_argument('--beta', type=float, default=1, help='Beta for generative model loss')
    parser.add_argument('--alpha', type=float, default=1, help='Alpha for transformer NLL model loss')

    # Conv Specific Arguments
    parser.add_argument('--stem_ratio', type=int, default=1, help='Stem ratio for ModernTCN')
    parser.add_argument('--downsample_ratio', type=int, default=2, help='Downsample ratio for ModernTCN')
    parser.add_argument('--ffn_ratio', type=int, default=1, help='FFN ratio for ModernTCN')
    parser.add_argument('--dw_dims', type=int, nargs='+', default=[64], help='Depthwise dimensions for each stage in ModernTCN')
    parser.add_argument('--small_kernel_merged', type=str2bool, default=False, help='Whether to merge small kernel in ReparamLargeKernelConv')
    parser.add_argument('--backbone_dropout', type=float, default=0.1, help='Dropout rate for ModernTCN backbone')
    parser.add_argument('--head_dropout', type=float, default=0.1, help='Dropout rate for ModernTCN head')
    parser.add_argument('--use_multi_scale', type=str2bool, default=True, help='Whether to use multi-scale feature in ModernTCN')
    parser.add_argument('--revin', type=str2bool, default=True, help='Whether to use RevIN in ModernTCN')
    parser.add_argument('--affine', type=str2bool, default=True, help='Whether to use affine transformation in RevIN')
    parser.add_argument('--subtract_last', type=str2bool, default=False, help='Whether to subtract last value in RevIN')
    parser.add_argument('--nvars', type=int, default=1, help='Number of variables')

    # Optimization
    parser.add_argument('--num_workers', type=int, default=10, help='Data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='Experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='Train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Adam weight decay')
    parser.add_argument('--des', type=str, default='test', help='Experiment description')
    parser.add_argument('--loss', type=str, default='MSE', help='Loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='Adjust learning rate')
    parser.add_argument('--use_amp', type=str2bool, default=False, help='Use automatic mixed precision training')
    parser.add_argument('--pct_start', type=float, default=0.1, help='Pct start for scheduler')

    # GPU
    parser.add_argument('--use_gpu', type=str2bool, default=True, help='Use GPU')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--use_multi_gpu', type=str2bool, default=False, help='Use multiple GPUs')
    parser.add_argument('--devices', type=str, default='0,1,2,3,4,5', help='Device IDs of multiple GPUs')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')

    # De-stationary Projector Params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='Hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='Number of hidden layers in projector')

    # Additional Params
    parser.add_argument('--low_pass', type=str2bool, default=False, help='Whether to apply low pass filter')
    parser.add_argument('--low_pass_threshold', type=float, default=3, help='Low pass threshold')

    args = parser.parse_args()

    # Post-processing Arguments
    args.d_latent = args.latent_dim  # Use latent_dim for d_latent
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    # Handle Multi-GPU
    if args.use_gpu and args.use_multi_gpu:
        args.device_ids = list(map(int, args.devices.split(',')))
        args.gpu = args.device_ids[0]
    else:
        args.use_multi_gpu = False

    # Settings
    args.task_name = 'long_term_forecast'
    args.is_training = 1 if args.is_training == 1 else 0
    args.root_path = './datasets/'
    args.checkpoints = os.path.join('./checkpoints', args.exp_type)

    return args


def generate_setting(args, iteration):
    if args.model.lower() == 'bayesformer':
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_eb{}_id{}_ld{}_ls{}_ss{}_ps{}_dt{}_revin{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.embed,
            args.inter_dim,
            args.latent_dim,
            args.stem_ratio,
            args.downsample_ratio,
            args.ffn_ratio,
            args.factor,
            args.revin,
            args.des, iteration)
    else:
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_revin{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.revin,
            args.des, iteration)
    return setting


def main():
    args = parse_arguments()

    print('Args in experiment:')
    print(args)
    print('Current Working Directory:', os.getcwd())
    print('Parent Directory:', os.path.abspath(os.path.join(os.getcwd(), "..")))
    print('Grandparent Directory:', os.path.abspath(os.path.join(os.getcwd(), "../..")))

    Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # Generate setting string
            setting = generate_setting(args, ii)

            # Check if setting already exists to avoid duplication
            results_path = os.path.join('results', args.exp_type)
            if os.path.exists(results_path):
                if setting in os.listdir(results_path):
                    print(f"Setting {setting} already exists in results. Skipping iteration {ii}.")
                    continue

            # Initialize and run experiment
            exp = Exp(args)
            print(f'>>>>>>> Start training: {setting} >>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train(setting)

            print(f'>>>>>>> Testing: {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        # Single testing run
        ii = 0
        setting = generate_setting(args, ii)

        exp = Exp(args)
        print(f'>>>>>>> Testing: {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
