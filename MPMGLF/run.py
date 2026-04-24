import argparse
import torch
import gc
from tasks import Long_term_Forecast

if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()
    
    parser = argparse.ArgumentParser(description='MyMoudle')
    
    parser.add_argument('--task_name', type=str, default='long_term_forecast',)
    parser.add_argument('--is_training', type=int, default=1,
                        help='Status 1 for training mode')
    
    parser.add_argument('--data', type=str, default='heifangtai',
                        help='Dataset name')
    parser.add_argument('--data_path', type=str, default='heifangtai.csv',
                        help='Data file name')
    parser.add_argument('--root_path', type=str, default='./data/',
                        help='Root folder for dataset')
    parser.add_argument('--features', type=str, default='MS',
                        help='Prediction task type: M, S, or MS')
    parser.add_argument('--target', type=str, default='displacement2',
                        help='Target column name for S or MS tasks')
    parser.add_argument('--freq', type=str, default='h',
                        help='Frequency for time features: s, t, h, d, b, w, m')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='Time feature embedding method: timeF, fixed, learned')
    parser.add_argument('--checkpoints', type=str, default='./chechpoints/',
                        help='Model weight save location')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly',
                        help='Subset for M4')
    
    parser.add_argument('--seq_len', type=int, default=8,
                        help='Input sequence length')
    parser.add_argument('--label_len', type=int, default=0,
                        help='Starting token length')
    parser.add_argument('--pred_len', type=int, default=8,
                        help='Prediction sequence length')
    
    parser.add_argument('--c_in', type=int, default=4,
                        help='Number of input features for embedding')
    parser.add_argument('--c_out', type=int, default=4,
                        help='Output size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Data loader num workers')
    parser.add_argument('--itr', type=int, default=1,
                        help='Number of runs for mean and variance')
    parser.add_argument('--train_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--logging', type=bool, default=True,
                        help='Save metrics to file')
    parser.add_argument('--out_dir', type=str, default='./out/canshuminganxing',
                        help='Output directory for saved data')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='Disable CUDA')
    parser.add_argument('--cuda_id', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--out_predictions', type=bool, default=True,
                        help='Output prediction results')
    
    parser.add_argument('--d_model', type=int, default=4,
                        help='Embedding dimension')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--e_layers', type=int, default=3,
                        help='Number of TimeBlock layers')
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed')
    
    parser.add_argument('--top_k', type=int, default=4,
                        help='Top k frequencies in TimesBlock FFT')
    parser.add_argument('--d_ff', type=int, default=32,
                        help='Hidden size in TimeBlock convolution')
    parser.add_argument('--num_kernels', type=int, default=3,
                        help='Number of kernels in Inception')
    
    parser.add_argument('--nhid', type=int, default=4,
                        help='GCN hidden size')
    parser.add_argument('--outfeature', type=int, default=4,
                        help='GCN output features')
    parser.add_argument('--graph_hops', type=int, default=3,
                        help='Number of neighbor hops to aggregate')
    parser.add_argument('--graph_drop', type=float, default=0.2,
                        help='Graph dropout rate')
    parser.add_argument('--graph_batchnorm', type=bool, default=True,
                        help='Apply batch normalization')
    
    parser.add_argument('--smoothness_ratio', type=float, default=0.1,
                        help='Smoothness parameter for graph loss')
    parser.add_argument('--degree_ratio', type=float, default=0.1,
                        help='Degree ratio for graph loss')
    parser.add_argument('--sparsity_ratio', type=float, default=0.2,
                        help='Sparsity ratio for graph loss')
    parser.add_argument('--input_graph_knn_size', type=int, default=6,
                        help='Initial KNN graph size')
    parser.add_argument('--graph_skip_conn', type=float, default=0.4,
                        help='Ratio between KNN and feature-generated graph')
    
    parser.add_argument('--graphiter', type=int, default=8,
                        help='Maximum graph iterations')
    parser.add_argument('--update_adj_ratio', type=float, default=0.5,
                        help='Ratio for updating adjacency matrix')
    parser.add_argument('--eps_adj', type=float, default=0.9,
                        help='Threshold for stopping iteration')
    
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--lradj', type=str, default='type1',
                        help='Learning rate adjustment type')
    parser.add_argument('--epsilon', type=float, default=0.5,
                        help='Attention threshold for graph generation')
    parser.add_argument('--num_pers', type=int, default=4,
                        help='Number of perspectives for graph learning')
    parser.add_argument('--markoff_value', type=float, default=0.5,
                        help='Value for attention below epsilon')
    
    parser.add_argument('--MLPNums', type=int, default=4,
                        help='Number of MLP layers for graph generation')
    parser.add_argument('--ggdmodel', type=int, nargs='+', default=[16, 32, 64, 128],
                        help='Dimensions for each MLP layer')
    parser.add_argument('--G', type=float, default=1.5,
                        help='Gravitational constant G')
    parser.add_argument('--K', type=int, default=4,
                        help='Top k strongest gravitational effects per node')
    parser.add_argument('--NumAttentionHeads', type=int, default=4,
                        help='Number of self-attention heads')
    parser.add_argument('--HiddenDropoutProb', type=float, default=0.2,
                        help='Dropout for multi-head attention')
    parser.add_argument('--HiddenSize', type=int, default=256,
                        help='Hidden size for multi-head attention')
    
    parser.add_argument('--CrossNumAttation', type=int, default=6,
                        help='Number of cross-attention heads')
    
    parser.add_argument('--rho', type=float, default=0.5,
                        help='SAM hyperparameter')
    parser.add_argument('--weight_decay', type=float, default=5e-8,
                        help='Weight decay')
    parser.add_argument('--lr_reduce_factor', type=float, default=0.5,
                        help='Learning rate reduction factor')
    parser.add_argument('--lr_patience', type=int, default=2,
                        help='Patience for learning rate reduction')
    
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--printeveryepochs', type=int, default=1,
                        help='Print every N epochs')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Print every N batches')
    parser.add_argument('--eary_stop_metric', type=str, default='MSE',
                        help='Metric for early stopping')
    
    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() else False
    
    print(torch.cuda.is_available())
    print("args:")
    print(args)
    
    if args.task_name == 'long_term_forecast':
        Taskitem = Long_term_Forecast
    elif args.task_name == 'short_term_forecast':
        Taskitem = Short_term_Forecast
    
    if args.is_training:
        for i in range(args.itr):
            setting = ('{}_{}_{}_fq{}_sl{}_ll{}_pl{}_ci{}_co{}_dm{}_nw{}_el{}_tk{}_df{}_e{}_np{}_mv{}_gh{}_sr{}_igks{}_gkc{}_g{}_uar{}_ea{}_r{}_{}').format(
                args.task_name, args.data, args.features, args.freq,
                args.seq_len, args.label_len, args.pred_len,
                args.c_in, args.c_out, args.d_model, args.num_workers,
                args.e_layers, args.top_k, args.d_ff,
                args.epsilon, args.num_pers, args.markoff_value,
                args.graph_hops, args.sparsity_ratio,
                args.input_graph_knn_size, args.graph_skip_conn,
                args.graphiter, args.update_adj_ratio, args.eps_adj,
                args.rho, i)
            
            task = Taskitem(args)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            score = task.train()
            torch.cuda.empty_cache()
    else:
        for i in range(args.itr):
            setting = ('{}_{}_{}_fq{}_sl{}_ll{}_pl{}_ci{}_co{}_dm{}_nw{}_el{}_tk{}_df{}_e{}_np{}_mv{}_gh{}_sr{}_igks{}_gkc{}_g{}_uar{}_ea{}_r{}_{}').format(
                args.task_name, args.data, args.features, args.freq,
                args.seq_len, args.label_len, args.pred_len,
                args.c_in, args.c_out, args.d_model, args.num_workers,
                args.e_layers, args.top_k, args.d_ff,
                args.epsilon, args.num_pers, args.markoff_value,
                args.graph_hops, args.sparsity_ratio,
                args.input_graph_knn_size, args.graph_skip_conn,
                args.graphiter, args.update_adj_ratio, args.eps_adj,
                args.rho, i)
            
            task = Taskitem(args)
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            task.test(setting, test=1)
            torch.cuda.empty_cache()