from model import *
import argparse
from dataset import get_seq_dataset
import matplotlib
import matplotlib.pyplot as plt
from utils import *
from tqdm import tqdm
import os
import numpy as np
import sys

def get_args(get_default = False):
    parser = argparse.ArgumentParser()
    # dataset config
    parser.add_argument("--dataset", type=str, default='harrypotter')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len",type=int,default=32)
    parser.add_argument("--hidden_size",type=int,default=128)
    parser.add_argument("--weight_learning_rate",type=float,default=1e-3)
    parser.add_argument("--epochs",type=int, default=1000000)
    parser.add_argument("--inference_steps",type=int, default=100)
    parser.add_argument("--inference_learning_rate",type=float,default=1e-1)
    parser.add_argument("--fixed_predictions",type=bool,default=True)
    parser.add_argument("--theta_update_discount",type=int,default=1)
    parser.add_argument("--fix_theta_until",type=int,default=0)
    parser.add_argument("--noise",type=float,default=0)
    parser.add_argument("--noclamp", action='store_true', help="disable dll clamp")

    parser.add_argument("--network_type",type=str,default="dll")
    
    parser.add_argument("--scheduler_type", type=str, default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fn", type=str, default="tanh")

    parser.add_argument("--num_steps", type=int, default=4)
    parser.add_argument("--save", type=bool, default=False)
    
    parser.add_argument("--earlystop_steps", type=int, default=-1)
    parser.add_argument("--verbose", action='store_true', help="Enable verbose output")
    parser.add_argument("--nolog", action='store_true', help="disable logging")
    parser.add_argument("--accelerate", action='store_true', help="enable pc accelerate")

    if __name__ != '__main__' or get_default:
        return parser.parse_args([])
    
    return parser.parse_args()
    

def run(args, temp_print=None):
    def custom_print(*args, **kwargs):
        message = " ".join(map(str, args)) + "\n"
        sys.stdout.write(message)

    print = temp_print if temp_print else custom_print
        
    for k,v in sorted(vars(args).items()):
        print(k,'=',v)
    setup_seed(args.seed)
    # global DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # B L V
    train_loader, test_loader, args.input_size, args.output_size, rnn_eval = get_seq_dataset(args.dataset, args.batch_size, args.seq_len, args.verbose)

    args.fn = tanh
    args.fn_deriv = tanh_deriv

    args.fa = False
    #define networks
    if args.network_type == "bp":
        model = BP_RNN_Model(args, DEVICE)
    elif args.network_type == "pc":
        model = PC_RNN_Model(args, DEVICE)
    elif args.network_type == "pcfa":
        args.fa = True
        model = PC_RNN_Model(args, DEVICE)
    elif args.network_type == "ipc":
        args.inference_steps = 1
        model = PC_RNN_Model(args, DEVICE)
    elif args.network_type == "dll":
        model = DLL_RNN_Model(args, DEVICE)
    else:
        raise Exception("Unknown network type entered")

    scheduler_functions = {
        "cosine": cosine_scheduler,
        "linear": linear_scheduler,
        "none": none_scheduler
    }

    best_score = -1
    for epoch in range(args.epochs):
        print(f"epoch: {epoch}")
        print("begin train...")
        rnn_eval.train_reset(epoch)
        for input_seq, target_seq in train_loader:
            # break
            if input_seq.shape[0] != args.batch_size: continue 
            input_seq = input_seq.permute(1, 2, 0).to(DEVICE) # L V B
            target_seq = target_seq.permute(1, 2, 0).to(DEVICE) # L V B
            output_seq = model.forward(input_seq)
            rnn_eval(output_seq, target_seq)

            model.update_weights(target_seq, epoch)

        rnn_eval.summarize(print)
        
        model.learning_rate = scheduler_functions[args.scheduler_type](epoch=epoch, total_epochs=args.epochs, initial_lr=args.weight_learning_rate)

        # eval
        print("begin eval...")
        rnn_eval.test_reset(epoch)
        for input_seq, target_seq in test_loader:
            if input_seq.shape[0] != args.batch_size: continue
            input_seq = input_seq.permute(1, 2, 0)
            target_seq = target_seq.permute(1, 2, 0)
            input_seq = input_seq.to(DEVICE)
            target_seq = target_seq.to(DEVICE)

            output_seq = model.forward(input_seq) # L V B
            rnn_eval(output_seq, target_seq)
            
        best_score, epoch_num, no_improve_cnt = rnn_eval.summarize(print)
        # print(no_improve_cnt)
        if args.earlystop_steps != -1 and no_improve_cnt >= args.earlystop_steps:
            break
    print(f"best_score:{best_score}\tepoch_num:{epoch_num}")
    return best_score, epoch_num

if __name__ == '__main__':
    args = get_args()
    if  not args.nolog:
        default_args = get_args(get_default=True)
        network_type = vars(args)["network_type"]
        dataset_folder = vars(args)["dataset"]
        diff_keys = {key for key in vars(args).keys() if key in vars(default_args).keys() and vars(args)[key] != vars(default_args)[key]}
        diff_keys.discard("network_type")
        diff_keys.discard("dataset")
        diff_keys.discard("accelerate")
        diff_keys.discard("verbose")
        diff_keys.discard("earlystop_steps")
        diff_keys = sorted(diff_keys)
        logger_name_parts = [f"{key[:3]}_{vars(args)[key]}" for key in diff_keys]
        logger_name = f"{network_type}_" + "_".join(logger_name_parts)
        print(f"{dataset_folder} {logger_name}")
        temp_logger = SimpleLogger(f"tmplogs/{dataset_folder}", f"{logger_name}")
        
        def temp_print(*args, **kwargs):
            temp_logger.info(" ".join(map(str, args)))
        run(args, temp_print)
    else: 
        run(args)
