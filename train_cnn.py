from model import *
import argparse
from dataset import *
import matplotlib.pyplot as plt
from utils import *
from tqdm import tqdm
import sys
import time
import yaml

def get_args(get_default = False):
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--batch_size", type=int, default=32)
    # parser.add_argument("--layer_type", type=str, default="DLL_OUTER_W_ConvLayer")
    parser.add_argument("--layer_type", type=str, default="DLL_INNER_W_ConvLayer")
    # parser.add_argument("--layer_type", type=str, default="ConvLayer")
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--network_type",type=str,default="dll")

    # train config
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-5)
    parser.add_argument("--scheduler_type", type=str, default="adam")
    parser.add_argument("--nolog", action='store_true', help="disable logging")
    parser.add_argument("--earlystop_steps", type=int, default=15)

    # subj and sst2 config
    parser.add_argument("--filters", type=str, default="3,4,5")
    parser.add_argument("--layers", type=str, default="512,256,128")
    parser.add_argument("--filter_num", type=int, default=100) 
    
    if __name__ != '__main__' or get_default:
        return parser.parse_args([])
    
    return parser.parse_args()

def run(args, temp_print=None):
    args.enable_adam = False
    args.enable_sgd = False
    if args.scheduler_type == "adam":
        args.enable_adam = True
    elif args.scheduler_type == "sgd":
        args.enable_sgd = True

    # dataset config
    with open('settings.yaml') as f:
        settings = yaml.safe_load(f)
    dataset_config = settings[args.dataset]
    
    args.channel = dataset_config['channel']
    args.input_size = dataset_config['input_size']
    if 'input_size_c' in dataset_config:
        args.input_size_c = dataset_config['input_size_c'] 
    args.num_class = dataset_config['num_class']

    def custom_print(*args, **kwargs):
        message = " ".join(map(str, args)) + "\n"
        sys.stdout.write(message)

    print = temp_print if temp_print else custom_print

    for k,v in sorted(vars(args).items()):
        print(k,'=',v)
    setup_seed(args.seed)
    # global DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data, train_loader, test_data, test_loader = get_dataset(args.dataset, args.batch_size)
    
    # model = PC_CNN_Model(args, device=DEVICE)
    if args.network_type == "pc":
        model = PC_CNN_Model(args, DEVICE)
    # elif args.network_type == "bp":
    #     model = BP_CNN_Model(args, DEVICE)
    # elif args.network_type == "pcfa":
    #     args.fa = True
    #     model = PC_CNN_Model(args, DEVICE)
    # elif args.network_type == "ipc":
    #     args.inference_steps = 1
    #     model = PC_CNN_Model(args, DEVICE)
    elif args.network_type == "dll":
        model = DLL_CNN_Model(args, device=DEVICE)
    else:
        raise Exception("Unknown network type entered")

    model.print_layers(print)

    scheduler_functions = {
        "cosine": cosine_scheduler,
        "linear": linear_scheduler,
        "adam": linear_scheduler,
        "sgd": none_scheduler,
        "none": none_scheduler
    }

    acc_max = -1
    for epoch in range(args.epochs):
        # print(f"epoch: {epoch}")
        
        start_time = time.time()
        total_loss = 0
        count = 0
        for inputs, label in tqdm(train_loader):
            inputs = inputs.float().to(DEVICE)
            label = label.to(DEVICE)
            # print(inputs.shape)
            bs, c, h, w= inputs.shape # B C H W
            assert c == args.channel, "channel unequal"
            one_hot_label = F.one_hot(label, num_classes=args.num_class).float()
            smoothed_label = label_smoothing(one_hot_label)
            model.forward(inputs)
            with torch.no_grad():
                model.iterate(smoothed_label.float())
                # model.iterate(one_hot_label.float())
            logits = model.u[-1] # B Label
            loss = F.cross_entropy(logits, smoothed_label).item()
            # loss = F.cross_entropy(logits, one_hot_label).item()
            total_loss += loss
            count += 1
            for layer in model.layers:
                if isinstance(layer, list):
                    for part in layer:
                        if hasattr(part, 'learning_rate'):
                            part.learning_rate = scheduler_functions[args.scheduler_type](epoch=epoch, total_epochs=args.epochs, initial_lr=args.learning_rate)
                else:
                    layer.learning_rate = scheduler_functions[args.scheduler_type](epoch=epoch, total_epochs=args.epochs, initial_lr=args.learning_rate)
            model.update_weights(epoch)
        
        end_time = time.time()
        print(f"avg_loss in epoch {epoch}: {total_loss/count}")
        # print(f"Time taken for epoch {epoch}: {end_time - start_time:.2f} seconds")

        # eval
        # print("begin eval...")
        y_true = []
        y_pred = []
        for inputs, label in tqdm(test_loader):
            inputs = inputs.to(DEVICE)
            label = label.to(DEVICE)
            bs, c, h, w= inputs.shape # B C H W
            assert c == args.channel, "channel unequal"
            y_true.extend(label.tolist())
            with torch.no_grad():
                model.forward(inputs)
            logits = model.u[-1] # B Label
            y_pred.extend(torch.argmax(logits,1))
            
        correct = 0
        for i in range(len(y_true)):
            correct += 1 if y_true[i] == y_pred[i] else 0
        test_acc = correct / len(y_pred)
        print("test_acc: ", test_acc)
        if acc_max < test_acc:
            acc_max = test_acc
            # torch.save(model, f"ckpt/{args.dataset}_{args.layer_type}_acc_{acc_max}_{str(args.dim_list)}.pt")
            early_stop = args.earlystop_steps
            acc_max_epoch = epoch
        else:
            early_stop -= 1
            if early_stop <= 0:
                break
    print("acc_max: ", acc_max)
    return acc_max, acc_max_epoch


if __name__ == '__main__':
    args = get_args()
    if not args.nolog:
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
        if logger_name[-1] == '_': logger_name = logger_name[:-1]
        print(f"{dataset_folder} {logger_name}")
        temp_logger = SimpleLogger(f"tmplogs/{dataset_folder}", f"{logger_name}")
        
        def temp_print(*args, **kwargs):
            temp_logger.info(" ".join(map(str, args)))
        run(args, temp_print)
    else: 
        run(args)