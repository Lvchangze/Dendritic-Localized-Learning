from model import *
import argparse
from dataset import get_dataset
import matplotlib.pyplot as plt
from utils import *
from tqdm import tqdm
import random
import ast
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dataset config
    # parser.add_argument("--dataset", type=str, default='fashionmnist')
    parser.add_argument("--dataset", type=str, default='mnist')
    parser.add_argument("--channel", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_class", type=int, default=10)
    
    # model config 
    # dim_list[0] = inputs.C*H*W, mnist = 784, fashionmnist=784, cifar10 = 3072, svhn = 3072
    parser.add_argument("--dim_list", type=str, default="[784, 1024, 512, 256, 10]")
    parser.add_argument("--layer_type", type=str, default="DLL_FCLayer")
    parser.add_argument("--sigma", type=float, default=1.0)

    # train config
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--scheduler_type", type=str, default="adam")
    parser.add_argument("--earlystop_steps", type=int, default=20)
    
    args = parser.parse_args()
    args.dim_list = eval(args.dim_list)

    args.enable_adam = False
    if args.scheduler_type == "adam":
        args.enable_adam = True

    for k,v in sorted(vars(args).items()):
        print(k,'=',v)
    setup_seed(args.seed)
    # global DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data, train_loader, test_data, test_loader = get_dataset(args.dataset, args.batch_size)
    model = MLP_Model(args, device=DEVICE)
    scheduler_functions = {
        "cosine": cosine_scheduler,
        "linear": linear_scheduler,
        "adam": linear_scheduler,
        "none": none_scheduler
    }

    acc_max = -1
    for epoch in range(args.epochs):
        # print(f"epoch: {epoch}")
                
        start_time = time.time()
        total_loss = 0
        count = 0
        for inputs, label in tqdm(train_loader):
            inputs = inputs.to(DEVICE)
            label = label.to(DEVICE)
            # print(inputs.shape)
            bs, c, h, w= inputs.shape # B C H W
            assert c == args.channel, "channel unequal"
            inputs = inputs.flatten(-3, -1) # B CHW
            one_hot_label = F.one_hot(label, num_classes=args.num_class).float()
            smoothed_label = label_smoothing(one_hot_label)
            model.forward(inputs)
            with torch.no_grad():
                model.iterate(smoothed_label.float())
                # model.iterate(one_hot_label.float())
            logits = model.u[-1]
            loss = F.cross_entropy(logits, smoothed_label).item()
            # loss = F.cross_entropy(logits, one_hot_label).item()
            total_loss += loss
            count += 1
            model.update_weights(epoch)
                
        end_time = time.time()
        print(f"avg_loss in epoch {epoch}: {total_loss/count}")
        print(f"Time taken for epoch {epoch}: {end_time - start_time:.2f} seconds")
        for layer in model.layers:
            layer.learning_rate = scheduler_functions[args.scheduler_type](epoch=epoch, total_epochs=args.epochs, initial_lr=args.learning_rate)

        # eval
        print("begin eval...")
        y_true = []
        y_pred = []
        for inputs, label in tqdm(test_loader):
            inputs = inputs.to(DEVICE)
            label = label.to(DEVICE)
            bs, c, h, w= inputs.shape # B C H W
            assert c == args.channel, "channel unequal"
            inputs = inputs.flatten(-3, -1) # B CHW
            y_true.extend(label.tolist())
            with torch.no_grad():
                model.forward(inputs)
            logits = model.x[-1]
            # print(f"torch.argmax(logits,1): {torch.argmax(logits,1)}")
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
