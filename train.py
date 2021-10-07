import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import math
import torch.optim as optim
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import pandas as pd
import torch_geometric.transforms as T

from dtb import DTBLayer
from dataset import *
from mdgnn import MDGNN
from utils import *
from config.config_dict import Config
from log.train_logger import TrainLogger

if __name__ == "__main__":
    config = Config()
    args = config.get_config()
    logger = TrainLogger(args)
    logger.info(__file__)

    save_model = args.get("save_model")
    data_path = args.get("data_root")

    property_names = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0',
                                'u298', 'h298', 'g298', 'cv']
    df = pd.read_csv(os.path.join(data_path, 'raw', 'gdb9.sdf.csv'))
    mu_list, std_list = [], [] 
    for pn in property_names:
        m = df[pn].mean()
        s = df[pn].std()
        mu_list.append(m)
        std_list.append(s)

    mu = torch.FloatTensor(mu_list).view(1, -1).cuda()
    std = torch.FloatTensor(std_list).view(1, -1).cuda()

    def val(model, dataloader, device):
        model.eval()
        running_loss = AverageMeter()
        running_loss_scale = AverageMeter()

        for data in dataloader:
            data = data.to(device)

            with torch.no_grad():
                pred = model(data)
                pred = pred.view(-1, 12)
                label = (data.y.view(-1, 12) - mu) / std
                pred_scale = (pred.view(-1, 12) * std) + mu
                label_scale = data.y.view(-1, 12)
                b, _ = label.shape
                loss = torch.abs(pred - label).mean(0) 
                loss_scale = torch.abs(pred_scale - label_scale).mean(0) 
                running_loss.update(loss, b)
                running_loss_scale.update(loss_scale, b)
            
        epoch_loss = running_loss.get_average()
        epoch_loss_scale = running_loss_scale.get_average()
        running_loss.reset()
        running_loss_scale.reset()

        model.train()

        return epoch_loss, epoch_loss_scale

    transform = T.Compose([Complete(), T.Distance(norm=True)])
    train_dataset, valid_dataset, test_dataset = load_dataset(path=data_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4)
    val_loader = DataLoader(valid_dataset, batch_size=512, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4)

    in_dim = train_dataset.num_node_features
    edge_in_dim = train_dataset.num_edge_features

    device = torch.device('cuda:0')
    model = MDGNN(in_dim, edge_in_dim).to(device)

    params_list = []
    for name, params in model.lin_share.named_parameters():
        if len(params.shape) > 1:
            params_list.append(params)

    shared_weights = params_list[-1]

    gn_layer = DTBLayer(shared_weights, 12, params_initial=[1.] * 12, alpha=1.0, lr=1e-3).cuda()

    epochs = 3000
    steps_per_epoch = 250
    num_iter = math.ceil((epochs * steps_per_epoch) / len(train_loader))
    break_flag = False

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss(reduction='none')

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=40, min_lr=1e-6)

    global_step = 0
    global_epoch = 0
    early_stop_epoch = 300

    running_loss = AverageMeter()
    running_task_loss = AverageMeter()
    running_task_difficulty = AverageMeter()
    running_best_mse = BestMeter("min")

    model.train()

    for i in range(num_iter):
        if break_flag:
            break

        for data in train_loader:
            global_step += 1       
            data = data.to(device)
            pred = model(data)

            pred = pred.view(-1, 12)
            label = (data.y.view(-1, 12) - mu) / std
            loss_task = criterion(pred, label.detach()).mean(0)
            
            loss = (loss_task * gn_layer.params_list.detach()).mean()

            gn_layer(*loss_task)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.update(loss.item(), data.y.size(0)) 
            running_task_loss.update(loss_task.detach().cpu().numpy(), label.size(0))
            running_task_difficulty.update(gn_layer.get_task_difficulty().detach().cpu().numpy(), 1)
    
            if global_step % steps_per_epoch == 0:
                global_epoch += 1

                epoch_loss = running_loss.get_average()
                running_loss.reset()
                
                epoch_task_loss = running_task_loss.get_average()
                running_task_loss.reset()

                epoch_task_difficulty = running_task_difficulty.get_average()
                running_task_difficulty.reset()

                val_loss, val_loss_scale = val(model, val_loader, device)
                val_loss = val_loss.detach().cpu().numpy()
                val_loss_scale = val_loss_scale.detach().cpu().numpy()

                msg_train = ["epoch-%d, train_loss-%.4f" % (global_epoch, epoch_loss)]
                for ip, prop in enumerate(property_names):
                    msg_train.append(f"{prop}_train: " + "%.4f" % epoch_task_loss[ip])
                msg_train = ', '.join(msg_train)
                logger.info(msg_train)

                msg_val_scale = ["epoch-%d, val_loss_scale-%.4f" % (global_epoch, val_loss_scale.mean())]
                for ip, prop in enumerate(property_names):
                    msg_val_scale.append(f"{prop}_val_scale: " + "%.4f" % val_loss_scale[ip])
                msg_val_scale = ', '.join(msg_val_scale)
                logger.info(msg_val_scale)

                msg_val = ["epoch-%d, val_loss-%.4f" % (global_epoch, val_loss.mean())]
                for ip, prop in enumerate(property_names):
                    msg_val.append(f"{prop}_val: " + "%.4f" % val_loss[ip])
                msg_val = ', '.join(msg_val)
                logger.info(msg_val)

                msg_weight = ["epoch-%d" % global_epoch]
                for w, prop in zip(gn_layer.params_list, property_names):
                    msg_weight.append(f"{prop}_weight: " + '%.3f' % w.detach().cpu().numpy())
                msg_weight = ', '. join(msg_weight)
                logger.info(msg_weight)

                msg_difficulty = ["epoch-%d" % global_epoch]
                for d, prop in zip(epoch_task_difficulty, property_names):
                    msg_difficulty.append(f"{prop}_difficulty: " + '%.3f' % d)
                msg_difficulty = ', '. join(msg_difficulty)
                logger.info(msg_difficulty)

                logger.info('-' * 10)

                scheduler.step(val_loss.mean())

                if val_loss.mean() < running_best_mse.get_best():
                    running_best_mse.update(val_loss.mean())
                    if save_model:
                        save_model_dict(model, logger.get_model_dir(), msg_val)
                else:
                    count = running_best_mse.counter()
                    if count > early_stop_epoch:
                        logger.info(f"early stop in epoch {global_epoch}")
                        break_flag = True
                        break


