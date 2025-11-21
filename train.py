from data import dataloader_train
import torch
import numpy as np
from tqdm import tqdm
import argparse
import os
import time
from ops.utils import str2bool, step_lr, get_lr
import pdb
from model.PETMamba import PETMamba

# time.sleep(21600)

parser = argparse.ArgumentParser()
#model
parser.add_argument("--gpus", '--list', action='append', type=int, help='GPU', default=[0])

#training
parser.add_argument("--lr", type=float, dest="lr", help="ADAM Learning rate", default=5e-3)
parser.add_argument("--lr_step", type=int, dest="lr_step", help="ADAM Learning rate step for decay", default=80)
parser.add_argument("--lr_decay", type=float, dest="lr_decay", help="ADAM Learning rate decay (on step)", default=0.35)
parser.add_argument("--eps", type=float, dest="eps", help="ADAM epsilon parameter", default=1e-3)
parser.add_argument("--num_epochs", type=int, dest="num_epochs", help="Total number of epochs to train", default=300)
parser.add_argument("--train_batch", type=int, default=2, help='batch size during training')

#data
parser.add_argument("--model_save_root", type=str, default='checkpoints')
parser.add_argument("--model_name", type=str, dest="model_name", help="The name of the model to be saved.", default=None)
parser.add_argument("--train_path", type=str, default="xxx")
parser.add_argument("--tqdm", type=str2bool, default=True)
parser.add_argument("--resume", type=str, dest="resume", help='Resume training of the model',default=None)

#model
parser.add_argument("--inner_num", type=int, default=24)
parser.add_argument("--iter_num", type=int, default=4)
parser.add_argument("--frame_num", type=int, default=24)
parser.add_argument("--which_model", type=str, default='PETMamba')
parser.add_argument("--img_hw", type=int, default=128)

parser.add_argument("--drop_rate", type=int, default=0)
parser.add_argument("--attn_drop_rate", type=int, default=0)
parser.add_argument("--d_state", type=int, default=32)
parser.add_argument("--patch", type=int, default=128)


args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
capability = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else os.cpu_count()
gpus=args.gpus
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
if device.type=='cuda':
    torch.cuda.set_device('cuda:{}'.format(gpus[0]))


train_loader = dataloader_train.get_dataloader(args)


if args.which_model == 'PETMamba':
    model = PETMamba(args).to(device=device)


if device.type=='cuda':
    model = torch.nn.DataParallel(model.to(device=device), device_ids=gpus)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=args.eps)

criterion = torch.nn.MSELoss(reduction='sum')

pytorch_total_params = sum(p.numel() for p in model.parameters())
print(f'Arguments: {vars(args)}')
print('Nb tensors: ',len(list(model.named_parameters())), "; Trainable Params: ", pytorch_total_params, "; device: ", device,
      "; name : ", device_name)

model_save_root = os.path.join(args.model_save_root, args.model_name)
if not os.path.exists(model_save_root):
    os.makedirs(model_save_root)
log_file_name = os.path.join(model_save_root, 'log.txt')

start_epoch = 0
if args.resume:
    try:
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model = checkpoint['state_dict']
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")
    except Exception as e:
        print(e)
        print(f'ckpt loading failed @{ckpt_path}, exit ...')
        exit()

print(f'... starting training ...\n')


for epoch in range(start_epoch, args.num_epochs):
    tic = time.time()
    num_iters = 0
    psnr_set = 0
    loss_set = 0

    model.train()

    if (epoch % args.lr_step) == 0 and (epoch != 0) :
        step_lr(optimizer, args.lr_decay)

    for batch in tqdm(train_loader, disable=not args.tqdm):
        img_clean, img_noise = batch
        img_clean, img_noise = img_clean.to(device=device), img_noise.to(device=device)

        output = model(img_noise)
        loss = criterion(output, img_clean) / args.train_batch

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_psnr = -10 * torch.log10((output - img_clean).pow(2).mean([1, 2, 3])).mean()

        psnr_set += loss_psnr.item()
        loss_set += loss.item()
        num_iters += 1

    tac = time.time()
    psnr_set /= num_iters
    loss_set /= num_iters

    tqdm.write(f'epoch {epoch} - train psnr: {psnr_set:0.4f} ({tac-tic:0.1f} s,  {(tac - tic) / num_iters:0.3f} s/iter, lr {get_lr(optimizer):0.1e}, loss: {loss_set:0.4f})')
    
    with open(f'{log_file_name}', 'a') as log_file:
        log_file = open(log_file_name, 'a')
        log_file.write(
            f'epoch {epoch} - train psnr: {psnr_set:0.4f} loss: {loss_set:0.4f} ({(tac - tic) / num_iters:0.3f} s/iter,  lr {get_lr(optimizer):0.2e})\n')

    torch.save({'epoch': epoch,
                     'config': vars(args),
                     'state_dict': model,
                     'optimizer': optimizer.state_dict()},  os.path.join(model_save_root+'/ckpt_'+str(epoch)))
