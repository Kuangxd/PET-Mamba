import torch
import numpy as np
import argparse
import os
import pdb



parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default='checkpoints/zubal/ckpt_285')
parser.add_argument("--data_path", type=str, default='test-imgs/zubal')
args = parser.parse_args()


model = torch.load(args.model_path)['state_dict']
model.eval()

pytorch_total_params = sum(p.numel() for p in model.parameters())
print('Nb tensors: ',len(list(model.named_parameters())), "; Trainable Params: ", pytorch_total_params)




for root, _, files in os.walk(args.data_path):
    for file in files:
        img_noise_path = os.path.join(root, file)
        img_noise = np.fromfile(img_noise_path, dtype=np.float32).reshape(128, 128, 24).transpose(2, 0, 1)

        img = torch.from_numpy(img_noise)
        img = img.unsqueeze(0).cuda().float()

        output = model(img).squeeze().detach().cpu().numpy()
        output[output<0] = 0


        save_path = os.path.join('results')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        output.transpose(1,2,0).tofile(os.path.join(save_path, file))