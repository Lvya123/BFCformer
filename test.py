from utils.data_RGB import get_validation_data
from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as PSNR
import numpy as np
import os
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import utils
from logger import *
import yaml
from pytorch_msssim import ssim
from einops import rearrange
with open('test.yml', mode='r') as f_yml:
    Loader, _ = ordered_yaml()
    opt = yaml.load(f_yml, Loader=Loader)

gpus = ','.join([str(i) for i in opt['GPU']])

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

model_restoration = utils.get_arch(opt['MODEL'])

dir_name = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(dir_name, 'log', opt['MODEL']['NAME'] + '_' + opt['MODEL']['MODE'])
result_dir = os.path.join(log_dir, 'results')
model_dir  = os.path.join(log_dir, 'models')


path_chk_rest = os.path.join(model_dir, opt['VAL']['PRETRAIN_MODEL'])
utils.load_checkpoint(model_restoration, path_chk_rest)
# val_epoch = utils.load_start_epoch(path_chk_rest)

model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

val_dataset = get_validation_data(opt['PATH']['VAL_DATASET'], {'patch_size':opt['VAL']['VAL_PS']})
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)


def calculate_psnr_torch(img1, img2):
  b, c, h, w = img1.shape
  v = torch.tensor([[65.481/255], [128.553/255], [24.966/255]]).cuda()
  img1 = torch.mm(img1.permute(0, 2, 3, 1).reshape(-1, c), v) + 16./255
  img2 = torch.mm(img2.permute(0, 2, 3, 1).reshape(-1, c), v) + 16./255
  img1 = img1.reshape(b, h, w, -1)
  img2 = img2.reshape(b, h, w, -1)
  mse_loss = F.mse_loss(img1, img2, reduction='none').mean((1, 2, 3))
  psnr_full = 10 * torch.log10(1 / mse_loss).mean()
  sim = ssim(img1.permute(0, 3, 1, 2), img2.permute(0, 3, 1, 2), data_range=1, size_average=False).mean()
  # print(mse)
  return psnr_full, sim


psnr_val_rgb = []
ssim_val_rgb = []

for ii, data_val in enumerate(val_loader, 1):

    tar_img = data_val[0].cuda()
    input_ = data_val[1].cuda()

    factor = 8
    h,w = input_.shape[2],input_.shape[3]
    H,W = ((h+factor)//factor)*factor,((w+factor)//factor)*factor
    padh = H-h if h%factor!=0 else 0
    padw = W-w if w%factor!=0 else 0
    input_ = F.pad(input_,(0,padw,0,padh),'reflect')

    with torch.no_grad():
        output = model_restoration(input_)
        out = output[0]
        out = out[:,:h,:w]


    output = rearrange(out, 'c h w -> 1 c h w')
    psnr_full, sim = calculate_psnr_torch(output, tar_img)

    psnr_val_rgb.append(psnr_full)
    ssim_val_rgb.append(sim)

    restored_img = torch.clamp(output,0,1).cpu().numpy().squeeze().transpose((1,2,0))
    


    ######### release testing gpu memory ##########
    del output
    del input_
    torch.cuda.empty_cache()

    log('%-6s \t %f \t %f' % (data_val[2][0], psnr_val_rgb[-1], ssim_val_rgb[-1]), os.path.join(log_dir, 'val_' + 'BFC'+'.txt'), P=True)
    if opt['VAL']['SAVE_IMG']:
        save_img_path = os.path.join(result_dir, 'python-' + 'BFC')
        file_name = data_val[2][0] + '.png'
        
        utils.mkdir(save_img_path)
        utils.save_img(os.path.join(save_img_path, file_name), restored_img)

avg_psnr  = sum(psnr_val_rgb)/ii
avg_ssim  = sum(ssim_val_rgb)/ii
log('total images = %d \t avg_psnr = %f \t avg_ssim = %f' % (ii,avg_psnr,avg_ssim), os.path.join(log_dir, 'val_' + 'BFC'+'.txt'), P=True)