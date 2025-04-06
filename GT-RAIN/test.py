# from utils.data_RGB import get_validation_data
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
from natsort import natsorted
from glob import glob
from tqdm.notebook import tqdm
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim 


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
val_epoch = utils.load_start_epoch(path_chk_rest)

print("===>Testing using weights of epoch: ",val_epoch)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

# val_dataset = get_validation_data(opt['PATH']['VAL_DATASET'], {'patch_size':opt['VAL']['VAL_PS']})
# val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)

scene_paths = natsorted(glob(f"{opt['PATH']['VAL_DATASET']}/*"))


total_PSNR_output = 0
total_SSIM_output = 0

for scene_path in tqdm(scene_paths):
    scene_name = scene_path.split('/')[-1]
    clean_img_path = glob(scene_path + '/*C-000.png')[0]
    rainy_img_paths = natsorted(glob(scene_path + '/*R-*.png'))
    scene_PSNR_output = 0
    scene_SSIM_output = 0
  
    for i in tqdm(range(len(rainy_img_paths))):
        filename = rainy_img_paths[i].split('/')[-1][:-4]
        img = Image.open(rainy_img_paths[i])
        gt_img = Image.open(clean_img_path)
        img = np.array(img, dtype=np.float32)
        img *= 1/255
        gt_img = np.array(gt_img, dtype=np.float32)
        gt_img *= 1/255
        

        input = torch.from_numpy(img).permute((2, 0, 1)) 
        input = torch.unsqueeze(input, 0).cuda()
        # print('gt',gt_img.shape)
        h1, w1 = input.shape[2], input.shape[3]
        if h1<128 or w1<128:
            ph = 128-h1
            pw = 128-w1
            input = F.pad(input, (0,pw,0,ph),mode='reflect')
            # print('input',input.shape)


        with torch.no_grad():
            # output = model_restoration(input)[0].squeeze().permute((1, 2, 0))
            output = model_restoration(input).squeeze(0).permute(1, 2, 0)
            # print('out',output.shape)
        output = output.detach().cpu().numpy()
        # print('output',output.shape)
        # print('output.max',output.max())
        # print('output.type',type(output.max))
        output = output[:h1,:w1:,]


        # USE THIS BLOCK TO SAVE
        im = Image.fromarray((output*255).astype(np.uint8))
        save_img_path = os.path.join(result_dir, 'epoch_' + str(val_epoch))

        

        if opt['VAL']['SAVE_IMG']:
            utils.mkdir(f"{save_img_path}/{scene_name}")
            im.save(f"{save_img_path}/{scene_name}/{filename}.png")

        # print('output',output.shape)
        # print('gt',gt_img.shape)
        # print('gt_img',gt_img.min())
        # print('output',output.min())
        scene_PSNR_output += psnr(gt_img, output)
        scene_SSIM_output += ssim(gt_img, output, data_range=1., channel_axis=-1)

    print(f"Scene PSNR Output: {scene_PSNR_output/len(rainy_img_paths)}")
    print(f"Scene SSIM Output: {scene_SSIM_output/len(rainy_img_paths)}")

    total_PSNR_output += scene_PSNR_output/len(rainy_img_paths)
    total_SSIM_output += scene_SSIM_output/len(rainy_img_paths)

num_scenes = len(scene_paths)
print(f"Total PSNR Output: {total_PSNR_output/num_scenes}")
print(f"Total SSIM Output: {total_SSIM_output/num_scenes}")

log('avg_psnr = %f \t avg_ssim = %f' % (total_PSNR_output/num_scenes, total_SSIM_output/num_scenes), os.path.join(log_dir, 'val_' + str(val_epoch)+'.txt'), P=True)


