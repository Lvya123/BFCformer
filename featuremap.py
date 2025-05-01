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
with open('test.yml', mode='r') as f_yml:
    Loader, _ = ordered_yaml()
    opt = yaml.load(f_yml, Loader=Loader)

gpus = ','.join([str(i) for i in opt['GPU']])

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

model_restoration = utils.get_arch(opt['MODEL'])
model_restoration.eval().cuda()

dir_name = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(dir_name, 'log', opt['MODEL']['NAME'] + '_' + opt['MODEL']['MODE'])
model_dir  = os.path.join(log_dir, 'models')
path_chk_rest = opt['VAL']['PRETRAIN_MODEL']
featuremap_dir = os.path.join(dir_name, 'featuremaps')
# print("path_chk_rest = ", path_chk_rest)
# path_chk_rest = os.path.join(log_dir, path_chk_rest)
path_chk_rest = r"/home/root1/data/lyn/DDN-72/log/UNet_small_baseline/models/model_epoch_180.pth"
utils.load_checkpoint(model_restoration, path_chk_rest)
val_epoch = utils.load_start_epoch(path_chk_rest)

val_dataset = get_validation_data(opt['PATH']['VAL_DATASET'], {'patch_size':opt['VAL']['VAL_PS']})
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)

out_list = []
inp_list = []

layer_dir = os.path.join(featuremap_dir, 'attention_D')


def forward_hooki(model, input, output):
    inp_list.append(input)

def forward_hooko(model, input, output):
    out_list.append(output)

# print(model_restoration.Decoder[2].layers[1].dau)
# exit(0)


hooki = model_restoration.Layers[0].layers[0].attention

hooki.register_forward_hook(forward_hooki)

hooko = model_restoration.Layers[-1].layers[-1].attention

hooko.register_forward_hook(forward_hooko)

print(hooki)
print('--------------------------')
print(hooko)

# exit(0)
for ii, data_val in enumerate(val_loader, 1):
    # if ii <= 618:
    #     continue
    img_path = os.path.join(layer_dir, str(ii))
    utils.mkdirs(img_path)

    # if ii == 1:
    input_ = data_val[1].cuda()
    with torch.no_grad():
        out_list = []
        inp_list = []

        input_ = data_val[1].cuda()
        factor = 8
        h,w = input_.shape[2],input_.shape[3]
        H,W = ((h+factor)//factor)*factor,((w+factor)//factor)*factor
        padh = H-h if h%factor!=0 else 0
        padw = W-w if w%factor!=0 else 0
        input_ = F.pad(input_,(0,padw,0,padh),'reflect')

        # print(input_.shape)

        restored = model_restoration(input_)[-1]

        # print('restored', restored.shape)

        # break
        restored = restored[...,:h,:w]
        featuremap = out_list[0]
        featuremap_in = inp_list[0]
        
        # print(featuremap.shape)
        # break
        # featuremapi = inp_list[0][0]
        # featuremapo = out_list[0]
        # featuremap = featuremapi + featuremapo
        featuremap = img_as_ubyte(torch.clamp(featuremap_in[0],0,1).cpu().squeeze().numpy().transpose(1,2,0))
        # featuremap = img_as_ubyte(torch.clamp(featuremap,0,1).cpu().squeeze().numpy().transpose(1,2,0))

        del input_
        del restored
        del out_list 
        del inp_list 

        torch.cuda.empty_cache()

    c = featuremap.shape[2]
    featuremap =featuremap[:h,:w,:]
    for i in range(c):
        file_name = str(i)+'.png'
        # utils.mkdirs(layer_dir)
        utils.save_img(os.path.join(img_path, file_name), featuremap[:,:,i])
        # import cv2
        # cv2.imwrite(os.path.join(img_path, file_name), featuremap[:,:,i])
    print(ii, 'finished')
    # break

# from natsort import natsorted
# import cv2
# import os
# import utils
