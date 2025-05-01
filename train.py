import os
from logger import ordered_yaml, dict2str, log
import yaml
with open('train.yml', mode='r') as f_yml:
    Loader, _ = ordered_yaml()
    opt = yaml.load(f_yml, Loader=Loader)

gpus = ','.join([str(i) for i in opt['GPU']])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import torch
torch.backends.cudnn.benchmark = True
from visdom import Visdom
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from collections import OrderedDict
import random
import time
import datetime
from tqdm import tqdm
from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as PSNR

import utils
from utils.data_RGB import get_training_data, get_validation_data
import utils.losses as losses

from ptflops import get_model_complexity_info

######### Logs dir ###########
dir_name = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(dir_name, 'log', opt['MODEL']['NAME'] + '_' + opt['MODEL']['MODE'])
utils.mkdir(log_dir)

train_log = os.path.join(log_dir, datetime.datetime.now().isoformat()+'.txt') 
print("Now time is : ",datetime.datetime.now().isoformat())
result_dir = os.path.join(log_dir, 'results')
model_dir  = os.path.join(log_dir, 'models')
utils.mkdir(result_dir)
utils.mkdir(model_dir)

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

######### Model ###########
model_restoration = utils.get_arch(opt['MODEL'])
num_params = 0
with open(train_log,'a') as f:
    f.write(dict2str(opt)+'\n')
    f.write(str(model_restoration)+'\n')
    
    for param in model_restoration.parameters():
        num_params += param.numel()
        
    f.write('parameters:' + str(num_params))
    

model_restoration.cuda()
# macs_hand = model_restoration.flops()
# log('macs_hand = %.2f GMac' % macs_hand, train_log,P=False)
macs, params = get_model_complexity_info(model_restoration, (3,256,256), as_strings=True,print_per_layer_stat = True, verbose=True)
log('macs = %s \t params = %s'%(macs, params), train_log)
x = torch.rand(1, 3, 256, 256)
# print(model_restoration.flops(x) / 1e9)
log('+ macs = %s GMac' % (model_restoration.flops(x) / 1e9), train_log)


######### Optimizer  ###########
start_epoch = 1
optimizer = torch.optim.Adam(model_restoration.parameters(), **opt['OPTIM'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **opt['SCHE'])

######### Resume ###########
if opt['TRAIN']['RESUME']:
    path_chk_rest = utils.get_last_path(model_dir, opt['TRAIN']['PRETRAIN_MODEL'])
    utils.load_checkpoint(model_restoration,path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)

    for param in optimizer.param_groups:
        param['lr'] = opt['OPTIM']['lr']

    for i in range(1, 1353*300+1):
        scheduler.step()
    new_lr = scheduler.get_last_lr()[0]
    
    log('------------------------------------------------------------------------------',train_log)
    log("==> Resuming Training with learning rate:%.20f"%new_lr, train_log)
    log('------------------------------------------------------------------------------',train_log)

######### Loss ###########
criterion_L1 = torch.nn.L1Loss()

######### DataLoaders ###########
train_dataset = get_training_data(opt['PATH']['TRAIN_DATASET'], {'patch_size':opt['TRAIN']['TRAIN_PS']})
train_loader = DataLoader(dataset=train_dataset, batch_size=opt['TRAIN']['BATCH_SIZE'], shuffle=True, num_workers=8, drop_last=False, pin_memory=False)

val_dataset = get_validation_data(opt['PATH']['VAL_DATASET'], {'patch_size':opt['TRAIN']['VAL_PS']})
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False, pin_memory=False)

trainset_len = train_dataset.__len__()
valset_len = val_dataset.__len__()
total_iters = trainset_len/opt['TRAIN']['BATCH_SIZE']
log('trainset length: %d \ttotal iters per epoch: %d \tvalset length: %d'%(trainset_len, total_iters, valset_len), train_log)

######### Visdom ###########

log("------------------------------------------------------------------",train_log)
print('==> visdom initial')
window_loss = Visdom(port=opt['PORT']) 
window_rain = Visdom(port=opt['PORT'])
window_restored = Visdom(port=opt['PORT'])
window_sharp = Visdom(port=opt['PORT'])
window_loss.line([[0.,0.]], [0.], win='train_loss',opts=dict(title='train_loss',legend=['loss_L1','loss_FFT'],xlabel='epoch',ylabel='loss'))
log("------------------------------------------------------------------\n",train_log)


best_psnr = 0
best_epoch = 0

# if not opt['TRAIN']['RESUME'] and opt['WARM_UP']['use_warm_up']:
#     log("start warm up, total warm up epoch:%-4d" % (opt['WARM_UP']['warm_up_epoch']), train_log)
#     log("------------------------------------------------------------------",train_log)
#     warm_up_optim = torch.optim.Adam(model_restoration.parameters(), lr=opt['WARM_UP']['warm_up_lr'])
#     for epo in range(1, (opt['WARM_UP']['warm_up_epoch']) + 1):
#         model_restoration.train()
#         start_warm_up_time = time.time()
#         warm_up_epoch_loss = 0
#         for iter, data in enumerate(train_loader, 1):
#             # zero_grad
#             optimizer.zero_grad()

#             target = data[0].cuda()
#             input_ = data[1].cuda()

#             restored = model_restoration(input_)

#             # Compute loss 
#             loss_L1 = criterion_L1(restored,target)
#             restored_fft = torch.fft.rfft2(restored,dim=(-2,-1))
#             target_fft = torch.fft.rfft2(target,dim=(-2,-1))
#             loss_fft = criterion_L1(restored_fft,target_fft)*0.1
            
#             loss = loss_L1 + loss_fft
        
#             loss.backward()
#             warm_up_epoch_loss += loss.item()
#             torch.nn.utils.clip_grad_norm_(model_restoration.parameters(),0.01)
#             optimizer.step()
#         log("WU_Epoch:%-4d\tWU_Time:%-4f\tLoss:%.6f \tLR:%.20f"%(epo, time.time()-start_warm_up_time, warm_up_epoch_loss,opt['WARM_UP']['warm_up_lr']),train_log)
#     log("end warm up, total using time :%-4f" % (time.time() - start_warm_up_time), train_log)
#     log("------------------------------------------------------------------\n",train_log)


######### Train ###########
log("------------------------------------------------------------------",train_log)
log('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt['TRAIN']['TOTAL_EPOCHS']),train_log)
log("------------------------------------------------------------------",train_log)

for epoch in range(start_epoch, opt['TRAIN']['TOTAL_EPOCHS'] + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    model_restoration.train()
    for iter, data in enumerate(train_loader, 1):

        # zero_grad
        optimizer.zero_grad()

        target = data[0].cuda()
        input_ = data[1].cuda()

        restored = model_restoration(input_)

        # Compute loss 
        loss_L1 = criterion_L1(restored,target)
        restored_fft = torch.fft.rfft2(restored,dim=(-2,-1))
        target_fft = torch.fft.rfft2(target,dim=(-2,-1))
        loss_fft = criterion_L1(restored_fft,target_fft)*0.1
        
        loss = loss_L1 + loss_fft
       
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_restoration.parameters(),0.01)
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()

        if iter % opt['TRAIN']['PRINT_FRE'] == 0:
            log("Epoch:%-4d \tIter:%-4d \tTime:%-4.f \tL1_loss:%.6f \tFFT_loss:%.6f \tLR:%.20f"%(epoch, iter, time.time()-epoch_start_time, 
            loss_L1, loss_fft*0.1,scheduler.get_last_lr()[0]),train_log)
            
            step = epoch + iter/total_iters
            window_loss.line([[loss_L1.item(),loss_fft.item()*0.1]], [step], win = 'train_loss', update='append')
            window_rain.images(input_[0], nrow=1, win='rain_Train', opts=dict(title='Rainy'))
            window_restored.images(torch.clamp(restored[0],0,1), nrow=1, win='Restored_Train', opts=dict(title='Restored'))
            window_sharp.images(target[0], nrow=1, win='Sharp_Train', opts=dict(title='Sharp'))
            
    #### Save ####
    if epoch >=  opt['TRAIN']['SAVE']:
        if epoch % opt['TRAIN']['SAVE_FRE'] == 0 or epoch>= 2000:
            torch.save({'epoch': epoch, 
                        'state_dict': model_restoration.state_dict(),
                        'optimizer' : optimizer.state_dict()
                        }, os.path.join(model_dir,f"model_epoch_{epoch}.pth")) 
    log("------------------------------------------------------------------",train_log)
    log("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f} LR: {:.10f}".format(epoch, time.time()-epoch_start_time, epoch_loss,scheduler.get_last_lr()[0]),train_log)
    log("------------------------------------------------------------------",train_log)
    
    #### Evaluation ####
    if epoch >= opt['TRAIN']['VAL']:
        if epoch % opt['TRAIN']['VAL_FRE'] == 0:
            ######### release training gpu memory ##########
            del input_
            del restored
            del target
            torch.cuda.empty_cache()
            log('eval epoch %d psnr'%(epoch), train_log)
            model_restoration = torch.nn.DataParallel(model_restoration)
            model_restoration.eval()
            psnr_val_rgb = []
            for ii, data_val in enumerate(tqdm(val_loader), 1):
                target_img = img_as_ubyte(data_val[0].numpy().squeeze().transpose((1,2,0)))
                input_ = data_val[1].cuda()
                factor = 8
                h,w = input_.shape[2],input_.shape[3]
                H,W = ((h+factor)//factor)*factor,((w+factor)//factor)*factor
                padh = H-h if h%factor!=0 else 0
                padw = W-w if w%factor!=0 else 0
                input_ = F.pad(input_,(0,padw,0,padh),'reflect')

                with torch.no_grad():
                    restored = model_restoration(input_)

                restored = restored[0]
                restored = torch.clamp(restored,0,1)
                restored = restored[:,:h,:w]
                restored_img = img_as_ubyte(torch.clamp(restored,0,1).cpu().numpy().squeeze().transpose((1,2,0)))
                
                ######### release testing gpu memory ##########
                del restored
                del input_
                torch.cuda.empty_cache()

                psnr_val_rgb.append(PSNR(restored_img, target_img))
                log('%-6s \t %f' % (data_val[2][0], psnr_val_rgb[-1]), os.path.join(log_dir, 'val_' + str(epoch)+'.txt'), P=False)
                if opt['TRAIN']['SAVE_IMG'] and epoch >= opt['TRAIN']['START_IMG']:
                    save_img_path = os.path.join(result_dir, 'epoch_' + str(epoch))
                    file_name = data_val[2][0] + '_restored.png'
                    utils.mkdir(save_img_path)
                    utils.save_img(os.path.join(save_img_path, file_name), restored_img)
            avg_psnr  = sum(psnr_val_rgb)/ii
            log('total images = %d \t avg_psnr = %f' % (ii,avg_psnr), os.path.join(log_dir, 'val_' + str(epoch)+'.txt'), P=False)

            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                best_epoch = epoch
                torch.save({'epoch': epoch, 
                            'state_dict': model_restoration.state_dict(),
                            'optimizer' : optimizer.state_dict()
                            }, os.path.join(model_dir,"model_best.pth"))

            log("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (epoch, avg_psnr, best_epoch, best_psnr),train_log)

    
    torch.save({'epoch': epoch, 
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,"model_latest.pth")) 