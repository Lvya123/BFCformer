 ###### A Bright Feature Selection Transformer for Single-Image Deraining ######

##  Training
---You first need to prepare the datasets for training.

---Different datasets may correspond to different training hyper-parameters. Please train the model with the arguments mentioned in the paper(A Bright Feature Selection Transformer for Single-Image Deraining), and modify them in train.yaml.

Before training the model, you need to open the visdom server in another command window to visualize the restoration results, please run
```
python -m visdom.server -port 2333
```

#### Start training ####
Please run:
```
python train.py
```
The results and weight will be saved in .\log.

#### Pre-trained Models
<a href="https://pan.baidu.com/s/1WU2m6HmHe6TXEZulH0HzWA">Download (cv8n)</a>


##  Testing
You also need to prepare the datasets for testing, and modify the arguments in test.yaml. Please run
```
python test.py
```

#### Visual Deraining Results
<a href="https://pan.baidu.com/s/1dqH3gGVvY_5G6rt1n1LkYg">Download (dg8w)</a> 


##  Performance Evaluation

#### To reproduce PSNR/SSIM scores of the paper on Rain200L/H, SPA-Data and GT-RAIN datasets, run this MATLAB script
```
.\test-200LH-spa-psnr-ssim\evaluate_PSNR_SSIM.m
```

#### To reproduce PSNR/SSIM scores of the paper on DID-Data and DDN-Data datasets, run this MATLAB script
```
.\test-ddn-did-psnr-ssim\statistic.m
```
