# A Bright Feature Selection Transformer for Single-Image Deraining

## Training
* Please download the corresponding training datasets and and modify the path in train.yml.
* Different datasets may require different training hyper-parameters. Please train the model with the arguments mentioned in the paper.
* Before training the model, you need to open the visdom server in another command window to visualize the deraining results, please run:
```
python -m visdom.server -port=2333
```
* Follow the instructions below to begin training our model.
```
python train.py
```
Run the script then you can find the generated model weights in the folder 'logs'.

## Testing
* Please use the corresponding testing datasets and and modify the path in test.yml.
* Follow the instructions below to begin training our model.
```
python test.py
```
Run the script then you can find the single-image deraining results in the folder 'logs'.


## Pre-trained Models
Our pre-trained models are available for download in Baidu Netdisk:
[Download Link](https://pan.baidu.com/s/1nn49nrsYdGruT_jyhToafw)  
Extraction Code: `3q1f`

## Visual Deraining Results
Our deraining results are available for download in Baidu Netdisk:
[Download Link](https://pan.baidu.com/s/1FVOfpCovQDOp5AGNmhf0wA)  
Extraction Code: `fjaq`
