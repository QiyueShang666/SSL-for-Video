## Contrastive Predictive Learning with Transformer for Video Representation Learning

This repository completes the video Self-Supervised Learning framework based on CPC idea and Vision Transformer 


### CPCTR Results

Best Performance result from this implementation(5pred3+7pred3):

| Pretrain Dataset| Resolution | Backbone | Finetune Acc@top-1 (UCF101) | Finetune Acc@Top-1 (HMDB51) |
|----|----|----|----|----|
|UCF101|224x224|2d-R18| <div align="center">63.5</div> |<div align="center">34.6</div>|


### Installation

The implementation should work with python >= 3.6, pytorch >= 0.4, torchvision >= 0.2.2. 

The repo also requires cv2, tensorboardX >= 1.7, joblib, tqdm, ipdb.

### Prepare data

Please download HMDB51 and UCF101 dataset along with their three splits, then use /ProcessData to extract frames from video.

### Self-supervised training (CPCTR)

Change directory `cd CPCTrans/CPCTrans/`

* example: train SSL model using 1 GPUs(about 28 hours with one RTX4090 GPU), with ResNet18 backbone, on UCF101 dataset with 224x224 resolution, for 300 epochs
  ```
  python main.py --net resnet18 --dataset ucf101 --batch_size 16 --img_dim 224 --epochs 300 --num_seq 5 --pred_step 3
  ```

### Evaluation: supervised action classification

Change directory `cd CPCTrans/Evaluate/`

* example: finetune pretrained weights (replace `{model.pth.tar}` with pretrained SSL model)
  ```
  python test.py --net resnet18 --dataset ucf101 --batch_size 16 --img_dim 224 --pretrain {model.pth.tar} --train_what ft --epochs 300 --num_seq 7 --pred_step 3
  ```

* example (continued): test the finetuned model (replace `{finetune_model.pth.tar}` with finetuned classifier model)
  ```
  python test.py --net resnet18 --dataset ucf101 --batch_size 16 --img_dim 224 --test {finetune_model.pth.tar}
  ```




