# Reference
https://github.com/yjn870/FSRCNN-pytorch

# FSRCNN

This repository is implementation of the ["Accelerating the Super-Resolution Convolutional Neural Network"](https://arxiv.org/abs/1608.00367).

## Differences from the original

- Added the zero-padding
- Used the Adam instead of the SGD

## Requirements

- PyTorch 1.4.0
- Numpy 1.19.2
- Pillow 8.1.0
- h5py 2.10.0
- tqdm 4.55.1

## Train

```bash
python train.py --train-file "BLAH_BLAH/91-image_x3.h5" \
                --eval-file "BLAH_BLAH/Set5_x3.h5" \
                --outputs-dir "BLAH_BLAH/outputs" \
                --scale 3 \
                --lr 1e-3 \
                --batch-size 16 \
                --num-epochs 20 \
                --num-workers 8 \
                --seed 123                
```

## Test

```bash
python test.py --weights-file "BLAH_BLAH/fsrcnn_x3.pth" \
               --image-file "data/butterfly_GT.bmp" \
               --scale 3
```

## Results 25.167

PSNR was calculated on the Y channel.

#### Hyperparameters
Scale factor: 3
* Learning rate: 1e-3
* Batch size: 16
* Epochs: 5
* Number of workers: 8



