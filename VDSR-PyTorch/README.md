# VDSR-PyTorch

# Reference
https://github.com/Lornatang/VDSR-PyTorch

### Overview
This repository contains an op-for-op PyTorch reimplementation of 
[Accurate Image Super-Resolution Using Very Deep Convolutional Networks](https://arxiv.org/abs/1511.04587).

### Test

Evaluate the overall performance of the network.
```bash
usage: test.py [-h] [--dataroot DATAROOT] [--scale-factor {2,3,4}]
               [--weights WEIGHTS] [--cuda]

Accurate Image Super-Resolution Using Very Deep Convolutional Networks

optional arguments:
  -h, --help            show this help message and exit
  --dataroot DATAROOT   The directory address where the image needs to be
                        processed. (default: `./data/Set5`).
  --scale-factor {2,3,4}
                        Image scaling ratio. (default: 4).
  --weights WEIGHTS     Generator model name. (default:`weights/vdsr_4x.pth`)
  --cuda                Enables cuda

# Example
python test.py --dataroot ./data/Set5 --scale-factor 4 --weights ./weights/vdsr_4x.pth --cuda
```

Evaluate the benchmark of validation data set in the network
```bash
usage: test_benchmark.py [-h] [--dataroot DATAROOT] [-j N]
                         [--image-size IMAGE_SIZE] --scale-factor {2,3,4}
                         --weights WEIGHTS [--cuda]

Accurate Image Super-Resolution Using Very Deep Convolutional Networks

optional arguments:
  -h, --help            show this help message and exit
  --dataroot DATAROOT   Path to datasets. (default:`./data/VOC2012`)
  -j N, --workers N     Number of data loading workers. (default:4)
  --image-size IMAGE_SIZE
                        Size of the data crop (squared assumed). (default:256)
  --scale-factor {2,3,4}
                        Low to high resolution scaling factor.
  --weights WEIGHTS     Path to weights.
  --cuda                Enables cuda

# Example
python test_benchmark.py --dataroot ./data/VOC2012 --scale-factor 4 --weights ./weights/vdsr_4x.pth --cuda
```

Test single picture
```bash
usage: test_image.py [-h] [--file FILE] [--scale-factor {2,3,4}]
                     [--weights WEIGHTS] [--cuda]

Accurate Image Super-Resolution Using Very Deep Convolutional Networks

optional arguments:
  -h, --help            show this help message and exit
  --file FILE           Test low resolution image name.
                        (default:`./assets/baby.png`)
  --scale-factor {2,3,4}
                        Super resolution upscale factor. (default:4)
  --weights WEIGHTS     Generator model name. (default:`weights/vdsr_4x.pth`)
  --cuda                Enables cuda

# Example
python test_image.py --file ./assets/baby.png --scale-factor 4 ---weights ./weights/vdsr_4x.pth -cuda
```

Test single video
```bash
usage: test_video.py [-h] --file FILE --scale-factor {2,3,4} --weights WEIGHTS
                     [--view] [--cuda]

Accurate Image Super-Resolution Using Very Deep Convolutional Networks

optional arguments:
  -h, --help            show this help message and exit
  --file FILE           Test low resolution video name.
  --scale-factor {2,3,4}
                        Super resolution upscale factor. (default:4)
  --weights WEIGHTS     Generator model name.
  --view                Super resolution real time to show.
  --cuda                Enables cuda

# Example
python test_video.py --file ./data/1.mp4 --scale-factor 4 --weights ./weights/vdsr_4x.pth --view --cuda
```

Low resolution / Recovered High Resolution / Ground Truth

<span align="center"><img src="assets/result.png" alt="">
</span>

### Train (e.g VOC2012)

```bash
usage: train.py [-h] [--dataroot DATAROOT] [-j N] [--epochs N]
                [--image-size IMAGE_SIZE] [-b N] [--lr LR]
                [--momentum MOMENTUM] [--weight-decay WEIGHT_DECAY]
                [--clip CLIP] [--scale-factor {2,3,4}] [--weights WEIGHTS]
                [-p N] [--manualSeed MANUALSEED] [--cuda]

Accurate Image Super-Resolution Using Very Deep Convolutional Networks

optional arguments:
  -h, --help            show this help message and exit
  --dataroot DATAROOT   Path to datasets. (default:`./data/VOC2012`)
  -j N, --workers N     Number of data loading workers. (default:4)
  --epochs N            Number of total epochs to run. (default:100)
  --image-size IMAGE_SIZE
                        Size of the data crop (squared assumed). (default:256)
  -b N, --batch-size N  mini-batch size (default: 16), this is the total
                        batch size of all GPUs on the current node when using
                        Data Parallel or Distributed Data Parallel.
  --lr LR               Learning rate. (default:0.1)
  --momentum MOMENTUM   Momentum, (default:0.9)
  --weight-decay WEIGHT_DECAY
                        Weight decay. (default:0.0001).
  --clip CLIP           Clipping Gradients. (default:0.4).
  --scale-factor {2,3,4}
                        Low to high resolution scaling factor. (default:4).
  --weights WEIGHTS     Path to weights (to continue training).
  -p N, --print-freq N  Print frequency. (default:5)
  --manualSeed MANUALSEED
                        Seed for initializing training. (default:0)
  --cuda                Enables cuda
```

#### Example (e.g VOC2012)

```bash
python train.py --dataroot ./data/VOC2012 --scale-factor 4 --cuda
```

### Hyperparameters
* Number of workers: 4
* Epochs: 100
* Batch size: 256
* Learning rate: 0.1
* Momentum: 0.9
* Weight decay: 1e-4
* Gradient clip: 0.4
* Scale factor: 3
