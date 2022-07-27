# ELIC: Efficient Learned Image Compression withUnevenly Grouped Space-Channel Contextual Adaptive Coding.
A Pytorch Implementation of "ELIC: Efficient Learned Image Compression withUnevenly Grouped Space-Channel Contextual Adaptive Coding."

Note that This Is Not An Official Implementation Code.

More details can be found in the following paper:

```
@inproceedings{he2022elic,
  title={Elic: Efficient learned image compression with unevenly grouped space-channel contextual adaptive coding},
  author={He, Dailan and Yang, Ziming and Peng, Weikun and Ma, Rui and Qin, Hongwei and Wang, Yan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5718--5727},
  year={2022}
}
```
 #dataset
 According to the paper, They train the models on the largest 8000 images picked from ImageNet dataset.
 so download the [ImageNet] (http://www.image-net.org/challenges/LSVRC/2012/dd31405981ef5f776aa17412e1f0c112/ILSVRC2012_img_train.tar)
 Run the code to prepare the Training dataset.
 
 
# Environment
   This code is based on the [CompressAI](https://github.com/InterDigitalInc/CompressAI).
   '''
   pip3 install torch torchvision torchaudio
   pip3 install compressai=1.1.5
   pip3 install thop, ptflops, timm.
   
   '''

#Usage

## Train Usage
   ```
   cd Code
   python train.py -d dataset --N 48 --angRes 13 --n_blocks 1 -e 100 -lr 1e-4 -n 20  --lambda 3e-3 --batch-size 8  --test-batch-size 8 --aux-learning-rate 1e-3 --patch-size 832 832 --cuda --save --seed 1926 --gpu-id  0,1,2,3 --savepath   ./checkpoint
   ```
   The training patches I used for training is available (https://pan.baidu.com/s/1hxWIrOC7ldIYV6zOzYIGpA 
Extraction code: uwrx)

## Update the entropy model
```
python updata.py checkpoint_path -n checkpoint
```
## Test 
Since the full test images are too large, I only upload a patch of the test image in Code/dataset/test. I re-trained the re-implementation algorithm in PyTorch with lambda=0.003, and the checkpoint is saved as the Code/checkpoint.pth.tar. 

In our original training stage, the algorithm was trained for 100 epochs, the provided checkpoint is only trained for 25 epochs since the re-training time is too long to wait. The checkpoint is provided to explain all the steps in our implementations.
```
python Inference.py --dataset/test --output_path Result_dir -p checkpoint.pth.tar
```
