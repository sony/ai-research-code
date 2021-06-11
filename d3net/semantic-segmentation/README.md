# D3Net (Semantic Segmentation)
This is NNabla implementation of D3Net based semantic segmentation.

## Quick Semantic Segmentation Demo by D3Net

From the Colab link below, you can try running D3Net (D3Net-L/D3Net-S) to generate semantic segmentation outputs for sample input images. Please give it a try!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/ai-research-code/blob/master/d3net/semantic-segmentation/D3Net-Semantic-Segmentation.ipynb)

## Getting started

## Prerequisites
* nnabla 
* cv2
* numpy
* PyYAML >= 5.3

## Inference: Semantic segmentation with pretrained model
We provide two D3Net models for Semantic Segmentation. The smaller architecture, denoted as D3Net-S, employs D3 blocks of (M, L, k, c) =(4, 8, 36, 0.2), while the larger architecture, D3Net-L, uses D3 blocks of (M, L, k, c) = (4, 10, 64, 0.2). Download the pre-trained weights from the links provided in the below table:

### Pre-trained Weights
| D3Net-L | D3Net-S |
|---|---|
|[D3Net-L weights](https://nnabla.org/pretrained-models/ai-research-code/d3net/semantic-segmentation/D3Net-L.h5)|[D3Net-S weights](https://nnabla.org/pretrained-models/ai-research-code/d3net/semantic-segmentation/D3Net-S.h5)|

Run the below command for D3Net-L inference:
```python
 # Assuming test.jpg file in the current directory.
 python ./infer.py -i ./test.jpg -cfg ./configs/D3Net_L.yaml -o output/ -m D3Net-L.h5 -c cudnn
 ```
 
Run the below command for D3Net-S inference:
```python
 # Assuming test.jpg file in the current directory.
 python ./infer.py -i ./test.jpg -cfg ./configs/D3Net_S.yaml -o output/ -m D3Net-S.h5 -c cudnn
 ```
Arguments:  
-i : Input file (Test image file.)  
-cfg : Configuration file (Configuration file('D3Net_L.yaml' or 'D3Net_S.yaml'))  
-m : Model file. (Pre-trained model('D3Net-L.h5' or 'D3Net-S.h5)')  
-c : Context. (Extension modules : `cpu` or `cudnn`)

## Training: Train the semantic segmentation model from scratch (**coming soon**)
