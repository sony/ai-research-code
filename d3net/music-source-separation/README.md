# D3Net (Music Source Separation)

This is inference code for D3Net based music source separation.

## Quick Music Source Separation Demo by D3Net

From the Colab link below, you can try using D3Net to generate and listen to separated audio sources of your audio music file. Please give it a try!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/ai-research-code/blob/master/d3net/music-source-separation/D3Net-MSS.ipynb)

## Getting started

## Prerequisites
* nnabla 
* librosa
* numpy
* soundfile
* yaml

## Inference: Music source separation with pretrained model

Download the pre-trained D3Net model for Music Source Separation [here](https://nnabla.org/pretrained-models/ai-research-code/d3net/mss/d3net-mss.h5).

Run the below command for inference:
```python
 # Assuming test.wav file in the current directory.
 python ./separate.py -i ./test.wav -o output/ -m d3net-mss.h5 -c cudnn
 ```
Arguments
-i : Input files (Multiple wave files can be specified.)  
-o : Output directory. (Output folder path to save separated instruments)  
-m : Model file. (Pre-trained model)  
-c : Context. (Extension modules : `cpu` or `cudnn`)

## Training: Train the music source separation model from scratch (**coming soon**)
