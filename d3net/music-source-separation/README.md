# D3Net (Music Source Separation)

This is inference code for D3Net based music source separation.

## Quick Music Source Separation Demo by D3Net

From the Colab link below, you can try using D3Net to generate and listen to separated audio sources of your audio music file. Please give it a try!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sony/ai-research-code/blob/master/d3net/music-source-separation/D3Net-MSS.ipynb)

## Getting started

## Prerequisites
* nnabla 
* librosa
* pydub
* numpy
* soundfile
* yaml

If you want to do test with openvino, you should also install openvino.

## Inference: Music source separation with pretrained model

Download the pre-trained D3Net model for Music Source Separation [here](https://nnabla.org/pretrained-models/ai-research-code/d3net/mss/d3net-mss.h5).

Run the below inference command for a sample audio file `test.wav` in current directory:
```python
 python ./separate.py -i ./test.wav -o output/ -m d3net-mss.h5 -c cudnn
 ```
### Inference with [OpenVINO](https://docs.openvinotoolkit.org/)
OpenVINO is a toolkit for quickly deploying a neural network model with high-performance especially on Intel hardware. It speeds up the D3Net inference on CPUs.

Download and extract the openvino weights [here](https://nnabla.org/pretrained-models/ai-research-code/d3net/mss/d3net-openvino.zip).
```bash
 unzip d3net-openvino.zip -d openvino_models
 ```
```python
 python ./separate.py -i ./test.wav -o output/ -b openvino -ovm ./openvino_models
 ```
Arguments:  
-i : Input files. (Any audio format files supported by FFMPEG.)  
-o : Output directory. (Output folder path to save separated instruments)  
-m : Model file. (Pre-trained model)  
-c : Context. (Extension modules : `cpu` or `cudnn`)  
-b: Backend framework for inference. (`nnabla` or `openvino`)
-ovm: Openvino models directory.

## Training: Train the music source separation model from scratch (**coming soon**)
