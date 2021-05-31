# D3Net (Music Source Separation)

This is inference code for D3Net based music source separation.


## Getting started

## Prerequisites
* nnabla 
* librosa
* numpy
* soundfile
* yaml

## Source separation with pretrained model

### How to separate using pre-trained D3Net 
Download [here](https://nnabla.org/pretrained-models/ai-research-code/d3net/mss/d3net-mss.h5) a pre-trained model of D3Net for Music Source Separation.

In order to use it, please use the following command:
```python
 # Assuming test.wav file in the current directory.
 python ./separate.py -i ./test.wav -o output/ -m d3net-mss.h5 -c cudnn
 ```

- Options  
-i : Input files (Multiple wave files can be specified.)  
-o : Output directory. (Output folder path to save separated instruments)  
-m : Model file. (Pre-trained model)  
-c : Context. (Extension modules : `cpu` or `cudnn`)
