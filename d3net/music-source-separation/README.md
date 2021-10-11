# D3Net (Music Source Separation)

This is the official NNabla implementation of D3Net based music source separation.

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

Download and extract the pre-trained weights [here](https://nnabla.org/pretrained-models/ai-research-code/d3net/mss/d3net-mss.zip).
```bash
 unzip d3net-mss.zip -d d3net-mss
 ```

Run the below inference command for a sample audio file `test.wav` in current directory:
```bash
 python ./separate.py -i ./test.wav -o output/ -m ./d3net-mss -c cudnn
 ```
Arguments:  
-i : Input files. (Any audio format files supported by FFMPEG.)  
-o : Output directory. (Output folder path to save separated instruments)  
-m : Model file. (Pre-trained model)  
-c : Context. (Extension modules : `cpu` or `cudnn`)  

### Inference with [OpenVINO](https://docs.openvinotoolkit.org/)
OpenVINO is a toolkit for quickly deploying a neural network model with high-performance especially on Intel hardware. It speeds up the D3Net inference about 5 times on single CPU.

Download and extract the openvino weights [here](https://nnabla.org/pretrained-models/ai-research-code/d3net/mss/d3net-openvino.zip).
```bash
 unzip d3net-openvino.zip -d openvino_models
 ```
```bash
 python ./separate_with_openvino.py -i ./test.wav -o output/ -m ./openvino_models -n 4
 ```
Arguments:  
-i : Input files. (Any audio format files supported by FFMPEG.)  
-o : Output directory. (Output folder path to save separated instruments)  
-m : Openvino models directory.  
-n : Specifies the number of threads that openvino should use for inference.

### Evaluation using `museval`

To perform evaluation in comparison to other SISEC systems, you would need to install the `museval` package using

```
pip install museval
```

and then run the below command for the evaluation:

```bash
python eval.py -m ./d3net-mss --root [Path of MUSDB18 dataset] --out-dir [Path to save musdb estimates and museval results]
 ```

#### Scores (Median of frames, Median of tracks)

|target|SDR  | SDR |
|------|-----|-----|
|`model`|Paper|NNabla|
|vocals|7.24 |7.14 |
|drums |7.01 |6.85 |
|bass |5.25 |5.32 |
|other |4.53 |4.82 |
|**Avg** |**6.01** |**6.03** |

Published pre-trained models yield SDR values as mentioned in the above table. The difference in the metrics between pre-trained models and the paper are due to randomness in the data preprocessing and weight initialization.

## Training: Train the music source separation model from scratch

D3Net model for Music Source Separation can be trained using the default parameters of the `train.py` function.

[MUSDB18](https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems) and [MUSDB18-HQ](https://sigsep.github.io/datasets/musdb.html#musdb18-hq-uncompressed-wav) are the largest freely available datasets for professionally produced music tracks (~10h duration) of different styles. They come with isolated `drums`, `bass`, `vocals` and `others` stems. _MUSDB18_ contains two subsets: "train", composed of 100 songs, and "test", composed of 50 songs.

To directly train a vocal model with _d3net-mss_, we would first need to download one of the datasets and place it in _unzipped_ in a directory of your choice (called `root`).

| Argument | Description | Default |
|----------|-------------|---------|
| `--root <str>` | path to root of dataset on disk.                                                  | `None`       |

Also note that, if `--root` is not specified, we automatically download a 7 second preview version of the MUSDB18 dataset. While this is comfortable for testing purposes, we wouldn't recommend to actually train your model on this.

#### Using WAV files

All files from the MUSDB18 dataset are encoded in the Native Instruments stems format (.mp4). If you want to use WAV files (e.g. for faster audio decoding), `musdb` also supports parsing and processing pre-decoded PCM/wav files. Downloaded STEMS dataset (.mp4) can be decoded into WAV version either by [docker based solution or running scripts manually as shown here](https://github.com/sigsep/sigsep-mus-io).

__When you use the decoded MUSDB18 dataset (WAV version), use the `--is-wav` argument while running train.py.__

### Single GPU training

#### For encoded MUSDB18 STEMS version
```bash
python train.py --root [Path of MUSDB18 dataset] --target [target track to be trained] --output [Path to save weights]
```

#### For decoded MUSDB18 WAV version
```bash
python train.py --root [Path of MUSDB18 dataset] --target [target track to be trained] --output [Path to save weights] --is-wav
```

### Distributed Training
For distributed training [install NNabla package compatible with Multi-GPU execution](https://nnabla.readthedocs.io/en/latest/python/pip_installation_cuda.html#pip-installation-distributed). Use the below code to start the distributed training.
```
export CUDA_VISIBLE_DEVICES=0,1,2,3 {device ids that you want to use}
```

#### For encoded MUSDB18 STEMS version
```bash
mpirun -n {no. of devices} python train.py --root [Path of MUSDB18 dataset] --target [target track to be trained] --output [Path to save weights]
```

#### For decoded MUSDB18 WAV version
```bash
mpirun -n {no. of devices} python train.py --root [Path of MUSDB18 dataset] --target [target track to be trained] --output [Path to save weights] --is-wav
```

Please note that above sample training scripts will work on high quality 'STEM' or low quality 'MP4 files'. In case you would like faster data loading, kindly look at [more details here](https://github.com/sigsep/sigsep-mus-db#using-wav-files-optional) to generate decoded 'WAV' files. In that case, please use `--is-wav` flag for training.

Training `MUSDB18` using _d3net-mss_ comes with several design decisions that we made as part of our defaults to improve efficiency and performance:

* __chunking__: we do not feed full audio tracks into _d3net-mss_ but instead chunk the audio into 6s excerpts (`--seq-dur 6.0`).
* __balanced track sampling__: to not create a bias for longer audio tracks we randomly yield one track from MUSDB18 and select a random chunk subsequently. In one epoch we select (on average) 64 samples from each track.
* __source augmentation__: we apply random gains between `0.75` and `1.25` to all sources before mixing. Furthermore, we randomly swap the channels the input mixture.
* __random track mixing__: for a given target we select a _random track_ with replacement. To yield a mixture we draw the interfering sources from different tracks (again with replacement) to increase generalization of the model.

Some of the parameters for the MUSDB sampling can be controlled using the following arguments:

| Argument      | Description                                                            | Default      |
|---------------------|-----------------------------------------------|--------------|
| `--is-wav`          | loads the decoded WAVs instead of STEMS for faster data loading. See [more details here](https://github.com/sigsep/sigsep-mus-db#using-wav-files-optional). | `True`      |
| `--samples-per-track <int>` | sets the number of samples that are randomly drawn from each track  | `64`       |
| `--source-augmentations <list[str]>` | applies augmentations to each audio source before mixing | `gain channelswap`       |

## Training and Model Parameters

An extensive list of additional training parameters allows researchers to quickly try out different parameterizations such as a different FFT size. The table below, we list the additional training parameters and their default values :

| Argument      | Description                                                                     | Default         |
|----------------------------|---------------------------------------------------------------------------------|-----------------|
| `--target <str>`           | name of target source (will be passed to the dataset)                         | `vocals`      |
| `--output <str>`           | path where to save the trained output model as well as checkpoints.                         | `./d3net-mss`      |
| `--epochs <int>`           | Number of epochs to train                                                       | `50`          |
| `--batch-size <int>`       | Batch size has influence on memory usage and performance of the LSTM layer      | `6`            |
| `--seq-dur <int>`          | Sequence duration in seconds of chunks taken from the dataset. A value of `<=0.0` results in full/variable length               | `6.0`           |
| `--lr <float>`             | learning rate                                                                   | `0.001`        |
| `--bandwidth <int>`        | maximum bandwidth in Hertz processed by the LSTM. Input and Output is always full bandwidth! | `16000`         |
| `--context <str>`          | Extension modules. ex) 'cpu', 'cudnn'.                                   | 'cudnn'         |
| `--seed <int>`             | Initial seed to set the random initialization                                   | `None`            |
