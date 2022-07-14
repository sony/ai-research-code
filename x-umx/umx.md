#  _Open-Unmix_ for NNabla

[![status](https://joss.theoj.org/papers/571753bc54c5d6dd36382c3d801de41d/status.svg)](https://joss.theoj.org/papers/571753bc54c5d6dd36382c3d801de41d) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/open-unmix-a-reference-implementation-for/music-source-separation-on-musdb18)](https://paperswithcode.com/sota/music-source-separation-on-musdb18?p=open-unmix-a-reference-implementation-for)

[![Gitter](https://badges.gitter.im/sigsep/open-unmix.svg)](https://gitter.im/sigsep/open-unmix?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge) [![Google group : Open-Unmix](https://img.shields.io/badge/discuss-on%20google%20groups-orange.svg)](https://groups.google.com/forum/#!forum/open-unmix)

This repository contains NNabla implementation of __Open-Unmix__, a deep neural network reference implementation for music source separation, applicable for researchers, audio engineers and artists. __Open-Unmix__ provides ready-to-use models that allow users to separate pop music into four stems: __vocals__, __drums__, __bass__ and remaining __other__ instruments. The models were pre-trained on the [MUSDB18](https://sigsep.github.io/datasets/musdb.html) dataset. See details [here](#getting-started).

__Related Projects:__ open-unmix-nnabla | :new: [x-umx-nnabla](x-umx.md) | [open-unmix-pytorch](https://github.com/sigsep/open-unmix-pytorch) | [musdb](https://github.com/sigsep/sigsep-mus-db) | [museval](https://github.com/sigsep/sigsep-mus-eval) | [norbert](https://github.com/sigsep/norbert)

## Model

![](https://docs.google.com/drawings/d/e/2PACX-1vTPoQiPwmdfET4pZhue1RvG7oEUJz7eUeQvCu6vzYeKRwHl6by4RRTnphImSKM0k5KXw9rZ1iIFnpGW/pub?w=959&h=308)

_Open-Unmix_ is based on a three-layer bidirectional deep LSTM. The model learns to predict the magnitude spectrogram of a target, like _vocals_, from the magnitude spectrogram of a mixture input. Internally, the prediction is obtained by applying a mask on the input. The model is optimized in the magnitude domain using mean squared error and the actual separation is done in a post-processing step involving a multichannel wiener filter implemented in [norbert](https://github.com/sigsep/norbert). To perform separation into multiple sources, multiple models are trained for each particular target. While this makes the training less comfortable, it allows great flexibility to customize the training data for each target source.

### Input Stage

__Open-Unmix__ operates in the time-frequency domain to perform its prediction. The input of the model is either:

* __A time domain__ signal tensor of shape `(nb_samples, nb_channels, nb_timesteps)`, where `nb_samples` are the samples in a batch, `nb_channels` is 1 or 2 for mono or stereo audio, respectively, and `nb_timesteps` is the number of audio samples in the recording.

 In that case, the model computes spectrograms with NNabla `STFT` on the fly.

* Alternatively _open-unmix_ also takes **magnitude spectrograms** directly (e.g. when pre-computed and loaded from disk).

 In that case, the input is of shape `(nb_frames, nb_samples, nb_channels, nb_bins)`, where `nb_frames` and `nb_bins` are the time and frequency-dimensions of a Short-Time-Fourier-Transform.

The input spectrogram is _standardized_ using the global mean and standard deviation for every frequency bin across all frames. Furthermore, we apply batch normalization in multiple stages of the model to make the training more robust against gain variation.

### Dimensionality reduction

The LSTM is not operating on the original input spectrogram resolution. Instead, in the first step after the normalization, the network learns to compresses the frequency and channel axis of the model to reduce redundancy and make the model converge faster.

### Bidirectional-LSTM

The core of __open-unmix__ is a three layer bidirectional [LSTM network](https://dl.acm.org/citation.cfm?id=1246450). Due to its recurrent nature, the model can be trained and evaluated on arbitrary length of audio signals. Since the model takes information from past and future simultaneously, the model cannot be used in an online/real-time manner.

### Output Stage

After LSTM, the signal is decoded back to its original input dimensionality. In the last steps the output is multiplied with the input magnitude spectrogram, so that the models is asked to learn a mask.

## Getting started

## Prerequisites
* nnabla >= v1.24.0
* musdb
* norbert
* resampy
* ffmpeg

### Installation

For installation we recommend to use the [Anaconda](https://anaconda.org/) python distribution. To create a conda environment for _open-unmix_, simply run:

`conda env create -f environment-X.yml` where `X` is either [`cpu`, `gpu`], depending on your system.


### Pre-trained models

We provide pre-trained models for __UMXHQ__ and __UMX__ in zip format. 
| UMXHQ weights | UMX weights |
|---|---|
|[UMXHQ pre-trained weights](https://nnabla.org/pretrained-models/open-unmix-nnabla/umxhq.zip)|[UMX pre-trained weights](https://nnabla.org/pretrained-models/open-unmix-nnabla/umx.zip)|

Download the weights files (zip format) from the above links and extract them.

* __`umxhq`__  trained on [MUSDB18-HQ](https://sigsep.github.io/datasets/musdb.html#musdb18-hq-uncompressed-wav) which comprises same tracks as in MUSDB18 but in uncompressed format, yielding full bandwidth of 22050 Hz.

* __`umx`__ is trained on uncompressed [MUSDB18](https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems) which is bandwidth limited to 16 kHz due to AAC compression. This model should be used for comparison with other (older) methods for evaluation in [SiSEC18](https://sisec18.unmix.app/).

In order to use it, please use the following command:
```bash
python test.py  --umx-infer --inputs [Input mixture (any audio format supported by FFMPEG)] --model [path to extracted base folder of umx/umxhq weights] --context cudnn --out-dir ./results 
```

### Evaluation using `museval`

To perform evaluation in comparison to other SISEC systems, you would need to install the `museval` package using

```
pip install museval
```

and then run the evaluation using

```bash
python eval.py --umx-infer --model [path to extracted base folder of umx/umxhq weights] --root [Path to MUSDB18 dataset] --out-dir [Path to save musdb estimates and museval results]
```

### Results compared to SiSEC 2018 (SDR/Vocals)

Open-Unmix yields state-of-the-art results compared to participants from [SiSEC 2018](https://sisec18.unmix.app/#/methods). Performances on `UMXHQ` and `UMX` are similar since evaluation was on compressed STEMS.

![boxplot_updated](https://user-images.githubusercontent.com/72940/63944652-3f624c80-ca72-11e9-8d33-bed701679fe6.png)

Note that

1. [`STL1`, `TAK2`, `TAK3`, `TAU1`, `UHL3`, `UMXHQ`] were omitted as they were _not_ trained on only _MUSDB18_.
2. [`HEL1`, `TAK1`, `UHL1`, `UHL2`] are not open-source.

#### Scores (Median of frames, Median of tracks)

#### For UMX
|target|SDR  |SDR  | SIR | SIR |SAR  |SAR  | ISR | ISR |
|------|-----|-----|-----|-----|-----|-----|-----|-----|
|`model`|Paper|NNabla|Paper|NNabla|Paper|NNabla|Paper|NNabla|
|vocals|6.32 |6.41 |13.33|13.79|6.52 |6.49 |12.95|11.4 |
|bass  |5.23 |5.30 |10.93|10.93|6.34 |5.87 |9.23 |9.17 |
|drums |5.73 |5.58 |11.12|10.66|6.02 |5.94 |10.51|11.33|
|other |4.02 |4.24 |6.59 |6.75 | 4.74|4.64 |9.31 |8.46 |

#### For UMXHQ
|target|SDR  |SDR  | SIR | SIR |SAR  |SAR  | ISR | ISR |
|------|-----|-----|-----|-----|-----|-----|-----|-----|
|`model`|Paper|NNabla|Paper|NNabla|Paper|NNabla|Paper|NNabla|
|vocals|6.25 |6.30 |12.95|12.02|6.50 |6.67 |12.70|12.20|
|bass  |5.07 |5.21 |10.35|10.23|6.02 |6.21 |9.71 |9.49 |
|drums |6.04 |5.78 |11.65|11.36|5.93 |6.27 |11.17|11.29|
|other |4.28 |4.17 |7.10 |6.74 |4.62 |4.59 |8.78 |8.63 |

## Training Open-Unmix

Both models, `umxhq` and `umx` that are provided with pre-trained weights, can be trained using the default parameters of the `train.py` function.

[MUSDB18](https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems) and [MUSDB18-HQ](https://sigsep.github.io/datasets/musdb.html#musdb18-hq-uncompressed-wav) are the largest freely available datasets for professionally produced music tracks (~10h duration) of different styles. They come with isolated `drums`, `bass`, `vocals` and `others` stems. _MUSDB18_ contains two subsets: "train", composed of 100 songs, and "test", composed of 50 songs.

To directly train a vocal model with _open-unmix_, we would first need to download one of the datasets and place it in _unzipped_ in a directory of your choice (called `root`).

| Argument | Description | Default |
|----------|-------------|---------|
| `--root <str>` | path to root of dataset on disk.                                                  | `None`       |
| `--source <str>` | target source to be trained. [_vocals, bass, drums, other_]               | `vocals`     |

For argument `--source`, _vocals_ is set as default value. Similarly we can train other target sources (_bass, drums, other_) by passing it for `--source` argument.

Also note that, if `--root` is not specified, we automatically download a 7 second preview version of the MUSDB18 dataset. While this is comfortable for testing purposes, we wouldn't recommend to actually train your model on this.
#### Using WAV files

All files from the MUSDB18 dataset are encoded in the Native Instruments stems format (.mp4). If you want to use WAV files (e.g. for faster audio decoding), `musdb` also supports parsing and processing pre-decoded PCM/wav files. Downloaded STEMS dataset (.mp4) can be decoded into WAV version either by [docker based solution or running scripts manually as shown here](https://github.com/sigsep/sigsep-mus-io).

__When you use the decoded MUSDB18 dataset (WAV version), use the `--is-wav` argument while running train.py.__

### Single GPU training
#### For encoded MUSDB18 STEMS version
```bash
python train.py --root [Path of MUSDB18] --source vocals --output [Path to save weights] --umx-train
```

#### For decoded MUSDB18 WAV version
```bash
python train.py --root [Path of MUSDB18] --source vocals --output [Path to save weights] --umx-train --is-wav
```

### Distributed Training
For distributed training [install NNabla package compatible with Multi-GPU execution](https://nnabla.readthedocs.io/en/latest/python/pip_installation_cuda.html#pip-installation-distributed). Use the below code to start the distributed training.
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3 {device ids that you want to use}
```

#### For encoded MUSDB18 STEMS version
```bash
mpirun -n {no. of devices} python train.py --root [Path of MUSDB18] --source vocals --output [Path to save weights] --umx-train
```

#### For decoded MUSDB18 WAV version
```bash
mpirun -n {no. of devices} python train.py --root [Path of MUSDB18] --source vocals --output [Path to save weights] --umx-train --is-wav
```

Please note that above sample training scripts will work on high quality 'STEM' or low quality 'MP4 files'. In case you would like faster data loading, kindly look at [more details here](https://github.com/sigsep/sigsep-mus-db#using-wav-files-optional) to generate decoded 'WAV' files. In that case, please use `--is-wav` flag for training.


Training `MUSDB18` using _open-unmix_ comes with several design decisions that we made as part of our defaults to improve efficiency and performance:

* __chunking__: we do not feed full audio tracks into _open-unmix_ but instead chunk the audio into 6s excerpts (`--seq-dur 6.0`).
* __balanced track sampling__: to not create a bias for longer audio tracks we randomly yield one track from MUSDB18 and select a random chunk subsequently. In one epoch we select (on average) 64 samples from each track.
* __source augmentation__: we apply random gains between `0.25` and `1.25` to all sources before mixing. Furthermore, we randomly swap the channels the input mixture.
* __random track mixing__: for a given target we select a _random track_ with replacement. To yield a mixture we draw the interfering sources from different tracks (again with replacement) to increase generalization of the model.
* __fixed validation split__: we provide a fixed validation split of [14 tracks](https://github.com/sigsep/sigsep-mus-db/blob/b283da5b8f24e84172a60a06bb8f3dacd57aa6cd/musdb/configs/mus.yaml#L41). We evaluate on these tracks in full length instead of using chunking to have evaluation as close as possible to the actual test data.

Some of the parameters for the MUSDB sampling can be controlled using the following arguments:

| Argument      | Description                                                            | Default      |
|---------------------|-----------------------------------------------|--------------|
| `--is-wav`          | loads the decoded WAVs instead of STEMS for faster data loading. See [more details here](https://github.com/sigsep/sigsep-mus-db#using-wav-files-optional). | `True`      |
| `--samples-per-track <int>` | sets the number of samples that are randomly drawn from each track  | `64`       |
| `--source-augmentations <list[str]>` | applies augmentations to each audio source before mixing | `gain channelswap`       |

## Training and Model Parameters

An extensive list of additional training parameters allows researchers to quickly try out different parameterizations such as a different FFT size. The table below, we list the additional training parameters and their default values (used for `umxhq` and `umx`L:

| Argument      | Description                                                                     | Default         |
|----------------------------|---------------------------------------------------------------------------------|-----------------|
| `--source <str>`           | name of target source (will be passed to the dataset)                         | `vocals`      |
| `--output <str>`           | path where to save the trained output model as well as checkpoints.                         | `./open-unmix`      |
| `--epochs <int>`           | Number of epochs to train                                                       | `1000`          |
| `--batch-size <int>`       | Batch size has influence on memory usage and performance of the LSTM layer      | `16`            |
| `--patience <int>`         | early stopping patience                                                         | `140`            |
| `--seq-dur <int>`          | Sequence duration in seconds of chunks taken from the dataset. A value of `<=0.0` results in full/variable length               | `6.0`           |
| `--unidirectional`           | changes the bidirectional LSTM to unidirectional (for real-time applications)  | `False`      |
| `--hidden-size <int>`             | Hidden size parameter of dense bottleneck layers  | `512`            |
| `--nfft <int>`             | STFT FFT window length in samples                                               | `4096`          |
| `--nhop <int>`             | STFT hop length in samples                                                      | `1024`          |
| `--lr <float>`             | learning rate                                                                   | `0.001`        |
| `--lr-decay-patience <int>`             | learning rate decay patience for plateau scheduler                                                                   | `80`        |
| `--lr-decay-gamma <float>`             | gamma of learning rate plateau scheduler.  | `0.3`        |
| `--weight-decay <float>`             | weight decay for regularization                                                                   | `0.00001`        |
| `--bandwidth <int>`        | maximum bandwidth in Hertz processed by the LSTM. Input and Output is always full bandwidth! | `16000`         |
| `--nb-channels <int>`      | set number of channels for model (1 for mono (spectral downmix is applied,) 2 for stereo)                     | `2`             |
| `--nb-workers <int>`      | Number of (parallel) workers for data-loader, can be safely increased for wav files   | `0` |
| `--context <str>`          | Extension modules. ex) 'cpu', 'cudnn'.                                   | 'cudnn'         |
| `--seed <int>`             | Initial seed to set the random initialization                                   | `42`            |



## Design Choices / Contributions

* we favored simplicity over performance to promote clearness of the code. The rationale is to have __open-unmix__ serve as a __baseline__ for future research while performance still meets current state-of-the-art (See [Evaluation](#Evaluation)). The results are comparable/better to those of `UHL1`/`UHL2` which obtained the best performance over all systems trained on MUSDB18 in the [SiSEC 2018 Evaluation campaign](https://sisec18.unmix.app).
* We designed the code to allow researchers to reproduce existing results, quickly develop new architectures and add own user data for training and testing. We favored framework specifics implementations instead of having a monolithic repository.
* _open-unmix_ is a community focused project, we therefore encourage the community to submit bug-fixes and comments and improve the computational performance. However, we are not looking for changes that only focused on improving the performance.

### Authors

[Fabian-Robert Stöter](https://www.faroit.com/), [Antoine Liutkus](https://github.com/aliutkus), Inria and LIRMM, Montpellier, France

## References

<details><summary>If you use open-unmix for your research – Cite Open-Unmix</summary>
  
```latex
@article{stoter19,  
  author={F.-R. St\\"oter and S. Uhlich and A. Liutkus and Y. Mitsufuji},  
  title={Open-unmix: a reference implementation for source separation},  
  journal={Journal of Open Source Software},  
  year=2019,  
  note={submitted}}
}
```

</p>
</details>

<details><summary>If you use the MUSDB dataset for your research - Cite the MUSDB18 Dataset</summary>
<p>

```latex
@misc{MUSDB18,
  author       = {Rafii, Zafar and
                  Liutkus, Antoine and
                  Fabian-Robert St{\"o}ter and
                  Mimilakis, Stylianos Ioannis and
                  Bittner, Rachel},
  title        = {The {MUSDB18} corpus for music separation},
  month        = dec,
  year         = 2017,
  doi          = {10.5281/zenodo.1117372},
  url          = {https://doi.org/10.5281/zenodo.1117372}
}
```

</p>
</details>


<details><summary>If compare your results with SiSEC 2018 Participants - Cite the SiSEC 2018 LVA/ICA Paper</summary>
<p>

```latex
@inproceedings{SiSEC18,
  author="St{\"o}ter, Fabian-Robert and Liutkus, Antoine and Ito, Nobutaka",
  title="The 2018 Signal Separation Evaluation Campaign",
  booktitle="Latent Variable Analysis and Signal Separation:
  14th International Conference, LVA/ICA 2018, Surrey, UK",
  year="2018",
  pages="293--305"
}
```

</p>
</details>

⚠️ Please note that the official acronym for _open-unmix_ is **UMX**.

### License

MIT


### Acknowledgements

<p align="center">
  <img src="https://raw.githubusercontent.com/sigsep/website/master/content/open-unmix/logo_INRIA.svg?sanitize=true" width="200" title="inria">
  <img src="https://raw.githubusercontent.com/sigsep/website/master/content/open-unmix/anr.jpg" width="100" alt="anr">
</p>
