# Sony Adapted FastSpeech 2 + TVCGMM - PyTorch Implementation

This is an adapted FastSpeech 2 pytorch implementation accompanying the paper [**Towards Robust FastSpeech 2 by Modelling Residual Multimodality**](https://sony.github.io/ai-research-code/tvc-gmm).
It is based on the PyTorch implementations of Microsoft's text-to-speech system [**FastSpeech 2: Fast and High-Quality End-to-End Text to Speech**](https://arxiv.org/abs/2006.04558v1) by [Chien 2020](https://github.com/ming024/FastSpeech2) and [Liu 2020](https://github.com/xcmyz/FastSpeech).
The hifigan implementation is taken from [Kong 2020](https://github.com/jik876/hifi-gan).

Along with bugfixes and code reorganization, two major changes have been introduced to the standard FastSpeech 2:
- Adapted Variance Predictors (`model/modules.py`)
- Trivariate-Chain Gaussian Mixture Modelling (Loss: `model/loss.py`, Sampling: `interactive_tts.py`)

## Demo
Audio samples of this implementation can be found [on its project page](https://sony.github.io/ai-research-code/tvc-gmm).
After downloading the checkpoints and putting them into `output/ckpt/<checkpoint name>` an interactive demo server can be started with `python demo.py --checkpoint libritts_tvcgmm_k5 --device cpu --port 9000` and accessed in the browser at [localhost:9000](http://localhost:9000).

## Quickstart

### Dependencies
We recommend installing the dependencies in a Python 3.8 conda environment (or venv) using:
```
conda create tvcgmm python=3.8
conda activate tvcgmm
pip install -r requirements.txt
```

### Inference

To run inference in our interactive demo it suffices to [download our pre-trained model checkpoints](https://github.com/sony/ai-research-code/releases). 
The downloaded `<checkpoint name>.zip` file contains a `40000.pth.tar` and a `used_config.yaml` which should both be placed together in the directory `output/ckpt/<checkpoint name>`. 

You can then import and use the `interactive_tts.py` class to generate samples.

## Training

### Datasets

The supported datasets are

- [LJSpeech](https://keithito.com/LJ-Speech-Dataset/): a single-speaker English dataset consisting of 13k short audio clips of a female speaker reading passages from 7 non-fiction books, approximately 24 hours in total.
- [VCTK](https://datashare.ed.ac.uk/handle/10283/3443): a multi-speaker English datasets consisting of 88k samples from 109 English speakers with various accents, deliberately selected for contextual and phonetic coverage, approximately 44 hours in total.
- [LibriTTS](https://research.google/tools/datasets/libri-tts/): a multi-speaker English dataset containing 585 hours of speech by 2456 speakers. We use the train-clean-360 split with 115k samples from 904 speakers over 192 hours.

We take LJSpeech as an example hereafter.

### Configuration
There are three config files for every dataset in `config/` containing the default parameters. 
Please edit/copy there and then run your experiments.
When training a new model the used configuration is copied next to the checkpoint in `output/ckpt/<experiment name>`.

Training with TVC-GMM can be enabled and the number of mixtures can be set in `model.yaml`.

### Preprocessing
 
First, run 
```
python prepare_align.py config/ljspeech/preprocess.yaml
```
to process the source datasets into a unified format and prepare them for alignment.

[Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/) (MFA) is used to obtain the alignments between the utterances and the phoneme sequences.
Download the official MFA package and run
```
./montreal-forced-aligner/bin/mfa_align raw_data/ljspeech/ lexicon/librispeech-lexicon.txt english preprocessed_data/ljspeech
```
to align the corpus and then run the preprocessing script, which will generate the mel, pitch, energy and duration targets for training.
```
python preprocess.py config/ljspeech/preprocess.yaml
```

### Training

Train your model with
```
python train.py --experiment <experiment name> -p config/ljspeech/preprocess.yaml -m config/ljspeech/model.yaml -t config/ljspeech/train.yaml
```

The model takes less than 10k steps of training to generate audio samples with acceptable quality and converges around 40k steps. Total training time in our tests was around 2-3 hours on a NVIDIA GeForce RTX 2080ti.

## TensorBoard

Use
```
tensorboard --logdir output/log
```
to serve TensorBoard on your localhost.
The loss curves, synthesized mel-spectrograms, and audio samples can be inspected.

## Citation
If you use TVC-GMM or any of our code in your project, please cite:
```
@inproceedings{koegel23_interspeech,
  author={Fabian Kögel and Bac Nguyen and Fabien Cardinaux},
  title={{Towards Robust FastSpeech 2 by Modelling Residual Multimodality}},
  year={2023},
  booktitle={Proc. Interspeech 2023}
}
```