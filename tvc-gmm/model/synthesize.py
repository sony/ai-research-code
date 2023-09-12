# Copyright 2023 Sony Group Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
######################################################################
#
# Implementation derived from
#   https://github.com/ming024/FastSpeech2/tree/d4e79e/synthesize.py
# available under MIT License.

import re
import os
import argparse
from string import punctuation

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style

from utils.model import get_model, get_vocoder
from utils.tools import to_device
from dataset import TextDataset
from utils.text import text_to_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_lexicon(lex_path):
    lexicon = {}
    allowed_symbols = set()
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
                for x in re.findall("([^\w\s])", word):
                    allowed_symbols.add("\\"+x)
    return lexicon, allowed_symbols


def preprocess_english(text, preprocess_config):
    text = text.strip(punctuation) # avoid silence at beginning/end
    lexicon, allowed_symbols = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    word_positions = []
    words = re.split(f"([^\w\s{''.join(allowed_symbols)}]|\s+)", text) # split on whitespace or symbols that are not allowed inside lexicon words
    words = list(filter(lambda i: not re.match("\s+|^$", i), words)) # remove empty matches and matched spaces
    for w in words:
        if w.lower() in lexicon:
            word_positions.append((len(phones), len(phones)+len(lexicon[w.lower()])))
            phones += lexicon[w.lower()] # add phonetization of recognized lexicon words
        elif re.match("\W", w):
            if len(phones) == 0 or phones[-1] != "sp":
                phones.append("sp") # add silence for pure symbols
        else:
            for w in re.split(f"[{''.join(allowed_symbols)}]", w): # split again on symbols allowed inside lexicon words (without capture) and re-try
                seq = list(filter(lambda p: p != " ", g2p(w))) # add estimated phonetization for unknown words
                word_positions.append((len(phones), len(phones)+len(seq)))
                phones += seq
    phones = "{" + " ".join(phones) + "}"

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return sequence, word_positions