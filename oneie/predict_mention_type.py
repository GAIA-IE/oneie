import sys
import os
import json
import glob
import tqdm
import traceback
# from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig

from model import OneIE
from model_fine_grained import OneIE as OneIEFG
from config import Config
from util import save_result
from data import IEDatasetEval
from convert import json_to_cs, json_to_mention_results, bio_to_cfet, json_to_cs_fg, json_to_cs_aida

cur_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(cur_dir, 'models')
mount_dir = '/data'


def load_model(model_path, device=0, gpu=False, beam_size=5):
    print('Loading the model from {}'.format(model_path))
    map_location = 'cuda:{}'.format(device) if gpu else 'cpu'
    state = torch.load(model_path, map_location=map_location)

    config = state['config']
    if type(config) is dict:
        config = Config.from_dict(config)
    vocabs = state['vocabs']
    valid_patterns = state['valid']

    # recover the model
    model = OneIEFG(config, vocabs, valid_patterns)
    model.load_state_dict(state['model'])
    model.beam_size = beam_size
    if gpu:
        model.cuda(device)

    tokenizer = BertTokenizer.from_pretrained(config.bert_model_name,
                                              do_lower_case=False)

    return model, tokenizer, config


def predict_mentions(input_path, annotation_path, output_path, batch_size=10,
                     max_length=128, device=0, gpu=True, file_extension='ltf.xml')