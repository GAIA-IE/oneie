import sys
import os
import json
import glob
import tqdm
import traceback
from argparse import ArgumentParser
from typing import Tuple, List, Dict, Any

import torch
from torch.utils.data import DataLoader
from transformers import (BertTokenizer,
                          BertConfig,
                          RobertaTokenizer,
                          RobertaConfig,
                          XLMRobertaTokenizer,
                          XLMRobertaConfig,
                          PreTrainedTokenizer,
                          PretrainedConfig
                          )

from model import OneIE
from config import Config
from util import save_result
from data import IEDatasetEval
from convert import json_to_cs, json_to_mention_results, bio_to_cfet, json_to_cs_fg, json_to_cs_aida
from merge import merge_json_result, merge_hidden_state

cur_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(cur_dir, 'models')
mount_dir = '/data'


lang_codes = {
    'english': 'en',
    'russian': 'ru',
    'spanish': 'es'
}


def load_model(model_path: str,
               device: int = 0,
               gpu: bool = True,
               beam_size: int = 5
               ) -> Tuple[OneIE, PreTrainedTokenizer, Config]:
    print('Loading the model from {}'.format(model_path))
    map_location = 'cuda:{}'.format(device) if gpu else 'cpu'
    state = torch.load(model_path, map_location=map_location)

    config = state['config']
    if type(config) is dict:
        config = Config.from_dict(config)
    vocabs = state['vocabs']
    valid_patterns = state['valid']

    # Recover the model
    model = OneIE(config, vocabs, valid_patterns)
    model.load_state_dict(state['model'])
    model.beam_size = beam_size
    if gpu:
        model.cuda(device)

    # Create the tokenizer
    if config.bert_model_name.startswith('bert-'):
        tokenizer = BertTokenizer.from_pretrained(config.bert_model_name)
    elif config.bert_model_name.startswith('roberta-'):
        tokenizer = RobertaTokenizer.from_pretrained(config.bert_model_name)
    elif config.bert_model_name.startswith('xlm-roberta-'):
        tokenizer = XLMRobertaTokenizer.from_pretrained(config.bert_model_name)
    else:
        raise ValueError('Unknown model name: {}'.format(config.bert_model_name))

    return model, tokenizer, config


def predict_document(path: str,
                     model: OneIE,
                     tokenizer: PreTrainedTokenizer,
                     config: Config,
                     batch_size: int = 10, 
                     max_length: int = 128,
                     gpu: bool = False,
                     language: str = 'english',
                     output_hidden: str = False):
    """
    :param path (str): path to the input file.
    :param model (OneIE): pre-trained model object.
    :param tokenizer (PreTrainedTokenizer): BERT tokenizer.
    :param config (Config): configuration object.
    :param batch_size (int): Batch size (default=20).
    :param max_length (int): Max word piece number (default=128).
    :param gpu (bool): Use GPU or not (default=False).
    :param langauge (str): Input document language (default='english').
    """
    test_set = IEDatasetEval(path, max_length=max_length, gpu=gpu,
                             input_format='ltf', language=language)
    test_set.numberize(tokenizer)
    # document info
    info = {
        'doc_id': test_set.doc_id,
        'ori_sent_num': test_set.ori_sent_num,
        'sent_num': len(test_set)
    }
    # prediction result
    result = []
    mention_hidden_states = []
    trigger_hidden_states = []
    for batch in DataLoader(test_set, batch_size=batch_size, shuffle=False, 
                            collate_fn=test_set.collate_fn):
        graphs, bert_outputs = model.predict(batch)
        for graph, tokens, sent_id, token_ids, bert_output in zip(
            graphs, batch.tokens, batch.sent_ids, batch.token_ids,
            bert_outputs):
            graph.clean(relation_directional=config.relation_directional,
                        symmetric_relations=config.symmetric_relations)
            result.append((sent_id, token_ids, tokens, graph))

            # hidden states
            if output_hidden:
                for s, e, _ in graph.entities:
                    for i in range(s, e):
                        t, tid = tokens[i], token_ids[i]
                        embed = bert_output[i].tolist()
                        mention_hidden_states.append((t, tid, ','.join(['{:.8f}'.format(e) for e in embed])))
                for s, e, _ in graph.triggers:
                    for i in range(s, e):
                        t, tid = tokens[i], token_ids[i]
                        embed = bert_output[i].tolist()
                        trigger_hidden_states.append((t, tid, ','.join(['{:.8f}'.format(e) for e in embed])))
                
    return result, info, mention_hidden_states, trigger_hidden_states


def predict_mention_type(src_path: str,
                         json_path: str,
                         model: OneIE,
                         tokenizer: PreTrainedTokenizer,
                         config: Config,
                         batch_size: int = 10,
                         max_length: int = 128,
                         gpu: bool = False,
                         language: str = 'english',
                         ):
    # Load IE result
    with open(json_path) as r:
        ie_result = [json.loads(line) for line in r]
    entity_result = {rst['sent_id']: [(ent[0], ent[1])
                                      for ent in rst['graph']['entities']]
                 for rst in ie_result}
    # Load source file
    test_set = IEDatasetEval(src_path,
                             max_length=max_length,
                             gpu=gpu,
                             input_format='ltf',
                             language='language'
                             )
    test_set.numberize(tokenizer)
    sent_mention_type_mapping = {}
    for batch in DataLoader(test_set,
                            batch_size=batch_size,
                            shuffle=False,
                            collate_fn=test_set.collate_fn):
        sent_ids = batch.sent_ids
        predicted_entities = [entity_result[sent_id]
                              for sent_id in sent_ids]
        mention_types = model.predict_mention_type(batch, predicted_entities)
        for sent_id, sent_mention_types in zip(sent_ids, mention_types):
            sent_mention_type_mapping[sent_id] = sent_mention_types
    
    # Add mention types
    updated_ie_result = []
    for rst in ie_result:
        sent_id = rst['sent_id']
        sent_mention_types = sent_mention_type_mapping[sent_id]
        rst['graph']['entities'] = [(start, end, enttype, mentype, score)
            for (start, end, enttype, _, score), mentype
            in zip(rst['graph']['entities'], sent_mention_types)
        ]
        updated_ie_result.append(rst)
    return updated_ie_result
        

def run_model(input_dir: str,
              output_dir: str,
              model_path: str,
              device: int = 0,
              gpu: bool = True,
              beam_size: int = 10,
              batch_size: int = 10,
              max_length: int = 128,
              language: str = 'english',
              output_hidden: str = False,
              file_extension: str = 'ltf.xml'):
    # Create the output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    json_output_dir = os.path.join(output_dir, 'json')
    if not os.path.exists(json_output_dir):
        os.mkdir(json_output_dir)
    if output_hidden:
        mention_output_dir = os.path.join(output_dir, 'mention')
        if not os.path.exists(mention_output_dir):
            os.mkdir(mention_output_dir)
        mention_hidden_file = os.path.join(
            mention_output_dir,
            '{}.mention.hidden.txt'.format(lang_codes[language])
        )
        trigger_hidden_file = os.path.join(
            mention_output_dir,
            '{}.trigger.hidden.txt'.format(lang_codes[language])
        )
        w_hid_men = open(mention_hidden_file, 'w')
        w_hid_trg = open(trigger_hidden_file, 'w')
        
    # Load the model
    model, tokenizer, config = load_model(model_path,
                                          device=device,
                                          gpu=gpu,
                                          beam_size=beam_size)

    # Get the list of documents
    input_files = glob.glob(os.path.join(input_dir,
                                         '*.{}'.format(file_extension)))
    input_files.sort()
    # input_files = input_files[:10]
    # Run the model, collect result and info
    doc_info_list = []
    progress = tqdm.tqdm(total=len(input_files), ncols=75)
    for input_file in input_files:
        progress.update(1)
        try:
            results = predict_document(input_file,
                                       model,
                                       tokenizer,
                                       config,
                                       gpu=gpu,
                                       language=language,
                                       batch_size=batch_size,
                                       max_length=max_length,
                                       output_hidden=output_hidden)
            doc_result, doc_info, mention_hiddens, trigger_hiddens = results
            
            # Save JSON format result
            doc_id = doc_info['doc_id']
            json_file = '{}.json'.format(doc_id)
            with open(os.path.join(json_output_dir, json_file), 'w') as w:
                for sent_id, token_ids, tokens, graph in doc_result:
                    output = {'doc_id': doc_id,
                              'sent_id': sent_id,
                              'tokens': tokens,
                              'token_ids': token_ids,
                              'graph': graph.to_dict()}
                    w.write(json.dumps(output) + '\n')
            
            # Save hidden states
            if output_hidden:
                for token, token_id, embed in mention_hiddens:
                    w_hid_men.write('{}\t{}\t{}\n'.format(token, token_id, embed))
                for token, token_id, embed in trigger_hiddens:
                    w_hid_trg.write('{}\t{}\t{}\n'.format(token, token_id, embed))
        except Exception as e:
            traceback.print_exc()
    progress.close()
    
    if output_hidden:
        w_hid_men.close()
        w_hid_trg.close()
            
    # Delete model
    if gpu:
        print('Deleting the model from GPU')
        del model
        torch.cuda.empty_cache()
   
    
def add_mention_types(src_path: str,
                      rst_path: str,
                      model_path: str,
                      device: int = 0,
                      gpu: bool = True,
                      batch_size: int = 10,
                      max_length: int = 128,
                      language: str = 'english',
                      file_extension: str = 'ltf.xml'
                      ):
    # Create the new JSON output dir
    output_json_dir = os.path.join(rst_path, 'json_mentypes')
    if not os.path.exists(output_json_dir):
        os.mkdir(output_json_dir)
    input_json_dir = os.path.join(rst_path, 'json')
    
    # Load the model
    model, tokenizer, config = load_model(model_path,
                                          device=device,
                                          gpu=gpu)
    
    # Get the list of documents
    input_files = glob.glob(os.path.join(src_path,
                                         '*.{}'.format(file_extension)))
    input_files.sort()
    # input_files = input_files[:10]
    progress = tqdm.tqdm(total=len(input_files), ncols=75)
    for input_file in input_files:
        progress.update(1)
        try:
            doc_id = (os.path.basename(input_file)
                  .replace('.{}'.format(file_extension), ''))
            json_file = os.path.join(input_json_dir, '{}.json'.format(doc_id))
            ie_result = predict_mention_type(input_file,
                                             json_file,
                                             model,
                                             tokenizer,
                                             config,
                                             batch_size=batch_size,
                                             max_length=max_length,
                                             gpu=gpu,
                                             language=language)
            output_file = os.path.join(output_json_dir, '{}.json'.format(doc_id))
            with open(output_file, 'w') as w:
                for rst in ie_result:
                    w.write(json.dumps(rst) + '\n')
        except Exception as e:
            traceback.print_exc()


def predict_english(input_path: str,
                    output_path: str,
                    batch_size: int = 10,
                    max_length: int = 128,
                    device: int = 0,
                    gpu: bool = True,
                    file_extension: str = 'ltf.xml',
                    beam_size: int = 10,
                    output_hidden: bool = False,
                    ):
    # Set GPU device
    if gpu:
        torch.cuda.set_device(device)
    
    # Run OneIE
    print('Running Model 1: OneIE trained on ERE+ACE')
    m1_output_path = os.path.join(output_path, 'm1')
    m1_model_path = os.path.join(model_dir, 'english.kairos.ace+ere.mdl')
    run_model(input_path,
              m1_output_path,
              m1_model_path,
              device=device,
              gpu=gpu,
              beam_size=beam_size,
              batch_size=batch_size,
              max_length=max_length,
              language='english',
              output_hidden=output_hidden)
    
    print('Running Model 2: OneIE trained on own annotations')
    m2_output_path = os.path.join(output_path, 'm2')
    m2_model_path = os.path.join(model_dir, 'english.kairos.m2.mdl')
    run_model(input_path,
              m2_output_path,
              m2_model_path,
              device=device,
              gpu=gpu,
              beam_size=beam_size,
              batch_size=batch_size,
              max_length=max_length,
              language='english',
              output_hidden=output_hidden)
    
    # Add mention types to the M2 results
    print('Adding mention types to the results using the AIDA model')
    add_mention_types(input_path,
                      m2_output_path,
                      m1_model_path,
                      device=device,
                      gpu=gpu,
                      batch_size=batch_size,
                      max_length=max_length,
                      language='english')
    
    # Merge two results
    print('Merging JSON files')
    m1_m2_output_path = os.path.join(output_path, 'm1_m2')
    m1_m2_json_output_path = os.path.join(m1_m2_output_path, 'json')
    if not os.path.exists(m1_m2_output_path):
        os.mkdir(m1_m2_output_path)
    if not os.path.exists(m1_m2_json_output_path):
        os.mkdir(m1_m2_json_output_path)
    m1_json_output_path = os.path.join(m1_output_path, 'json')
    m2_json_output_path = os.path.join(m2_output_path, 'json_mentypes')
    merge_json_result(m1_json_output_path,
                      m2_json_output_path,
                      m1_m2_json_output_path)
    
    # Convert to mention and cs
    print('Converting to BIO and TAB format')
    # M1 result
    m1_mention_dir = os.path.join(m1_output_path, 'mention')
    if not os.path.exists(m1_mention_dir):
        os.mkdir(m1_mention_dir)
    json_to_mention_results(m1_json_output_path,
                            m1_mention_dir,
                            'en',
                            rev_entity_type=False)
    bio_to_cfet(os.path.join(m1_mention_dir, 'en.nam.bio'),
                os.path.join(m1_mention_dir, 'en.nam.cfet.json'))
    # M2 result
    m2_mention_dir = os.path.join(m2_output_path, 'mention')
    if not os.path.exists(m2_mention_dir):
        os.mkdir(m2_mention_dir)
    json_to_mention_results(m2_json_output_path,
                            m2_mention_dir,
                            'en',
                            rev_entity_type=False)
    bio_to_cfet(os.path.join(m2_mention_dir, 'en.nam.bio'),
                os.path.join(m2_mention_dir, 'en.nam.cfet.json'))
    # M1M2 result
    m1_m2_mention_dir = os.path.join(m1_m2_output_path, 'mention')
    if not os.path.exists(m1_m2_mention_dir):
        os.mkdir(m1_m2_mention_dir)
    json_to_mention_results(m1_m2_json_output_path,
                            m1_m2_mention_dir,
                            'en',
                            rev_entity_type=False)
    
    # Merge hidden state
    # if output_hidden:
    #     print('Merging hidden states')
    #     merge_hidden_state(
    #         os.path.join(m18_mention_dir, 'en.mention.hidden.txt'),
    #         os.path.join(m36_mention_dir, 'en.mention.hidden.txt'),
    #         os.path.join(mention_dir, 'en.mention.hidden.txt')
    #     )
    #     merge_hidden_state(
    #         os.path.join(m18_mention_dir, 'en.trigger.hidden.txt'),
    #         os.path.join(m36_mention_dir, 'en.trigger.hidden.txt'),
    #         os.path.join(mention_dir, 'en.trigger.hidden.txt')
    #     )
    
    # Convert to CS
    print('Converting to cold start format')
    # M1 result
    m1_cs_dir = os.path.join(m1_output_path, 'cs')
    if not os.path.exists(m1_cs_dir):
        os.mkdir(m1_cs_dir)
    json_to_cs_fg(m1_json_output_path, m1_cs_dir)
    # M2 result
    m2_cs_dir = os.path.join(m2_output_path, 'cs')
    if not os.path.exists(m2_cs_dir):
        os.mkdir(m2_cs_dir)
    json_to_cs_fg(m2_json_output_path, m2_cs_dir)
    # M1 M2 result
    m1_m2_cs_dir = os.path.join(m1_m2_output_path, 'cs')
    if not os.path.exists(m1_m2_cs_dir):
        os.mkdir(m1_m2_cs_dir)
    json_to_cs_fg(m1_m2_json_output_path, m1_m2_cs_dir)
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', help='Path to the input directory')
    parser.add_argument('-o', '--output', help='Path to the output directory')
    parser.add_argument('-l', '--language', default='en', help='Dataset language')
    parser.add_argument('--batch_size', default=10, type=int, help='Batch size')
    parser.add_argument('--beam_size', default=10, type=int, help='Beam size')
    parser.add_argument('--max_len', default=128, type=int, help='Max sequence length')
    parser.add_argument('--device', default=0, type=int, help='GPU index')
    parser.add_argument('--output_hidden', action='store_true')
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    
    input_dir = os.path.join(mount_dir, args.input)
    output_dir = os.path.join(mount_dir, args.output)
    if args.language == 'en':
        predict_english(input_dir,
                        output_dir,
                        batch_size=args.batch_size,
                        max_length=args.max_len,
                        device=args.device,
                        gpu=use_gpu,
                        beam_size=args.beam_size,
                        output_hidden=args.output_hidden
                        )