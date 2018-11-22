import re
import json
import os
import sys
import random
import string
import logging
import argparse
from os.path import basename
from shutil import copyfile
from datetime import datetime
from collections import Counter
import torch
import msgpack
import pickle
import pandas as pd
import numpy as np
from QA_model.model_CoQA import QAModel
from CoQA_eval import CoQAEvaluator
from general_utils import score, BatchGen_CoQA

parser = argparse.ArgumentParser(
    description='Predict using a Dialog QA model.'
)
parser.add_argument('--dev_dir', default='CoQA/')

parser.add_argument('-o', '--output_dir', default='pred_out/')
parser.add_argument('--number', type=int, default=-1, help='id of the current prediction')
parser.add_argument('-m', '--model', default='',
                    help='testing model pathname, e.g. "models/checkpoint_epoch_11.pt"')

parser.add_argument('-bs', '--batch_size', default=1)

parser.add_argument('--show', type=int, default=3)
parser.add_argument('--seed', type=int, default=1023,
                    help='random seed for data shuffling, dropout, etc.')
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
                    help='whether to use GPU acceleration.')

args = parser.parse_args()
if args.model == '':
    print("model file is not provided")
    sys.exit(-1)
if args.model[-3:] != '.pt':
    print("does not recognize the model file")
    sys.exit(-1)

# create prediction output dir
os.makedirs(args.output_dir, exist_ok=True)
# count the number of prediction files
if args.number == -1:
    args.number = len(os.listdir(args.output_dir))+1
args.output = args.output_dir + 'pred' + str(args.number) + '.pckl'

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.seed)

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
ch.setFormatter(formatter)
log.addHandler(ch)

def main():
    log.info('[program starts.]')
    checkpoint = torch.load(args.model)
    opt = checkpoint['config']
    opt['task_name'] = 'CoQA'
    opt['cuda'] = args.cuda
    opt['seed'] = args.seed
    if opt.get('do_hierarchical_query') is None:
        opt['do_hierarchical_query'] = False
    state_dict = checkpoint['state_dict']
    log.info('[model loaded.]')

    test, test_embedding = load_dev_data(opt)
    model = QAModel(opt, state_dict = state_dict)
    CoQAEval = CoQAEvaluator("CoQA/dev.json")
    log.info('[Data loaded.]')

    model.setup_eval_embed(test_embedding)

    if args.cuda:
        model.cuda()

    batches = BatchGen_CoQA(test, batch_size=args.batch_size, evaluation=True, gpu=args.cuda, dialog_ctx=opt['explicit_dialog_ctx'], precompute_elmo=16 // args.batch_size)
    sample_idx = random.sample(range(len(batches)), args.show)

    with open("CoQA/dev.json", "r", encoding="utf8") as f:
        dev_data = json.load(f)

    list_of_ids = []
    for article in dev_data['data']:
        id = article["id"]
        for Qs in article["questions"]:
            tid = Qs["turn_id"]
            list_of_ids.append((id, tid))

    predictions = []
    for i, batch in enumerate(batches):
        prediction = model.predict(batch)
        predictions.extend(prediction)

        if not (i in sample_idx):
            continue

        print("Story: ", batch[-4][0])
        for j in range(len(batch[-2][0])):
            print("Q: ", batch[-2][0][j])
            print("A: ", prediction[j])
            print("Gold A: ", batch[-1][0][j])
            print("---")
        print("")

    assert(len(list_of_ids) == len(predictions))
    official_predictions = []
    for ids, pred in zip(list_of_ids, predictions):
        official_predictions.append({
         "id": ids[0],
         "turn_id": ids[1],
         "answer": pred})
    with open("model_prediction.json", "w", encoding="utf8") as f:
        json.dump(official_predictions, f)

    f1 = CoQAEval.compute_turn_score_seq(predictions)
    log.warning("Test F1: {:.3f}".format(f1 * 100.0))

def load_dev_data(opt): # can be extended to true test set
    with open(os.path.join(args.dev_dir, 'dev_meta.msgpack'), 'rb') as f:
        meta = msgpack.load(f, encoding='utf8')
    embedding = torch.Tensor(meta['embedding'])
    assert opt['embedding_dim'] == embedding.size(1)

    with open(os.path.join(args.dev_dir, 'dev_data.msgpack'), 'rb') as f:
        data = msgpack.load(f, encoding='utf8')

    assert opt['num_features'] == len(data['context_features'][0][0]) + opt['explicit_dialog_ctx'] * 3

    dev = {'context': list(zip(
                        data['context_ids'],
                        data['context_tags'],
                        data['context_ents'],
                        data['context'],
                        data['context_span'],
                        data['1st_question'],
                        data['context_tokenized'])),
           'qa': list(zip(
                        data['question_CID'],
                        data['question_ids'],
                        data['context_features'],
                        data['answer_start'],
                        data['answer_end'],
                        data['rationale_start'],
                        data['rationale_end'],
                        data['answer_choice'],
                        data['question'],
                        data['answer'],
                        data['question_tokenized']))
          }

    return dev, embedding

if __name__ == '__main__':
    main()
