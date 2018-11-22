import re
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
from QA_model.model_QuAC import QAModel
from general_utils import score, BatchGen_QuAC, find_best_score_and_thresh

parser = argparse.ArgumentParser(
    description='Predict using a Dialog QA model.'
)
parser.add_argument('--dev_dir', default='QuAC_data/')

parser.add_argument('-o', '--output_dir', default='pred_out/')
parser.add_argument('--number', type=int, default=-1, help='id of the current prediction')
parser.add_argument('-m', '--model', default='',
                    help='testing model pathname, e.g. "models/checkpoint_epoch_11.pt"')

parser.add_argument('-bs', '--batch_size', type=int, default=4)
parser.add_argument('--no_ans', type=float, default=0)
parser.add_argument('--min_f1', type=float, default=0.4)

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
    opt['task_name'] = 'QuAC'
    opt['cuda'] = args.cuda
    opt['seed'] = args.seed
    if opt.get('disperse_flow') is None:
        opt['disperse_flow'] = False
    if opt.get('rationale_lambda') is None:
        opt['rationale_lambda'] = 0.0
    if opt.get('no_dialog_flow') is None:
        opt['no_dialog_flow'] = False
    if opt.get('do_hierarchical_query') is None:
        opt['do_hierarchical_query'] = False
    state_dict = checkpoint['state_dict']
    log.info('[model loaded.]')

    test, test_embedding, test_answer = load_dev_data(opt)
    model = QAModel(opt, state_dict = state_dict)
    log.info('[Data loaded.]')

    model.setup_eval_embed(test_embedding)

    if args.cuda:
        model.cuda()

    batches = BatchGen_QuAC(test, batch_size=args.batch_size, evaluation=True, gpu=args.cuda, dialog_ctx=opt['explicit_dialog_ctx'], use_dialog_act=opt['use_dialog_act'], precompute_elmo=opt['elmo_batch_size'] // args.batch_size)
    sample_idx = random.sample(range(len(batches)), args.show)

    predictions = []
    no_ans_scores = []
    for i, batch in enumerate(batches):
        prediction, noans = model.predict(batch, No_Ans_Threshold=args.no_ans)
        predictions.extend(prediction)
        no_ans_scores.extend(noans)

        if not (i in sample_idx):
            continue
        
        print("Context: ", batch[-4][0])
        for j in range(len(batch[-2][0])):
            print("Q: ", batch[-2][0][j])
            print("A: ", prediction[0][j])
            print("     True A: ", batch[-1][0][j], "| Follow up" if batch[-6][0][j].item() // 10 else "| Don't follow up")
            print("     Val. A: ", test_answer[args.batch_size * i][j])
        print("")


    pred_out = {'predictions': predictions, 'no_ans_scores': no_ans_scores}
    with open(args.output, 'wb') as f:
        pickle.dump(pred_out, f)

    f1, h_f1, HEQ_Q, HEQ_D = score(predictions, test_answer, min_F1=args.min_f1)
    log.warning("Test F1: {:.2f}, HEQ_Q: {:.2f}, HEQ_D: {:.2f}".format(f1, HEQ_Q, HEQ_D))

def load_dev_data(opt): # can be extended to true test set
    with open(os.path.join(args.dev_dir, 'dev_meta.msgpack'), 'rb') as f:
        meta = msgpack.load(f, encoding='utf8')
    embedding = torch.Tensor(meta['embedding'])
    assert opt['embedding_dim'] == embedding.size(1)

    with open(os.path.join(args.dev_dir, 'dev_data.msgpack'), 'rb') as f:
        data = msgpack.load(f, encoding='utf8')

    assert opt['num_features'] == len(data['context_features'][0][0]) + opt['explicit_dialog_ctx'] * (opt['use_dialog_act']*3 + 2)
    
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
                        data['answer_choice'],
                        data['question'],
                        data['answer'],
                        data['question_tokenized']))
          }
    
    dev_answer = []
    for i, CID in enumerate(data['question_CID']):
        if len(dev_answer) <= CID:
            dev_answer.append([])
        dev_answer[CID].append(data['all_answer'][i])
    
    return dev, embedding, dev_answer

if __name__ == '__main__':
    main()
