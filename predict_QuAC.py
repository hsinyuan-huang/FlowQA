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
from tqdm import tqdm
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

parser.add_argument('-bs', '--batch_size', type=int, default=1)
parser.add_argument('--no_ans', type=float, default=0)
parser.add_argument('--min_f1', type=float, default=0.4)

parser.add_argument('--show', type=int, default=3)
parser.add_argument('--seed', type=int, default=1023,
                    help='random seed for data shuffling, dropout, etc.')
parser.add_argument('--use_bert', type=int, default=1,
                            help='pass 1 to use bert')
parser.add_argument('--finetune_bert', type=int, default=1,
                            help='pass 1 to finetune bert')
parser.add_argument('--bert_type', type=str, default='bert-base-uncased')
parser.add_argument('--bert_num_layers', type=int, default=4)
parser.add_argument('--bert_lr', type=float, default=1e-5)
parser.add_argument('--bert_warmup', type=float, default=-1)
parser.add_argument('--bert_t_total', type=int, default=-1)
parser.add_argument('--bert_schedule', type=str, default='warmup_constant')
parser.add_argument('--bert_stride', type=int, default=256)
parser.add_argument('--bert_start_idx', type=int, default=8)
parser.add_argument('--bert_agg_type', type=str, default='max')
parser.add_argument('--aggregate_grad_steps', type=int, default=3)
parser.add_argument('--load_optimizer', type=int, default=0)
parser.add_argument('--use_positional', type=int, default=1)
parser.add_argument('--max_seq_length', type=int, default=1925)
parser.add_argument('--positional_emb_dim', type=int, default=15)
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
    for i, batch in tqdm(enumerate(batches)):
        prediction, noans = model.predict(batch, No_Ans_Threshold=args.no_ans)
        predictions.extend([{"best_span_str" : preds, "qid" : qid} for preds, qid in zip(prediction, batch[-1])])
        no_ans_scores.extend(noans)

    pred_ans = [x["best_span_str"] for x in predictions]
    question_f1 = []
    for testarr, predarr in zip(test_answer, pred_ans):
        question_f1.append([])
        for ta, pa in zip(testarr, predarr):
            qf1, _, _, _  = score([[pa]], [[ta]], 0)
            question_f1[-1].append(qf1)

    for i in range(len(predictions)):
        predictions[i]['F1'] = question_f1[i]

    pred_out = {'predictions': predictions, 'no_ans_scores': no_ans_scores}
    with open(args.output, 'wb') as f:
        pickle.dump(pred_out, f)

    f1, h_f1, HEQ_Q, HEQ_D = score([x["best_span_str"] for x in predictions], test_answer, min_F1=args.min_f1)
    log.warning("Test F1: {:.2f}, HEQ_Q: {:.2f}, HEQ_D: {:.2f}".format(f1, HEQ_Q, HEQ_D))

def load_dev_data(opt): # can be extended to true test set
    with open(os.path.join(args.dev_dir, 'dev_meta.msgpack'), 'rb') as f:
        meta = msgpack.load(f, encoding='utf8')
    embedding = torch.Tensor(meta['embedding'])
    assert opt['embedding_dim'] == embedding.size(1)

    with open(os.path.join(args.dev_dir, 'dev_data.msgpack'), 'rb') as f:
        data = msgpack.load(f, encoding='utf8')
    #data_orig = pd.read_csv(os.path.join(args.dev_dir, 'dev.csv'))

    if opt['use_bert']:
        dev = {'context': list(zip(
                            data['context_ids'],
                            data['context_tags'],
                            data['context_ents'],
                            data['context'],
                            data['context_span'],
                            data['1st_question'],
                            data['context_tokenized'],
                            data['context_bertidx'],
                            data['context_bert_spans'])),
               'qa': list(zip(
                            data['question_CID'],
                            data['question_ids'],
                            data['context_features'],
                            data['answer_start'],
                            data['answer_end'],
                            data['answer_choice'],
                            data['question'],
                            data['answer'],
                            data['question_tokenized'],
                            data['question_bertidx'],
                            data['question_bert_spans'],
                            data['qids']))
              }
    else:
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
                            data['question_tokenized'],
                            data['question_tokenized'],
                            data['qids']))
              }

    dev_answer = []
    for i, CID in enumerate(data['question_CID']):
        if len(dev_answer) <= CID:
            dev_answer.append([])
        dev_answer[CID].append(data['all_answer'][i])

    return dev, embedding, dev_answer

if __name__ == '__main__':
    main()
