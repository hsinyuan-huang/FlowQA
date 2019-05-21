import os
import re
import sys
import random
import string
import logging
import argparse
from shutil import copyfile
from datetime import datetime
from collections import Counter
import torch
import msgpack
import pandas as pd
import numpy as np
from QA_model.model_QuAC import QAModel
from general_utils import find_best_score_and_thresh, BatchGen_QuAC
from QA_model import constants

parser = argparse.ArgumentParser(
    description='Train a Dialog QA model.'
)

# system
parser.add_argument('--task_name', default='QuAC')
parser.add_argument('--name', default='', help='additional name of the current run')
parser.add_argument('--log_file', default='output.log',
                    help='path for log file.')
parser.add_argument('--log_per_updates', type=int, default=20,
                    help='log model loss per x updates (mini-batches).')

parser.add_argument('--train_dir', default='QuAC_data/')
parser.add_argument('--dev_dir', default='QuAC_data/')
parser.add_argument('--answer_type_num', type=int, default=1)

parser.add_argument('--model_dir', default='models',
                    help='path to store saved models.')
parser.add_argument('--eval_per_epoch', type=int, default=1,
                    help='perform evaluation per x epoches.')
parser.add_argument('--MTLSTM_path', default='glove/MT-LSTM.pth')
parser.add_argument('--save_all', dest='save_best_only', action='store_false', help='save all models.')
parser.add_argument('--do_not_save', action='store_true', help='don\'t save any model')
parser.add_argument('--save_for_predict', action='store_true')
parser.add_argument('--seed', type=int, default=1023,
                    help='random seed for data shuffling, dropout, etc.')
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
                    help='whether to use GPU acceleration.')
# training
parser.add_argument('-e', '--epoches', type=int, default=30)
parser.add_argument('-bs', '--batch_size', type=int, default=1)
parser.add_argument('-ebs', '--elmo_batch_size', type=int, default=12)
parser.add_argument('-rs', '--resume', default='',
                    help='previous model pathname. '
                         'e.g. "models/checkpoint_epoch_11.pt"')
parser.add_argument('-ro', '--resume_options', action='store_true',
                    help='use previous model options, ignore the cli and defaults.')
parser.add_argument('-rlr', '--reduce_lr', type=float, default=0.,
                    help='reduce initial (resumed) learning rate by this factor.')
parser.add_argument('-op', '--optimizer', default='adamax',
                    help='supported optimizer: adamax, sgd, adadelta, adam')
parser.add_argument('-gc', '--grad_clipping', type=float, default=10)
parser.add_argument('-wd', '--weight_decay', type=float, default=0)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.1,
                    help='only applied to SGD.')
parser.add_argument('-mm', '--momentum', type=float, default=0,
                    help='only applied to SGD.')
parser.add_argument('-tp', '--tune_partial', type=int, default=1000,
                    help='finetune top-x embeddings (including <PAD>, <UNK>).')
parser.add_argument('--fix_embeddings', action='store_true',
                    help='if true, `tune_partial` will be ignored.')
parser.add_argument('--elmo_lambda', type=float, default=0.0)
parser.add_argument('--no_question_normalize', dest='question_normalize', action='store_false') # when set, do dialog normalize
parser.add_argument('--pretrain', default='')

# model
parser.add_argument('--explicit_dialog_ctx', type=int, default=2)
parser.add_argument('--use_dialog_act', action='store_true')
parser.add_argument('--no_dialog_flow', action='store_true')
parser.add_argument('--no_hierarchical_query', dest='do_hierarchical_query', action='store_false')
parser.add_argument('--no_prealign', dest='do_prealign', action='store_false')

parser.add_argument('--final_output_att_hidden', type=int, default=250)
parser.add_argument('--question_merge', default='linear_self_attn')
parser.add_argument('--no_ptr_update', dest='do_ptr_update', action='store_false')
parser.add_argument('--no_ptr_net_indep_attn', dest='ptr_net_indep_attn', action='store_false')
parser.add_argument('--ptr_net_attn_type', default='Bilinear', help="Attention for answer span output: Bilinear, MLP or Default")

parser.add_argument('--do_residual_rnn', dest='do_residual_rnn', action='store_true')
parser.add_argument('--do_residual_everything', dest='do_residual_everything', action='store_true')
parser.add_argument('--do_residual', dest='do_residual', action='store_true')
parser.add_argument('--rnn_layers', type=int, default=1, help="Default number of RNN layers")
parser.add_argument('--rnn_type', default='lstm',
                    help='supported types: rnn, gru, lstm')
parser.add_argument('--concat_rnn', dest='concat_rnn', action='store_true')

parser.add_argument('--hidden_size', type=int, default=125)
parser.add_argument('--self_attention_opt', type=int, default=1) # 0: no self attention

parser.add_argument('--deep_inter_att_do_similar', type=int, default=0)
parser.add_argument('--deep_att_hidden_size_per_abstr', type=int, default=250)

parser.add_argument('--no_elmo', dest='use_elmo', action='store_false')
parser.add_argument('--no_em', action='store_true')

parser.add_argument('--no_wemb', dest='use_wemb', action='store_false') # word embedding
parser.add_argument('--CoVe_opt', type=int, default=1) # contexualized embedding option
parser.add_argument('--no_pos', dest='use_pos', action='store_false') # pos tagging
parser.add_argument('--pos_size', type=int, default=51, help='how many kinds of POS tags.')
parser.add_argument('--pos_dim', type=int, default=12, help='the embedding dimension for POS tags.')
parser.add_argument('--no_ner', dest='use_ner', action='store_false') # named entity
parser.add_argument('--ner_size', type=int, default=19, help='how many kinds of named entity tags.')
parser.add_argument('--ner_dim', type=int, default=8, help='the embedding dimension for named entity tags.')

parser.add_argument('--prealign_hidden', type=int, default=300)
parser.add_argument('--prealign_option', type=int, default=2, help='0: No prealign, 1, 2, ...: Different options')

parser.add_argument('--no_seq_dropout', dest='do_seq_dropout', action='store_false')
parser.add_argument('--my_dropout_p', type=float, default=0.4)
parser.add_argument('--dropout_emb', type=float, default=0.4)

parser.add_argument('--max_len', type=int, default=35)
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
parser.add_argument('--bert_stride', type=int, default=constants.BERT_MAXLEN)
parser.add_argument('--bert_start_idx', type=int, default=6)
parser.add_argument('--bert_agg_type', type=str, default='mean')
parser.add_argument('--aggregate_grad_steps', type=int, default=3)
parser.add_argument('--load_optimizer', type=int, default=1)
parser.add_argument('--use_positional', type=int, default=1)
parser.add_argument('--max_seq_length', type=int, default=1925)
parser.add_argument('--positional_emb_dim', type=int, default=15)

args = parser.parse_args()
assert 0 <= args.bert_stride <= constants.BERT_MAXLEN, "bert stride should be less than or equal to %d" % constants.BERT_MAXLEN

if args.name != '':
    args.model_dir = args.model_dir + '_' + args.name
    args.log_file = os.path.dirname(args.log_file) + 'output_' + args.name + '.log'

if args.bert_lr == 0 or args.use_bert == 0:
    args.finetune_bert = 0

# set model dir
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)
model_dir = os.path.abspath(model_dir)

# set random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.seed)

# setup logger
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
fh = logging.FileHandler(args.log_file)
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
log.addHandler(fh)
log.addHandler(ch)

def main():
    log.info('[program starts.]')
    log.info('seed: {}'.format(args.seed))
    log.info(str(vars(args)))
    opt = vars(args) # changing opt will change args
    train, train_embedding, opt = load_train_data(opt)
    dev, dev_embedding, dev_answer = load_dev_data(opt)
    opt['num_features'] += args.explicit_dialog_ctx * (args.use_dialog_act*3 + 2) # dialog_act + previous answer
    if opt['use_elmo'] == False:
        opt['elmo_batch_size'] = 0
    log.info('[Data loaded.]')

    if args.resume:
        log.info('[loading previous model...]')
        if args.cuda:
            checkpoint = torch.load(args.resume, map_location={'cpu': 'cuda:0'})
        else:
            checkpoint = torch.load(args.resume, map_location={'cuda:0': 'cpu'})
        if args.resume_options:
            opt = checkpoint['config']
        state_dict = checkpoint['state_dict']
        model = QAModel(opt, train_embedding, state_dict)
        epoch_0 = checkpoint['epoch'] + 1
        for i in range(checkpoint['epoch']):
            random.shuffle(list(range(len(train))))  # synchronize random seed
        if args.reduce_lr:
            lr_decay(model.optimizer, lr_decay=args.reduce_lr)
    else:
        model = QAModel(opt, train_embedding)
        epoch_0 = 1

    if args.pretrain:
        pretrain_model = torch.load(args.pretrain)
        state_dict = pretrain_model['state_dict']['network']

        model.get_pretrain(state_dict)

    model.setup_eval_embed(dev_embedding)
    log.info("[dev] Total number of params: {}".format(model.total_param))

    if args.cuda:
        model.cuda()

    if args.resume:
        batches = BatchGen_QuAC(dev, batch_size=args.batch_size, evaluation=True, gpu=args.cuda, dialog_ctx=args.explicit_dialog_ctx, use_dialog_act=args.use_dialog_act, use_bert=args.use_bert)
        predictions, no_ans_scores = [], []
        for batch in batches:
            phrases, noans = model.predict(batch)
            predictions.extend(phrases)
            no_ans_scores.extend(noans)
        f1, na, thresh = find_best_score_and_thresh(predictions, dev_answer, no_ans_scores)
        log.info("[dev F1: {} NA: {} TH: {}]".format(f1, na, thresh))
        best_val_score, best_na, best_thresh = f1, na, thresh
    else:
        best_val_score, best_na, best_thresh = 0.0, 0.0, 0.0
    
    aggregate_grad_steps = 1
    if opt['use_bert']:
        aggregate_grad_steps = opt['aggregate_grad_steps']

    for epoch in range(epoch_0, epoch_0 + args.epoches):

        log.warning('Epoch {}'.format(epoch))
        # train
        batches = BatchGen_QuAC(train, batch_size=args.batch_size, gpu=args.cuda, dialog_ctx=args.explicit_dialog_ctx, use_dialog_act=args.use_dialog_act, precompute_elmo=args.elmo_batch_size // args.batch_size, use_bert=args.use_bert)
        start = datetime.now()
        
        total_batches = len(batches)
        loss = 0
        model.optimizer.zero_grad()
        if opt['finetune_bert']:
            model.bertadam.zero_grad()
        
        for i, batch in enumerate(batches):
            loss += model.update(batch)
            if (i+1) % aggregate_grad_steps == 0 or total_batches == (i+1):
                # Update the gradients
                model.take_step()
                loss = 0
            if i % args.log_per_updates == 0:
                log.info('updates[{0:6}] train loss[{1:.5f}] remaining[{2}]'.format(
                    model.updates, model.train_loss.avg,
                    str((datetime.now() - start) / (i + 1) * (len(batches) - i - 1)).split('.')[0]))
        # eval
        if epoch % args.eval_per_epoch == 0:
            batches = BatchGen_QuAC(dev, batch_size=args.batch_size, evaluation=True, gpu=args.cuda, dialog_ctx=args.explicit_dialog_ctx, use_dialog_act=args.use_dialog_act, precompute_elmo=args.elmo_batch_size // args.batch_size, use_bert=args.use_bert)
            predictions, no_ans_scores = [], []
            for batch in batches:
                phrases, noans = model.predict(batch)
                predictions.extend(phrases)
                no_ans_scores.extend(noans)
            f1, na, thresh = find_best_score_and_thresh(predictions, dev_answer, no_ans_scores)

        # save
        if args.save_best_only:
            if f1 > best_val_score:
                best_val_score, best_na, best_thresh = f1, na, thresh
                model_file = os.path.join(model_dir, 'best_model.pt')
                model.save(model_file, epoch)
                log.info('[new best model saved.]')
        else:
            model_file = os.path.join(model_dir, 'checkpoint_epoch_{}.pt'.format(epoch))
            model.save(model_file, epoch)
            if f1 > best_val_score:
                best_val_score, best_na, best_thresh = f1, na, thresh
                copyfile(os.path.join(model_dir, model_file),
                         os.path.join(model_dir, 'best_model.pt'))
                log.info('[new best model saved.]')

        log.warning("Epoch {} - dev F1: {:.3f} NA: {:.3f} TH: {:.3f} (best F1: {:.3f} NA: {:.3f} TH: {:.3f})".format(epoch, f1, na, thresh, best_val_score, best_na, best_thresh))

def lr_decay(optimizer, lr_decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    log.info('[learning rate reduced by {}]'.format(lr_decay))
    return optimizer

def load_train_data(opt):
    with open(os.path.join(args.train_dir, 'train_meta.msgpack'), 'rb') as f:
        meta = msgpack.load(f, encoding='utf8')
    embedding = torch.Tensor(meta['embedding'])
    opt['vocab_size'] = embedding.size(0)
    opt['embedding_dim'] = embedding.size(1)

    with open(os.path.join(args.train_dir, 'train_data.msgpack'), 'rb') as f:
        data = msgpack.load(f, encoding='utf8')
    #data_orig = pd.read_csv(os.path.join(args.train_dir, 'train.csv'))

    opt['num_features'] = len(data['context_features'][0][0])
    
    if opt['use_bert']:
        train = {'context': list(zip(
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
                            data['question_bert_spans']))
                }
    else:
        train = {'context': list(zip(
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
    return train, embedding, opt

def load_dev_data(opt): # can be extended to true test set
    with open(os.path.join(args.dev_dir, 'dev_meta.msgpack'), 'rb') as f:
        meta = msgpack.load(f, encoding='utf8')
    embedding = torch.Tensor(meta['embedding'])
    assert opt['embedding_dim'] == embedding.size(1)

    with open(os.path.join(args.dev_dir, 'dev_data.msgpack'), 'rb') as f:
        data = msgpack.load(f, encoding='utf8')
    #data_orig = pd.read_csv(os.path.join(args.dev_dir, 'dev.csv'))

    assert opt['num_features'] == len(data['context_features'][0][0])

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
                            data['question_bert_spans']))
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
