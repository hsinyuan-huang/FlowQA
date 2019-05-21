import re
import pickle
import json
import spacy
import msgpack
import unicodedata
import numpy as np
import pandas as pd
import argparse
import collections
import multiprocessing
import logging
import random
import torch
from pytorch_pretrained_bert import BertTokenizer
from allennlp.modules.elmo import batch_to_ids
from general_utils import flatten_json, normalize_text, build_embedding, load_glove_vocab, pre_proc, get_context_span, find_answer_span, feature_gen, token2id
from QA_model import constants

parser = argparse.ArgumentParser()
parser.add_argument('-i', default='./input_file')
parser.add_argument('-o', default='./output_file')
args = parser.parse_args()

bertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize(lis):
    global bertTokenizer
    tokens = [bertTokenizer.tokenize(" ".join(x)) for x in lis]
    return tokens

def bert_tokens_to_ids(tokens):
    global bertTokenizer
    ids = []
    for i in range(len(tokens) // constants.BERT_MAXLEN + 1):
        ids.extend(bertTokenizer.convert_tokens_to_ids(tokens[i * constants.BERT_MAXLEN : (i + 1) * constants.BERT_MAXLEN]))
    return ids

def proc_dev(ith, article):
    rows = []
    
    for paragraph in article['paragraphs']:
        context = paragraph['context']
        for qa in paragraph['qas']:
            qid = qa['id']
            question = qa['question']
            answers = qa['orig_answer']
            
            answer = answers['text']
            answer_start = answers['answer_start']
            answer_end = answers['answer_start'] + len(answers['text'])
            answer_choice = 0 if answer == 'CANNOTANSWER' else\
                            1 if qa['yesno'] == 'y' else\
                            2 if qa['yesno'] == 'n' else\
                            3 # Not a yes/no question
            if answer_choice != 0:
                """
                0: Do not ask a follow up question!
                1: Definitely ask a follow up question!
                2: Not too important, but you can ask a follow up.
                """
                answer_choice += 10 * (0 if qa['followup'] == "n" else\
                                       1 if qa['followup'] == "y" else\
                                       2)
            else:
                answer_start, answer_end = -1, -1
            
            ans_ls = []
            for ans in qa['answers']:
                ans_ls.append(ans['text'])
            
            rows.append((ith, question, answer, answer_start, answer_end, answer_choice, ans_ls, qid))

    return rows, context

def build_dev_vocab(questions, contexts): # most vocabulary comes from tr_vocab
    # TODO: Needs train vocab
    tr_vocab = pickle.load(open('train_vocab.pkl', 'rb'))
    existing_vocab = set(tr_vocab)
    glove_vocab = load_glove_vocab('./glove/glove.840B.300d.txt', 300) # return a "set" of vocabulary
    new_vocab = list(set([w for doc in questions + contexts for w in doc if w not in existing_vocab and w in glove_vocab]))
    vocab = tr_vocab + new_vocab
    print('train vocab {0}, total vocab {1}'.format(len(tr_vocab), len(vocab)))
    return vocab

def calc_bert_spans(berttokens, tokens):
    span_idx = [[]]
    currlen = 0
    tidx = 0
    for i, x in enumerate(berttokens):
        currlen += len(x) if (x[0] != '#' or len(x) == 1) else len(x) - 2
        span_idx[-1].append(i)
        if currlen == len(tokens[tidx]):
            span_idx.append([])
            currlen = 0
            tidx += 1
    if span_idx[-1] == []:
        del span_idx[-1]
    return span_idx


def preprocess_data(dev_file):
    dev, dev_context = flatten_json(dev_file, proc_dev)

    dev = pd.DataFrame(dev, columns=['context_idx', 'question', 'answer',
                                         'answer_start', 'answer_end', 'answer_choice', 'all_answer', 'qid'])
    print('dev json data flattened.')

    devC_iter = (pre_proc(c) for c in dev_context)
    devQ_iter = (pre_proc(q) for q in dev.question)
    nlp = spacy.load('en', disable=['parser'])
    devC_docs = [doc for doc in nlp.pipe(
        devC_iter, batch_size=64, n_threads=multiprocessing.cpu_count())]
    devQ_docs = [doc for doc in nlp.pipe(
        devQ_iter, batch_size=64, n_threads=multiprocessing.cpu_count())]
    del nlp

    devC_tokens = [[normalize_text(w.text) for w in doc] for doc in devC_docs]
    devQ_tokens = [[normalize_text(w.text) for w in doc] for doc in devQ_docs]
    devC_unnorm_tokens = [[w.text for w in doc] for doc in devC_docs]
    print('All tokens for dev are obtained.')

    dev_context_span = [get_context_span(a, b) for a, b in zip(dev_context, devC_unnorm_tokens)]
    print('context span for dev is generated.')

    ans_st_token_ls, ans_end_token_ls = [], []
    for ans_st, ans_end, idx in zip(dev.answer_start, dev.answer_end, dev.context_idx):
        ans_st_token, ans_end_token = find_answer_span(dev_context_span[idx], ans_st, ans_end)
        ans_st_token_ls.append(ans_st_token)
        ans_end_token_ls.append(ans_end_token)

    dev['answer_start_token'], dev['answer_end_token'] = ans_st_token_ls, ans_end_token_ls
    initial_len = len(dev)
    dev.dropna(inplace=True) # modify self DataFrame
    print('drop {0}/{1} inconsistent samples.'.format(initial_len - len(dev), initial_len))
    print('answer span for dev is generated.')

    devC_tags, devC_ents, devC_features = feature_gen(devC_docs, dev.context_idx, devQ_docs, False)
    print('features for dev is generated: {}, {}, {}'.format(len(devC_tags), len(devC_ents), len(devC_features)))

    dev_vocab = build_dev_vocab(devQ_tokens, devC_tokens) # tr_vocab is a subset of dev_vocab
    devC_ids = token2id(devC_tokens, dev_vocab, unk_id=1)
    devQ_ids = token2id(devQ_tokens, dev_vocab, unk_id=1)
    devQ_tokens = [["<S>"] + doc + ["</S>"] for doc in devQ_tokens]
    devQ_ids = [[2] + qsent + [3] for qsent in devQ_ids]

    # BERT stuff
    devC_bert_tokens = tokenize(devC_tokens)
    devC_bert_ids = [bert_tokens_to_ids(x) for x in devC_bert_tokens]
    devQ_bert_tokens = tokenize(devQ_tokens)
    devQ_bert_ids = [bert_tokens_to_ids(x) for x in devQ_bert_tokens]

    devC_bert_spans = [calc_bert_spans(b, t) for b, t in zip(devC_bert_tokens, devC_tokens)]
    devQ_bert_spans = [calc_bert_spans(b, t) for b, t in zip(devQ_bert_tokens, devQ_tokens)]

    vocab_tag = pickle.load(open('./vocab_tag.pkl', 'rb'))
    vocab_ent = pickle.load(open('./vocab_ent.pkl', 'rb'))

    devC_tag_ids = token2id(devC_tags, vocab_tag) # vocab_tag same as training
    # entities
    devC_ent_ids = token2id(devC_ents, vocab_ent, unk_id=0) # vocab_ent same as training
    print('vocabulary for dev is built.')

    dev_embedding = build_embedding('glove/glove.840B.300d.txt', dev_vocab, 300)

    meta = {
        'vocab': dev_vocab,
        'embedding': dev_embedding.tolist()
    }

    prev_CID, first_question = -1, []
    for i, CID in enumerate(dev.context_idx):
        if not (CID == prev_CID):
            first_question.append(i)
        prev_CID = CID

    result = {
        'qids': dev.qid.tolist(),
        'question_ids': devQ_ids,
        'context_ids': devC_ids,
        'context_features': devC_features, # exact match, tf
        'context_tags': devC_tag_ids, # POS tagging
        'context_ents': devC_ent_ids, # Entity recognition
        'context': dev_context,
        'context_span': dev_context_span,
        '1st_question': first_question,
        'question_CID': dev.context_idx.tolist(),
        'question': dev.question.tolist(),
        'answer': dev.answer.tolist(),
        'answer_start': dev.answer_start_token.tolist(),
        'answer_end': dev.answer_end_token.tolist(),
        'answer_choice': dev.answer_choice.tolist(),
        'all_answer': dev.all_answer.tolist(),
        'context_tokenized': devC_tokens,
        'question_tokenized': devQ_tokens,
        'context_bertidx': devC_bert_ids,
        'context_bert_spans': devC_bert_spans,
        'question_bertidx': devQ_bert_ids,
        'question_bert_spans': devQ_bert_spans
    }

    return meta, result

def load_dev_data(meta, data): # can be extended to true test set
    embedding = torch.Tensor(meta['embedding'])

    #data_orig = pd.read_csv(os.path.join(args.dev_dir, 'dev.csv'))

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

    dev_answer = []
    for i, CID in enumerate(data['question_CID']):
        if len(dev_answer) <= CID:
            dev_answer.append([])
        dev_answer[CID].append(data['all_answer'][i])

    return dev, embedding, dev_answer


def main():
    meta, test = preprocess_data(args.i)
    test, test_embedding, test_answer = load_dev_data(meta, test)
    checkpoint = torch.load('best_model.pt')

if __name__ == '__main__':
    main()
