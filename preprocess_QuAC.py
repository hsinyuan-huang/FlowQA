import re
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

parser = argparse.ArgumentParser(
    description='Preprocessing train + dev files, about 20 minutes to run on Servers.'
)
parser.add_argument('--wv_file', default='glove/glove.840B.300d.txt',
                    help='path to word vector file.')
parser.add_argument('--wv_dim', type=int, default=300,
                    help='word vector dimension.')
parser.add_argument('--sort_all', action='store_true',
                    help='sort the vocabulary by frequencies of all words.'
                         'Otherwise consider question words first.')
parser.add_argument('--threads', type=int, default=multiprocessing.cpu_count(),
                    help='number of threads for preprocessing.')
parser.add_argument('--use_bert', type=int, default=1,
                    help='pass 1 to save preprocessed indices for bert tokens')
parser.add_argument('--bert_type', type=str, default='bert-base-uncased')
parser.add_argument('--no_match', action='store_true',
                    help='do not extract the three exact matching features.')
parser.add_argument('--seed', type=int, default=1023,
                    help='random seed for data shuffling, embedding init, etc.')

args = parser.parse_args()
if args.use_bert:
    bertTokenizer = BertTokenizer.from_pretrained(args.bert_type)
trn_file = 'QuAC_data/train.json'
dev_file = 'QuAC_data/dev.json'
wv_file = args.wv_file
wv_dim = args.wv_dim
nlp = spacy.load('en', disable=['parser'])

random.seed(args.seed)
np.random.seed(args.seed)

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG,
                    datefmt='%m/%d/%Y %I:%M:%S')
log = logging.getLogger(__name__)

log.info('start data preparing... (using {} threads)'.format(args.threads))

glove_vocab = load_glove_vocab(wv_file, wv_dim) # return a "set" of vocabulary
log.info('glove loaded.')

#===============================================================
#=================== Work on training data =====================
#===============================================================

def proc_train(ith, article):
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
            rows.append((ith, question, answer, answer_start, answer_end, answer_choice, qid))

    return rows, context

train, train_context = flatten_json(trn_file, proc_train, args.use_bert)
train = pd.DataFrame(train, columns=['context_idx', 'question', 'answer',
                                    'answer_start', 'answer_end', 'answer_choice', 'qid'])

log.info('train json data flattened.')

print(train)

trC_iter = (pre_proc(c) for c in train_context)
trQ_iter = (pre_proc(q) for q in train.question)
trC_docs = [doc for doc in nlp.pipe(trC_iter, batch_size=64, n_threads=args.threads)]
trQ_docs = [doc for doc in nlp.pipe(trQ_iter, batch_size=64, n_threads=args.threads)]

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

# tokens
trC_tokens = [[normalize_text(w.text) for w in doc] for doc in trC_docs]
trQ_tokens = [[normalize_text(w.text) for w in doc] for doc in trQ_docs]


trC_unnorm_tokens = [[w.text for w in doc] for doc in trC_docs]
log.info('All tokens for training are obtained.')

train_context_span = [get_context_span(a, b) for a, b in zip(train_context, trC_unnorm_tokens)]

ans_st_token_ls, ans_end_token_ls = [], []
for ans_st, ans_end, idx in zip(train.answer_start, train.answer_end, train.context_idx):
    ans_st_token, ans_end_token = find_answer_span(train_context_span[idx], ans_st, ans_end)
    ans_st_token_ls.append(ans_st_token)
    ans_end_token_ls.append(ans_end_token)

train['answer_start_token'], train['answer_end_token'] = ans_st_token_ls, ans_end_token_ls
initial_len = len(train)
train.dropna(inplace=True) # modify self DataFrame
log.info('drop {0}/{1} inconsistent samples.'.format(initial_len - len(train), initial_len))
log.info('answer span for training is generated.')

# features
trC_tags, trC_ents, trC_features = feature_gen(trC_docs, train.context_idx, trQ_docs, args.no_match)
log.info('features for training is generated: {}, {}, {}'.format(len(trC_tags), len(trC_ents), len(trC_features)))

def build_train_vocab(questions, contexts): # vocabulary will also be sorted accordingly
    if args.sort_all:
        counter = collections.Counter(w for doc in questions + contexts for w in doc)
        vocab = sorted([t for t in counter if t in glove_vocab], key=counter.get, reverse=True)
    else:
        counter_c = collections.Counter(w for doc in contexts for w in doc)
        counter_q = collections.Counter(w for doc in questions for w in doc)
        counter = counter_c + counter_q
        vocab = sorted([t for t in counter_q if t in glove_vocab], key=counter_q.get, reverse=True)
        vocab += sorted([t for t in counter_c.keys() - counter_q.keys() if t in glove_vocab],
                        key=counter.get, reverse=True)
    total = sum(counter.values())
    matched = sum(counter[t] for t in vocab)
    log.info('vocab {1}/{0} OOV {2}/{3} ({4:.4f}%)'.format(
        len(counter), len(vocab), (total - matched), total, (total - matched) / total * 100))
    vocab.insert(0, "<PAD>")
    vocab.insert(1, "<UNK>")
    vocab.insert(2, "<S>")
    vocab.insert(3, "</S>")
    return vocab

# vocab
tr_vocab = build_train_vocab(trQ_tokens, trC_tokens)
trC_ids = token2id(trC_tokens, tr_vocab, unk_id=1)
trQ_ids = token2id(trQ_tokens, tr_vocab, unk_id=1)
trQ_tokens = [["<S>"] + doc + ["</S>"] for doc in trQ_tokens]
trQ_ids = [[2] + qsent + [3] for qsent in trQ_ids]
print(trQ_ids[:10])


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

if args.use_bert:
    trC_bert_tokens = tokenize(trC_tokens)
    trC_bert_ids = [bert_tokens_to_ids(x) for x in trC_bert_tokens]
    trQ_bert_tokens = tokenize(trQ_tokens)
    trQ_bert_ids = [bert_tokens_to_ids(x) for x in trQ_bert_tokens]
    trC_bert_spans = [calc_bert_spans(b, t) for b, t in zip(trC_bert_tokens, trC_tokens)]
    trQ_bert_spans = [calc_bert_spans(b, t) for b, t in zip(trQ_bert_tokens, trQ_tokens)]

# tags
vocab_tag = [''] + list(nlp.tagger.labels)
trC_tag_ids = token2id(trC_tags, vocab_tag)
# entities
vocab_ent = list(set([ent for sent in trC_ents for ent in sent]))
trC_ent_ids = token2id(trC_ents, vocab_ent, unk_id=0)

log.info('Found {} POS tags.'.format(len(vocab_tag)))
log.info('Found {} entity tags: {}'.format(len(vocab_ent), vocab_ent))
log.info('vocabulary for training is built.')

tr_embedding = build_embedding(wv_file, tr_vocab, wv_dim)
log.info('got embedding matrix for training.')

# don't store row name in csv
#train.to_csv('QuAC_data/train.csv', index=False, encoding='utf8')

meta = {
    'vocab': tr_vocab,
    'embedding': tr_embedding.tolist()
}
with open('QuAC_data/train_meta.msgpack', 'wb') as f:
    msgpack.dump(meta, f)

prev_CID, first_question = -1, []
for i, CID in enumerate(train.context_idx):
    if not (CID == prev_CID):
        first_question.append(i)
    prev_CID = CID

if args.use_bert:
    result = {
        'qids': train.qid.tolist(),
        'question_ids': trQ_ids,
        'context_ids': trC_ids,
        'context_features': trC_features, # exact match, tf
        'context_tags': trC_tag_ids, # POS tagging
        'context_ents': trC_ent_ids, # Entity recognition
        'context': train_context,
        'context_span': train_context_span,
        '1st_question': first_question,
        'question_CID': train.context_idx.tolist(),
        'question': train.question.tolist(),
        'answer': train.answer.tolist(),
        'answer_start': train.answer_start_token.tolist(),
        'answer_end': train.answer_end_token.tolist(),
        'answer_choice': train.answer_choice.tolist(),
        'context_tokenized': trC_tokens,
        'question_tokenized': trQ_tokens,
        'context_bertidx': trC_bert_ids,
        'context_bert_spans': trC_bert_spans,
        'question_bertidx': trQ_bert_ids,
        'question_bert_spans': trQ_bert_spans
    }
else:
    result = {
        'qids': train.qid.tolist(),
        'question_ids': trQ_ids,
        'context_ids': trC_ids,
        'context_features': trC_features, # exact match, tf
        'context_tags': trC_tag_ids, # POS tagging
        'context_ents': trC_ent_ids, # Entity recognition
        'context': train_context,
        'context_span': train_context_span,
        '1st_question': first_question,
        'question_CID': train.context_idx.tolist(),
        'question': train.question.tolist(),
        'answer': train.answer.tolist(),
        'answer_start': train.answer_start_token.tolist(),
        'answer_end': train.answer_end_token.tolist(),
        'answer_choice': train.answer_choice.tolist(),
        'context_tokenized': trC_tokens,
        'question_tokenized': trQ_tokens
    }
with open('QuAC_data/train_data.msgpack', 'wb') as f:
    msgpack.dump(result, f)

del train, trQ_ids, trC_ids, trC_features, trC_tag_ids, trC_ent_ids, train_context, train_context_span
del first_question, trC_tokens, trQ_tokens, trC_bert_ids, trC_bert_spans, trQ_bert_ids, trQ_bert_spans
del trC_iter, trQ_iter, trC_docs, trQ_docs

log.info('saved training to disk.')

#==========================================================
#=================== Work on dev data =====================
#==========================================================

def proc_dev(ith, article):
    global args, bertTokenizer
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

dev, dev_context = flatten_json(dev_file, proc_dev)
dev = pd.DataFrame(dev, columns=['context_idx', 'question', 'answer',
                                     'answer_start', 'answer_end', 'answer_choice', 'all_answer', 'qid'])
log.info('dev json data flattened.')

print(dev)

devC_iter = (pre_proc(c) for c in dev_context)
devQ_iter = (pre_proc(q) for q in dev.question)
devC_docs = [doc for doc in nlp.pipe(
    devC_iter, batch_size=64, n_threads=args.threads)]
devQ_docs = [doc for doc in nlp.pipe(
    devQ_iter, batch_size=64, n_threads=args.threads)]

# tokens
devC_tokens = [[normalize_text(w.text) for w in doc] for doc in devC_docs]
devQ_tokens = [[normalize_text(w.text) for w in doc] for doc in devQ_docs]
devC_unnorm_tokens = [[w.text for w in doc] for doc in devC_docs]
log.info('All tokens for dev are obtained.')

dev_context_span = [get_context_span(a, b) for a, b in zip(dev_context, devC_unnorm_tokens)]
log.info('context span for dev is generated.')

ans_st_token_ls, ans_end_token_ls = [], []
for ans_st, ans_end, idx in zip(dev.answer_start, dev.answer_end, dev.context_idx):
    ans_st_token, ans_end_token = find_answer_span(dev_context_span[idx], ans_st, ans_end)
    ans_st_token_ls.append(ans_st_token)
    ans_end_token_ls.append(ans_end_token)

dev['answer_start_token'], dev['answer_end_token'] = ans_st_token_ls, ans_end_token_ls
initial_len = len(dev)
dev.dropna(inplace=True) # modify self DataFrame
log.info('drop {0}/{1} inconsistent samples.'.format(initial_len - len(dev), initial_len))
log.info('answer span for dev is generated.')

# features
devC_tags, devC_ents, devC_features = feature_gen(devC_docs, dev.context_idx, devQ_docs, args.no_match)
log.info('features for dev is generated: {}, {}, {}'.format(len(devC_tags), len(devC_ents), len(devC_features)))

def build_dev_vocab(questions, contexts): # most vocabulary comes from tr_vocab
    existing_vocab = set(tr_vocab)
    new_vocab = list(set([w for doc in questions + contexts for w in doc if w not in existing_vocab and w in glove_vocab]))
    vocab = tr_vocab + new_vocab
    log.info('train vocab {0}, total vocab {1}'.format(len(tr_vocab), len(vocab)))
    return vocab

# vocab
dev_vocab = build_dev_vocab(devQ_tokens, devC_tokens) # tr_vocab is a subset of dev_vocab
devC_ids = token2id(devC_tokens, dev_vocab, unk_id=1)
devQ_ids = token2id(devQ_tokens, dev_vocab, unk_id=1)
devQ_tokens = [["<S>"] + doc + ["</S>"] for doc in devQ_tokens]
devQ_ids = [[2] + qsent + [3] for qsent in devQ_ids]
print(devQ_ids[:10])

if args.use_bert:
    devC_bert_tokens = tokenize(devC_tokens)
    devC_bert_ids = [bert_tokens_to_ids(x) for x in devC_bert_tokens]
    devQ_bert_tokens = tokenize(devQ_tokens)
    devQ_bert_ids = [bert_tokens_to_ids(x) for x in devQ_bert_tokens]

    devC_bert_spans = [calc_bert_spans(b, t) for b, t in zip(devC_bert_tokens, devC_tokens)]
    devQ_bert_spans = [calc_bert_spans(b, t) for b, t in zip(devQ_bert_tokens, devQ_tokens)]


# tags
devC_tag_ids = token2id(devC_tags, vocab_tag) # vocab_tag same as training
# entities
devC_ent_ids = token2id(devC_ents, vocab_ent, unk_id=0) # vocab_ent same as training
log.info('vocabulary for dev is built.')

dev_embedding = build_embedding(wv_file, dev_vocab, wv_dim)
# tr_embedding is a submatrix of dev_embedding
log.info('got embedding matrix for dev.')

# don't store row name in csv
#dev.to_csv('QuAC_data/dev.csv', index=False, encoding='utf8')

meta = {
    'vocab': dev_vocab,
    'embedding': dev_embedding.tolist()
}
with open('QuAC_data/dev_meta.msgpack', 'wb') as f:
    msgpack.dump(meta, f)

prev_CID, first_question = -1, []
for i, CID in enumerate(dev.context_idx):
    if not (CID == prev_CID):
        first_question.append(i)
    prev_CID = CID

if args.use_bert:
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
else:
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
        'question_tokenized': devQ_tokens
    }
with open('QuAC_data/dev_data.msgpack', 'wb') as f:
    msgpack.dump(result, f)

log.info('saved dev to disk.')
