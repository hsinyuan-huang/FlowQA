import re
import os
import sys
import random
import string
import logging
import argparse
import unicodedata
from shutil import copyfile
from datetime import datetime
from collections import Counter
import torch
import msgpack
import json
import numpy as np
import pandas as pd
from allennlp.modules.elmo import batch_to_ids

#===========================================================================
#================= All for preprocessing SQuAD data set ====================
#===========================================================================

def len_preserved_normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def len_preserved_space(matchobj):
        return ' ' * len(matchobj.group(0))

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', len_preserved_space, text)

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    return remove_articles(remove_punc(lower(s)))

def split_with_span(s):
    if s.split() == []:
        return [], []
    else:
        return zip(*[(m.group(0), (m.start(), m.end()-1)) for m in re.finditer(r'\S+', s)])

def free_text_to_span(free_text, full_text):
    if free_text == "unknown":
        return "__NA__", -1, -1
    if normalize_answer(free_text) == "yes":
        return "__YES__", -1, -1
    if normalize_answer(free_text) == "no":
        return "__NO__", -1, -1

    free_ls = len_preserved_normalize_answer(free_text).split()
    full_ls, full_span = split_with_span(len_preserved_normalize_answer(full_text))
    if full_ls == []:
        return full_text, 0, len(full_text)

    max_f1, best_index = 0.0, (0, len(full_ls)-1)
    free_cnt = Counter(free_ls)
    for i in range(len(full_ls)):
        full_cnt = Counter()
        for j in range(len(full_ls)):
            if i+j >= len(full_ls): break
            full_cnt[full_ls[i+j]] += 1

            common = free_cnt & full_cnt
            num_same = sum(common.values())
            if num_same == 0: continue

            precision = 1.0 * num_same / (j + 1)
            recall = 1.0 * num_same / len(free_ls)
            f1 = (2 * precision * recall) / (precision + recall)

            if max_f1 < f1:
                max_f1 = f1
                best_index = (i, j)

    assert(best_index is not None)
    (best_i, best_j) = best_index
    char_i, char_j = full_span[best_i][0], full_span[best_i+best_j][1]+1

    return full_text[char_i:char_j], char_i, char_j

def flatten_json(file, proc_func):
    with open(file, encoding="utf8") as f:
        data = json.load(f)['data']
    rows, contexts = [], []
    for i in range(len(data)):
        partial_rows, context = proc_func(i, data[i])
        rows.extend(partial_rows)
        contexts.append(context)
    return rows, contexts

def normalize_text(text):
    return unicodedata.normalize('NFD', text)

def load_glove_vocab(file, wv_dim):
    vocab = set()
    with open(file, encoding="utf8") as f:
        for line in f:
            elems = line.split()
            token = normalize_text(''.join(elems[0:-wv_dim]))
            vocab.add(token)
    return vocab

def space_extend(matchobj):
    return ' ' + matchobj.group(0) + ' '

def pre_proc(text):
    # make hyphens, spaces clean
    text = re.sub(u'-|\u2010|\u2011|\u2012|\u2013|\u2014|\u2015|%|\[|\]|:|\(|\)|/', space_extend, text)
    text = text.strip(' \n')
    text = re.sub('\s+', ' ', text)
    return text

def feature_gen(C_docs, Q_CID, Q_docs, no_match):
    C_tags = [[w.tag_ for w in doc] for doc in C_docs]
    C_ents = [[w.ent_type_ for w in doc] for doc in C_docs]
    C_features = []

    for question, context_id in zip(Q_docs, Q_CID):
        context = C_docs[context_id]

        counter_ = Counter(w.text.lower() for w in context)
        total = sum(counter_.values())
        term_freq = [counter_[w.text.lower()] / total for w in context]

        if no_match:
            C_features.append(list(zip(term_freq)))
        else:
            question_word = {w.text for w in question}
            question_lower = {w.text.lower() for w in question}
            question_lemma = {w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower() for w in question}
            match_origin = [w.text in question_word for w in context]
            match_lower = [w.text.lower() in question_lower for w in context]
            match_lemma = [(w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower()) in question_lemma for w in context]
            C_features.append(list(zip(match_origin, match_lower, match_lemma, term_freq)))

    return C_tags, C_ents, C_features

def get_context_span(context, context_token):
    p_str = 0
    p_token = 0
    t_span = []
    while p_str < len(context):
        if re.match('\s', context[p_str]):
            p_str += 1
            continue

        token = context_token[p_token]
        token_len = len(token)
        if context[p_str:p_str + token_len] != token:
            log.info("Something wrong with get_context_span()")
            return []
        t_span.append((p_str, p_str + token_len))

        p_str += token_len
        p_token += 1
    return t_span

def find_answer_span(context_span, answer_start, answer_end):
    if answer_start == -1 and answer_end == -1:
        return (-1, -1)

    t_start, t_end = 0, 0
    for token_id, (s, t) in enumerate(context_span):
        if s <= answer_start:
            t_start = token_id
        if t <= answer_end:
            t_end = token_id

    if t_start == -1 or t_end == -1:
        print(context_span, answer_start, answer_end)
        return (None, None)
    else:
        return (t_start, t_end)

def build_embedding(embed_file, targ_vocab, wv_dim):
    vocab_size = len(targ_vocab)
    emb = np.random.uniform(-1, 1, (vocab_size, wv_dim))
    emb[0] = 0 # <PAD> should be all 0 (using broadcast)

    w2id = {w: i for i, w in enumerate(targ_vocab)}
    with open(embed_file, encoding="utf8") as f:
        for line in f:
            elems = line.split()
            token = normalize_text(''.join(elems[0:-wv_dim]))
            if token in w2id:
                emb[w2id[token]] = [float(v) for v in elems[-wv_dim:]]
    return emb

def token2id(docs, vocab, unk_id=None):
    w2id = {w: i for i, w in enumerate(vocab)}
    ids = [[w2id[w] if w in w2id else unk_id for w in doc] for doc in docs]
    return ids

#===========================================================================
#================ For batch generation (train & predict) ===================
#===========================================================================

class BatchGen_CoQA:
    def __init__(self, data, batch_size, gpu, dialog_ctx=0, evaluation=False, context_maxlen=100000, precompute_elmo=0):
        '''
        input:
            data - see train.py
            batch_size - int
        '''
        self.dialog_ctx = dialog_ctx
        self.batch_size = batch_size
        self.context_maxlen = context_maxlen
        self.precompute_elmo = precompute_elmo

        self.eval = evaluation
        self.gpu = gpu

        self.context_num = len(data['context'])
        self.question_num = len(data['qa'])
        self.data = data

    def __len__(self):
        return (self.context_num + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        # Random permutation for the context
        idx_perm = range(0, self.context_num)
        if not self.eval:
            idx_perm = np.random.permutation(idx_perm)

        batch_size = self.batch_size
        for batch_i in range((self.context_num + self.batch_size - 1) // self.batch_size):

            batch_idx = idx_perm[self.batch_size * batch_i: self.batch_size * (batch_i+1)]

            context_batch = [self.data['context'][i] for i in batch_idx]
            batch_size = len(context_batch)

            context_batch = list(zip(*context_batch))

            # Process Context Tokens
            context_len = max(len(x) for x in context_batch[0])
            if not self.eval:
                context_len = min(context_len, self.context_maxlen)
            context_id = torch.LongTensor(batch_size, context_len).fill_(0)
            for i, doc in enumerate(context_batch[0]):
                select_len = min(len(doc), context_len)
                context_id[i, :select_len] = torch.LongTensor(doc[:select_len])

            # Process Context POS Tags
            context_tag = torch.LongTensor(batch_size, context_len).fill_(0)
            for i, doc in enumerate(context_batch[1]):
                select_len = min(len(doc), context_len)
                context_tag[i, :select_len] = torch.LongTensor(doc[:select_len])

            # Process Context Named Entity
            context_ent = torch.LongTensor(batch_size, context_len).fill_(0)
            for i, doc in enumerate(context_batch[2]):
                select_len = min(len(doc), context_len)
                context_ent[i, :select_len] = torch.LongTensor(doc[:select_len])

            if self.precompute_elmo > 0:
                if batch_i % self.precompute_elmo == 0:
                    precompute_idx = idx_perm[self.batch_size * batch_i: self.batch_size * (batch_i+self.precompute_elmo)]
                    elmo_tokens = [self.data['context'][i][6] for i in precompute_idx]
                    context_cid = batch_to_ids(elmo_tokens)
                else:
                    context_cid = torch.LongTensor(1).fill_(0)
            else:
                context_cid = batch_to_ids(context_batch[6])

            # Process Questions (number = batch * Qseq)
            qa_data = self.data['qa']

            question_num, question_len = 0, 0
            question_batch = []
            for first_QID in context_batch[5]:
                i, question_seq = 0, []
                while True:
                    if first_QID + i >= len(qa_data) or qa_data[first_QID + i][0] != qa_data[first_QID][0]: # their corresponding context ID is different
                        break
                    question_seq.append(first_QID + i)
                    question_len = max(question_len, len(qa_data[first_QID + i][1]))
                    i += 1
                question_batch.append(question_seq)
                question_num = max(question_num, i)

            question_id = torch.LongTensor(batch_size, question_num, question_len).fill_(0)
            question_tokens = []
            for i, q_seq in enumerate(question_batch):
                for j, id in enumerate(q_seq):
                    doc = qa_data[id][1]
                    question_id[i, j, :len(doc)] = torch.LongTensor(doc)
                    question_tokens.append(qa_data[id][10])

                for j in range(len(q_seq), question_num):
                    question_id[i, j, :2] = torch.LongTensor([2, 3])
                    question_tokens.append(["<S>", "</S>"])

            question_cid = batch_to_ids(question_tokens)

            # Process Context-Question Features
            feature_len = len(qa_data[0][2][0])
            context_feature = torch.Tensor(batch_size, question_num, context_len, feature_len + (self.dialog_ctx * 3)).fill_(0)
            for i, q_seq in enumerate(question_batch):
                for j, id in enumerate(q_seq):
                    doc = qa_data[id][2]
                    select_len = min(len(doc), context_len)
                    context_feature[i, j, :select_len, :feature_len] = torch.Tensor(doc[:select_len])

                    for prv_ctx in range(0, self.dialog_ctx):
                        if j > prv_ctx:
                            prv_id = id - prv_ctx - 1
                            prv_ans_st, prv_ans_end, prv_rat_st, prv_rat_end, prv_ans_choice = qa_data[prv_id][3], qa_data[prv_id][4], qa_data[prv_id][5], qa_data[prv_id][6], qa_data[prv_id][7]

                            if prv_ans_choice == 3:
                                # There is an answer
                                for k in range(prv_ans_st, prv_ans_end + 1):
                                    if k >= context_len:
                                        break
                                    context_feature[i, j, k, feature_len + prv_ctx * 3 + 1] = 1
                            else:
                                context_feature[i, j, :select_len, feature_len + prv_ctx * 3 + 2] = 1

            # Process Answer (w/ raw question, answer text)
            answer_s = torch.LongTensor(batch_size, question_num).fill_(0)
            answer_e = torch.LongTensor(batch_size, question_num).fill_(0)
            rationale_s = torch.LongTensor(batch_size, question_num).fill_(0)
            rationale_e = torch.LongTensor(batch_size, question_num).fill_(0)
            answer_c = torch.LongTensor(batch_size, question_num).fill_(0)
            overall_mask = torch.ByteTensor(batch_size, question_num).fill_(0)
            question, answer = [], []
            for i, q_seq in enumerate(question_batch):
                question_pack, answer_pack = [], []
                for j, id in enumerate(q_seq):
                    answer_s[i, j], answer_e[i, j], rationale_s[i, j], rationale_e[i, j], answer_c[i, j] = qa_data[id][3], qa_data[id][4], qa_data[id][5], qa_data[id][6], qa_data[id][7]
                    overall_mask[i, j] = 1
                    question_pack.append(qa_data[id][8])
                    answer_pack.append(qa_data[id][9])
                question.append(question_pack)
                answer.append(answer_pack)

            # Process Masks
            context_mask = torch.eq(context_id, 0)
            question_mask = torch.eq(question_id, 0)

            text = list(context_batch[3]) # raw text
            span = list(context_batch[4]) # character span for each words

            if self.gpu: # page locked memory for async data transfer
                context_id = context_id.pin_memory()
                context_feature = context_feature.pin_memory()
                context_tag = context_tag.pin_memory()
                context_ent = context_ent.pin_memory()
                context_mask = context_mask.pin_memory()
                question_id = question_id.pin_memory()
                question_mask = question_mask.pin_memory()
                answer_s = answer_s.pin_memory()
                answer_e = answer_e.pin_memory()
                rationale_s = rationale_s.pin_memory()
                rationale_e = rationale_e.pin_memory()
                answer_c = answer_c.pin_memory()
                overall_mask = overall_mask.pin_memory()
                context_cid = context_cid.pin_memory()
                question_cid = question_cid.pin_memory()

            yield (context_id, context_cid, context_feature, context_tag, context_ent, context_mask,
                   question_id, question_cid, question_mask, overall_mask,
                   answer_s, answer_e, answer_c, rationale_s, rationale_e,
                   text, span, question, answer)

class BatchGen_QuAC:
    def __init__(self, data, batch_size, gpu, dialog_ctx=0, use_dialog_act=False, evaluation=False, context_maxlen=100000, precompute_elmo=0):
        '''
        input:
            data - see train.py
            batch_size - int
        '''
        self.dialog_ctx = dialog_ctx
        self.use_dialog_act = use_dialog_act
        self.batch_size = batch_size
        self.context_maxlen = context_maxlen
        self.precompute_elmo = precompute_elmo

        self.eval = evaluation
        self.gpu = gpu

        self.context_num = len(data['context'])
        self.question_num = len(data['qa'])
        self.data = data

    def __len__(self):
        return (self.context_num + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        # Random permutation for the context
        idx_perm = range(0, self.context_num)
        if not self.eval:
            idx_perm = np.random.permutation(idx_perm)

        batch_size = self.batch_size
        for batch_i in range((self.context_num + self.batch_size - 1) // self.batch_size):

            batch_idx = idx_perm[self.batch_size * batch_i: self.batch_size * (batch_i+1)]

            context_batch = [self.data['context'][i] for i in batch_idx]
            batch_size = len(context_batch)

            context_batch = list(zip(*context_batch))

            # Process Context Tokens
            context_len = max(len(x) for x in context_batch[0])
            if not self.eval:
                context_len = min(context_len, self.context_maxlen)
            context_id = torch.LongTensor(batch_size, context_len).fill_(0)
            for i, doc in enumerate(context_batch[0]):
                select_len = min(len(doc), context_len)
                context_id[i, :select_len] = torch.LongTensor(doc[:select_len])

            # Process Context POS Tags
            context_tag = torch.LongTensor(batch_size, context_len).fill_(0)
            for i, doc in enumerate(context_batch[1]):
                select_len = min(len(doc), context_len)
                context_tag[i, :select_len] = torch.LongTensor(doc[:select_len])

            # Process Context Named Entity
            context_ent = torch.LongTensor(batch_size, context_len).fill_(0)
            for i, doc in enumerate(context_batch[2]):
                select_len = min(len(doc), context_len)
                context_ent[i, :select_len] = torch.LongTensor(doc[:select_len])

            if self.precompute_elmo > 0:
                if batch_i % self.precompute_elmo == 0:
                    precompute_idx = idx_perm[self.batch_size * batch_i: self.batch_size * (batch_i+self.precompute_elmo)]
                    elmo_tokens = [self.data['context'][i][6] for i in precompute_idx]
                    context_cid = batch_to_ids(elmo_tokens)
                else:
                    context_cid = torch.LongTensor(1).fill_(0)
            else:
                context_cid = batch_to_ids(context_batch[6])

            # Process Questions (number = batch * Qseq)
            qa_data = self.data['qa']

            question_num, question_len = 0, 0
            question_batch = []
            for first_QID in context_batch[5]:
                i, question_seq = 0, []
                while True:
                    if first_QID + i >= len(qa_data) or qa_data[first_QID + i][0] != qa_data[first_QID][0]: # their corresponding context ID is different
                        break
                    question_seq.append(first_QID + i)
                    question_len = max(question_len, len(qa_data[first_QID + i][1]))
                    i += 1
                question_batch.append(question_seq)
                question_num = max(question_num, i)

            question_id = torch.LongTensor(batch_size, question_num, question_len).fill_(0)
            question_tokens = []
            for i, q_seq in enumerate(question_batch):
                for j, id in enumerate(q_seq):
                    doc = qa_data[id][1]
                    question_id[i, j, :len(doc)] = torch.LongTensor(doc)
                    question_tokens.append(qa_data[id][8])

                for j in range(len(q_seq), question_num):
                    question_id[i, j, :2] = torch.LongTensor([2, 3])
                    question_tokens.append(["<S>", "</S>"])

            question_cid = batch_to_ids(question_tokens)

            # Process Context-Question Features
            feature_len = len(qa_data[0][2][0])
            context_feature = torch.Tensor(batch_size, question_num, context_len, feature_len + (self.dialog_ctx * (self.use_dialog_act*3+2))).fill_(0)
            for i, q_seq in enumerate(question_batch):
                for j, id in enumerate(q_seq):
                    doc = qa_data[id][2]
                    select_len = min(len(doc), context_len)
                    context_feature[i, j, :select_len, :feature_len] = torch.Tensor(doc[:select_len])

                    for prv_ctx in range(0, self.dialog_ctx):
                        if j > prv_ctx:
                            prv_id = id - prv_ctx - 1
                            prv_ans_st, prv_ans_end, prv_ans_choice = qa_data[prv_id][3], qa_data[prv_id][4], qa_data[prv_id][5]

                            # dialog act: don't follow-up, follow-up, maybe follow-up (prv_ans_choice // 10)
                            if self.use_dialog_act:
                                context_feature[i, j, :select_len, feature_len + prv_ctx * (self.use_dialog_act*3+2) + 2 + (prv_ans_choice // 10)] = 1

                            if prv_ans_choice == 0: # indicating that the previous reply is NO ANSWER
                                context_feature[i, j, :select_len, feature_len + prv_ctx * (self.use_dialog_act*3+2) + 1] = 1
                                continue

                            # There is an answer
                            for k in range(prv_ans_st, prv_ans_end + 1):
                                if k >= context_len:
                                    break
                                context_feature[i, j, k, feature_len + prv_ctx * (self.use_dialog_act*3+2)] = 1

            # Process Answer (w/ raw question, answer text)
            answer_s = torch.LongTensor(batch_size, question_num).fill_(0)
            answer_e = torch.LongTensor(batch_size, question_num).fill_(0)
            answer_c = torch.LongTensor(batch_size, question_num).fill_(0)
            overall_mask = torch.ByteTensor(batch_size, question_num).fill_(0)
            question, answer = [], []
            for i, q_seq in enumerate(question_batch):
                question_pack, answer_pack = [], []
                for j, id in enumerate(q_seq):
                    answer_s[i, j], answer_e[i, j], answer_c[i, j] = qa_data[id][3], qa_data[id][4], qa_data[id][5]
                    overall_mask[i, j] = 1
                    question_pack.append(qa_data[id][6])
                    answer_pack.append(qa_data[id][7])
                question.append(question_pack)
                answer.append(answer_pack)

            # Process Masks
            context_mask = torch.eq(context_id, 0)
            question_mask = torch.eq(question_id, 0)

            text = list(context_batch[3]) # raw text
            span = list(context_batch[4]) # character span for each words

            if self.gpu: # page locked memory for async data transfer
                context_id = context_id.pin_memory()
                context_feature = context_feature.pin_memory()
                context_tag = context_tag.pin_memory()
                context_ent = context_ent.pin_memory()
                context_mask = context_mask.pin_memory()
                question_id = question_id.pin_memory()
                question_mask = question_mask.pin_memory()
                answer_s = answer_s.pin_memory()
                answer_e = answer_e.pin_memory()
                answer_c = answer_c.pin_memory()
                overall_mask = overall_mask.pin_memory()
                context_cid = context_cid.pin_memory()
                question_cid = question_cid.pin_memory()

            yield (context_id, context_cid, context_feature, context_tag, context_ent, context_mask,
                   question_id, question_cid, question_mask, overall_mask,
                   answer_s, answer_e, answer_c,
                   text, span, question, answer)

#===========================================================================
#========================== For QuAC evaluation ============================
#===========================================================================

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def single_score(prediction, ground_truth):
    if prediction == "CANNOTANSWER" and ground_truth == "CANNOTANSWER":
        return 1.0
    elif prediction == "CANNOTANSWER" or ground_truth == "CANNOTANSWER":
        return 0.0
    else:
        return f1_score(prediction, ground_truth)

def handle_cannot(refs):
    num_cannot = 0
    num_spans = 0
    for ref in refs:
        if ref == 'CANNOTANSWER': num_cannot += 1
        else: num_spans += 1

    if num_cannot >= num_spans:
        refs = ['CANNOTANSWER']
    else:
        refs = [x for x in refs if x != 'CANNOTANSWER']
    return refs

def leave_one_out(refs):
    if len(refs) == 1:
        return 1.0

    t_f1 = 0.0
    for i in range(len(refs)):
        m_f1 = 0
        new_refs = refs[:i] + refs[i+1:]

        for j in range(len(new_refs)):
            f1_ij = single_score(refs[i], new_refs[j])

            if f1_ij > m_f1:
                m_f1 = f1_ij
        t_f1 += m_f1

    return t_f1 / len(refs)

def leave_one_out_max(prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        scores_for_ground_truths.append(single_score(prediction, ground_truth))

    if len(scores_for_ground_truths) == 1:
        return scores_for_ground_truths[0]
    else:
        # leave out one ref every time
        t_f1 = []
        for i in range(len(scores_for_ground_truths)):
            t_f1.append(max(scores_for_ground_truths[:i] + scores_for_ground_truths[i+1:]))
        return 1.0 * sum(t_f1) / len(t_f1)

def find_best_score_and_thresh(pred, truth, no_ans_score, min_F1=0.4):
    pred = [p for dialog_p in pred for p in dialog_p]
    truth = [t for dialog_t in truth for t in dialog_t]
    no_ans_score = [n for dialog_n in no_ans_score for n in dialog_n]

    clean_pred, clean_truth, clean_noans = [], [], []

    all_f1 = []
    for p, t, n in zip(pred, truth, no_ans_score):
        clean_t = handle_cannot(t)
        human_F1 = leave_one_out(clean_t)
        if human_F1 < min_F1: continue

        clean_pred.append(p)
        clean_truth.append(clean_t)
        clean_noans.append(n)
        all_f1.append(leave_one_out_max(p, clean_t))

    cur_f1, best_f1 = sum(all_f1), sum(all_f1)
    best_thresh = max(clean_noans) + 1

    cur_noans, best_noans, noans_cnt = 0, 0, 0
    sort_idx = sorted(range(len(clean_noans)), key=lambda k: clean_noans[k], reverse=True)
    for i in sort_idx:
        if clean_truth[i] == ['CANNOTANSWER']:
            cur_f1 += 1
            cur_noans += 1
            noans_cnt += 1
        else:
            cur_f1 -= all_f1[i]
            cur_noans -= 1

        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_noans = cur_noans
            best_thresh = clean_noans[i] - 1e-7

    return 100.0 * best_f1 / len(clean_pred), 100.0 * (len(clean_pred) - noans_cnt + best_noans) / len(clean_pred), best_thresh

def score(model_results, human_results, min_F1=0.4):
    Q_at_least_human, total_Qs = 0.0, 0.0
    D_at_least_human, total_Ds = 0.0, 0.0
    total_machine_f1, total_human_f1 = 0.0, 0.0

    assert len(human_results) == len(model_results)
    for human_diag_ans, model_diag_ans in zip(human_results, model_results):
        good_dialog = 1.0

        assert len(human_diag_ans) == len(model_diag_ans)
        for human_ans, model_ans in zip(human_diag_ans, model_diag_ans):
            # model_ans is (text, choice)
            # human_ans is a list of (text, choice)

            # human_ans[0] is the original dialog answer
            clean_human_ans = handle_cannot(human_ans)
            human_F1 = leave_one_out(clean_human_ans)

            if human_F1 < min_F1: continue

            machine_f1 = leave_one_out_max(model_ans, clean_human_ans)
            total_machine_f1 += machine_f1
            total_human_f1 += human_F1

            if machine_f1 >= human_F1:
                Q_at_least_human += 1.0
            else:
                good_dialog = 0.0
            total_Qs += 1.0

        D_at_least_human += good_dialog
        total_Ds += 1.0

    return 100.0 * total_machine_f1 / total_Qs, 100.0 * total_human_f1 / total_Qs, 100.0 * Q_at_least_human / total_Qs, 100.0 * D_at_least_human / total_Ds
