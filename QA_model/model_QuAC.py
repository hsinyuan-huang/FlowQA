import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging

from torch.nn import Parameter
from torch.autograd import Variable
from .utils import AverageMeter
from .detail_model import FlowQA

logger = logging.getLogger(__name__)


class QAModel(object):
    """
    High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    def __init__(self, opt, embedding=None, state_dict=None):
        # Book-keeping.
        self.opt = opt
        self.updates = state_dict['updates'] if state_dict else 0
        self.eval_embed_transfer = True
        self.train_loss = AverageMeter()

        # Building network.
        self.network = FlowQA(opt, embedding)
        if state_dict:
            new_state = set(self.network.state_dict().keys())
            for k in list(state_dict['network'].keys()):
                if k not in new_state:
                    del state_dict['network'][k]
            self.network.load_state_dict(state_dict['network'])

        # Building optimizer.
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if opt['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(parameters, opt['learning_rate'],
                                       momentum=opt['momentum'],
                                       weight_decay=opt['weight_decay'])
        elif opt['optimizer'] == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay=opt['weight_decay'])
        elif opt['optimizer'] == 'adadelta':
            self.optimizer = optim.Adadelta(parameters, rho=0.95, weight_decay=opt['weight_decay'])
        else:
            raise RuntimeError('Unsupported optimizer: %s' % opt['optimizer'])
        if state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])

        if opt['fix_embeddings']:
            wvec_size = 0
        else:
            wvec_size = (opt['vocab_size'] - opt['tune_partial']) * opt['embedding_dim']
        self.total_param = sum([p.nelement() for p in parameters]) - wvec_size

    def update(self, batch):
        # Train mode
        self.network.train()
        torch.set_grad_enabled(True)

        # Transfer to GPU
        if self.opt['cuda']:
            inputs = [e.cuda(non_blocking=True) for e in batch[:9]]
            overall_mask = batch[9].cuda(non_blocking=True)

            answer_s = batch[10].cuda(non_blocking=True)
            answer_e = batch[11].cuda(non_blocking=True)
            answer_c = batch[12].cuda(non_blocking=True)
        else:
            inputs = [e for e in batch[:9]]
            overall_mask = batch[9]

            answer_s = batch[10]
            answer_e = batch[11]
            answer_c = batch[12]

        # Run forward
        # output: [batch_size, question_num, context_len], [batch_size, question_num]
        score_s, score_e, score_no_answ = self.network(*inputs)

        # Compute loss and accuracies
        loss = self.opt['elmo_lambda'] * (self.network.elmo.scalar_mix_0.scalar_parameters[0] ** 2
                                        + self.network.elmo.scalar_mix_0.scalar_parameters[1] ** 2
                                        + self.network.elmo.scalar_mix_0.scalar_parameters[2] ** 2) # ELMo L2 regularization
        all_no_answ = (answer_c == 0)
        answer_s.masked_fill_(all_no_answ, -100) # ignore_index is -100 in F.cross_entropy
        answer_e.masked_fill_(all_no_answ, -100)

        for i in range(overall_mask.size(0)):
            q_num = sum(overall_mask[i]) # the true question number for this sampled context

            target_s = answer_s[i, :q_num] # Size: q_num
            target_e = answer_e[i, :q_num]
            target_c = answer_c[i, :q_num]
            target_no_answ = all_no_answ[i, :q_num]

            # single_loss is averaged across q_num
            if self.opt['question_normalize']:
                single_loss = F.binary_cross_entropy_with_logits(score_no_answ[i, :q_num], target_no_answ.float()) * q_num.item() / 8.0
                single_loss = single_loss + F.cross_entropy(score_s[i, :q_num], target_s) * (q_num - sum(target_no_answ)).item() / 7.0
                single_loss = single_loss + F.cross_entropy(score_e[i, :q_num], target_e) * (q_num - sum(target_no_answ)).item() / 7.0
            else:
                single_loss = F.binary_cross_entropy_with_logits(score_no_answ[i, :q_num], target_no_answ.float()) \
                            + F.cross_entropy(score_s[i, :q_num], target_s) + F.cross_entropy(score_e[i, :q_num], target_e)

            loss = loss + (single_loss / overall_mask.size(0))
        self.train_loss.update(loss.item(), overall_mask.size(0))

        # Clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.network.parameters(),
                                       self.opt['grad_clipping'])

        # Update parameters
        self.optimizer.step()
        self.updates += 1

        # Reset any partially fixed parameters (e.g. rare words)
        self.reset_embeddings()
        self.eval_embed_transfer = True

    def predict(self, batch, No_Ans_Threshold=None):
        # Eval mode
        self.network.eval()
        torch.set_grad_enabled(False)

        # Transfer trained embedding to evaluation embedding
        if self.eval_embed_transfer:
            self.update_eval_embed()
            self.eval_embed_transfer = False

        # Transfer to GPU
        if self.opt['cuda']:
            inputs = [e.cuda(non_blocking=True) for e in batch[:9]]
        else:
            inputs = [e for e in batch[:9]]

        # Run forward
        # output: [batch_size, question_num, context_len], [batch_size, question_num]
        score_s, score_e, score_no_answ = self.network(*inputs)
        score_s = F.softmax(score_s, dim=2)
        score_e = F.softmax(score_e, dim=2)

        # Transfer to CPU/normal tensors for numpy ops
        score_s = score_s.data.cpu()
        score_e = score_e.data.cpu()
        score_no_answ = score_no_answ.data.cpu()

        # Get argmax text spans
        text = batch[13]
        spans = batch[14]
        overall_mask = batch[9]

        predictions, no_ans_scores = [], []
        max_len = self.opt['max_len'] or score_s.size(2)

        for i in range(overall_mask.size(0)):
            dialog_pred, dialog_noans = [], []

            for j in range(overall_mask.size(1)):
                if overall_mask[i, j] == 0: # this dialog has ended
                    break

                dialog_noans.append(score_no_answ[i, j].item())
                if No_Ans_Threshold is not None and score_no_answ[i, j] > No_Ans_Threshold:
                    dialog_pred.append("CANNOTANSWER")
                else:
                    scores = torch.ger(score_s[i, j], score_e[i, j])
                    scores.triu_().tril_(max_len - 1)
                    scores = scores.numpy()
                    s_idx, e_idx = np.unravel_index(np.argmax(scores), scores.shape)

                    s_offset, e_offset = spans[i][s_idx][0], spans[i][e_idx][1]
                    dialog_pred.append(text[i][s_offset:e_offset])

            predictions.append(dialog_pred)
            no_ans_scores.append(dialog_noans)

        return predictions, no_ans_scores # list of (list of strings), list of (list of floats)

    # allow the evaluation embedding be larger than training embedding
    # this is helpful if we have pretrained word embeddings
    def setup_eval_embed(self, eval_embed, padding_idx = 0):
        # eval_embed should be a supermatrix of training embedding
        self.network.eval_embed = nn.Embedding(eval_embed.size(0),
                                               eval_embed.size(1),
                                               padding_idx = padding_idx)
        self.network.eval_embed.weight.data = eval_embed
        for p in self.network.eval_embed.parameters():
            p.requires_grad = False
        self.eval_embed_transfer = True

        if hasattr(self.network, 'CoVe'):
            self.network.CoVe.setup_eval_embed(eval_embed)

    def update_eval_embed(self):
        # update evaluation embedding to trained embedding
        if self.opt['tune_partial'] > 0:
            offset = self.opt['tune_partial']
            self.network.eval_embed.weight.data[0:offset] \
                = self.network.embedding.weight.data[0:offset]
        else:
            offset = 10
            self.network.eval_embed.weight.data[0:offset] \
                = self.network.embedding.weight.data[0:offset]

    def reset_embeddings(self):
        # Reset fixed embeddings to original value
        if self.opt['tune_partial'] > 0:
            offset = self.opt['tune_partial']
            if offset < self.network.embedding.weight.data.size(0):
                self.network.embedding.weight.data[offset:] \
                    = self.network.fixed_embedding

    def get_pretrain(self, state_dict):
        own_state = self.network.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, Parameter):
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print("Skip", name)
                continue

    def save(self, filename, epoch):
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'updates': self.updates # how many updates
            },
            'config': self.opt,
            'epoch': epoch
        }
        try:
            torch.save(params, filename)
            logger.info('model saved to {}'.format(filename))
        except BaseException:
            logger.warn('[ WARN: Saving failed... continuing anyway. ]')

    def save_for_predict(self, filename, epoch):
        network_state = dict([(k, v) for k, v in self.network.state_dict().items() if k[0:4] != 'CoVe'])
        if 'eval_embed.weight' in network_state:
            del network_state['eval_embed.weight']
        if 'fixed_embedding' in network_state:
            del network_state['fixed_embedding']
        params = {
            'state_dict': {'network': network_state},
            'config': self.opt,
        }
        try:
            torch.save(params, filename)
            logger.info('model saved to {}'.format(filename))
        except BaseException:
            logger.warn('[ WARN: Saving failed... continuing anyway. ]')

    def cuda(self):
        self.network.cuda()
