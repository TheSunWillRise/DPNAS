import torch
import torch.nn as nn
import torch.nn.functional as F
from genotypes import PRIMITIVES, LSTM_PRIMITIVES, PRIV_PRIMITIVES
from operations import *
import utils
import numpy as np
from utils import arch_to_genotype, draw_genotype, infinite_get, arch_to_string
import os

from backpack import backpack, extend, context
from backpack.extensions import BatchGrad




class MeWOp(nn.Module):     # this is nolonger a mixed op as in darts
    def __init__(self, C, stride):
        super(MeWOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, index):
        return self._ops[index](x)

class LSTMMeWOp(nn.Module):
    def __init__(self, C, stride):
        super(LSTMMeWOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in LSTM_PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, index):
        return self._ops[index](x)

class PrivOp(nn.Module):
    def __init__(self, C, stride):
        super(PrivOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIV_PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            # if 'pool' in primitive:
            #     op = nn.Sequential(op, GN(C))
            self._ops.append(op)

    def forward(self, x, index):
        return self._ops[index](x)

class MeWCell(nn.Module):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, cell_type):
        super(MeWCell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)

        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        for i in range(self._steps):     # all possible ops
            for j in range(i + 2):
                stride = 2 if reduction and j < 2 else 1
                if cell_type == "SAMPLE":
                    op = MeWOp(C, stride)
                elif cell_type == "LSTM" or cell_type=='ENAS' or cell_type=='LSTM2':
                    op = LSTMMeWOp(C, stride)
                else:
                    assert False, "unsupported controller_type: %s" % cell_type
                self._ops.append(op)

    def forward(self, s0, s1, arch):
        """

        :param s0:
        :param s1:
        :param arch: a list, the element is (op_id, from_node, to_node), sorted by to_node (!!not check
                     the ordering for efficiency, but must be assured when generating!!)
                     from_node/to_node starts from 0, 0 is the prev_prev_node, 1 is prev_node
                     The mapping from (F, T) pair to edge_ID is (T-2)(T+1)/2+S,

        :return:
        """
        s0 = self.preprocess0.forward(s0)
        s1 = self.preprocess1.forward(s1)
        # states = [s0, s1]
        states = {0: s0, 1: s1}

        for op, f, t in arch:   #                      # The LSTM way, directly generate the arch.
            edge_id = int((t - 2) * (t + 1) / 2 + f)   # TODO: this can give the edge_id. which op is used in the ops_list. self._ops is no longer a sequence list, but a list here to take different orders. the order is defined here
            if t in states:
                states[t] = states[t] + self._ops[edge_id](states[f], op)   # op forward: self._ops[op](x)  self._ops is a list of ops
            else:
                states[t] = self._ops[edge_id](states[f], op)
        return torch.cat([states[i] for i in range(2, self._steps + 2)], dim=1)

class PrivCell(nn.Module):
    def __init__(self, steps, C_prev, C, reduction, cell_type):
        super(PrivCell, self).__init__()
        self.reduction = reduction
        self._steps = steps

        self.preprocess1 = nn.Sequential(
            nn.Conv2d(C_prev, C, 1, bias=False),
            GN(C),
        )
        self._ops = nn.ModuleList()
        for i in range(self._steps): # all possible ops
            for j in range(i + 1):
                stride = 1
                if cell_type == "LSTM" or cell_type=='ENAS' or cell_type=='LSTM2':
                    op = PrivOp(C, stride)
                else:
                    assert False, "unsupported controller_type: %s" % cell_type
                self._ops.append(op)

    def forward(self, s1, arch):
        s1 = self.preprocess1.forward(s1)
        states = {0: s1}
        for op, f, t in arch:
            edge_id = int(f + np.sum([list(range(t))]))

            if t in states.keys():
                states[t] = states[t] + self._ops[edge_id](states[f], op)
            else:
                states[t] = self._ops[edge_id](states[f], op)

        return torch.cat([states[i] for i in range(1, self._steps + 1)], dim=1)


class ArchMaster(nn.Module):
    def __init__(self, n_ops, n_nodes, device, controller_type='SAMPLE', controller_hid=None,
                 controller_temperature=None, controller_tanh_constant=None, controller_op_tanh_reduce=None, lstm_num_layers=2):
        super(ArchMaster, self).__init__()
        self.K = sum([x + 2 for x in range(n_nodes)])
        self.n_ops = n_ops
        self.n_nodes = n_nodes     #  self._steps
        self.device = device
        self.controller_type = controller_type

        if controller_type in ['LSTM', 'LSTM2', 'ENAS']:  # TODO: ??? why 2 LSTMs??
            self.controller_hid = controller_hid
            self.attention_hid = self.controller_hid
            self.temperature = controller_temperature
            self.tanh_constant = controller_tanh_constant
            self.op_tanh_reduce = controller_op_tanh_reduce
            self.lstm_num_layers = lstm_num_layers


        if self.controller_type == 'LSTM':
            # Embedding of (n_nodes+1) nodes
            # Note that the (n_nodes+2)-th node will not be used
            self.node_op_hidden = nn.Embedding(n_nodes + 1 + n_ops, self.controller_hid)
            self.emb_attn = nn.Linear(self.controller_hid, self.attention_hid, bias=False)   # self.controller_hid, self.attention_hid
            self.hid_attn = nn.Linear(self.controller_hid, self.attention_hid, bias=False)
            self.v_attn = nn.Linear(self.controller_hid, 1, bias=False)
            self.w_soft = nn.Linear(self.controller_hid, self.n_ops)
            self.lstm = nn.LSTMCell(self.controller_hid, self.controller_hid)  # input size, hidden size
            self.reset_parameters()
            self.static_init_hidden = utils.keydefaultdict(self.init_hidden)
            self.static_inputs = utils.keydefaultdict(self._get_default_hidden)
            self.tanh = nn.Tanh()
            self.prev_nodes, self.prev_ops = [], []
            self.query_index = torch.LongTensor(range(0, n_nodes+1)).to(device)
        elif self.controller_type == 'LSTM2':
            # Embedding of (n_nodes+1) nodes
            # Note that the (n_nodes+2)-th node will not be used
            self.node_op_hidden = nn.Embedding(n_nodes + 1 + n_ops, self.controller_hid)
            self.g_emb = nn.Embedding(1, self.controller_hid) # Starting Token Embedding
            self.emb_attn = nn.Linear(self.controller_hid, self.attention_hid, bias=False)
            self.hid_attn = nn.Linear(self.controller_hid, self.attention_hid, bias=False)
            self.v_attn = nn.Linear(self.controller_hid, 1, bias=False)
            self.w_soft = nn.Linear(self.controller_hid, self.n_ops)
            self.lstm = nn.LSTMCell(self.controller_hid, self.controller_hid)
            self.reset_parameters()
            self.static_init_hidden = utils.keydefaultdict(self.init_hidden)
            self.tanh = nn.Tanh()
            self.prev_nodes, self.prev_ops = [], []
            self.anchors_w_1 = []
            self.query_index = torch.LongTensor(range(0, n_nodes + 1)).to(device)
        elif self.controller_type == 'ENAS':
            self.emb_attn = nn.Linear(self.controller_hid, self.attention_hid, bias=False)
            self.hid_attn = nn.Linear(self.controller_hid, self.attention_hid, bias=False)
            self.v_attn = nn.Linear(self.controller_hid, 1, bias=False)
            self.w_soft = nn.Linear(self.controller_hid, self.n_ops)
            # self.lstm = nn.LSTMCell(self.controller_hid, self.controller_hid)
            self.lstm = nn.ModuleList()
            for i in range(self.lstm_num_layers):
                self.lstm.append(nn.LSTMCell(self.controller_hid, self.controller_hid).to(device))
            # self.register_buffer('anchors', torch.zeros(n_nodes+2, self.controller_hid))
            # self.register_buffer('anchors_w_1', torch.zeros(n_nodes+2, self.controller_hid))
            # self.g_emb = nn.Embedding(1, self.controller_hid)
            self.anchors = []
            self.anchors_w_1 = []
            self.w_emb = nn.Embedding(self.n_ops + 1, self.controller_hid)
            self.reset_parameters()
            # self.static_init_hidden = utils.keydefaultdict(self.init_hidden)
            # self.static_inputs = utils.keydefaultdict(self._get_default_hidden)
            self.tanh = nn.Tanh()
            self.prev_nodes, self.prev_ops = [], []
            self.query_index = torch.LongTensor(range(0, n_nodes + 1)).to(self.device)
        else:
            assert False, "unsupported controller_type: %s" % self.controller_type

    def _init_nodes(self):
        self.anchors = []
        self.anchors_w_1 = []
        prev_c = [torch.zeros(1, self.controller_hid).to(self.device) for _ in range(self.lstm_num_layers)]
        prev_h = [torch.zeros(1, self.controller_hid).to(self.device) for _ in range(self.lstm_num_layers)]
        # initialize the first two nodes
        # inputs = self.g_emb(torch.LongTensor([0]))
        inputs = self.w_emb(torch.LongTensor([self.n_ops]).to(self.device))
        for node_id in range(2):
            next_c, next_h = self.stack_lstm(inputs, prev_c, prev_h)
            prev_c, prev_h = next_c, next_h
            # self.anchors[node_id].copy_(torch.zeros_like(next_h[-1].view(-1)))
            self.anchors.append(torch.zeros_like(next_h[-1]))
            new_hidden = self.emb_attn(next_h[-1])
            # self.anchors_w_1[node_id].copy_(new_hidden.view(-1))
            self.anchors_w_1.append(new_hidden)
        return inputs, prev_c, prev_h

    def stack_lstm(self, x, prev_c, prev_h):
        next_c, next_h = [], []
        for layer_id, (_c, _h) in enumerate(zip(prev_c, prev_h)):
            inputs = x if layer_id == 0 else next_h[-1]
            curr_c, curr_h = self.lstm[layer_id](inputs, (_c, _h))
            # curr_c, curr_h = self.lstm(inputs, (_c, _h))
            next_c.append(curr_c)
            next_h.append(curr_h)
        return next_c, next_h

    def _body(self, node_id, inputs, prev_c, prev_h, arc_seq, entropy, log_prob):
        # indices = range(0, node_id+2)
        start_id = 4 * node_id
        prev_layers = []
        self.node_logits_ops_list = []
        for i in range(2):  # index_1, index_2
            next_c, next_h = self.stack_lstm(inputs, prev_c, prev_h)
            prev_c, prev_h = next_c, next_h
            # query = self.anchors_w_1.index_select(
            #     0, self.query_index[0:node_id + 2]
            # )
            query = torch.cat(self.anchors_w_1[0:node_id + 2])
            # print(self.anchors.is_cuda)
            # print(self.anchors_w_1.is_cuda)
            # print(query.is_cuda)
            # print(next_h[-1].is_cuda)
            query = self.tanh(query + self.hid_attn(next_h[-1]))  # attention used tanh, used as in LSTM
            logits = self.v_attn(query).view(-1)    # attention here, no tanh on the logits. see self.tanh_constant
            if self.temperature is not None:
                logits /= self.temperature
            if self.tanh_constant is not None:
                logits = self.tanh_constant * self.tanh(logits)
            if self.force_uniform:
                probs = F.softmax(torch.zeros_like(logits), dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)
            try:
                index = probs.multinomial(num_samples=1)
            except:
                probs = F.softmax(torch.zeros_like(logits), dim=-1)
                index = probs.multinomial(num_samples=1)

            log_probs = torch.log(probs)
            arc_seq[start_id + 2 * i] = index
            curr_log_prob = log_probs.gather(0, index)[0]
            log_prob += curr_log_prob
            curr_ent = -(log_probs * probs).sum()
            entropy += curr_ent
            prev_layers.append(self.anchors[index])
            inputs = prev_layers[-1]
        for i in range(2):  # op_1, op_2
            next_c, next_h = self.stack_lstm(inputs, prev_c, prev_h)
            prev_c, prev_h = next_c, next_h
            logits = self.w_soft(next_h[-1]).view(-1)
            if self.temperature is not None:
                logits /= self.temperature
            if self.tanh_constant is not None:
                op_tanh = self.tanh_constant / self.op_tanh_reduce
                logits = op_tanh * self.tanh(logits)
            if self.force_uniform:
                probs = F.softmax(torch.zeros_like(logits), dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)
            self.node_logits_ops_list.append(logits)
            try:
                op_id = probs.multinomial(num_samples=1)
            except:
                probs = F.softmax(torch.zeros_like(logits), dim=-1)
                op_id = probs.multinomial(num_samples=1)

            log_probs = torch.log(probs)
            arc_seq[start_id + 2 * i + 1] = op_id
            curr_log_prob = log_probs.gather(0, op_id)[0]
            log_prob += curr_log_prob
            curr_ent = -(log_probs * probs).sum()
            entropy += curr_ent
            inputs = self.w_emb(op_id)

        # TODO: why additional run one step forward?
        next_c, next_h = self.stack_lstm(inputs, prev_c, prev_h)
        # self.anchors[node_id+2].copy_(next_h[-1].view(-1))
        self.anchors.append(next_h[-1])
        new_hidden = self.emb_attn(next_h[-1])
        # self.anchors_w_1[node_id+2].copy_(new_hidden.view(-1))
        self.anchors_w_1.append(new_hidden)
        # inputs = self.g_emb(torch.LongTensor([0]))
        inputs = self.w_emb(torch.LongTensor([self.n_ops]).to(self.device))   # TODO: reset the inputs?
        return inputs, next_c, next_h, arc_seq, entropy, log_prob

    def _convert_lstm_output(self, prev_nodes, prev_ops):
        """

        :param prev_nodes: vector, each element is the node ID, int64, in the range of [0,1,...,n_node]
        :param prev_ops: vector, each element is the op_id, int64, in the range [0,1,...,n_ops-1]
        :return: arch list, (op, f, t) is the elements
        """
        assert len(prev_nodes) == 2 * self.n_nodes
        assert len(prev_ops) == 2 * self.n_nodes
        arch_list = []
        for i in range(self.n_nodes):
            t_node = i + 2
            f1_node = prev_nodes[i * 2].item()
            f2_node = prev_nodes[i * 2 + 1].item()
            f1_op = prev_ops[i * 2].item()
            f2_op = prev_ops[i * 2 + 1].item()
            arch_list.append((f1_op, f1_node, t_node))
            arch_list.append((f2_op, f2_node, t_node))
        return arch_list

    def _get_default_hidden(self, key):
        return utils.get_variable(
            torch.zeros(key, self.controller_hid), self.device, requires_grad=False)

    # device
    def init_hidden(self, batch_size):
        zeros = torch.zeros(batch_size, self.controller_hid)
        return (utils.get_variable(zeros, self.device, requires_grad=False),
                utils.get_variable(zeros.clone(), self.device, requires_grad=False))

    def reset_parameters(self):
        # params initialization
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        self.w_soft.bias.data.fill_(0)

    def _clear_arch(self, arch):
        new_arch = [arch[0]]
        for e in arch[1:]:
            if not (e == new_arch[-1]):
                new_arch.append(e)
        return new_arch

    def forward(self):
        if self.controller_type == 'LSTM':
            log_p, entropy = 0, 0
            self.prev_nodes, self.prev_ops = [], []
            self.logits_nodes_list, self.logits_ops_list = [],[]
            batch_size = 1
            inputs = self.static_inputs[batch_size]  # batch_size x hidden_dim
            hidden = self.static_init_hidden[batch_size]
            for node_idx in range(self.n_nodes):
                for i in range(2):  # index_1, index_2
                    if node_idx == 0 and i == 0:
                        embed = inputs
                    else:
                        embed = self.node_op_hidden(inputs)
                    hx, cx = self.lstm(embed, hidden)
                    query = self.node_op_hidden.weight.index_select(
                        0, self.query_index[0:node_idx + 2]
                    )
                    query = self.tanh(self.emb_attn(query) + self.hid_attn(hx))

                    self.query_index[0:node_idx + 2]
                    logits = self.v_attn(query).view(-1)
                    if self.temperature is not None:
                        logits /= self.temperature
                    if self.tanh_constant is not None:
                        logits = self.tanh_constant * self.tanh(logits)   # TODD: check here now
                    if self.force_uniform:
                        probs = F.softmax(torch.zeros_like(logits), dim=-1)
                    else:
                        probs = F.softmax(logits, dim=-1)  # absent in ENAS code
                    self.logits_nodes_list.append(logits)
                    try:
                        action = probs.multinomial(num_samples=1)
                    except:
                        probs = F.softmax(torch.zeros_like(logits), dim=-1)
                        action = probs.multinomial(num_samples=1)

                    log_probs = torch.log(probs)
                    # selected_probs = probs.gather(0, action)
                    selected_log_p = log_probs.gather(0, action)[0]
                    self.prev_nodes.append(action)
                    log_p += selected_log_p
                    entropy += -(log_probs * probs).sum()
                    # reset hidden and inputs
                    hidden = (hx, cx)
                    inputs = utils.get_variable(action, self.device, requires_grad=False)
                for i in range(2):  # op_1, op_2
                    embed = self.node_op_hidden(inputs)
                    hx, cx = self.lstm(embed, hidden)
                    logits = self.w_soft(hx).view(-1)
                    if self.temperature is not None:
                        logits /= self.temperature
                    if self.tanh_constant is not None:
                        op_tanh = self.tanh_constant / self.op_tanh_reduce
                        logits = op_tanh * self.tanh(logits)
                    if self.force_uniform:
                        probs = F.softmax(torch.zeros_like(logits), dim=-1)
                    else:
                        probs = F.softmax(logits, dim=-1)  # absent in ENAS code
                    self.logits_ops_list.append(logits)
                    try:
                        action = probs.multinomial(num_samples=1)
                    except:
                        probs = F.softmax(torch.zeros_like(logits), dim=-1)
                        action = probs.multinomial(num_samples=1)

                    log_probs = torch.log(probs)
                    self.prev_ops.append(action)
                    # selected_probs = probs.gather(1, action).view(-1)
                    selected_log_p = log_probs.gather(0, action)[0]
                    log_p += selected_log_p
                    entropy += -(log_probs * probs).sum()
                    # reset hidden and inputs
                    hidden = (hx, cx)
                    inputs = utils.get_variable(action + self.n_nodes + 1, self.device, requires_grad=False)
            arch = self._convert_lstm_output(torch.cat(self.prev_nodes), torch.cat(self.prev_ops))
            # print(len(self.logits_nodes_list), self.logits_nodes_list)
            # print(len(self.logits_ops_list), self.logits_ops_list)
            # self.logits_nodes = torch.stack(tuple(self.logits_nodes_list))
            self.logits_ops = torch.stack(tuple(self.logits_ops_list))
            # print('current embedding of all the candidates are:', self.node_op_hidden.weight)
            # print('The node logits are:', self.logits_nodes)
            # print('The ops logits are:', self.logits_ops)
            return arch, log_p, entropy
        elif self.controller_type == 'LSTM2':
            log_p, entropy = 0, 0
            self.prev_nodes, self.prev_ops = [], []
            self.logits_nodes_list, self.logits_ops_list = [],[]
            # init controller
            batch_size = 1
            inputs = self.g_emb(torch.LongTensor([0]).to(self.device))
            hidden = self.static_init_hidden[batch_size]
            self.anchors_w_1 = []
            for node_idx in range(2):
                hx, cx = self.lstm(inputs, hidden)
                hidden = (hx, cx)
                new_hidden = self.emb_attn(hx)
                self.anchors_w_1.append(new_hidden)

            for node_idx in range(self.n_nodes):
                for i in range(2):  # index_1, index_2
                    if i == 0:
                        embed = inputs
                    else:
                        embed = self.node_op_hidden(inputs)
                    hx, cx = self.lstm(embed, hidden)
                    query = torch.cat(self.anchors_w_1[0:node_idx+2])
                    # query = self.node_op_hidden.weight.index_select(
                    #     0, self.query_index[0:node_idx + 2]
                    # )
                    query = self.tanh(query + self.hid_attn(hx))
                    logits = self.v_attn(query).view(-1)
                    if self.temperature is not None:
                        logits /= self.temperature
                    if self.tanh_constant is not None:
                        logits = self.tanh_constant * self.tanh(logits)
                    if self.force_uniform:
                        probs = F.softmax(torch.zeros_like(logits), dim=-1)
                    else:
                        probs = F.softmax(logits, dim=-1)  # absent in ENAS code
                    self.logits_nodes_list.append(logits)
                    try:
                        action = probs.multinomial(num_samples=1)
                    except:
                        probs = F.softmax(torch.zeros_like(logits), dim=-1)
                        action = probs.multinomial(num_samples=1)

                    log_probs = torch.log(probs)
                    # selected_probs = probs.gather(0, action)
                    selected_log_p = log_probs.gather(0, action)[0]
                    self.prev_nodes.append(action)
                    log_p += selected_log_p
                    entropy += -(log_probs * probs).sum()
                    # reset hidden and inputs
                    hidden = (hx, cx)
                    inputs = utils.get_variable(action, self.device, requires_grad=False)
                for i in range(2):  # op_1, op_2
                    embed = self.node_op_hidden(inputs)
                    hx, cx = self.lstm(embed, hidden)
                    logits = self.w_soft(hx).view(-1)
                    if self.temperature is not None:
                        logits /= self.temperature
                    if self.tanh_constant is not None:
                        op_tanh = self.tanh_constant / self.op_tanh_reduce
                        logits = op_tanh * self.tanh(logits)
                    if self.force_uniform:
                        probs = F.softmax(torch.zeros_like(logits), dim=-1)
                    else:
                        probs = F.softmax(logits, dim=-1)  # absent in ENAS code
                    self.logits_ops_list.append(logits)
                    try:
                        action = probs.multinomial(num_samples=1)
                    except:
                        probs = F.softmax(torch.zeros_like(logits), dim=-1)
                        action = probs.multinomial(num_samples=1)

                    log_probs = torch.log(probs)
                    self.prev_ops.append(action)
                    # selected_probs = probs.gather(1, action).view(-1)
                    selected_log_p = log_probs.gather(0, action)[0]
                    log_p += selected_log_p
                    entropy += -(log_probs * probs).sum()
                    # reset hidden and inputs
                    hidden = (hx, cx)
                    inputs = utils.get_variable(action + self.n_nodes + 1, self.device, requires_grad=False)
                embed = self.node_op_hidden(inputs)
                hx, cx = self.lstm(embed, hidden)
                hidden = (hx, cx)
                self.anchors_w_1.append(self.emb_attn(hx))
                inputs = self.g_emb(torch.LongTensor([0]).to(self.device))
            arch = self._convert_lstm_output(torch.cat(self.prev_nodes), torch.cat(self.prev_ops))
            self.logits_ops = torch.stack(tuple(self.logits_ops_list))
            # print('current embedding of all the candidates are:', self.node_op_hidden.weight)
            # print('The node logits are:', self.logits_nodes)
            # print('The ops logits are:', self.logits_ops)
            return arch, log_p, entropy
        elif self.controller_type == 'ENAS':
            log_prob, entropy = 0, 0
            self.prev_nodes, self.prev_ops = [], []
            self.logits_ops_list = []
            arc_seq = [0] * (self.n_nodes * 4)
            inputs, prev_c, prev_h = self._init_nodes()
            for node_id in range(self.n_nodes):
                inputs, next_c, next_h, arc_seq, entropy, log_prob = \
                self._body(node_id, inputs, prev_c, prev_h,
                           arc_seq, entropy, log_prob)
                prev_c = next_c
                prev_h = next_h
                self.logits_ops_list.extend(self.node_logits_ops_list)
            # transform arc_seq to prev_nodes and prev_ops
            for node_id in range(self.n_nodes):
                for i in range(2):
                    self.prev_nodes.append(arc_seq[node_id * 4 + i * 2])
                    self.prev_ops.append(arc_seq[node_id * 4 + i * 2 + 1])
            arch = self._convert_lstm_output(torch.cat(self.prev_nodes),
                                             torch.cat(self.prev_ops))
            self.logits_ops = torch.stack(tuple(self.logits_ops_list))

            arch = self._clear_arch(arch)
            return arch, log_prob, entropy
        else:
            assert False, "unsupported controller_type: %s" % self.controller_type


class DPArchMaster(nn.Module):
    def __init__(self, n_ops, n_nodes, device,
                 controller_type='SAMPLE',
                 controller_hid=None,
                 controller_temperature=None,
                 controller_tanh_constant=None,
                 controller_op_tanh_reduce=None,
                 lstm_num_layers=2):
        super(DPArchMaster, self).__init__()
        self.K = sum([x + 2 for x in range(n_nodes)])
        self.n_ops = n_ops
        self.n_nodes = n_nodes     #  self._steps
        self.device = device
        self.controller_type = controller_type

        if controller_type in ['LSTM', 'LSTM2', 'ENAS']:  # TODO: ??? why 2 LSTMs??
            self.controller_hid = controller_hid
            self.attention_hid = self.controller_hid
            self.temperature = controller_temperature
            self.tanh_constant = controller_tanh_constant
            self.op_tanh_reduce = controller_op_tanh_reduce
            self.lstm_num_layers = lstm_num_layers


        if self.controller_type == 'LSTM':
            # Embedding of (n_nodes+1) nodes
            # Note that the (n_nodes+2)-th node will not be used
            self.node_op_hidden = nn.Embedding(n_nodes + 1 + n_ops, self.controller_hid)
            self.emb_attn = nn.Linear(self.controller_hid, self.attention_hid, bias=False)   # self.controller_hid, self.attention_hid
            self.hid_attn = nn.Linear(self.controller_hid, self.attention_hid, bias=False)
            self.v_attn = nn.Linear(self.controller_hid, 1, bias=False)
            self.w_soft = nn.Linear(self.controller_hid, self.n_ops)
            self.lstm = nn.LSTMCell(self.controller_hid, self.controller_hid)  # input size, hidden size
            self.reset_parameters()
            self.static_init_hidden = utils.keydefaultdict(self.init_hidden)
            self.static_inputs = utils.keydefaultdict(self._get_default_hidden)
            self.tanh = nn.Tanh()
            self.prev_nodes, self.prev_ops = [], []
            self.query_index = torch.LongTensor(range(0, n_nodes+1)).to(device)
        elif self.controller_type == 'LSTM2':
            # Embedding of (n_nodes+1) nodes
            # Note that the (n_nodes+2)-th node will not be used
            self.node_op_hidden = nn.Embedding(n_nodes + 1 + n_ops, self.controller_hid)
            self.g_emb = nn.Embedding(1, self.controller_hid) # Starting Token Embedding
            self.emb_attn = nn.Linear(self.controller_hid, self.attention_hid, bias=False)
            self.hid_attn = nn.Linear(self.controller_hid, self.attention_hid, bias=False)
            self.v_attn = nn.Linear(self.controller_hid, 1, bias=False)
            self.w_soft = nn.Linear(self.controller_hid, self.n_ops)
            self.lstm = nn.LSTMCell(self.controller_hid, self.controller_hid)
            self.reset_parameters()
            self.static_init_hidden = utils.keydefaultdict(self.init_hidden)
            self.tanh = nn.Tanh()
            self.prev_nodes, self.prev_ops = [], []
            self.anchors_w_1 = []
            self.query_index = torch.LongTensor(range(0, n_nodes + 1)).to(device)
        elif self.controller_type == 'ENAS':
            self.emb_attn = nn.Linear(self.controller_hid, self.attention_hid, bias=False)
            self.hid_attn = nn.Linear(self.controller_hid, self.attention_hid, bias=False)
            self.v_attn = nn.Linear(self.controller_hid, 1, bias=False)
            self.w_soft = nn.Linear(self.controller_hid, self.n_ops)
            # self.lstm = nn.LSTMCell(self.controller_hid, self.controller_hid)
            self.lstm = nn.ModuleList()
            for i in range(self.lstm_num_layers):
                self.lstm.append(nn.LSTMCell(self.controller_hid, self.controller_hid).to(device))
            # self.register_buffer('anchors', torch.zeros(n_nodes+2, self.controller_hid))
            # self.register_buffer('anchors_w_1', torch.zeros(n_nodes+2, self.controller_hid))
            # self.g_emb = nn.Embedding(1, self.controller_hid)
            self.anchors = []
            self.anchors_w_1 = []
            self.w_emb = nn.Embedding(self.n_ops + 1, self.controller_hid)
            self.reset_parameters()
            # self.static_init_hidden = utils.keydefaultdict(self.init_hidden)
            # self.static_inputs = utils.keydefaultdict(self._get_default_hidden)
            self.tanh = nn.Tanh()
            self.prev_nodes, self.prev_ops = [], []
            self.query_index = torch.LongTensor(range(0, n_nodes + 1)).to(self.device)
        else:
            assert False, "unsupported controller_type: %s" % self.controller_type

    def _init_nodes(self):
        self.anchors = []
        self.anchors_w_1 = []
        prev_c = [torch.zeros(1, self.controller_hid).to(self.device) for _ in range(self.lstm_num_layers)]
        prev_h = [torch.zeros(1, self.controller_hid).to(self.device) for _ in range(self.lstm_num_layers)]
        # initialize the first two nodes
        # inputs = self.g_emb(torch.LongTensor([0]))
        inputs = self.w_emb(torch.LongTensor([self.n_ops]).to(self.device))
        for node_id in range(2):
            next_c, next_h = self.stack_lstm(inputs, prev_c, prev_h)
            prev_c, prev_h = next_c, next_h
            # self.anchors[node_id].copy_(torch.zeros_like(next_h[-1].view(-1)))
            self.anchors.append(torch.zeros_like(next_h[-1]))
            new_hidden = self.emb_attn(next_h[-1])
            # self.anchors_w_1[node_id].copy_(new_hidden.view(-1))
            self.anchors_w_1.append(new_hidden)
        return inputs, prev_c, prev_h

    def stack_lstm(self, x, prev_c, prev_h):
        next_c, next_h = [], []
        for layer_id, (_c, _h) in enumerate(zip(prev_c, prev_h)):
            inputs = x if layer_id == 0 else next_h[-1]
            curr_c, curr_h = self.lstm[layer_id](inputs, (_c, _h))
            # curr_c, curr_h = self.lstm(inputs, (_c, _h))
            next_c.append(curr_c)
            next_h.append(curr_h)
        return next_c, next_h

    def _body(self, node_id, inputs, prev_c, prev_h, arc_seq, entropy, log_prob):
        start_id = 0 if node_id==0 \
            else int(np.sum([(i+2) for i in range(node_id)]))*2
        num_ops = node_id + 2
        self.node_logits_ops_list = []

        for i in range(num_ops): # loop for nodes
            arc_seq[start_id + 2 * i] = torch.LongTensor([i]).to(inputs.device)

        for i in range(num_ops): # loop for ops
            next_c, next_h = self.stack_lstm(inputs, prev_c, prev_h)
            prev_c, prev_h = next_c, next_h
            logits = self.w_soft(next_h[-1]).view(-1)
            if self.temperature is not None:
                logits /= self.temperature
            if self.tanh_constant is not None:
                op_tanh = self.tanh_constant / self.op_tanh_reduce
                logits = op_tanh * self.tanh(logits)
            if self.force_uniform:
                probs = F.softmax(torch.zeros_like(logits), dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)
            self.node_logits_ops_list.append(logits)
            try:
                op_id = probs.multinomial(num_samples=1)
            except:
                probs = F.softmax(torch.zeros_like(logits), dim=-1)
                op_id = probs.multinomial(num_samples=1)

            log_probs = torch.log(probs)
            arc_seq[start_id + 2 * i + 1] = op_id
            curr_log_prob = log_probs.gather(0, op_id)[0]
            log_prob += curr_log_prob
            curr_ent = -(log_probs * probs).sum()
            entropy += curr_ent
            inputs = self.w_emb(op_id)

        # TODO: why additional run one step forward?
        next_c, next_h = self.stack_lstm(inputs, prev_c, prev_h)
        # self.anchors[node_id+2].copy_(next_h[-1].view(-1))
        self.anchors.append(next_h[-1])
        new_hidden = self.emb_attn(next_h[-1])
        # self.anchors_w_1[node_id+2].copy_(new_hidden.view(-1))
        self.anchors_w_1.append(new_hidden)
        # inputs = self.g_emb(torch.LongTensor([0]))
        inputs = self.w_emb(torch.LongTensor([self.n_ops]).to(self.device))   # TODO: reset the inputs?
        return inputs, next_c, next_h, arc_seq, entropy, log_prob

    def _convert_lstm_output(self, prev_nodes, prev_ops):
        """
        :param prev_nodes: vector, each element is the node ID, int64, in the range of [0,1,...,n_node]
        :param prev_ops: vector, each element is the op_id, int64, in the range [0,1,...,n_ops-1]
        :return: arch list, (op, f, t) is the elements
        """
        _count_list = [(i+2) for i in range(self.n_nodes)]
        assert len(prev_nodes) == np.sum(_count_list)
        assert len(prev_ops) == np.sum(_count_list)
        arch_list = []
        offset = 0
        for i in range(self.n_nodes):
            for _ in range(i+2):
                arch_list.append((prev_ops[offset].item(),
                                  prev_nodes[offset].item(),
                                  i+2))
                offset += 1

        return arch_list

    def _get_default_hidden(self, key):
        return utils.get_variable(
            torch.zeros(key, self.controller_hid), self.device, requires_grad=False)

    # device
    def init_hidden(self, batch_size):
        zeros = torch.zeros(batch_size, self.controller_hid)
        return (utils.get_variable(zeros, self.device, requires_grad=False),
                utils.get_variable(zeros.clone(), self.device, requires_grad=False))

    def reset_parameters(self):
        # params initialization
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        self.w_soft.bias.data.fill_(0)

    def _clear_arch(self, arch):
        new_arch = [arch[0]]
        for e in arch[1:]:
            if not (e == new_arch[-1]):
                new_arch.append(e)
        return new_arch


    def forward(self):
        if self.controller_type == 'LSTM':
            log_p, entropy = 0, 0
            self.prev_nodes, self.prev_ops = [], []
            self.logits_nodes_list, self.logits_ops_list = [],[]
            batch_size = 1
            inputs = self.static_inputs[batch_size]  # batch_size x hidden_dim
            hidden = self.static_init_hidden[batch_size]
            for node_idx in range(self.n_nodes):
                for i in range(2):  # index_1, index_2
                    if node_idx == 0 and i == 0:
                        embed = inputs
                    else:
                        embed = self.node_op_hidden(inputs)
                    hx, cx = self.lstm(embed, hidden)
                    query = self.node_op_hidden.weight.index_select(
                        0, self.query_index[0:node_idx + 2]
                    )
                    query = self.tanh(self.emb_attn(query) + self.hid_attn(hx))

                    self.query_index[0:node_idx + 2]
                    logits = self.v_attn(query).view(-1)
                    if self.temperature is not None:
                        logits /= self.temperature
                    if self.tanh_constant is not None:
                        logits = self.tanh_constant * self.tanh(logits)   # TODD: check here now
                    if self.force_uniform:
                        probs = F.softmax(torch.zeros_like(logits), dim=-1)
                    else:
                        probs = F.softmax(logits, dim=-1)  # absent in ENAS code
                    self.logits_nodes_list.append(logits)
                    try:
                        action = probs.multinomial(num_samples=1)
                    except:
                        probs = F.softmax(torch.zeros_like(logits), dim=-1)
                        action = probs.multinomial(num_samples=1)

                    log_probs = torch.log(probs)
                    # selected_probs = probs.gather(0, action)
                    selected_log_p = log_probs.gather(0, action)[0]
                    self.prev_nodes.append(action)
                    log_p += selected_log_p
                    entropy += -(log_probs * probs).sum()
                    # reset hidden and inputs
                    hidden = (hx, cx)
                    inputs = utils.get_variable(action, self.device, requires_grad=False)
                for i in range(2):  # op_1, op_2
                    embed = self.node_op_hidden(inputs)
                    hx, cx = self.lstm(embed, hidden)
                    logits = self.w_soft(hx).view(-1)
                    if self.temperature is not None:
                        logits /= self.temperature
                    if self.tanh_constant is not None:
                        op_tanh = self.tanh_constant / self.op_tanh_reduce
                        logits = op_tanh * self.tanh(logits)
                    if self.force_uniform:
                        probs = F.softmax(torch.zeros_like(logits), dim=-1)
                    else:
                        probs = F.softmax(logits, dim=-1)  # absent in ENAS code
                    self.logits_ops_list.append(logits)
                    try:
                        action = probs.multinomial(num_samples=1)
                    except:
                        probs = F.softmax(torch.zeros_like(logits), dim=-1)
                        action = probs.multinomial(num_samples=1)

                    log_probs = torch.log(probs)
                    self.prev_ops.append(action)
                    # selected_probs = probs.gather(1, action).view(-1)
                    selected_log_p = log_probs.gather(0, action)[0]
                    log_p += selected_log_p
                    entropy += -(log_probs * probs).sum()
                    # reset hidden and inputs
                    hidden = (hx, cx)
                    inputs = utils.get_variable(action + self.n_nodes + 1, self.device, requires_grad=False)
            arch = self._convert_lstm_output(torch.cat(self.prev_nodes), torch.cat(self.prev_ops))
            # print(len(self.logits_nodes_list), self.logits_nodes_list)
            # print(len(self.logits_ops_list), self.logits_ops_list)
            # self.logits_nodes = torch.stack(tuple(self.logits_nodes_list))
            self.logits_ops = torch.stack(tuple(self.logits_ops_list))
            # print('current embedding of all the candidates are:', self.node_op_hidden.weight)
            # print('The node logits are:', self.logits_nodes)
            # print('The ops logits are:', self.logits_ops)
            return arch, log_p, entropy
        elif self.controller_type == 'LSTM2':
            log_p, entropy = 0, 0
            self.prev_nodes, self.prev_ops = [], []
            self.logits_nodes_list, self.logits_ops_list = [],[]
            # init controller
            batch_size = 1
            inputs = self.g_emb(torch.LongTensor([0]).to(self.device))
            hidden = self.static_init_hidden[batch_size]
            self.anchors_w_1 = []
            for node_idx in range(2):
                hx, cx = self.lstm(inputs, hidden)
                hidden = (hx, cx)
                new_hidden = self.emb_attn(hx)
                self.anchors_w_1.append(new_hidden)

            for node_idx in range(self.n_nodes):
                for i in range(2):  # index_1, index_2
                    if i == 0:
                        embed = inputs
                    else:
                        embed = self.node_op_hidden(inputs)
                    hx, cx = self.lstm(embed, hidden)
                    query = torch.cat(self.anchors_w_1[0:node_idx+2])
                    # query = self.node_op_hidden.weight.index_select(
                    #     0, self.query_index[0:node_idx + 2]
                    # )
                    query = self.tanh(query + self.hid_attn(hx))
                    logits = self.v_attn(query).view(-1)
                    if self.temperature is not None:
                        logits /= self.temperature
                    if self.tanh_constant is not None:
                        logits = self.tanh_constant * self.tanh(logits)
                    if self.force_uniform:
                        probs = F.softmax(torch.zeros_like(logits), dim=-1)
                    else:
                        probs = F.softmax(logits, dim=-1)  # absent in ENAS code
                    self.logits_nodes_list.append(logits)
                    try:
                        action = probs.multinomial(num_samples=1)
                    except:
                        probs = F.softmax(torch.zeros_like(logits), dim=-1)
                        action = probs.multinomial(num_samples=1)

                    log_probs = torch.log(probs)
                    # selected_probs = probs.gather(0, action)
                    selected_log_p = log_probs.gather(0, action)[0]
                    self.prev_nodes.append(action)
                    log_p += selected_log_p
                    entropy += -(log_probs * probs).sum()
                    # reset hidden and inputs
                    hidden = (hx, cx)
                    inputs = utils.get_variable(action, self.device, requires_grad=False)
                for i in range(2):  # op_1, op_2
                    embed = self.node_op_hidden(inputs)
                    hx, cx = self.lstm(embed, hidden)
                    logits = self.w_soft(hx).view(-1)
                    if self.temperature is not None:
                        logits /= self.temperature
                    if self.tanh_constant is not None:
                        op_tanh = self.tanh_constant / self.op_tanh_reduce
                        logits = op_tanh * self.tanh(logits)
                    if self.force_uniform:
                        probs = F.softmax(torch.zeros_like(logits), dim=-1)
                    else:
                        probs = F.softmax(logits, dim=-1)  # absent in ENAS code
                    self.logits_ops_list.append(logits)
                    try:
                        action = probs.multinomial(num_samples=1)
                    except:
                        probs = F.softmax(torch.zeros_like(logits), dim=-1)
                        action = probs.multinomial(num_samples=1)

                    log_probs = torch.log(probs)
                    self.prev_ops.append(action)
                    # selected_probs = probs.gather(1, action).view(-1)
                    selected_log_p = log_probs.gather(0, action)[0]
                    log_p += selected_log_p
                    entropy += -(log_probs * probs).sum()
                    # reset hidden and inputs
                    hidden = (hx, cx)
                    inputs = utils.get_variable(action + self.n_nodes + 1, self.device, requires_grad=False)
                embed = self.node_op_hidden(inputs)
                hx, cx = self.lstm(embed, hidden)
                hidden = (hx, cx)
                self.anchors_w_1.append(self.emb_attn(hx))
                inputs = self.g_emb(torch.LongTensor([0]).to(self.device))
            arch = self._convert_lstm_output(torch.cat(self.prev_nodes), torch.cat(self.prev_ops))
            self.logits_ops = torch.stack(tuple(self.logits_ops_list))
            # print('current embedding of all the candidates are:', self.node_op_hidden.weight)
            # print('The node logits are:', self.logits_nodes)
            # print('The ops logits are:', self.logits_ops)
            return arch, log_p, entropy
        elif self.controller_type == 'ENAS':
            log_prob, entropy = 0, 0
            self.prev_nodes, self.prev_ops = [], []
            self.logits_ops_list = []
            arc_seq = [0] * np.sum([(i+2) for i in range(self.n_nodes)])*2

            inputs, prev_c, prev_h = self._init_nodes()

            for node_id in range(self.n_nodes):
                inputs, next_c, next_h, arc_seq, entropy, log_prob = \
                self._body(node_id, inputs, prev_c, prev_h,
                           arc_seq, entropy, log_prob)
                prev_c = next_c
                prev_h = next_h
                self.logits_ops_list.extend(self.node_logits_ops_list)

            for node_id in range(self.n_nodes):
                for i in range(node_id+2):
                    start_id = 0 if node_id == 0 \
                        else int(np.sum([(i + 2) for i in range(node_id)]))*2
                    self.prev_nodes.append(arc_seq[start_id + i * 2])
                    self.prev_ops.append(arc_seq[start_id + i * 2 + 1])

            arch = self._convert_lstm_output(torch.cat(self.prev_nodes),
                                             torch.cat(self.prev_ops))
            self.logits_ops = torch.stack(tuple(self.logits_ops_list))

            arch = self._clear_arch(arch)
            return arch, log_prob, entropy
        else:
            assert False, "unsupported controller_type: %s" % self.controller_type


class DPDARTSArchMaster(nn.Module):
    def __init__(self, n_ops, n_nodes, device,
                 controller_type='SAMPLE',
                 controller_hid=None,
                 controller_temperature=None,
                 controller_tanh_constant=None,
                 controller_op_tanh_reduce=None,
                 lstm_num_layers=2):
        super(DPDARTSArchMaster, self).__init__()
        self.K = sum([x + 2 for x in range(n_nodes)])
        self.n_ops = n_ops
        self.n_nodes = n_nodes     #  self._steps
        self.device = device
        self.controller_type = controller_type

        if controller_type in ['LSTM', 'LSTM2', 'ENAS']:  # TODO: ??? why 2 LSTMs??
            self.controller_hid = controller_hid
            self.attention_hid = self.controller_hid
            self.temperature = controller_temperature
            self.tanh_constant = controller_tanh_constant
            self.op_tanh_reduce = controller_op_tanh_reduce
            self.lstm_num_layers = lstm_num_layers


        if self.controller_type == 'LSTM':
            # Embedding of (n_nodes+1) nodes
            # Note that the (n_nodes+2)-th node will not be used
            self.node_op_hidden = nn.Embedding(n_nodes + 1 + n_ops, self.controller_hid)
            self.emb_attn = nn.Linear(self.controller_hid, self.attention_hid, bias=False)   # self.controller_hid, self.attention_hid
            self.hid_attn = nn.Linear(self.controller_hid, self.attention_hid, bias=False)
            self.v_attn = nn.Linear(self.controller_hid, 1, bias=False)
            self.w_soft = nn.Linear(self.controller_hid, self.n_ops)
            self.lstm = nn.LSTMCell(self.controller_hid, self.controller_hid)  # input size, hidden size
            self.reset_parameters()
            self.static_init_hidden = utils.keydefaultdict(self.init_hidden)
            self.static_inputs = utils.keydefaultdict(self._get_default_hidden)
            self.tanh = nn.Tanh()
            self.prev_nodes, self.prev_ops = [], []
            self.query_index = torch.LongTensor(range(0, n_nodes+1)).to(device)
        elif self.controller_type == 'LSTM2':
            # Embedding of (n_nodes+1) nodes
            # Note that the (n_nodes+2)-th node will not be used
            self.node_op_hidden = nn.Embedding(n_nodes + 1 + n_ops, self.controller_hid)
            self.g_emb = nn.Embedding(1, self.controller_hid) # Starting Token Embedding
            self.emb_attn = nn.Linear(self.controller_hid, self.attention_hid, bias=False)
            self.hid_attn = nn.Linear(self.controller_hid, self.attention_hid, bias=False)
            self.v_attn = nn.Linear(self.controller_hid, 1, bias=False)
            self.w_soft = nn.Linear(self.controller_hid, self.n_ops)
            self.lstm = nn.LSTMCell(self.controller_hid, self.controller_hid)
            self.reset_parameters()
            self.static_init_hidden = utils.keydefaultdict(self.init_hidden)
            self.tanh = nn.Tanh()
            self.prev_nodes, self.prev_ops = [], []
            self.anchors_w_1 = []
            self.query_index = torch.LongTensor(range(0, n_nodes + 1)).to(device)
        elif self.controller_type == 'ENAS':
            self.emb_attn = nn.Linear(self.controller_hid, self.attention_hid, bias=False)
            self.hid_attn = nn.Linear(self.controller_hid, self.attention_hid, bias=False)
            self.v_attn = nn.Linear(self.controller_hid, 1, bias=False)
            self.w_soft = nn.Linear(self.controller_hid, self.n_ops)
            # self.lstm = nn.LSTMCell(self.controller_hid, self.controller_hid)
            self.lstm = nn.ModuleList()
            for i in range(self.lstm_num_layers):
                self.lstm.append(nn.LSTMCell(self.controller_hid, self.controller_hid).to(device))
            # self.register_buffer('anchors', torch.zeros(n_nodes+2, self.controller_hid))
            # self.register_buffer('anchors_w_1', torch.zeros(n_nodes+2, self.controller_hid))
            # self.g_emb = nn.Embedding(1, self.controller_hid)
            self.anchors = []
            self.anchors_w_1 = []
            self.w_emb = nn.Embedding(self.n_ops + 1, self.controller_hid)
            self.reset_parameters()
            # self.static_init_hidden = utils.keydefaultdict(self.init_hidden)
            # self.static_inputs = utils.keydefaultdict(self._get_default_hidden)
            self.tanh = nn.Tanh()
            self.prev_nodes, self.prev_ops = [], []
            self.query_index = torch.LongTensor(range(0, n_nodes + 1)).to(self.device)
        else:
            assert False, "unsupported controller_type: %s" % self.controller_type

    def _init_nodes(self):
        self.anchors = []
        self.anchors_w_1 = []
        prev_c = [torch.zeros(1, self.controller_hid).to(self.device) for _ in range(self.lstm_num_layers)]
        prev_h = [torch.zeros(1, self.controller_hid).to(self.device) for _ in range(self.lstm_num_layers)]
        # initialize the first two nodes
        # inputs = self.g_emb(torch.LongTensor([0]))
        inputs = self.w_emb(torch.LongTensor([self.n_ops]).to(self.device))
        for node_id in range(2):
            next_c, next_h = self.stack_lstm(inputs, prev_c, prev_h)
            prev_c, prev_h = next_c, next_h
            # self.anchors[node_id].copy_(torch.zeros_like(next_h[-1].view(-1)))
            self.anchors.append(torch.zeros_like(next_h[-1]))
            new_hidden = self.emb_attn(next_h[-1])
            # self.anchors_w_1[node_id].copy_(new_hidden.view(-1))
            self.anchors_w_1.append(new_hidden)
        return inputs, prev_c, prev_h

    def stack_lstm(self, x, prev_c, prev_h):
        next_c, next_h = [], []
        for layer_id, (_c, _h) in enumerate(zip(prev_c, prev_h)):
            inputs = x if layer_id == 0 else next_h[-1]
            curr_c, curr_h = self.lstm[layer_id](inputs, (_c, _h))
            # curr_c, curr_h = self.lstm(inputs, (_c, _h))
            next_c.append(curr_c)
            next_h.append(curr_h)
        return next_c, next_h

    def _body(self, node_id, inputs, prev_c, prev_h, arc_seq, entropy, log_prob):
        start_id = 0 if node_id==0 \
            else int(np.sum([(i+1) for i in range(node_id)]))
        num_ops = node_id + 1
        self.node_logits_ops_list = []

        for i in range(num_ops): # loop for ops
            next_c, next_h = self.stack_lstm(inputs, prev_c, prev_h)
            prev_c, prev_h = next_c, next_h
            logits = self.w_soft(next_h[-1]).view(-1)
            if self.temperature is not None:
                logits /= self.temperature
            if self.tanh_constant is not None:
                op_tanh = self.tanh_constant / self.op_tanh_reduce
                logits = op_tanh * self.tanh(logits)
            if self.force_uniform:
                probs = F.softmax(torch.zeros_like(logits), dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)
            self.node_logits_ops_list.append(logits)
            op_id = probs.multinomial(num_samples=1)

            log_probs = torch.log(probs)
            arc_seq[start_id + i] = op_id
            curr_log_prob = log_probs.gather(0, op_id)[0]
            log_prob += curr_log_prob
            curr_ent = -(log_probs * probs).sum()
            entropy += curr_ent
            inputs = self.w_emb(op_id)

        # TODO: why additional run one step forward?
        next_c, next_h = self.stack_lstm(inputs, prev_c, prev_h)
        # self.anchors[node_id+2].copy_(next_h[-1].view(-1))
        self.anchors.append(next_h[-1])
        new_hidden = self.emb_attn(next_h[-1])
        # self.anchors_w_1[node_id+2].copy_(new_hidden.view(-1))
        self.anchors_w_1.append(new_hidden)
        # inputs = self.g_emb(torch.LongTensor([0]))
        inputs = self.w_emb(torch.LongTensor([self.n_ops]).to(self.device))   # TODO: reset the inputs?
        return inputs, next_c, next_h, arc_seq, entropy, log_prob

    def _convert_lstm_output(self, prev_ops):
        """
        :param prev_nodes: vector, each element is the node ID, int64, in the range of [0,1,...,n_node]
        :param prev_ops: vector, each element is the op_id, int64, in the range [0,1,...,n_ops-1]
        :return: arch list, (op, f, t) is the elements
        """
        _count_list = [(i+1) for i in range(self.n_nodes)]
        assert len(prev_ops) == np.sum(_count_list)
        arch_list = []
        offset = 0
        for i in range(self.n_nodes):
            t = i+1
            for f in range(t):
                arch_list.append(( prev_ops[offset].item(), f, t))
                offset += 1

        return arch_list

    def _get_default_hidden(self, key):
        return utils.get_variable(
            torch.zeros(key, self.controller_hid), self.device, requires_grad=False)

    # device
    def init_hidden(self, batch_size):
        zeros = torch.zeros(batch_size, self.controller_hid)
        return (utils.get_variable(zeros, self.device, requires_grad=False),
                utils.get_variable(zeros.clone(), self.device, requires_grad=False))

    def reset_parameters(self):
        # params initialization
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        self.w_soft.bias.data.fill_(0)

    def _clear_arch(self, arch):
        new_arch = [arch[0]]
        for e in arch[1:]:
            if not (e == new_arch[-1]):
                new_arch.append(e)
        return new_arch


    def forward(self):
        if self.controller_type == 'ENAS':
            log_prob, entropy = 0, 0
            self.prev_nodes, self.prev_ops = [], []
            self.logits_ops_list = []
            arc_seq = [0] * np.sum([(i+1) for i in range(self.n_nodes)])

            inputs, prev_c, prev_h = self._init_nodes()

            for node_id in range(self.n_nodes):
                inputs, next_c, next_h, arc_seq, entropy, log_prob = \
                self._body(node_id, inputs, prev_c, prev_h,
                           arc_seq, entropy, log_prob)
                prev_c = next_c
                prev_h = next_h
                self.logits_ops_list.extend(self.node_logits_ops_list)

            for node_id in range(self.n_nodes):
                for i in range(node_id+1):
                    start_id = 0 if node_id == 0 \
                        else int(np.sum([(i + 1) for i in range(node_id)]))
                    self.prev_ops.append(arc_seq[start_id + i])

            arch = self._convert_lstm_output(torch.cat(self.prev_ops))
            self.logits_ops = torch.stack(tuple(self.logits_ops_list))

            arch = self._clear_arch(arch)
            return arch, log_prob, entropy
        else:
            assert False, "unsupported controller_type: %s" % self.controller_type


class MeWNetwork(nn.Module):
    def __init__(self, C, num_classes, layers, criterion, device,
                 steps=4, multiplier=4, stem_multiplier=3,
                 controller_type='LSTM', controller_hid=None,
                 controller_temperature=None, controller_tanh_constant=None,
                 controller_op_tanh_reduce=None, entropy_coeff=[0.0, 0.0],
                 dp_train=False):
        super(MeWNetwork, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps   # use this default value
        self._multiplier = multiplier
        self._device = device

        self.controller_type = controller_type
        self.controller_hid = controller_hid
        self.controller_temperature = controller_temperature
        self.controller_tanh_constant = controller_tanh_constant
        self.controller_op_tanh_reduce = controller_op_tanh_reduce
        self.entropy_coeff = entropy_coeff

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = MeWCell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev,
                           self.controller_type)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_archmaster()
        # for name, param in self.named_parameters():
        #     print(name)
        # print(self._arch_parameters)
        # assert False

    def _initialize_archmaster(self):
        if self.controller_type == "LSTM" or self.controller_type=="LSTM2" or self.controller_type=='ENAS':
            num_ops = len(LSTM_PRIMITIVES)
        else:
            assert False, "unsupported controller_type: %s" % self.controller_type
        self.arch_normal_master = ArchMaster(
            num_ops, self._steps, self._device, self.controller_type,
            self.controller_hid, self.controller_temperature,
            self.controller_tanh_constant, self.controller_op_tanh_reduce)
        self.arch_reduce_master = ArchMaster(
            num_ops, self._steps, self._device, self.controller_type,
            self.controller_hid, self.controller_temperature,
            self.controller_tanh_constant, self.controller_op_tanh_reduce)

        self._arch_parameters = list(self.arch_normal_master.parameters()) + list(self.arch_reduce_master.parameters())


    def _test_acc(self, test_queue, arch_normal, arch_reduce, return_loss=False):
        # TODO: go over all the testing data to obtain the accuracy
        top1 = utils.AvgrageMeter()
        objs = utils.AvgrageMeter()
        for step, (test_input, test_target) in enumerate(test_queue):
            test_input = test_input.to(self._device)
            test_target = test_target.to(self._device)
            n = test_input.size(0)
            with torch.no_grad():
                logits = self.forward(test_input, arch_normal, arch_reduce)
            loss = self._criterion(logits, test_target)
            accuracy = utils.accuracy(logits, test_target)[0]
            top1.update(accuracy.item(), n)
            objs.update(loss.item(), n)

        if return_loss:
            return top1.avg, objs.avg
        else:
            return top1.avg


    def test(self, test_queue, n_archs, logger, folder, suffix, benchmark_model=None):
        # test on n_archs
        best_acc = -np.inf
        best_acc_bm = -np.inf
        best_arch_normal_logP = None
        best_arch_reduce_logP = None
        best_arch_normal_ent = None
        best_arch_reduce_ent = None
        best_arch_normal = None
        best_arch_reduce = None
        for i in range(n_archs):     # test several archs
            arch_normal, arch_normal_logP, arch_normal_ent = self.arch_normal_master.forward()
            arch_reduce, arch_reduce_logP, arch_reduce_ent = self.arch_reduce_master.forward()
            top1 = self._test_acc(test_queue, arch_normal, arch_reduce)
            # if benchmark_model is not None:
            #     top1_benchmark, train_iterator = benchmark_model._test_acc(test_queue, train_queue, train_iterator, n_steps, arch_normal, arch_reduce)
            # else:
            top1_benchmark = 0

            logger.info('Candidate Arch#%d, Top1=%f, Top1_bm=%f, -LogP(NOR,RED)=%f(%f,%f), ENT(NOR,RED)=%f(%f,%f), NormalCell=%s, ReduceCell=%s',
                        i, top1, top1_benchmark,  -arch_normal_logP-arch_reduce_logP, -arch_normal_logP, arch_normal_ent,
                        arch_normal_ent+arch_reduce_ent, arch_normal_ent, arch_reduce_ent,
                        arch_normal,
                        arch_reduce)
            if top1 > best_acc:
                best_acc = top1
                best_acc_bm = top1_benchmark
                best_arch_normal = arch_normal
                best_arch_reduce = arch_reduce
                best_arch_normal_logP = arch_normal_logP
                best_arch_reduce_logP = arch_reduce_logP
                best_arch_normal_ent = arch_normal_ent
                best_arch_reduce_ent = arch_reduce_ent

        # TODO: draw best genotype, and logging genotype
        logger.info("Best: Accuracy %f Accuracy_bm %f -LogP %f ENT %f", best_acc, best_acc_bm, -best_arch_normal_logP-best_arch_reduce_logP, best_arch_normal_ent+best_arch_reduce_ent)
        logger.info("Normal: -logP %f, Entropy %f\n%s", -best_arch_normal_logP, best_arch_normal_ent, best_arch_normal)
        logger.info("Reduction: -logP %f, Entropy %f\n%s", -best_arch_reduce_logP, best_arch_reduce_ent, best_arch_reduce)
        genotype = arch_to_genotype(best_arch_normal, best_arch_reduce, self._steps, self.controller_type)
        draw_genotype(genotype.normal, self._steps, os.path.join(folder, "normal_%s" % suffix))
        draw_genotype(genotype.reduce, self._steps, os.path.join(folder, "reduce_%s" % suffix))
        logger.info('genotype = %s', genotype)

    def forward(self, input, arch_normal, arch_reduce):
        # arch_normal, arch_normal_logP, arch_normal_entropy = self.arch_normal_master.forward()
        # arch_reduce, arch_reduce_logP, arch_reduce_entropy = self.arch_reduce_master.forward()
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                archs = arch_reduce
            else:
                archs = arch_normal
            s0, s1 = s1, cell(s0, s1, archs)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits


    def step(self, input, target):
        # one step update of the controller
        arch_normal, arch_normal_logP, arch_normal_entropy = self.arch_normal_master.forward()  # see here, arch is generated in this step func.
        arch_reduce, arch_reduce_logP, arch_reduce_entropy = self.arch_reduce_master.forward()  # the model should contain the arch generate process so that each call of this step build a new graph
        # arch_normal does not requirs grad(discrete)

        self.optimizer.zero_grad()
        logits = self.forward(input, arch_normal, arch_reduce)
        loss = self._criterion(logits, target)
        loss.backward()
        self.optimizer.step()
        return logits, loss, arch_normal, arch_normal_logP, arch_normal_entropy, arch_reduce, arch_reduce_logP, arch_reduce_entropy


    def _loss_arch(self, input, target, baseline=None, epoch=0):
        arch_normal, arch_normal_logP, arch_normal_entropy = self.arch_normal_master.forward()  # see here, arch is generated in this step func.
        arch_reduce, arch_reduce_logP, arch_reduce_entropy = self.arch_reduce_master.forward()
        with torch.no_grad():
            logits = self.forward(input, arch_normal, arch_reduce)
        accuracy = utils.accuracy(logits, target)[0] / 100.0
        reward = accuracy - baseline if baseline else accuracy
        policy_loss = -(arch_normal_logP + arch_reduce_logP) * reward - (
        self.entropy_coeff[0] * arch_normal_entropy + self.entropy_coeff[1] * arch_reduce_entropy)
        # reward = reward + self.entropy_coeff * (arch_normal_entropy + arch_reduce_entropy)
        # action_likelihood * Reward. this is the policy-gradient. then the entropy part is delt with as usual
        return policy_loss, accuracy, arch_normal_entropy, arch_reduce_entropy


    def arch_parameters(self):
        return self._arch_parameters


    def model_parameters(self):
        for k, v in self.named_parameters():
            if 'arch' not in k:
                yield v


class MaxPool2x2(nn.Module):
    def __init__(self, reduction=True):
        super(MaxPool2x2, self).__init__()
        self.reduction = reduction

        self.layers = nn.Sequential(
            nn.MaxPool2d(2, 2),
        )

    def forward(self, s1, arch):
        return self.layers(s1)


class DPNetwork(nn.Module):
    def __init__(self, C, num_classes, layers, criterion, device,
                 steps=5, multiplier=5, stem_multiplier=3,
                 controller_type='LSTM', controller_hid=None,
                 controller_temperature=None, controller_tanh_constant=None,
                 controller_op_tanh_reduce=None, entropy_coeff=[0.0, 0.0],
                 args=None):
        super(DPNetwork, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = extend( criterion)
        self._steps = steps   # use this default value
        self._multiplier = multiplier
        self._device = device

        self._dp = args.private
        self._args = args

        self.controller_type = controller_type
        self.controller_hid = controller_hid
        self.controller_temperature = controller_temperature
        self.controller_tanh_constant = controller_tanh_constant
        self.controller_op_tanh_reduce = controller_op_tanh_reduce
        self.entropy_coeff = entropy_coeff

        C_prev = C_curr = stem_multiplier * C
        self.stem = extend( nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            GN(C_curr)
        ))

        cells = nn.ModuleList()
        for i in range(layers):
            if i in [1,3,5]:
                cell = MaxPool2x2()
            else:
                if i in [2,4]:
                    C_curr *= 2
                cell = PrivCell(steps, C_prev, C_curr, False,
                                self.controller_type)
                C_prev = multiplier * C_curr
            cells += [cell]

        self.cells = extend(cells)
        self.adapt = extend(nn.Sequential(
            nn.Conv2d(C_prev, 128, 1, bias=False),
            GN(128),
        ))
        self.classifier = extend(nn.Sequential(
            nn.Linear(128 *4*4, 10),
            nn.ReLU(),
            nn.Linear(128, 10)
        ))

        self._initialize_archmaster()


    def _initialize_archmaster(self):
        if self.controller_type == "LSTM" or self.controller_type=="LSTM2" or self.controller_type=='ENAS':
            num_ops = len(PRIV_PRIMITIVES)
        else:
            assert False, "unsupported controller_type: %s" % self.controller_type

        master = DPDARTSArchMaster

        self.arch_normal_master = master(
            num_ops, self._steps, self._device, self.controller_type,
            self.controller_hid, self.controller_temperature,
            self.controller_tanh_constant, self.controller_op_tanh_reduce)

        self._arch_parameters = list(self.arch_normal_master.parameters())

    def _test_acc(self, test_queue, arch_normal, arch_reduce, return_loss=False):
        # TODO: go over all the testing data to obtain the accuracy
        top1 = utils.AvgrageMeter()
        objs = utils.AvgrageMeter()
        for step, (test_input, test_target) in enumerate(test_queue):
            test_input = test_input.to(self._device)
            test_target = test_target.to(self._device)
            n = test_input.size(0)
            with torch.no_grad():
                logits = self.forward(test_input, arch_normal, None)
            loss = self._criterion(logits, test_target)
            if self._dp:
                loss /= n
            accuracy = utils.accuracy(logits, test_target)[0]
            top1.update(accuracy.item(), n)
            objs.update(loss.item(), n)

        if return_loss:
            return top1.avg, objs.avg
        else:
            return top1.avg

    def test(self, test_queue, n_archs, logger, folder, suffix, benchmark_model=None):
        # test on n_archs
        best_acc = -np.inf
        best_acc_bm = -np.inf
        best_arch_normal_logP = None
        best_arch_normal_ent = None
        best_arch_normal = None
        for i in range(n_archs):     # test several archs
            arch_normal, arch_normal_logP, arch_normal_ent = self.arch_normal_master.forward()
            top1 = self._test_acc(test_queue, arch_normal, None)

            top1_benchmark = 0

            logger.info('Candidate Arch#%d, Top1=%f, Top1_bm=%f, '
                        '-LogP(NOR)=%f, ENT(NOR)=%f, NormalCell=%s',
                        i, top1, top1_benchmark, -arch_normal_logP, arch_normal_ent, arch_normal,)
            if top1 > best_acc:
                best_acc = top1
                best_acc_bm = top1_benchmark
                best_arch_normal = arch_normal
                best_arch_normal_logP = arch_normal_logP
                best_arch_normal_ent = arch_normal_ent

        # TODO: draw best genotype, and logging genotype
        logger.info("Best: Accuracy %f Accuracy_bm %f -LogP %f ENT %f", best_acc, best_acc_bm, -best_arch_normal_logP, best_arch_normal_ent)
        logger.info("Normal: -logP %f, Entropy %f\n%s", -best_arch_normal_logP, best_arch_normal_ent, best_arch_normal)
        genotype = arch_to_genotype(best_arch_normal, best_arch_normal,
                                    self._steps, self.controller_type,
                                    private=self._dp)
        draw_genotype(genotype.normal, self._steps, os.path.join(folder, "normal_%s" % suffix))
        logger.info('genotype = %s', genotype)

    def forward(self, input, arch_normal, arch_reduce):
        s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            archs = None if cell.reduction else arch_normal
            s1 = cell(s1, archs)

        out = self.adapt(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def step(self, inputs, targets):
        arch_normal, arch_normal_logP, arch_normal_entropy = self.arch_normal_master.forward()

        batchsize = inputs.size(0)

        if self._dp:
            _clip = self._args.dp_clip
            _sigma = self._args.dp_sigma
            _micro_bs = 150

            if self._args.multi_forward:
                _steps = batchsize//_micro_bs if batchsize%_micro_bs==0 \
                    else batchsize//_micro_bs+1

                grad_sums = {}
                logits = []
                losses = 0

                ## sum grads
                for i in range(_steps):
                    _b, _e = i*_micro_bs, min((i+1)*_micro_bs, batchsize)
                    input = inputs[_b:_e]
                    target = targets[_b:_e]

                    self.optimizer.zero_grad()
                    logit = self.forward(input, arch_normal, None)
                    loss = self._criterion(logit, target)
                    logits.append(logit)
                    losses += loss

                    with backpack(BatchGrad()):
                        loss.backward()

                    all_norm = []
                    for p in self.model_parameters():
                        try:
                            g_i = p.grad_batch.reshape(p.grad_batch.shape[0], -1)
                        except:
                            continue
                        g_i_norm = g_i.norm(dim=-1).view(-1, 1)
                        all_norm.append(g_i_norm)
                    all_norm = torch.cat(all_norm, dim=-1).norm(dim=-1, keepdim=True)
                    clip_scale = torch.clamp(_clip / all_norm, max=1.0)
                    for name, p in self.named_parameters():
                        if 'arch' in name: continue
                        try:
                            g_i = p.grad_batch.reshape(p.grad_batch.size(0), -1)
                        except:
                            continue
                        g_clip = g_i * clip_scale
                        g_sum = torch.sum(g_clip, dim=0).reshape(p.shape)

                        if name in grad_sums.keys():
                            grad_sums[name] += g_sum
                        else:
                            grad_sums[name] = g_sum
                        del p.grad_batch

                ## add noise
                for name, p in self.named_parameters():
                    if not (name in grad_sums.keys()):
                        continue
                    noise = torch.randn_like(p) * _sigma * _clip
                    p.grad.data = (grad_sums[name] + noise) / batchsize

                logits = torch.cat(logits)

            else:
                self.optimizer.zero_grad()
                logits = self.forward(inputs, arch_normal, None)
                losses = self._criterion(logits, targets)
                with backpack(BatchGrad()):
                    losses.backward()

                all_norm = []
                for p in self.model_parameters():
                    try:
                        g_i = p.grad_batch.reshape(batchsize, -1)
                    except:
                        continue
                    g_i_norm = g_i.norm(dim=-1).view(-1, 1)
                    all_norm.append(g_i_norm)
                all_norm = torch.cat(all_norm, dim=-1).norm(dim=-1, keepdim=True)
                clip_scale = torch.clamp(_clip / all_norm, max=1.0)
                for p in self.model_parameters():
                    try:
                        g_i = p.grad_batch.reshape(batchsize, -1)
                    except:
                        continue
                    g_clip = g_i * clip_scale
                    g_sum = torch.sum(g_clip, dim=0).reshape(p.shape)
                    noise = torch.randn_like(p) * _sigma * _clip
                    p.grad.data = (g_sum + noise) / batchsize
                    del p.grad_batch

        else:
            self.optimizer.zero_grad()
            logits = self.forward(inputs, arch_normal, None)
            losses = self._criterion(logits, targets)
            losses.backward()
            try:
                for p in self.model_parameters():
                    del p.grad_batch
            except:
                pass

        self.optimizer.step()
        losses /= batchsize
        return logits, losses, \
               arch_normal, arch_normal_logP, arch_normal_entropy, \
               arch_normal, arch_normal_logP, arch_normal_entropy


    def _loss_arch(self, input, target, baseline=None, epoch=0):
        arch_normal, arch_normal_logP, arch_normal_entropy = self.arch_normal_master.forward()
        with torch.no_grad():
            logits = self.forward(input, arch_normal, None)
        accuracy = utils.accuracy(logits, target)[0] / 100.0
        reward = accuracy - baseline if baseline else accuracy
        policy_loss = -arch_normal_logP * reward - self.entropy_coeff[0] * arch_normal_entropy

        return policy_loss, accuracy, arch_normal_entropy, arch_normal_entropy


    def arch_parameters(self):
        return self._arch_parameters


    def model_parameters(self):
        for k, v in self.named_parameters():
            if 'arch' not in k:
                yield v
