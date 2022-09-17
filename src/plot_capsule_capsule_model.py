import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
from torch.nn.init import xavier_normal


class CapsModel(nn.Module):
    def __init__(self, args, label_dim, t_in, a_in, v_in):
        super(CapsModel, self).__init__()
        self.t_in = t_in
        self.a_in = a_in
        self.v_in = v_in
        self.label_dim = label_dim
        self.dim_capsule = args.dim_capsule
        self.routing = args.routing
        self.dropout = args.dropout
        self.dataset = args.dataset

        # encoding stage
        self.biLSTM_t = nn.LSTM(input_size=self.t_in, hidden_size=self.dim_capsule//2, batch_first=True, bidirectional=True)
        self.biLSTM_a = nn.LSTM(input_size=self.a_in, hidden_size=self.dim_capsule//2, batch_first=True, bidirectional=True)
        self.biLSTM_v = nn.LSTM(input_size=self.v_in, hidden_size=self.dim_capsule//2, batch_first=True, bidirectional=True)

        # fusion stage
        self.Wt = {}  # pose matrix for textual unimodal capsule
        self.Wa = {}  # pose matrix for audio unimodal capsule
        self.Wv = {}  # pose matrix for video unimodal capsule
        for i in range(4):
            self.Wt[i] = nn.Parameter(torch.Tensor(self.dim_capsule, self.dim_capsule))
            self.Wa[i] = nn.Parameter(torch.Tensor(self.dim_capsule, self.dim_capsule))
            self.Wv[i] = nn.Parameter(torch.Tensor(self.dim_capsule, self.dim_capsule))
        self.lstm_ta = nn.LSTM(input_size=self.dim_capsule, hidden_size=self.dim_capsule, batch_first=True, dropout=self.dropout)
        self.lstm_tv = nn.LSTM(input_size=self.dim_capsule, hidden_size=self.dim_capsule, batch_first=True, dropout=self.dropout)
        self.lstm_av = nn.LSTM(input_size=self.dim_capsule, hidden_size=self.dim_capsule, batch_first=True, dropout=self.dropout)
        self.lstm_tav = nn.LSTM(input_size=self.dim_capsule, hidden_size=self.dim_capsule, batch_first=True, dropout=self.dropout)
        self.lstm_deci = nn.LSTM(input_size=self.dim_capsule, hidden_size=self.dim_capsule, batch_first=True, bidirectional=True)

        # decoding stage
        self.fc1 = nn.Linear(in_features=self.dim_capsule, out_features=self.dim_capsule//2)
        self.fc2 = nn.Linear(in_features=self.dim_capsule//2, out_features=self.label_dim)
        for i in range(4):
            xavier_normal(self.Wt[i])
            xavier_normal(self.Wa[i])
            xavier_normal(self.Wv[i])

    def forward(self, text, audio, video, batch_size):
        # encoding stage
        dump, (tuc, c_tuc) = self.biLSTM_t(text)  # tuc means text unimodal capsule
        dump, (auc, c_auc) = self.biLSTM_a(audio)  # auc means audio unimodal capsule
        dump, (vuc, c_vuc) = self.biLSTM_v(video)  # vuc means video unimodal capsule
        tuc = torch.tanh(tuc.permute(1, 0, 2))    # the output dimension from LSTM is (2, batch, dim_capsule//2)
        # tuc = F.elu(tuc.permute(1, 0, 2))
        tuc = tuc.reshape(batch_size, 1, self.dim_capsule)  # concatenate two direction
        auc = torch.tanh(auc.permute(1, 0, 2))
        # auc = F.elu(auc.permute(1, 0, 2))
        auc = auc.reshape(batch_size, 1, self.dim_capsule)
        vuc = torch.tanh(vuc.permute(1, 0, 2))
        # vuc = F.elu(vuc.permute(1, 0, 2))
        vuc = vuc.reshape(batch_size, 1, self.dim_capsule)

        # fusion stage
        tusc = {}  # textual unimodal sub-capsule
        ausc = {}
        vusc = {}

        capsule_to_save = {}

        rc = {}    # routing coefficient, 0 for ta, 1 for tv, 2 for av, 3 for tav, 4 for decision
        for i in range(4):
            tusc[i] = torch.matmul(tuc, self.Wt[i])
            ausc[i] = torch.matmul(auc, self.Wa[i])
            vusc[i] = torch.matmul(vuc, self.Wv[i])
        ta_concat = torch.cat([tusc[0], ausc[0]], 1)   # concat sub_capsule to implement fusion
        tv_concat = torch.cat([tusc[1], vusc[0]], 1)
        av_concat = torch.cat([ausc[1], vusc[1]], 1)
        tav_concat = torch.cat([tusc[2], ausc[2], vusc[2]], 1)
        for r in range(self.routing + 1):
            if r == 0:
                rc[0] = torch.ones(batch_size, 2, self.dim_capsule)
                rc[1] = torch.ones(batch_size, 2, self.dim_capsule)
                rc[2] = torch.ones(batch_size, 2, self.dim_capsule)
                rc[3] = torch.ones(batch_size, 3, self.dim_capsule)
                rc[4] = torch.ones(batch_size, 7, self.dim_capsule)
            for i in range(5):
                rc[i] = F.softmax(rc[i], 1)
            dump, (bc_ta, c_dump) = self.lstm_ta(rc[0] * ta_concat)   # bimodal capsule
            bc_ta = torch.tanh(bc_ta.permute(1, 0, 2))
            dump, (bc_tv, c_dump) = self.lstm_tv(rc[1] * tv_concat)
            bc_tv = torch.tanh(bc_tv.permute(1, 0, 2))
            dump, (bc_av, c_dump) = self.lstm_av(rc[2] * av_concat)
            bc_av = torch.tanh(bc_av.permute(1, 0, 2))
            dump, (tc_tav, c_dump) = self.lstm_tav(rc[3] * tav_concat)  # trimodal capsule
            tc_tav = torch.tanh(tc_tav.permute(1, 0, 2))
            deci_concat = torch.cat([tusc[3], ausc[3], vusc[3], bc_ta, bc_tv, bc_av, tc_tav], 1)
            dump, (dc, c_dump) = self.lstm_deci(rc[4] * deci_concat)
            dc = torch.tanh(dc.permute(1, 0, 2))
            dc = torch.sum(dc, 1)
            dc = dc.unsqueeze(1)

            capsule_to_save[r] = torch.cat([deci_concat, dc], 1)

            # routing mechanism
            if r < self.routing:
                last = rc[4]
                rc[4] = torch.bmm(deci_concat, dc.permute(0, 2, 1))  # inner-product
                rc[4] = last + rc[4].repeat(1, 1, self.dim_capsule)
                last = rc[3]
                rc[3] = torch.bmm(tav_concat, tc_tav.permute(0, 2, 1))
                rc[3] = last + rc[3].repeat(1, 1, self.dim_capsule)
                last = rc[2]
                rc[2] = torch.bmm(av_concat, bc_av.permute(0, 2, 1))
                rc[2] = last + rc[2].repeat(1, 1, self.dim_capsule)
                last = rc[1]
                rc[1] = torch.bmm(tv_concat, bc_tv.permute(0, 2, 1))
                rc[1] = last + rc[1].repeat(1, 1, self.dim_capsule)
                last = rc[0]
                rc[0] = torch.bmm(ta_concat, bc_ta.permute(0, 2, 1))
                rc[0] = last + rc[0].repeat(1, 1, self.dim_capsule)

        # decoding stage
        dc = dc.squeeze(1)
        output1 = torch.tanh(self.fc1(dc))
        if self.dataset == 'iemocap':
            preds = torch.tanh(self.fc2(output1))
        elif self.dataset == 'mosei_senti':
            preds = self.fc2(output1)
        return preds, capsule_to_save








