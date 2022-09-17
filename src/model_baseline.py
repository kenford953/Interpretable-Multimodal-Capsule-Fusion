import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
from torch.nn.init import xavier_normal


class EF_LSTM(nn.Module):
    def __init__(self, hidden_size_in_lstm, label_dim, t_in, a_in, v_in):
        super(EF_LSTM, self).__init__()
        self.t_in = t_in
        self.a_in = a_in
        self.v_in = v_in
        self.label_dim = label_dim
        self.lstm = nn.LSTM(input_size=self.t_in+self.a_in+self.v_in, hidden_size=hidden_size_in_lstm, batch_first=True)
        self.fc1 = nn.Linear(in_features=hidden_size_in_lstm, out_features=hidden_size_in_lstm // 2)
        self.fc2 = nn.Linear(in_features=hidden_size_in_lstm // 2, out_features=self.label_dim)

    def forward(self, text, audio, video, batch_size):
        fusion = torch.cat([text, audio, video], 2)
        _, (fusion, _) = self.lstm(fusion)
        fusion = fusion.permute(1, 0, 2).squeeze(1)
        output1 = torch.tanh(self.fc1(fusion))
        preds = torch.tanh(self.fc2(output1))
        return preds


class LF_LSTM(nn.Module):
    def __init__(self, hidden_size_in_lstm, label_dim, t_in, a_in, v_in):
        super(LF_LSTM, self).__init__()
        self.lstm_t = nn.LSTM(input_size=t_in, hidden_size=hidden_size_in_lstm, batch_first=True)
        self.lstm_a = nn.LSTM(input_size=a_in, hidden_size=hidden_size_in_lstm, batch_first=True)
        self.lstm_v = nn.LSTM(input_size=v_in, hidden_size=hidden_size_in_lstm, batch_first=True)
        self.fc1 = nn.Linear(in_features=3 * hidden_size_in_lstm, out_features=3 * hidden_size_in_lstm // 2)
        self.fc2 = nn.Linear(in_features=3 * hidden_size_in_lstm // 2, out_features=label_dim)

    def forward(self, text, audio, video, batch_size):
        _, (deci_t, _) = self.lstm_t(text)
        deci_t = deci_t.permute(1, 0, 2).squeeze(1)
        _, (deci_a, _) = self.lstm_a(audio)
        deci_a = deci_a.permute(1, 0, 2).squeeze(1)
        _, (deci_v, _) = self.lstm_v(video)
        deci_v = deci_v.permute(1, 0, 2).squeeze(1)
        deci = torch.cat([deci_t, deci_a, deci_v], -1)
        output1 = torch.tanh(self.fc1(deci))
        preds = torch.tanh(self.fc2(output1))
        return preds


class RAVEN(nn.Module):
    def __init__(self, hidden_a, hidden_v, post_fusion_prob, post_fusion_dim, label_dim, t_in, a_in, v_in):
        super(RAVEN, self).__init__()
        self.orig_d_a = a_in
        self.orig_d_l = t_in
        self.orig_d_v = v_in
        self.hidden_a = hidden_a
        self.hidden_v = hidden_v
        self.post_fusion_prob = post_fusion_prob
        self.post_fusion_dim = post_fusion_dim
        self.output_dim = label_dim

        self.wa = nn.Parameter(torch.Tensor(self.orig_d_l + self.hidden_a, self.orig_d_l).cuda())
        self.ba = nn.Parameter(torch.Tensor(self.orig_d_l, ).cuda())
        self.wv = nn.Parameter(torch.Tensor(self.orig_d_l + self.hidden_v, self.orig_d_l).cuda())
        self.bv = nn.Parameter(torch.Tensor(self.orig_d_l,).cuda())
        self.bh = nn.Parameter(torch.Tensor(self.orig_d_l, ).cuda())

        self.wa2 = nn.Parameter(torch.Tensor(self.hidden_a, self.orig_d_l).cuda())
        self.wv2 = nn.Parameter(torch.Tensor(self.hidden_v, self.orig_d_l).cuda())

        xavier_normal(self.wa)
        xavier_normal(self.wv)
        xavier_normal(self.wa2)
        xavier_normal(self.wv2)

        self.LSTM = nn.LSTM(input_size=self.orig_d_l, hidden_size=100, num_layers=1, \
                            bias=True, batch_first=True, dropout=0, bidirectional=False)

        self.LSTM_a = nn.LSTM(input_size=self.orig_d_a, hidden_size=self.hidden_a, num_layers=1, \
                              bias=True, batch_first=True, dropout=0, bidirectional=False)

        self.LSTM_v = nn.LSTM(input_size=self.orig_d_v, hidden_size=self.hidden_v, num_layers=1, \
                              bias=True, batch_first=True, dropout=0, bidirectional=False)

        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.post_fusion_layer_1 = nn.Linear(100, self.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, self.output_dim)

    def forward(self, text_x, audio_x, video_x, batch_size):
        # print(x.shape)
        # if (len(x.shape) == 4):
        #     x = x.squeeze(0)
        # if (len(x.shape) == 2):
        #     x = x.unsqueeze(0)

        # text_x = x[:, :, :self.orig_d_l]
        # audio_x = x[:, :, self.orig_d_l:self.orig_d_l + self.orig_d_a]

        acoustic_feature = self.LSTM_a(audio_x)[0]
        visual_feature = self.LSTM_v(video_x)[0]
        wa = F.sigmoid(torch.matmul(torch.cat([acoustic_feature, text_x], dim=-1), self.wa) + self.ba)

        wv = F.sigmoid(torch.matmul(torch.cat([visual_feature, text_x], dim=-1), self.wv) + self.bv)

        hm = wa * torch.matmul(acoustic_feature, self.wa2) + self.bh + wv * torch.matmul(visual_feature, self.wv2)
        aa = torch.norm(text_x, p=2, dim=2) / torch.norm(hm, p=2, dim=2)

        for i in range(text_x.shape[0]):
            for j in range(text_x.shape[1]):
                if aa[i, j] > 1:
                    aa[i, j] = 1
        em = text_x + aa.unsqueeze(2) * hm
        U, _ = self.LSTM(em)[1]

        post_fusion_dropped = self.post_fusion_dropout(U.squeeze(0))
        post_fusion_y_1 = F.relu(self.post_fusion_layer_1(post_fusion_dropped))
        post_fusion_y_2 = F.relu(self.post_fusion_layer_2(post_fusion_y_1))
        post_fusion_y_3 = (self.post_fusion_layer_3(post_fusion_y_2))
        output = post_fusion_y_3
        output = output.view(-1, self.output_dim)

        return output


class ATT(nn.Module):
    def __init__(self, dim_capsule, label_dim, t_in, a_in, v_in, args):
        super(ATT, self).__init__()
        self.t_in = t_in
        self.a_in = a_in
        self.v_in = v_in
        self.label_dim = label_dim
        self.dim_capsule = dim_capsule
        self.dataset = args.dataset

        # encoding stage
        self.biLSTM_t = nn.LSTM(input_size=self.t_in, hidden_size=self.dim_capsule//2, batch_first=True, bidirectional=True)
        self.biLSTM_a = nn.LSTM(input_size=self.a_in, hidden_size=self.dim_capsule//2, batch_first=True, bidirectional=True)
        self.biLSTM_v = nn.LSTM(input_size=self.v_in, hidden_size=self.dim_capsule//2, batch_first=True, bidirectional=True)

        self.Wt_a = nn.Parameter(torch.Tensor(self.dim_capsule, self.dim_capsule))
        self.Wt_v = nn.Parameter(torch.Tensor(self.dim_capsule, self.dim_capsule))
        self.Wt_av = nn.Parameter(torch.Tensor(self.dim_capsule, self.dim_capsule))
        self.Wt_deci = nn.Parameter(torch.Tensor(self.dim_capsule, self.dim_capsule))

        self.Wa_t = nn.Parameter(torch.Tensor(self.dim_capsule, self.dim_capsule))
        self.Wa_v = nn.Parameter(torch.Tensor(self.dim_capsule, self.dim_capsule))
        self.Wa_tv = nn.Parameter(torch.Tensor(self.dim_capsule, self.dim_capsule))
        self.Wa_deci = nn.Parameter(torch.Tensor(self.dim_capsule, self.dim_capsule))

        self.Wv_t = nn.Parameter(torch.Tensor(self.dim_capsule, self.dim_capsule))
        self.Wv_a = nn.Parameter(torch.Tensor(self.dim_capsule, self.dim_capsule))
        self.Wv_ta = nn.Parameter(torch.Tensor(self.dim_capsule, self.dim_capsule))
        self.Wv_deci = nn.Parameter(torch.Tensor(self.dim_capsule, self.dim_capsule))

        self.Wta_deci = nn.Parameter(torch.Tensor(self.dim_capsule, self.dim_capsule))
        self.Wtv_deci = nn.Parameter(torch.Tensor(self.dim_capsule, self.dim_capsule))
        self.Wav_deci = nn.Parameter(torch.Tensor(self.dim_capsule, self.dim_capsule))
        self.Wtav_deci = nn.Parameter(torch.Tensor(self.dim_capsule, self.dim_capsule))

        # decoding stage
        self.fc1 = nn.Linear(in_features=self.dim_capsule, out_features=self.dim_capsule//2)
        self.fc2 = nn.Linear(in_features=self.dim_capsule//2, out_features=self.label_dim)

        xavier_normal(self.Wt_a)
        xavier_normal(self.Wt_v)
        xavier_normal(self.Wt_av)
        xavier_normal(self.Wt_deci)
        xavier_normal(self.Wa_t)
        xavier_normal(self.Wa_v)
        xavier_normal(self.Wa_tv)
        xavier_normal(self.Wa_deci)
        xavier_normal(self.Wv_t)
        xavier_normal(self.Wv_a)
        xavier_normal(self.Wv_ta)
        xavier_normal(self.Wv_deci)
        xavier_normal(self.Wta_deci)
        xavier_normal(self.Wtv_deci)
        xavier_normal(self.Wav_deci)
        xavier_normal(self.Wtav_deci)

    def forward(self, text, audio, video, batch_size):
        _, (text_context, _) = self.biLSTM_t(text)  # text_context means features containing contextual information
        _, (audio_context, _) = self.biLSTM_a(audio)
        _, (video_context, _) = self.biLSTM_v(video)
        text_context = text_context.permute(1, 0, 2).reshape(batch_size, 1, self.dim_capsule)
        audio_context = audio_context.permute(1, 0, 2).reshape(batch_size, 1, self.dim_capsule)
        video_context = video_context.permute(1, 0, 2).reshape(batch_size, 1, self.dim_capsule)
        # the output dimension from LSTM is (2, batch, dim_capsule//2)

        att_t_a = (torch.bmm(torch.matmul(text_context, self.Wt_a), audio_context.permute(0, 2, 1))).squeeze(-1)
        att_t_v = (torch.bmm(torch.matmul(text_context, self.Wt_v), video_context.permute(0, 2, 1))).squeeze(-1)
        att_t_av = (torch.bmm(torch.matmul(text_context, self.Wt_av), torch.cat([audio_context.permute(0, 2, 1), video_context.permute(0, 2, 1)], -1))).sum(-1)

        att_a_t = (torch.bmm(torch.matmul(audio_context, self.Wa_t), text_context.permute(0, 2, 1))).squeeze(-1)
        att_a_v = (torch.bmm(torch.matmul(audio_context, self.Wa_v), video_context.permute(0, 2, 1))).squeeze(-1)
        att_a_tv = (torch.bmm(torch.matmul(audio_context, self.Wa_tv), torch.cat([text_context.permute(0, 2, 1), video_context.permute(0, 2, 1)], -1))).sum(-1)

        att_v_t = (torch.bmm(torch.matmul(video_context, self.Wv_t), text_context.permute(0, 2, 1))).squeeze(-1)
        att_v_a = (torch.bmm(torch.matmul(video_context, self.Wv_a), audio_context.permute(0, 2, 1))).squeeze(-1)
        att_v_ta = (torch.bmm(torch.matmul(video_context, self.Wv_ta), torch.cat([text_context.permute(0, 2, 1), audio_context.permute(0, 2, 1)], -1))).sum(-1)


        inter_ta = att_t_a * text_context.squeeze(1) + att_a_t * audio_context.squeeze(1)
        inter_tv = att_t_v * text_context.squeeze(1) + att_v_t * video_context.squeeze(1)
        inter_av = att_a_v * audio_context.squeeze(1) + att_v_a * video_context.squeeze(1)
        inter_tav = att_t_av * text_context.squeeze(1) + att_a_tv * audio_context.squeeze(1) + att_v_ta * video_context.squeeze(1)

        att_t_deci = (torch.bmm(torch.matmul(text_context, self.Wt_deci), torch.cat([audio_context.permute(0, 2, 1),
                                                                                     video_context.permute(0, 2, 1),
                                                                                     inter_tav.unsqueeze(-1),
                                                                                     inter_ta.unsqueeze(-1),
                                                                                     inter_tv.unsqueeze(-1),
                                                                                     inter_av.unsqueeze(-1)], -1))).sum(-1)
        att_a_deci = (torch.bmm(torch.matmul(audio_context, self.Wa_deci), torch.cat([text_context.permute(0, 2, 1),
                                                                                      video_context.permute(0, 2, 1),
                                                                                      inter_tav.unsqueeze(-1),
                                                                                      inter_ta.unsqueeze(-1),
                                                                                      inter_tv.unsqueeze(-1),
                                                                                      inter_av.unsqueeze(-1)], -1))).sum(-1)
        att_v_deci = (torch.bmm(torch.matmul(video_context, self.Wv_deci), torch.cat([audio_context.permute(0, 2, 1),
                                                                                      text_context.permute(0, 2, 1),
                                                                                      inter_tav.unsqueeze(-1),
                                                                                      inter_ta.unsqueeze(-1),
                                                                                      inter_tv.unsqueeze(-1),
                                                                                      inter_av.unsqueeze(-1)], -1))).sum(-1)
        att_ta_deci = (torch.bmm(torch.matmul(inter_ta.unsqueeze(1), self.Wta_deci), torch.cat([audio_context.permute(0, 2, 1),
                                                                                                video_context.permute(0, 2, 1),
                                                                                                inter_tav.unsqueeze(-1),
                                                                                                text_context.permute(0, 2, 1),
                                                                                                inter_tv.unsqueeze(-1),
                                                                                                inter_av.unsqueeze(-1)], -1))).sum(-1)
        att_tv_deci = (torch.bmm(torch.matmul(inter_tv.unsqueeze(1), self.Wtv_deci), torch.cat([audio_context.permute(0, 2, 1),
                                                                                                video_context.permute(0, 2, 1),
                                                                                                inter_tav.unsqueeze(-1),
                                                                                                text_context.permute(0, 2, 1),
                                                                                                inter_ta.unsqueeze(-1),
                                                                                                inter_av.unsqueeze(-1)], -1))).sum(-1)
        att_av_deci = (torch.bmm(torch.matmul(inter_av.unsqueeze(1), self.Wav_deci), torch.cat([audio_context.permute(0, 2, 1),
                                                                                                video_context.permute(0, 2, 1),
                                                                                                inter_tav.unsqueeze(-1),
                                                                                                text_context.permute(0, 2, 1),
                                                                                                inter_tv.unsqueeze(-1),
                                                                                                inter_ta.unsqueeze(-1)], -1))).sum(-1)
        att_tav_deci = (torch.bmm(torch.matmul(inter_tav.unsqueeze(1), self.Wtav_deci), torch.cat([audio_context.permute(0, 2, 1),
                                                                                                   video_context.permute(0, 2, 1),
                                                                                                   inter_ta.unsqueeze(-1),
                                                                                                   text_context.permute(0, 2, 1),
                                                                                                   inter_tv.unsqueeze(-1),
                                                                                                   inter_av.unsqueeze(-1)], -1))).sum(-1)
        att_t = {'att_t_a':att_t_a, 'att_t_v':att_t_v, 'att_t_av':att_t_av, 'att_t_deci':att_t_deci}
        att_a = {'att_a_t':att_a_t, 'att_a_v':att_a_v, 'att_a_tv':att_a_tv, 'att_a_deci':att_a_deci}
        att_v = {'att_v_t':att_v_t, 'att_v_a':att_v_a, 'att_v_ta':att_v_ta, 'att_v_deci':att_v_deci}
        att_inter = {'att_ta_deci':att_ta_deci, 'att_tv_deci':att_tv_deci, 'att_av_deci':att_av_deci, 'att_tav_deci':att_tav_deci}
        deci = att_t_deci * text_context.squeeze(1) + att_a_deci * audio_context.squeeze(1) + att_v_deci * video_context.squeeze(1) \
               + att_ta_deci * inter_ta + att_tv_deci * inter_tv + att_av_deci * inter_av + att_tav_deci * inter_tav

        output1 = torch.tanh(self.fc1(deci))
        if self.dataset == 'iemocap':
            preds = torch.tanh(self.fc2(output1))
        elif self.dataset in ['mosi', 'mosei_senti']:
            preds = self.fc2(output1)
        return preds, att_t, att_a, att_v, att_inter


class WOBC(nn.Module):
    def __init__(self, args, dim_capsule, routing, dropout, label_dim, t_in, a_in, v_in):
        super(WOBC, self).__init__()
        self.t_in = t_in
        self.a_in = a_in
        self.v_in = v_in
        self.label_dim = label_dim
        self.dim_capsule = dim_capsule
        self.routing = routing
        self.dropout = dropout
        self.dataset = args.dataset
        # self.device = device

        # encoding stage
        self.biLSTM_t = nn.LSTM(input_size=self.t_in, hidden_size=self.dim_capsule//2, batch_first=True, bidirectional=True)
        self.biLSTM_a = nn.LSTM(input_size=self.a_in, hidden_size=self.dim_capsule//2, batch_first=True, bidirectional=True)
        self.biLSTM_v = nn.LSTM(input_size=self.v_in, hidden_size=self.dim_capsule//2, batch_first=True, bidirectional=True)

        # fusion stage
        self.Wt = {}  # pose matrix for textual unimodal capsule
        self.Wa = {}  # pose matrix for audio unimodal capsule
        self.Wv = {}  # pose matrix for video unimodal capsule
        for i in range(2):
            self.Wt[i] = nn.Parameter(torch.Tensor(self.dim_capsule, self.dim_capsule))
            self.Wa[i] = nn.Parameter(torch.Tensor(self.dim_capsule, self.dim_capsule))
            self.Wv[i] = nn.Parameter(torch.Tensor(self.dim_capsule, self.dim_capsule))
        # self.lstm_ta = nn.LSTM(input_size=self.dim_capsule, hidden_size=self.dim_capsule, batch_first=True, dropout=self.dropout)
        # self.lstm_tv = nn.LSTM(input_size=self.dim_capsule, hidden_size=self.dim_capsule, batch_first=True, dropout=self.dropout)
        # self.lstm_av = nn.LSTM(input_size=self.dim_capsule, hidden_size=self.dim_capsule, batch_first=True, dropout=self.dropout)
        self.lstm_tav = nn.LSTM(input_size=self.dim_capsule, hidden_size=self.dim_capsule, batch_first=True, dropout=self.dropout)
        self.lstm_deci = nn.LSTM(input_size=self.dim_capsule, hidden_size=self.dim_capsule, batch_first=True, bidirectional=True)

        # decoding stage
        self.fc1 = nn.Linear(in_features=self.dim_capsule, out_features=self.dim_capsule//2)
        self.fc2 = nn.Linear(in_features=self.dim_capsule//2, out_features=self.label_dim)
        for i in range(2):
            xavier_normal(self.Wt[i])
            xavier_normal(self.Wa[i])
            xavier_normal(self.Wv[i])

    def forward(self, text, audio, video, batch_size):
        # encoding stage
        _, (text_context, _) = self.biLSTM_t(text)    # text_context means features containing contextual information
        _, (audio_context, _) = self.biLSTM_a(audio)
        _, (video_context, _) = self.biLSTM_v(video)
        text_context = text_context.permute(1, 0, 2).reshape(batch_size, 1, self.dim_capsule)
        audio_context = audio_context.permute(1, 0, 2).reshape(batch_size, 1, self.dim_capsule)
        video_context = video_context.permute(1, 0, 2).reshape(batch_size, 1, self.dim_capsule)
        # the output dimension from LSTM is (2, batch, dim_capsule//2)

        # fusion stage
        tusc = {}  # textual unimodal sub-capsule
        ausc = {}
        vusc = {}
        rc = {}    # routing coefficient, 0 for ta, 1 for tv, 2 for av, 3 for tav, 4 for decision
        for i in range(2):
            tusc[i] = torch.matmul(text_context, self.Wt[i])
            ausc[i] = torch.matmul(audio_context, self.Wa[i])
            vusc[i] = torch.matmul(video_context, self.Wv[i])
        # ta_concat = torch.cat([tusc[0], ausc[0]], 1)   # concat sub_capsule to implement fusion
        # tv_concat = torch.cat([tusc[1], vusc[0]], 1)
        # av_concat = torch.cat([ausc[1], vusc[1]], 1)
        tav_concat = torch.cat([tusc[0], ausc[0], vusc[0]], 1)

        # rc[0] = torch.ones(batch_size, 2, self.dim_capsule)
        # rc[1] = torch.ones(batch_size, 2, self.dim_capsule)
        # rc[2] = torch.ones(batch_size, 2, self.dim_capsule)
        rc[0] = torch.ones(batch_size, 3, self.dim_capsule)
        rc[1] = torch.ones(batch_size, 4, self.dim_capsule)

        for r in range(self.routing + 1):
            for i in range(2):
                rc[i] = F.softmax(rc[i], 1)
            # dump, (bc_ta, c_dump) = self.lstm_ta(rc[0] * ta_concat)   # bimodal capsule
            # bc_ta = bc_ta.permute(1, 0, 2)
            # dump, (bc_tv, c_dump) = self.lstm_tv(rc[1] * tv_concat)
            # bc_tv = bc_tv.permute(1, 0, 2)
            # dump, (bc_av, c_dump) = self.lstm_av(rc[2] * av_concat)
            # bc_av = bc_av.permute(1, 0, 2)
            dump, (tc_tav, c_dump) = self.lstm_tav(rc[0] * tav_concat)  # trimodal capsule
            tc_tav = tc_tav.permute(1, 0, 2)
            deci_concat = torch.cat([tusc[0], ausc[0], vusc[0], tc_tav], 1)
            dump, (dc, c_dump) = self.lstm_deci(rc[1] * deci_concat)
            dc = dc.permute(1, 0, 2)
            dc = torch.sum(dc, 1)
            dc = dc.unsqueeze(1)

            # routing mechanism
            if r < self.routing:
                last = rc[1]
                rc[1] = torch.bmm(deci_concat, dc.permute(0, 2, 1))  # inner-product
                rc[1] = last + rc[1].repeat(1, 1, self.dim_capsule)
                last = rc[0]
                rc[0] = torch.bmm(tav_concat, tc_tav.permute(0, 2, 1))
                rc[0] = last + rc[0].repeat(1, 1, self.dim_capsule)
                # last = rc[2]
                # rc[2] = torch.bmm(av_concat, bc_av.permute(0, 2, 1))
                # rc[2] = last + rc[2].repeat(1, 1, self.dim_capsule)
                # last = rc[1]
                # rc[1] = torch.bmm(tv_concat, bc_tv.permute(0, 2, 1))
                # rc[1] = last + rc[1].repeat(1, 1, self.dim_capsule)
                # last = rc[0]
                # rc[0] = torch.bmm(ta_concat, bc_ta.permute(0, 2, 1))
                # rc[0] = last + rc[0].repeat(1, 1, self.dim_capsule)

        # decoding stage
        dc = dc.squeeze(1)
        output1 = torch.tanh(self.fc1(dc))
        if self.dataset == 'iemocap':
            preds = torch.tanh(self.fc2(output1))
        elif self.dataset in ['mosi', 'mosei_senti']:
            preds = self.fc2(output1)
        return preds


class WOUC(nn.Module):
    def __init__(self, args, dim_capsule, routing, dropout, label_dim, t_in, a_in, v_in):
        super(WOUC, self).__init__()
        self.t_in = t_in
        self.a_in = a_in
        self.v_in = v_in
        self.label_dim = label_dim
        self.dim_capsule = dim_capsule
        self.routing = routing
        self.dropout = dropout
        self.dataset = args.dataset
        # self.device = device

        # encoding stage
        self.biLSTM_t = nn.LSTM(input_size=self.t_in, hidden_size=self.dim_capsule//2, batch_first=True, bidirectional=True)
        self.biLSTM_a = nn.LSTM(input_size=self.a_in, hidden_size=self.dim_capsule//2, batch_first=True, bidirectional=True)
        self.biLSTM_v = nn.LSTM(input_size=self.v_in, hidden_size=self.dim_capsule//2, batch_first=True, bidirectional=True)

        # fusion stage
        self.Wt = {}  # pose matrix for textual unimodal capsule
        self.Wa = {}  # pose matrix for audio unimodal capsule
        self.Wv = {}  # pose matrix for video unimodal capsule
        for i in range(3):
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
        for i in range(3):
            xavier_normal(self.Wt[i])
            xavier_normal(self.Wa[i])
            xavier_normal(self.Wv[i])

    def forward(self, text, audio, video, batch_size):
        # encoding stage
        _, (text_context, _) = self.biLSTM_t(text)    # text_context means features containing contextual information
        _, (audio_context, _) = self.biLSTM_a(audio)
        _, (video_context, _) = self.biLSTM_v(video)
        text_context = text_context.permute(1, 0, 2).reshape(batch_size, 1, self.dim_capsule)
        audio_context = audio_context.permute(1, 0, 2).reshape(batch_size, 1, self.dim_capsule)
        video_context = video_context.permute(1, 0, 2).reshape(batch_size, 1, self.dim_capsule)
        # the output dimension from LSTM is (2, batch, dim_capsule//2)

        # fusion stage
        tusc = {}  # textual unimodal sub-capsule
        ausc = {}
        vusc = {}
        rc = {}    # routing coefficient, 0 for ta, 1 for tv, 2 for av, 3 for tav, 4 for decision
        for i in range(3):
            tusc[i] = torch.matmul(text_context, self.Wt[i])
            ausc[i] = torch.matmul(audio_context, self.Wa[i])
            vusc[i] = torch.matmul(video_context, self.Wv[i])
        ta_concat = torch.cat([tusc[0], ausc[0]], 1)   # concat sub_capsule to implement fusion
        tv_concat = torch.cat([tusc[1], vusc[0]], 1)
        av_concat = torch.cat([ausc[1], vusc[1]], 1)
        tav_concat = torch.cat([tusc[2], ausc[2], vusc[2]], 1)

        rc[0] = torch.ones(batch_size, 2, self.dim_capsule)
        rc[1] = torch.ones(batch_size, 2, self.dim_capsule)
        rc[2] = torch.ones(batch_size, 2, self.dim_capsule)
        rc[3] = torch.ones(batch_size, 3, self.dim_capsule)
        rc[4] = torch.ones(batch_size, 4, self.dim_capsule)

        for r in range(self.routing + 1):
            for i in range(5):
                rc[i] = F.softmax(rc[i], 1)
            dump, (bc_ta, c_dump) = self.lstm_ta(rc[0] * ta_concat)   # bimodal capsule
            bc_ta = bc_ta.permute(1, 0, 2)
            dump, (bc_tv, c_dump) = self.lstm_tv(rc[1] * tv_concat)
            bc_tv = bc_tv.permute(1, 0, 2)
            dump, (bc_av, c_dump) = self.lstm_av(rc[2] * av_concat)
            bc_av = bc_av.permute(1, 0, 2)
            dump, (tc_tav, c_dump) = self.lstm_tav(rc[3] * tav_concat)  # trimodal capsule
            tc_tav = tc_tav.permute(1, 0, 2)
            deci_concat = torch.cat([bc_ta, bc_tv, bc_av, tc_tav], 1)
            dump, (dc, c_dump) = self.lstm_deci(rc[4] * deci_concat)
            dc = dc.permute(1, 0, 2)
            dc = torch.sum(dc, 1)
            dc = dc.unsqueeze(1)

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
        elif self.dataset in ['mosi', 'mosei_senti']:
            preds = self.fc2(output1)
        return preds
