# Codes From https://github.com/cedias/Hierarchical-Sentiment

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
from torch.autograd import Variable


class Attention(nn.Module):
    def __init__(self, hid_size, att_size):
        super(Attention, self).__init__()
        self.lin = nn.Linear(hid_size, att_size)
        self.att_w = nn.Linear(att_size, 1, bias=False)

    def forward(self, enc_sents, enc_sent_com_attent, len_s):
        emb_h = F.tanh(self.lin(enc_sent_com_attent))
        att = self.att_w(emb_h).squeeze(-1)
        att_soft_res = self._masked_softmax(att, len_s)
        out = att_soft_res.unsqueeze(-1)
        attended = out * enc_sents
        # return attended.sum(0, True).squeeze(0)
        return attended.sum(0, True).squeeze(0), att_soft_res

    def _masked_softmax(self, mat, len_s):
        len_s = torch.FloatTensor(len_s).type_as(mat.data).long()
        idxes = torch.arange(0, int(len_s[0]), out=mat.data.new(int(len_s[0])).long()).unsqueeze(1)
        mask = Variable((idxes < len_s.unsqueeze(0)).float(), requires_grad=False)
        exp = torch.exp(mat) * mask
        sum_exp = exp.sum(0, True) + 0.0001
        return exp / sum_exp.expand_as(exp)

class BiRNN(nn.Module):
    def __init__(self, inp_size, hid_size, dropout=0, RNN_cell=nn.GRU):
        super(BiRNN, self).__init__()

        self.rnn = RNN_cell(input_size=inp_size,
                            hidden_size=hid_size,
                            num_layers=1,
                            bias=True,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=True)

    def forward(self, packed_batch):
        rnn_sents, _ = self.rnn(packed_batch)
        enc_sents, len_s = torch.nn.utils.rnn.pad_packed_sequence(rnn_sents)
        return enc_sents, len_s

class HAN(nn.Module):

    def __init__(self, ntoken, num_class, emb_size=200, hid_size=50):
        super(HAN, self).__init__()

        self.emb_size = emb_size
        self.embed = nn.Embedding(ntoken, emb_size,padding_idx=0)
        self.word_rnn = BiRNN(emb_size, hid_size)
        self.sent_rnn = BiRNN(hid_size * 2, hid_size)
        self.word_attent = Attention(hid_size * 2, hid_size * 2)
        self.sent_attent = Attention(hid_size * 2, hid_size * 2)
        self.lin_out = nn.Linear(hid_size*2,num_class)

    def set_emb_tensor(self,emb_tensor):
        self.emb_size = emb_tensor.size(-1)
        self.embed.weight.data = emb_tensor

    
    def _reorder_sent(self,sents,sent_order):
        
        sents = F.pad(sents,(0,0,1,0)) #adds a 0 to the top
        revs = sents[sent_order.view(-1)]
        revs = revs.view(sent_order.size(0),sent_order.size(1),sents.size(1))

        return revs
 

    def forward(self, batch_reviews,sent_order,ls,lr):
        emb_w = F.dropout(self.embed(batch_reviews),training=self.training)
        packed_sents = torch.nn.utils.rnn.pack_padded_sequence(emb_w, ls,batch_first=True)
        wrd_rnn_res, len_s = self.word_rnn(packed_sents)
        sent_embs = self.word_attent(wrd_rnn_res, wrd_rnn_res, len_s)
        rev_embs = self._reorder_sent(sent_embs,sent_order)
        packed_rev = torch.nn.utils.rnn.pack_padded_sequence(rev_embs, lr,batch_first=True)
        sen_rnn_res, len_r = self.sent_rnn(packed_rev)
        doc_embs = self.sent_attent(sen_rnn_res, sen_rnn_res, len_r)
        out = self.lin_out(doc_embs)
        return out

class MHAN(nn.Module):
    def __init__(self,
                 ntoken,
                 n_users,
                 n_rating,
                 num_class,
                 aspect_label_num,
                 emb_size=200,
                 hid_size=100,
                 mode = '',
                 rnn_type = 'GRU',
                 drop_out_rate=0.0,
                 model_init_needed=False
                 ):
        super(MHAN, self).__init__()

        self.mode = mode
        self.emb_size = emb_size
        self.aspect_label_num = aspect_label_num
        self.hid_size = hid_size
        self.rnn_type = rnn_type
        self.drop_out_rate = drop_out_rate
        self.model_init_needed = model_init_needed


        self.embed = nn.Embedding(ntoken, emb_size, padding_idx=0)
        self.users = nn.Embedding(n_users, emb_size)
        I.normal(self.users.weight.data, 0.01, 0.01)
        self.rating_embed = nn.Embedding(n_rating, emb_size)
        I.normal(self.rating_embed.weight.data, 0.01, 0.01)

        if self.rnn_type == 'LSTM':
            RNN_cell = nn.LSTM
        elif self.rnn_type == 'GRU':
            RNN_cell = nn.GRU
        else:
            RNN_cell = nn.GRU


        self.word_rnn = BiRNN(inp_size = emb_size,
                              hid_size = self.hid_size,
                              dropout = self.drop_out_rate,
                              RNN_cell = RNN_cell)

        self.sent_rnn = BiRNN(inp_size = self.hid_size * 2,
                              hid_size = self.hid_size,
                              dropout = self.drop_out_rate,
                              RNN_cell = RNN_cell)

        self.word_attent = Attention(self.hid_size * 2 , self.hid_size * 2)
        self.sent_attent = Attention(self.hid_size * 2 , self.hid_size * 2)

        self.lin_out1 = nn.Linear(self.hid_size * 2, num_class).cuda()
        self.lin_out2 = nn.Linear(self.hid_size * 2, num_class).cuda()
        self.lin_out3 = nn.Linear(self.hid_size * 2, num_class).cuda()
        self.lin_out4 = nn.Linear(self.hid_size * 2, num_class).cuda()
        self.lin_out5 = nn.Linear(self.hid_size * 2, num_class).cuda()
        self.lin_out6 = nn.Linear(self.hid_size * 2, num_class).cuda()
        self.lin_out7 = nn.Linear(self.hid_size * 2, num_class).cuda()

        self._init_para()


    def _init_para(self):
        if self.model_init_needed:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.normal_(0, 0.01)

    def set_emb_tensor(self, emb_tensor):
        self.emb_size = emb_tensor.size(-1)
        self.embed.weight.data = emb_tensor

    def _reorder_sent(self, sents, sent_order):
        sents = F.pad(sents, (0, 0, 1, 0))  # adds a 0 to the top
        revs = sents[sent_order.view(-1)]
        revs = revs.view(sent_order.size(0), sent_order.size(1), sents.size(1))
        return revs

    def forward(self, batch_reviews, sent_order, ls, lr, batch_overRating,batch_doc_usrs,batch_sen_usrs):

        if self.drop_out_rate > 0.0:
            emb_w = F.dropout(self.embed(batch_reviews), training=self.training, p=self.drop_out_rate)
        else:
            emb_w = self.embed(batch_reviews)

        packed_sents = torch.nn.utils.rnn.pack_padded_sequence(emb_w, ls, batch_first=True)
        wrd_rnn_res, len_s = self.word_rnn(packed_sents)

        wrd_rnn_asp_join_tmp_tensor = wrd_rnn_res

        sent_emb, wrd_att_res = self.word_attent(wrd_rnn_res, wrd_rnn_asp_join_tmp_tensor, len_s)
        rev_emb = self._reorder_sent(sent_emb, sent_order)
        packed_rev = torch.nn.utils.rnn.pack_padded_sequence(rev_emb, lr, batch_first=True)
        sen_rnn_res, len_r = self.sent_rnn(packed_rev)

        sent_rnn_asp_join_tmp_tensor = sen_rnn_res

        doc_embs, sent_att_res = self.sent_attent(sen_rnn_res, sent_rnn_asp_join_tmp_tensor, len_r)
        doc_embs_new = doc_embs

        out = []
        if self.aspect_label_num == 7:
            out.append(self.lin_out1(doc_embs_new))
            out.append(self.lin_out2(doc_embs_new))
            out.append(self.lin_out3(doc_embs_new))
            out.append(self.lin_out4(doc_embs_new))
            out.append(self.lin_out5(doc_embs_new))
            out.append(self.lin_out6(doc_embs_new))
            out.append(self.lin_out7(doc_embs_new))
        if self.aspect_label_num == 4:
            out.append(self.lin_out1(doc_embs_new))
            out.append(self.lin_out2(doc_embs_new))
            out.append(self.lin_out3(doc_embs_new))
            out.append(self.lin_out4(doc_embs_new))
        return out, [wrd_att_res], [sent_att_res]

# class MBGRU(nn.Module):
#     def __init__(self,
#                  ntoken,
#                  num_class,
#                  aspect_label_num,
#                  emb_size=200,
#                  hid_size=100,
#                  mode = '',
#                  rnn_type = 'GRU',
#                  drop_out_rate=0.0,
#                  model_init_needed=False
#                  ):
#         super(MBGRU, self).__init__()
#
#         self.mode = mode
#         self.emb_size = emb_size
#         self.hid_size = hid_size
#         self.rnn_type = rnn_type
#         self.drop_out_rate = drop_out_rate
#         self.model_init_needed = model_init_needed
#         self.aspect_label_num = aspect_label_num
#
#         self.embed = nn.Embedding(ntoken, emb_size, padding_idx=0)
#
#
#         if self.rnn_type == 'LSTM':
#             RNN_cell = nn.LSTM
#         elif self.rnn_type == 'GRU':
#             RNN_cell = nn.GRU
#         else:
#             RNN_cell = nn.GRU
#
#
#         self.word_rnn = BiRNN(inp_size = emb_size,
#                               hid_size = self.hid_size,
#                               dropout = self.drop_out_rate,
#                               RNN_cell = RNN_cell)
#
#         self.sent_rnn = BiRNN(inp_size = self.hid_size * 2,
#                               hid_size = self.hid_size,
#                               dropout = self.drop_out_rate,
#                               RNN_cell = RNN_cell)
#
#         self.word_attent = Attention(self.hid_size * 2 , self.hid_size * 2)
#         self.sent_attent = Attention(self.hid_size * 2 , self.hid_size * 2)
#
#         self.lin_out1 = nn.Linear(self.hid_size * 2, num_class).cuda()
#         self.lin_out2 = nn.Linear(self.hid_size * 2, num_class).cuda()
#         self.lin_out3 = nn.Linear(self.hid_size * 2, num_class).cuda()
#         self.lin_out4 = nn.Linear(self.hid_size * 2, num_class).cuda()
#         self.lin_out5 = nn.Linear(self.hid_size * 2, num_class).cuda()
#         self.lin_out6 = nn.Linear(self.hid_size * 2, num_class).cuda()
#         self.lin_out7 = nn.Linear(self.hid_size * 2, num_class).cuda()
#
#         self._init_para()
#
#
#     def _init_para(self):
#         if self.model_init_needed:
#             for m in self.modules():
#                 if isinstance(m, nn.Linear):
#                     nn.init.xavier_normal(m.weight.data)
#                     if m.bias is not None:
#                         m.bias.data.normal_(0, 0.01)
#
#     def set_emb_tensor(self, emb_tensor):
#         self.emb_size = emb_tensor.size(-1)
#         self.embed.weight.data = emb_tensor
#
#     def _reorder_sent(self, sents, sent_order):
#         sents = F.pad(sents, (0, 0, 1, 0))  # adds a 0 to the top
#         revs = sents[sent_order.view(-1)]
#         revs = revs.view(sent_order.size(0), sent_order.size(1), sents.size(1))
#         return revs
#
#     def forward(self, batch_reviews, sent_order, ls, lr, batch_overRating,batch_doc_usrs,batch_sen_usrs):
#
#         if self.drop_out_rate > 0.0:
#             emb_w = F.dropout(self.embed(batch_reviews), training=self.training, p=self.drop_out_rate)
#         else:
#             emb_w = self.embed(batch_reviews)
#
#         packed_sents = torch.nn.utils.rnn.pack_padded_sequence(emb_w, ls, batch_first=True)
#         wrd_rnn_res, len_s = self.word_rnn(packed_sents)
#
#         wrd_rnn_fill_1_tensor = torch.ones_like(wrd_rnn_res).float().cuda()
#         sent_emb, wrd_att_res = self.word_attent(wrd_rnn_res, wrd_rnn_fill_1_tensor, len_s)
#         rev_emb = self._reorder_sent(sent_emb, sent_order)
#         packed_rev = torch.nn.utils.rnn.pack_padded_sequence(rev_emb, lr, batch_first=True)
#         sen_rnn_res, len_r = self.sent_rnn(packed_rev)
#         sen_rnn_fill_1_tensor = torch.ones_like(sen_rnn_res).float().cuda()
#         doc_embs, sent_att_res = self.sent_attent(sen_rnn_res, sen_rnn_fill_1_tensor, len_r)
#         doc_embs_new = doc_embs
#
#         out = []
#         if self.aspect_label_num == 7:
#             out.append(self.lin_out1(doc_embs_new))
#             out.append(self.lin_out2(doc_embs_new))
#             out.append(self.lin_out3(doc_embs_new))
#             out.append(self.lin_out4(doc_embs_new))
#             out.append(self.lin_out5(doc_embs_new))
#             out.append(self.lin_out6(doc_embs_new))
#             out.append(self.lin_out7(doc_embs_new))
#         if self.aspect_label_num == 4:
#             out.append(self.lin_out1(doc_embs_new))
#             out.append(self.lin_out2(doc_embs_new))
#             out.append(self.lin_out3(doc_embs_new))
#             out.append(self.lin_out4(doc_embs_new))
#         return out, [wrd_att_res], [sen_rnn_res]

class MBGRU(nn.Module):
    def __init__(self,
                 ntoken,
                 n_users,
                 n_rating,
                 num_class,
                 aspect_label_num,
                 emb_size=200,
                 hid_size=100,
                 mode = '',
                 rnn_type = 'GRU',
                 drop_out_rate=0.0,
                 model_init_needed=False
                 ):
        super(MBGRU, self).__init__()

        self.mode = mode
        self.emb_size = emb_size
        self.aspect_label_num = aspect_label_num
        self.hid_size = hid_size
        self.rnn_type = rnn_type
        self.drop_out_rate = drop_out_rate
        self.model_init_needed = model_init_needed


        self.embed = nn.Embedding(ntoken, emb_size, padding_idx=0)
        if self.mode.find('U') != -1 or self.mode.find('R') != -1:
            self.users = nn.Embedding(n_users, emb_size)
            I.normal(self.users.weight.data, 0.01, 0.01)
            self.rating_embed = nn.Embedding(n_rating, emb_size)
            I.normal(self.rating_embed.weight.data, 0.01, 0.01)

        if self.rnn_type == 'LSTM':
            RNN_cell = nn.LSTM
        elif self.rnn_type == 'GRU':
            RNN_cell = nn.GRU
        else:
            RNN_cell = nn.GRU


        self.word_rnn = BiRNN(inp_size = emb_size,
                              hid_size = self.hid_size,
                              dropout = self.drop_out_rate,
                              RNN_cell = RNN_cell)

        self.sent_rnn = BiRNN(inp_size = self.hid_size * 2,
                              hid_size = self.hid_size,
                              dropout = self.drop_out_rate,
                              RNN_cell = RNN_cell)


        if self.mode.find('U') != -1:
            self.word_attent = Attention(self.hid_size * 2 + emb_size, self.hid_size * 2)
            self.sent_attent = Attention(self.hid_size * 2 + emb_size, self.hid_size * 2)
        else:
            self.word_attent = Attention(self.hid_size * 2 , self.hid_size * 2)
            self.sent_attent = Attention(self.hid_size * 2 , self.hid_size * 2)

        if self.mode.find('U') != -1 and self.mode.find('R') != -1:
            self.lin_out1 = nn.Linear(hid_size * 2 + emb_size * 2, num_class).cuda()
            self.lin_out2 = nn.Linear(hid_size * 2 + emb_size * 2, num_class).cuda()
            self.lin_out3 = nn.Linear(hid_size * 2 + emb_size * 2, num_class).cuda()
            self.lin_out4 = nn.Linear(hid_size * 2 + emb_size * 2, num_class).cuda()
            self.lin_out5 = nn.Linear(hid_size * 2 + emb_size * 2, num_class).cuda()
            self.lin_out6 = nn.Linear(hid_size * 2 + emb_size * 2, num_class).cuda()
            self.lin_out7 = nn.Linear(hid_size * 2 + emb_size * 2, num_class).cuda()
        elif self.mode.find('U') != -1 or self.mode.find('R') != -1:
            self.lin_out1 = nn.Linear(self.hid_size * 2 + emb_size, num_class).cuda()
            self.lin_out2 = nn.Linear(self.hid_size * 2 + emb_size, num_class).cuda()
            self.lin_out3 = nn.Linear(self.hid_size * 2 + emb_size, num_class).cuda()
            self.lin_out4 = nn.Linear(self.hid_size * 2 + emb_size, num_class).cuda()
            self.lin_out5 = nn.Linear(self.hid_size * 2 + emb_size, num_class).cuda()
            self.lin_out6 = nn.Linear(self.hid_size * 2 + emb_size, num_class).cuda()
            self.lin_out7 = nn.Linear(self.hid_size * 2 + emb_size, num_class).cuda()
        else:
            self.lin_out1 = nn.Linear(self.hid_size * 2, num_class).cuda()
            self.lin_out2 = nn.Linear(self.hid_size * 2, num_class).cuda()
            self.lin_out3 = nn.Linear(self.hid_size * 2, num_class).cuda()
            self.lin_out4 = nn.Linear(self.hid_size * 2, num_class).cuda()
            self.lin_out5 = nn.Linear(self.hid_size * 2, num_class).cuda()
            self.lin_out6 = nn.Linear(self.hid_size * 2, num_class).cuda()
            self.lin_out7 = nn.Linear(self.hid_size * 2, num_class).cuda()

        self._init_para()


    def _init_para(self):
        if self.model_init_needed:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.normal_(0, 0.01)

    def set_emb_tensor(self, emb_tensor):
        self.emb_size = emb_tensor.size(-1)
        self.embed.weight.data = emb_tensor

    def _reorder_sent(self, sents, sent_order):
        sents = F.pad(sents, (0, 0, 1, 0))  # adds a 0 to the top
        revs = sents[sent_order.view(-1)]
        revs = revs.view(sent_order.size(0), sent_order.size(1), sents.size(1))
        return revs

    def forward(self, batch_reviews, sent_order, ls, lr, batch_overRating,batch_doc_usrs,batch_sen_usrs):

        if self.drop_out_rate > 0.0:
            emb_w = F.dropout(self.embed(batch_reviews), training=self.training, p=self.drop_out_rate)
            if self.mode.find('U') != -1 or self.mode.find('R') != -1:
                emb_u_doc_level = F.dropout(self.users(batch_doc_usrs), training=self.training, p=self.drop_out_rate)
                emb_u_sen_level = F.dropout(self.users(batch_sen_usrs), training=self.training, p=self.drop_out_rate)
                emb_overRating = F.dropout(self.rating_embed(batch_overRating), training=self.training, p=self.drop_out_rate)
        else:
            emb_w = self.embed(batch_reviews)
            if self.mode.find('U') != -1 or self.mode.find('R') != -1:
                emb_u_doc_level = self.users(batch_doc_usrs)
                emb_u_sen_level = self.users(batch_sen_usrs)
                emb_overRating = self.rating_embed(batch_overRating)

        packed_sents = torch.nn.utils.rnn.pack_padded_sequence(emb_w, ls, batch_first=True)
        wrd_rnn_res, len_s = self.word_rnn(packed_sents)
        if self.mode.find('U') == -1:
            wrd_rnn_asp_join_tmp_tensor = torch.ones_like(wrd_rnn_res).float().cuda()
        else:
            wrd_rnn_asp_join_tmp_tensor = torch.cat([emb_u_sen_level.expand(wrd_rnn_res.size()[0],emb_u_sen_level.size()[0],emb_u_sen_level.size()[1]),
                                                     wrd_rnn_res], dim=-1)

        sent_emb, wrd_att_res = self.word_attent(wrd_rnn_res, wrd_rnn_asp_join_tmp_tensor, len_s)
        rev_emb = self._reorder_sent(sent_emb, sent_order)
        packed_rev = torch.nn.utils.rnn.pack_padded_sequence(rev_emb, lr, batch_first=True)
        sen_rnn_res, len_r = self.sent_rnn(packed_rev)
        if self.mode.find('U') == -1:
            sent_rnn_asp_join_tmp_tensor = torch.ones_like(sen_rnn_res).float().cuda()
        else:
            sent_rnn_asp_join_tmp_tensor = torch.cat([emb_u_doc_level.expand(sen_rnn_res.size()[0],
                                                                             emb_u_doc_level.size()[0],
                                                                             emb_u_doc_level.size()[1]),
                                                      sen_rnn_res], dim=-1)
        doc_embs, sent_att_res = self.sent_attent(sen_rnn_res, sent_rnn_asp_join_tmp_tensor, len_r)
        if self.mode.find('U') != -1 and self.mode.find('R') != -1:
            doc_embs_new = torch.cat([emb_u_doc_level, doc_embs, emb_overRating], dim=-1)
        elif self.mode.find('R') != -1:
            doc_embs_new = torch.cat([doc_embs, emb_overRating], dim=-1)
        elif self.mode.find('U') != -1:
            doc_embs_new = torch.cat([emb_u_doc_level, doc_embs], dim=-1)
        else:
            doc_embs_new = doc_embs

        out = []
        if self.aspect_label_num == 7:
            out.append(self.lin_out1(doc_embs_new))
            out.append(self.lin_out2(doc_embs_new))
            out.append(self.lin_out3(doc_embs_new))
            out.append(self.lin_out4(doc_embs_new))
            out.append(self.lin_out5(doc_embs_new))
            out.append(self.lin_out6(doc_embs_new))
            out.append(self.lin_out7(doc_embs_new))
        if self.aspect_label_num == 4:
            out.append(self.lin_out1(doc_embs_new))
            out.append(self.lin_out2(doc_embs_new))
            out.append(self.lin_out3(doc_embs_new))
            out.append(self.lin_out4(doc_embs_new))
        return out, [wrd_att_res], [sent_att_res]

class MBGRUAsp(nn.Module):
    # 以 Attention 的方式 添加 Aspect
    def __init__(self,
                 ntoken,
                 n_users,
                 n_rating,
                 num_class,
                 aspect_label_num,
                 emb_size=200,
                 hid_size=100,
                 mode = '',
                 rnn_type = 'GRU',
                 drop_out_rate=0.0,
                 model_init_needed=False
                 ):
        super(MBGRUAsp, self).__init__()

        self.mode = mode
        self.emb_size = emb_size
        self.aspect_label_num = aspect_label_num
        self.hid_size = hid_size
        self.rnn_type = rnn_type
        self.drop_out_rate = drop_out_rate
        self.model_init_needed = model_init_needed

        self.embed = nn.Embedding(ntoken, emb_size, padding_idx=0)
        self.users = nn.Embedding(n_users, emb_size)
        I.normal(self.users.weight.data, 0.01, 0.01)
        self.rating_embed = nn.Embedding(n_rating, emb_size)
        I.normal(self.rating_embed.weight.data, 0.01, 0.01)

        if self.rnn_type == 'LSTM':
            RNN_cell = nn.LSTM
        elif self.rnn_type == 'GRU':
            RNN_cell = nn.GRU
        else:
            RNN_cell = nn.GRU

        self.word_rnn = BiRNN(inp_size=emb_size,
                              hid_size=self.hid_size,
                              dropout=self.drop_out_rate,
                              RNN_cell=RNN_cell)

        self.sent_rnn = BiRNN(inp_size=self.hid_size * 2,
                              hid_size=self.hid_size,
                              dropout=self.drop_out_rate,
                              RNN_cell=RNN_cell)

        self.asp_embed = nn.Embedding(aspect_label_num, emb_size)

        if self.mode.find('U') != -1:
            self.word_attent = Attention(hid_size * 2 + emb_size * 2, hid_size * 2)
            self.sent_attent = Attention(hid_size * 2 + emb_size * 2, hid_size * 2)
        else:
            self.word_attent = Attention(hid_size * 2 + emb_size, hid_size * 2)
            self.sent_attent = Attention(hid_size * 2 + emb_size, hid_size * 2)

        if self.mode.find('U') != -1 and self.mode.find('R') != -1:
            self.lin_out1 = nn.Linear(hid_size * 2 + emb_size * 2, num_class).cuda()
            self.lin_out2 = nn.Linear(hid_size * 2 + emb_size * 2, num_class).cuda()
            self.lin_out3 = nn.Linear(hid_size * 2 + emb_size * 2, num_class).cuda()
            self.lin_out4 = nn.Linear(hid_size * 2 + emb_size * 2, num_class).cuda()
            self.lin_out5 = nn.Linear(hid_size * 2 + emb_size * 2, num_class).cuda()
            self.lin_out6 = nn.Linear(hid_size * 2 + emb_size * 2, num_class).cuda()
            self.lin_out7 = nn.Linear(hid_size * 2 + emb_size * 2, num_class).cuda()
        elif self.mode.find('U') != -1 or self.mode.find('R') != -1:
            self.lin_out1 = nn.Linear(self.hid_size * 2 + emb_size, num_class).cuda()
            self.lin_out2 = nn.Linear(self.hid_size * 2 + emb_size, num_class).cuda()
            self.lin_out3 = nn.Linear(self.hid_size * 2 + emb_size, num_class).cuda()
            self.lin_out4 = nn.Linear(self.hid_size * 2 + emb_size, num_class).cuda()
            self.lin_out5 = nn.Linear(self.hid_size * 2 + emb_size, num_class).cuda()
            self.lin_out6 = nn.Linear(self.hid_size * 2 + emb_size, num_class).cuda()
            self.lin_out7 = nn.Linear(self.hid_size * 2 + emb_size, num_class).cuda()
        else:
            self.lin_out1 = nn.Linear(self.hid_size * 2, num_class).cuda()
            self.lin_out2 = nn.Linear(self.hid_size * 2, num_class).cuda()
            self.lin_out3 = nn.Linear(self.hid_size * 2, num_class).cuda()
            self.lin_out4 = nn.Linear(self.hid_size * 2, num_class).cuda()
            self.lin_out5 = nn.Linear(self.hid_size * 2, num_class).cuda()
            self.lin_out6 = nn.Linear(self.hid_size * 2, num_class).cuda()
            self.lin_out7 = nn.Linear(self.hid_size * 2, num_class).cuda()

        self._init_para()


    def _init_para(self):
        if self.model_init_needed:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.normal_(0, 0.01)


    def set_emb_tensor(self, emb_tensor):
        self.emb_size = emb_tensor.size(-1)
        self.embed.weight.data = emb_tensor


    def _reorder_sent(self, sents, sent_order):
        sents = F.pad(sents, (0, 0, 1, 0))  # adds a 0 to the top
        revs = sents[sent_order.view(-1)]
        revs = revs.view(sent_order.size(0), sent_order.size(1), sents.size(1))
        return revs

    def set_asp_emb_tensor(self, asp_emb_tensor):
        self.asp_emb_size = asp_emb_tensor.size(-1)
        self.asp_embed.weight.data = asp_emb_tensor


    def get_aspect_tensor(self,emb_w,aspect_id):
        return self.asp_embed(Variable(torch.zeros(emb_w.size()[1]).long().fill_(aspect_id)).cuda())

    def forward(self, batch_reviews, sent_order, ls, lr, batch_overRating, batch_doc_usrs, batch_sen_usrs):

        if self.drop_out_rate > 0.0:
            emb_w = F.dropout(self.embed(batch_reviews), training=self.training, p=self.drop_out_rate)
            emb_u_doc_level = F.dropout(self.users(batch_doc_usrs), training=self.training, p=self.drop_out_rate)
            emb_u_sen_level = F.dropout(self.users(batch_sen_usrs), training=self.training, p=self.drop_out_rate)
            emb_overRating = F.dropout(self.rating_embed(batch_overRating), training=self.training, p=self.drop_out_rate)
        else:
            emb_w = self.embed(batch_reviews)
            emb_u_doc_level = self.users(batch_doc_usrs)
            emb_u_sen_level = self.users(batch_sen_usrs)
            emb_overRating = self.rating_embed(batch_overRating)


        packed_sents = torch.nn.utils.rnn.pack_padded_sequence(emb_w, ls, batch_first=True)
        wrd_rnn_res, len_s = self.word_rnn(packed_sents)
        wrd_rnn_asp_join_list = []
        sent_emb_list = []
        rev_embs_list = []
        doc_embs_list = []
        word_attent_res_list = []
        sen_attent_res_list = []

        for aspect_id in range(0,self.aspect_label_num):
            temp_asp_tensor = self.get_aspect_tensor(wrd_rnn_res, aspect_id)
            if self.mode.find('U') == -1:
                wrd_rnn_asp_join_tmp_tensor = torch.cat([temp_asp_tensor.expand(wrd_rnn_res.size()[0],temp_asp_tensor.size()[0],temp_asp_tensor.size()[1]),
                                                         wrd_rnn_res], dim=-1)
            else:
                wrd_rnn_asp_join_tmp_tensor = torch.cat([emb_u_sen_level.expand(wrd_rnn_res.size()[0],
                                                                                emb_u_sen_level.size()[0],
                                                                                emb_u_sen_level.size()[1]),
                                                         temp_asp_tensor.expand(wrd_rnn_res.size()[0],
                                                                                temp_asp_tensor.size()[0],
                                                                                temp_asp_tensor.size()[1]),
                                                         wrd_rnn_res], dim=-1)
            sent_emb, wrd_att_res = self.word_attent(wrd_rnn_res, wrd_rnn_asp_join_tmp_tensor, len_s)
            word_attent_res_list.append(wrd_att_res)
            sent_emb_list.append(sent_emb)
            rev_embs = self._reorder_sent(sent_emb, sent_order)
            rev_embs_list.append(rev_embs)
            packed_rev = torch.nn.utils.rnn.pack_padded_sequence(rev_embs, lr, batch_first=True)
            sen_rnn_res, len_r = self.sent_rnn(packed_rev)

            temp_asp_tensor = self.get_aspect_tensor(sen_rnn_res, aspect_id)
            if self.mode.find('U') == -1:
                sent_rnn_asp_join_tmp_tensor = torch.cat([temp_asp_tensor.expand(sen_rnn_res.size()[0],
                                                                                temp_asp_tensor.size()[0],
                                                                                temp_asp_tensor.size()[1]),
                                                          sen_rnn_res], dim=-1)
            else:
                sent_rnn_asp_join_tmp_tensor = torch.cat([emb_u_doc_level.expand(sen_rnn_res.size()[0],
                                                                                 emb_u_doc_level.size()[0],
                                                                                 emb_u_doc_level.size()[1]),
                                                          temp_asp_tensor.expand(sen_rnn_res.size()[0],
                                                                                 temp_asp_tensor.size()[0],
                                                                                 temp_asp_tensor.size()[1]),
                                                          sen_rnn_res], dim=-1)
            doc_embs, sent_att_res = self.sent_attent(sen_rnn_res, sent_rnn_asp_join_tmp_tensor, len_r)
            sen_attent_res_list.append(sent_att_res)
            if self.mode.find('U') != -1 and self.mode.find('R') != -1:
                doc_embs_new = torch.cat([emb_u_doc_level, doc_embs, emb_overRating], dim=-1)
            elif self.mode.find('R') != -1:
                doc_embs_new = torch.cat([doc_embs, emb_overRating], dim=-1)
            elif self.mode.find('U') != -1:
                doc_embs_new = torch.cat([emb_u_doc_level, doc_embs], dim=-1)
            else:
                doc_embs_new = doc_embs

            doc_embs_list.append(doc_embs_new)

        out = []
        if self.aspect_label_num == 7:
            out.append(self.lin_out1(doc_embs_list[0]))
            out.append(self.lin_out2(doc_embs_list[1]))
            out.append(self.lin_out3(doc_embs_list[2]))
            out.append(self.lin_out4(doc_embs_list[3]))
            out.append(self.lin_out5(doc_embs_list[4]))
            out.append(self.lin_out6(doc_embs_list[5]))
            out.append(self.lin_out7(doc_embs_list[6]))
        if self.aspect_label_num == 4:
            out.append(self.lin_out1(doc_embs_list[0]))
            out.append(self.lin_out2(doc_embs_list[1]))
            out.append(self.lin_out3(doc_embs_list[2]))
            out.append(self.lin_out4(doc_embs_list[3]))
        return out, word_attent_res_list, sen_attent_res_list



