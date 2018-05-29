#-*- coding: UTF-8 -*-  
import numpy
import torch
from tqdm import tqdm
import time
import logging

def get_current_date_time():
    return time.strftime('%Y-%m-%d %H:%M:%S')

def load_embeddings(embed_file, word_dict_file, embed_size = 200,offset=0):
    word2index = {}
    word_list = open(word_dict_file).readlines()
    for i,word in enumerate(word_list):
        word = word.strip()
        word2index[word] = i + offset

    tensor = torch.randn(len(word2index) + offset, embed_size)  ## adding offset
    logging.info(tensor.size())
    wordIndex2embedExist = {}

    embed_con_list = open(embed_file).readlines()
    for i,line in tqdm(enumerate(embed_con_list),desc="Creating embedding tensor",total=len(embed_con_list)):
        spl = line.strip().split(" ")
        word = spl[0]
        wordIndex2embedExist[word2index[word]] = 1
        vector_value = spl[1:]
        if word in word2index.keys():
            tensor[word2index[word]] = torch.FloatTensor(list(float(x) for x in vector_value))
    return tensor, word2index, wordIndex2embedExist

def tuple_batch(review,rating):
    """
    Prepare batch
    - Reorder reviews by length
    - Split reviews by sentences which are reordered by length
    - Build sentence ordering index to extract each sentences in training loop
    """
    list_rev = review

    r_t = torch.Tensor(rating).long()

    sorted_r = sorted([(len(r),r_n,r) for r_n,r in enumerate(list_rev)],reverse=True) #index by desc rev_le
    lr,r_n,ordered_list_rev = zip(*sorted_r)

    max_sents = lr[0]
    r_t = [r_t[x] for x in r_n]
    review = [review[x] for x in r_n]


    stat =  sorted([(len(s),r_n,s_n,s) for r_n,r in enumerate(ordered_list_rev) for s_n,s in enumerate(r)],reverse=True)
    max_words = stat[0][0]

    ls = []
    batch_t = torch.zeros(len(stat),max_words).long()                         # (sents ordered by len)
    sent_order = torch.zeros(len(ordered_list_rev),max_sents).long().fill_(0) # (rev_n,sent_n)

    for i,s in enumerate(stat):
        sent_order[s[1],s[2]] = i+1   #i+1 because 0 is for empty.
        batch_t[i,0:len(s[3])] = torch.LongTensor(s[3])
        ls.append(s[0])

    return batch_t,r_t,sent_order,ls,lr,review

def tuple_batch_aspect(batch_size,batch_i,docs,overall_rating,aspect_rating_list,usrs):
    review = docs[batch_i * batch_size:(batch_i + 1) * batch_size]
    overall_rating_np = numpy.asarray(overall_rating[batch_i * batch_size:(batch_i + 1) * batch_size], dtype=numpy.float32)
    usrs_np = numpy.asarray(usrs[batch_i * batch_size:(batch_i + 1) * batch_size], dtype=numpy.int32)
    aspect_rating_np = []
    for i in range(0,len(aspect_rating_list)):
        aspect_rating_np.append(numpy.asarray(aspect_rating_list[i][batch_i * batch_size:(batch_i + 1) * batch_size], dtype=numpy.float32))

    """
    Prepare batch
    - Reorder reviews by length
    - Split reviews by sentences which are reordered by length
    - Build sentence ordering index to extract each sentences in training loop
    """
    list_rev = review

    r_t = torch.Tensor(overall_rating_np).long()
    asp_r_t_list = []
    for i in range(0,len(aspect_rating_list)):
        asp_r_t_list.append(torch.Tensor(aspect_rating_np[i]).long())
    u_t = torch.Tensor(usrs_np).long()


    sorted_r = sorted([(len(r),r_n,r) for r_n,r in enumerate(list_rev)],reverse=True) #index by desc rev_le
    lr,r_n,ordered_list_rev = zip(*sorted_r)

    max_sents = lr[0]

    for i in range(0, len(aspect_rating_list)):
        asp_r_t_list[i] = [asp_r_t_list[i][x] for x in r_n]
    r_t = [r_t[x] for x in r_n]
    doc_level_u_t = [u_t[x] for x in r_n]
    review = [review[x] for x in r_n]

    stat =  sorted([(len(s),r_n,s_n,s) for r_n,r in enumerate(ordered_list_rev) for s_n,s in enumerate(r)],reverse=True)
    max_words = stat[0][0]

    ls = []
    batch_t = torch.zeros(len(stat),max_words).long()                         # (sents ordered by len)
    sent_order = torch.zeros(len(ordered_list_rev),max_sents).long().fill_(0) # (rev_n,sent_n)
    sen_level_u_t = []
    for i,s in enumerate(stat):
        sent_order[s[1],s[2]] = i+1   #i+1 because 0 is for empty.
        batch_t[i,0:len(s[3])] = torch.LongTensor(s[3])
        ls.append(s[0])
        sen_level_u_t.append(doc_level_u_t[s[1]])

    return (batch_t,r_t,sent_order,ls,lr,review,asp_r_t_list,doc_level_u_t,sen_level_u_t)

class DatasetPrepare(object):
    def __init__(self,
                 args,
                 dataset_dict,
                 batch_size = 32,
                 mode = 'train'):
        self.mode = mode
        self.dataset_dict = dataset_dict
        self.batch_size = batch_size
        self.args = args

    def get_usr_dict(self,usr_list):
        usr2index = {}
        for i in range(0,len(usr_list)):
            if (usr_list[i] in usr2index.keys()) == False:
                usr2index[usr_list[i]] = len(usr2index)
        return usr2index

    def _load_word_embedding(self):
        wrd_embed_tensor, word2index, wordIndex2embedExist= load_embeddings(embed_file = self.dataset_dict['emb'],
                                                                            word_dict_file = self.dataset_dict['word_dict'],
                                                                            offset = 2,
                                                                            embed_size = self.args['embed_size'])
        self.wordIndex2embedExist = wordIndex2embedExist
        word2index["_pad_"] = 0
        word2index["_unk_"] = 1
        index2word = {v: k for k, v in word2index.items()}
        self.token_size = len(word2index)
        self.embed_size = wrd_embed_tensor.size()[1]
        self.wrd_embed_tensor = wrd_embed_tensor
        self.index2word = index2word
        self.word2index = word2index

    def _get_index(self,word2index,wrd):
        if (wrd in word2index.keys()) == False:
            return word2index["_unk_"]
        else:
            return word2index[wrd]

    def _load_aspect_keywords_embedding(self,file,word2index,word_embed_tensor):
        lines = list(map(lambda line: line.strip().split(' '), open(file).readlines()))
        aspect_value_list = list(map(lambda line: list(map(lambda word: self._get_index(word2index, word), line)), lines))
        self.aspect_embedding = torch.zeros(len(lines), word_embed_tensor.size()[1]).float()
        for i,aspect_value in enumerate(aspect_value_list):
            aspect_value_new = aspect_value
            temp_embedding = torch.zeros(len(aspect_value_new), word_embed_tensor.size()[1]).float()
            for k in range(0,len(aspect_value_new)):
                temp_embedding[k] = word_embed_tensor[aspect_value_new[k]]
            self.aspect_embedding[i] = temp_embedding.mean(0)

    def _get_asp_rat_dis(self,aspect_rating_list,overall_rating,asp_spec_bool = True):
        # print('_get_asp_rat_dis: begin...')
        aspect_id_log_prob_matrix_list = []
        rating_num = self.dataset_dict['class_num']
        def compute_appear_times(overall_rat,asp_rat,dis_matrix):
            for i in range(0,overall_rat.shape[0]):
                if overall_rat[i] < 0 or asp_rat[i] < 0: continue
                dis_matrix[overall_rat[i]][asp_rat[i]] += 1

        def compute_log_prob(dis_matrix):
            sum_matrix = dis_matrix.sum(axis=1)
            sum_matrix = sum_matrix.repeat(rating_num).reshape(rating_num, rating_num)
            prob_matrix = dis_matrix / sum_matrix
            return prob_matrix


        if asp_spec_bool == False:
            dis_matrix = numpy.zeros([rating_num, rating_num], dtype=float) + 1e-4
            for aspect_rating in aspect_rating_list:
                compute_appear_times(overall_rating,aspect_rating,dis_matrix)
            log_prob_matrix = compute_log_prob(dis_matrix)
            for aspect_rating in aspect_rating_list:
                aspect_id_log_prob_matrix_list.append(log_prob_matrix)

        else:
            for aspect_rating in aspect_rating_list:
                dis_matrix = numpy.zeros([rating_num, rating_num], dtype=float) + 1e-4
                compute_appear_times(overall_rating,aspect_rating,dis_matrix)
                log_prob_matrix = compute_log_prob(dis_matrix)
                aspect_id_log_prob_matrix_list.append(log_prob_matrix)

        self.aspect_id_log_prob_matrix_list = aspect_id_log_prob_matrix_list

    def _get_dis_from_overrat_aspectid(self,overall_rating,aspect_id):
        log_prob_matrix = self.aspect_id_log_prob_matrix_list[aspect_id]
        return log_prob_matrix[int(overall_rating)]

    def _tuple_batch_aspect(self, batch_size, batch_i, docs, overall_rating, aspect_rating_list, usrs):
        review = docs[batch_i * batch_size:(batch_i + 1) * batch_size]
        overall_rating_np = numpy.asarray(overall_rating[batch_i * batch_size:(batch_i + 1) * batch_size],
                                          dtype=numpy.float32)
        usrs_np = numpy.asarray(usrs[batch_i * batch_size:(batch_i + 1) * batch_size], dtype=numpy.int32)
        aspect_rating_np = []
        for i in range(0, len(aspect_rating_list)):
            aspect_rating_np.append(
                numpy.asarray(aspect_rating_list[i][batch_i * batch_size:(batch_i + 1) * batch_size],
                              dtype=numpy.float32))

        """
        Prepare batch
        - Reorder reviews by length
        - Split reviews by sentences which are reordered by length
        - Build sentence ordering index to extract each sentences in training loop
        """
        list_rev = review

        r_t = torch.Tensor(overall_rating_np).long()
        rating_dis_list = []
        asp_r_t_list = []
        for i in range(0, len(aspect_rating_list)):
            asp_r_t_list.append(torch.Tensor(aspect_rating_np[i]).long())
        u_t = torch.Tensor(usrs_np).long()

        sorted_r = sorted([(len(r), r_n, r) for r_n, r in enumerate(list_rev)], reverse=True)  # index by desc rev_le
        lr, r_n, ordered_list_rev = zip(*sorted_r)

        max_sents = lr[0]

        for i in range(0, len(aspect_rating_list)):
            asp_r_t_list[i] = [asp_r_t_list[i][x] for x in r_n]
        r_t = [r_t[x] for x in r_n]
        doc_level_u_t = [u_t[x] for x in r_n]
        review = [review[x] for x in r_n]
        if self.mode == 'train':
            for i in range(0, len(aspect_rating_list)):
                tmp_matrix = numpy.zeros([len(review),self.dataset_dict['class_num']], dtype=numpy.float32)
                for j in range(0,len(review)):
                    overall_rat_score = r_t[j]
                    aspect_id = i
                    tmp_matrix[j] = self._get_dis_from_overrat_aspectid(overall_rat_score,aspect_id)
                rating_dis_list.append(torch.Tensor(tmp_matrix).float())

        stat = sorted([(len(s), r_n, s_n, s) for r_n, r in enumerate(ordered_list_rev) for s_n, s in enumerate(r)],
                      reverse=True)
        max_words = stat[0][0]

        ls = []
        batch_t = torch.zeros(len(stat), max_words).long()  # (sents ordered by len)
        sent_order = torch.zeros(len(ordered_list_rev), max_sents).long().fill_(0)  # (rev_n,sent_n)
        sen_level_u_t = []
        for i, s in enumerate(stat):
            sent_order[s[1], s[2]] = i + 1  # i+1 because 0 is for empty.
            batch_t[i, 0:len(s[3])] = torch.LongTensor(s[3])
            ls.append(s[0])
            sen_level_u_t.append(doc_level_u_t[s[1]])

        return (batch_t, r_t, sent_order, ls, lr, review, asp_r_t_list, doc_level_u_t, sen_level_u_t, rating_dis_list)

    def get_batch_dat(self,word2index = {}, usr2index = {}):

        if self.mode == 'train':
            self._load_word_embedding()
            word2index = self.word2index

        filename = self.dataset_dict[self.mode]
        self.aspect_num = self.dataset_dict['aspect_num']
        lines = list(map(lambda x: x.split('\t\t'), open(filename, 'r', encoding='ISO-8859-1').readlines()))

        overall_rating = numpy.asarray(
            list(map(lambda x: int(x[2]) - 1, lines)),
            dtype=numpy.int32
        )

        aspect_rating = list(map(lambda x: x[3], lines))
        aspect_rating_list = []
        for i in range(0,self.aspect_num):
            aspect_rating_list.append(numpy.asarray(list(map(lambda x: int(x.split(' ')[i]) - 1, aspect_rating)),dtype=numpy.int32))

        id_list = [line[0] for line in lines]
        usr_list = [line[1] for line in lines]

        if self.mode == 'train':
            self.usr2index = self.get_usr_dict(usr_list)
            self.usr_size = len(self.usr2index)
            usr2index = self.usr2index
            file = self.dataset_dict['aspect_words']
            self._load_aspect_keywords_embedding(file,self.word2index,self.wrd_embed_tensor)
            self._get_asp_rat_dis(aspect_rating_list,overall_rating,self.args['dis_loss_aspect_spec'])

        docs = list(map(lambda x: x[4], lines))
        docs = list(map(lambda x: x.split('<ssssss>'), docs))
        docs = list(map(lambda doc: list(map(lambda sentence: sentence.split(' '), doc)), docs))
        docs = list(map(lambda doc:
                        list(map(lambda sentence:
                                 list(map(lambda word: self._get_index(word2index, word), sentence)), doc)), docs))

        usrs =  list(map(lambda usr: self._get_index(usr2index,usr), usr_list))

        self.epoch = int(len(docs) / int(self.batch_size))
        if len(docs) % self.batch_size != 0:
            self.epoch += 1

        self.batch_t = []
        self.sent_order = []
        self.ls = []
        self.lr = []
        self.review = []
        self.r_t = []
        self.doc_level_u_t = []
        self.sen_level_u_t = []
        self.asp_r_t = []
        self.id_list = []
        self.rating_dis_list = []

        epoch_list = range(0,int(self.epoch))
        desc_str = 'Load ' + self.mode + 'Data:'
        for i, epoch in tqdm(enumerate(epoch_list), desc=desc_str, total=len(epoch_list)):
            batch_res = self._tuple_batch_aspect(self.batch_size, i, docs, overall_rating, aspect_rating_list, usrs)
            batch_t, r_t, sent_order, ls, lr, review, asp_r_t_list, doc_level_u_t, sen_level_u_t, rating_dis_t = batch_res
            self.rating_dis_list.append(rating_dis_t)
            self.batch_t.append(batch_t)
            self.sent_order.append(sent_order)
            self.ls.append(ls)
            self.lr.append(lr)
            self.review.append(review)
            self.doc_level_u_t.append(torch.from_numpy(numpy.asarray(doc_level_u_t, dtype=numpy.int32)).long())
            self.sen_level_u_t.append(torch.from_numpy(numpy.asarray(sen_level_u_t, dtype=numpy.int32)).long())
            self.r_t.append(torch.from_numpy(numpy.asarray(r_t, dtype=numpy.int32)).long())
            asp_rating_list = []
            for k in range(0,self.aspect_num):
                asp_rating_list.append(torch.from_numpy(numpy.asarray(asp_r_t_list[k], dtype=numpy.int32)).long())
            self.asp_r_t.append(asp_rating_list)
            self.id_list.append(id_list[i * self.batch_size:(i + 1) * self.batch_size])

        logging.info('load {} data end, word number: {}, usr number: {}'.format(self.mode,len(word2index), len(usr2index)))

