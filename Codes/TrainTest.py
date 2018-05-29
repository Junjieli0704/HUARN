#-*- coding: UTF-8 -*-
import numpy
import torch
from tqdm import tqdm
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import math
import os
from Nets import *
from DatasetPrepare import *
import logging

def accuracy(out,truth,ignore_index=-2):
    def sm(mat):
        exp = torch.exp(mat)
        sum_exp = exp.sum(1,True)+0.0001
        return exp/sum_exp.expand_as(exp)

    _,max_i = torch.max(sm(out),1)

    tmp = torch.eq(truth, ignore_index)
    select_list = []
    for i in range(0,tmp.size()[0]):
        if tmp[i].data[0] == 0:
            select_list.append(i)

    select_tensor = Variable(torch.from_numpy(numpy.asarray(select_list,dtype=numpy.int32))).long().cuda()
    max_i_new = torch.index_select(max_i, 0, select_tensor)
    truth_new = torch.index_select(truth, 0, select_tensor)

    eq = torch.eq(max_i_new,truth_new).float()
    all_eq = torch.sum(eq)

    return all_eq, all_eq/truth_new.size(0)*100, max_i_new.float(),truth_new

def get_dir_files(dir,is_contain_dir = False):
    file_list = []
    if os.path.exists(dir):
        dir_file_list = os.listdir(dir)
        for dir_file in dir_file_list:
            if is_contain_dir:    file_list.append(dir + dir_file);
            else:     file_list.append(dir_file);
    return file_list

def mk_dir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exist = os.path.exists(path)
    if not is_exist:
        temp_str = path + ' create successfully!'
        print(temp_str)
        os.makedirs(path)
        return True
    else:
        return False

class RunBase(object):
    def __init__(self,
                 args,
                 dataset_dict,
                 optimizer_fun = 'Adam',
                 criterion_fun = 'CrossEntropy',
                 ignore_index = -2,
                 model_filefold = '../Model/'):


        self.dataset_dict = dataset_dict
        self.is_load_train = False
        self.model_filefold = model_filefold
        self.criterion_fun = criterion_fun
        self.optimizer_fun = optimizer_fun
        self.ignore_index = ignore_index
        self.args = args
        self.cuda = self.args['cuda']
        self.epoch_size = self.args['epoch']

    def _prepare_net_info(self):
        self.killDiv_criterion = torch.nn.KLDivLoss()
        if self.criterion_fun == 'CrossEntropy':
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        if self.cuda:
            self.net.cuda()
        if self.optimizer_fun == 'Adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()))
        torch.nn.utils.clip_grad_norm(self.net.parameters(), 1)


    def _load_dataset(self, mode, word2index = {}, usr2index = {}):
        if mode.find('train') != -1:
            self.dataset_train = DatasetPrepare(self.args,self.dataset_dict, mode='train', batch_size=self.args['batch_size'])
            self.dataset_train.get_batch_dat()
            self.is_load_train = True
        if mode.find('test') != -1:
            self.dataset_test = DatasetPrepare(self.args,self.dataset_dict, mode='test', batch_size=self.args['batch_size'])
            if self.is_load_train:
                word2index = self.dataset_train.word2index
                usr2index = self.dataset_train.usr2index
            self.dataset_test.get_batch_dat(word2index,usr2index)
        if mode.find('dev') != -1:
            self.dataset_dev = DatasetPrepare(self.args,self.dataset_dict, mode='dev', batch_size=self.args['batch_size'])
            if self.is_load_train:
                word2index = self.dataset_train.word2index
                usr2index = self.dataset_train.usr2index
            self.dataset_dev.get_batch_dat(word2index,usr2index)

    def _construct_net(self, token_size, n_users):
        self.net = None
        pass

    def _mkdir_model_filefold(self):
        mk_dir(self._get_model_filefold())

    def _get_model_filefold(self):
        model_filefold = self.model_filefold
        dataname = self.args['dataset']
        model_type = self.args['model_type']
        model_mode = self.args['model_mode']
        hidden_size = self.args['hidden_size']
        temp_file_fold = model_filefold + dataname + '/' + model_type + model_mode
        temp_file_fold += '_hidden(' + str(hidden_size) + ')'
        temp_file_fold += '_rnnType(' + str(self.args['rnn_type']) + ')'
        temp_file_fold += '_dropOutRate(' + str(self.args['drop_out_rate']) + ')'
        if self.args['model_init_needed']:
            temp_file_fold += '_init(T)'
        else:
            temp_file_fold += '_init(F)'
        temp_file_fold += '_disLossAlpha(' + str(self.args['dis_loss_alpha']) + ')'
        temp_file_fold += '/'
        return temp_file_fold


    def _save_model(self,model_str,model_acc_res = {}):
        if self.args['save_model']:
            temp_dict = {}
            temp_dict['state_dict'] = self.net.state_dict()
            temp_dict['acc'] = model_acc_res
            temp_dict['word_dic'] = self.dataset_train.word2index
            temp_dict['usr_dic'] = self.dataset_train.usr2index
            model_file = self._get_model_file(model_str)
            torch.save(temp_dict, model_file)
            logging.info('Saving model ({}) finished.'.format(model_file))
        else:
            logging.info('No saving model.')

    def _load_model_para(self,model_file):
        temp_dict = torch.load(model_file,map_location=lambda storage, loc: storage.cuda(self.args['gpu_id']))
        state = temp_dict['state_dict']
        model_acc_res = temp_dict['acc']
        word2index = temp_dict['word_dic']
        usr2index = temp_dict['usr_dic']
        return word2index,usr2index,state,model_acc_res

    def _train(self, epoch):
        pass

    def _tst(self, dataset_pre, msg="Evaluating"):
        pass

    def _test(self):
        return self._tst(self.dataset_test,msg="Testing")

    def _develop(self):
        return self._tst(self.dataset_dev,msg="Validation")

    def test_best_dev_model(self,is_need_run = False):
        pass

    def _adjust_learning_rate(self, decay_rate=0.8):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate

    def run(self):
        pass

    def count_parameters(self):
        total_param = 0
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                num_param = numpy.prod(param.size())
                # if param.dim() > 1:
                #     print(name , ':' , 'x'.join(str(x) for x in list(param.size())) , '=' , num_param)
                # else:
                #     print(name , ':' , num_param)
                total_param += num_param
        logging.info('number of trainable parameters = {}'.format(total_param))


    def _get_last_model_save(self):
        file_fold = self._get_model_filefold()
        file_list = get_dir_files(file_fold, False)
        iter_time_max = -1
        for file_name in file_list:
            if file_name.find('model_') != -1:
                iter_time = int(file_name.split('_')[-1])
                if iter_time > iter_time_max:
                    iter_time_max = iter_time

        if iter_time_max == -1:
            return iter_time_max, None
        else:
            return iter_time_max, file_fold + 'model_' + str(iter_time_max)


    def _get_best_dev_model(self,):
        file_fold = self._get_model_filefold()
        best_model_file = file_fold + 'best_dev_model'
        logging.info(best_model_file)
        if os.path.exists(best_model_file):
            return best_model_file
        else:
            return 'NULL'

    def _get_model_file(self,model_str):
        return self._get_model_filefold() + model_str + ''

class RunMODELA(RunBase):
    def _construct_net(self,token_size,n_users):
        self.net = MODELA(ntoken=token_size,
                        n_rating=self.dataset_dict['class_num'],
                        n_users = n_users,
                        emb_size = self.args['embed_size'],
                        hid_size = self.args['hidden_size'],
                        num_class = self.dataset_dict['class_num'],
                        aspect_label_num = self.dataset_dict['aspect_num'],
                        mode = self.args['model_mode'],
                        rnn_type = self.args['rnn_type'],
                        drop_out_rate = self.args['drop_out_rate'],
                        model_init_needed = self.args['model_init_needed']
                        )

    def _train(self, epoch, eval_per_time = -1, is_last = False):

        self.net.train()
        epoch_loss = 0
        ce_loss = 0
        mean_mse = 0
        mean_rmse = 0
        ok_all = 0
        dataset_pre = self.dataset_train
        aspect_label_num = self.dataset_train.aspect_num
        if eval_per_time == -1:
            iter_begin = 0
            iter_end = len(dataset_pre.batch_t)
            num_train_all = len(dataset_pre.batch_t)

        else:
            iter_begin = eval_per_time * self.args['eval_period']
            if is_last == False:
                iter_end = (eval_per_time + 1) * self.args['eval_period'] - 1
            else:
                iter_end = len(dataset_pre.batch_t)
            num_train_all = iter_end - iter_begin + 1

        with tqdm(total=num_train_all, desc='Training') as pbar:
            for iteration in range(iter_begin, iter_end):
                batch_t = dataset_pre.batch_t[iteration]
                r_t = dataset_pre.r_t[iteration]
                doc_level_u_t = dataset_pre.doc_level_u_t[iteration]
                sen_level_u_t = dataset_pre.sen_level_u_t[iteration]
                sent_order = dataset_pre.sent_order[iteration]
                ls = dataset_pre.ls[iteration]
                lr = dataset_pre.lr[iteration]
                aspect_r_t_list = dataset_pre.asp_r_t[iteration]
                rating_dis_list = dataset_pre.rating_dis_list[iteration]

                batch_t_tensor = Variable(batch_t).cuda()
                r_t_tensor = Variable(r_t).cuda()
                doc_u_t_tensor = Variable(doc_level_u_t).cuda()
                sen_u_t_tensor = Variable(sen_level_u_t).cuda()
                sent_order_tensor = Variable(sent_order).cuda()
                aspect_r_t_tensor_list = []
                for aspect_r_t in aspect_r_t_list:
                    aspect_r_t_tensor_list.append(Variable(aspect_r_t).cuda())

                aspect_rating_dis_tensor_list = []
                for rating_dis in rating_dis_list:
                    aspect_rating_dis_tensor_list.append(Variable(rating_dis).cuda())

                self.optimizer.zero_grad()
                out,_,_ = self.net(batch_t_tensor, sent_order_tensor, ls, lr, r_t_tensor, doc_u_t_tensor,sen_u_t_tensor)
                out_all = torch.cat(tuple(out), 0)
                aspect_r_t_tensor_all = torch.cat(tuple(aspect_r_t_tensor_list), 0)

                ok, per, val_i, truth_new = accuracy(out_all, aspect_r_t_tensor_all)

                ok_all += per.data[0]
                mseloss = F.mse_loss(val_i, truth_new.float())
                mean_rmse += math.sqrt(mseloss.data[0])
                mean_mse += mseloss.data[0]

                loss = self.criterion(out[0], aspect_r_t_tensor_list[0])
                for i in range(0, aspect_label_num):
                    loss = loss + self.criterion(out[i], aspect_r_t_tensor_list[i])
                ce_loss += loss.data[0]
                for i in range(0,aspect_label_num):
                    loss = loss + self.args['dis_loss_alpha'] * self.killDiv_criterion(F.log_softmax(out[i],dim=1), aspect_rating_dis_tensor_list[i])
                epoch_loss += loss.data[0]

                loss.backward()
                self.optimizer.step()

                pbar.update(1)
                pbar.set_postfix({"label_size": truth_new.size()[0],
                                  "acc": ok_all / (iteration + 1 - iter_begin),
                                  "CELoss": ce_loss / (iteration + 1 - iter_begin),
                                  "CE+DisLoss": epoch_loss / (iteration + 1 - iter_begin),
                                  "mseloss": mean_mse / (iteration + 1 - iter_begin)})

        logging.info("===> Epoch {} Complete: Avg. Loss: {:.4f}, {}% accuracy".format(epoch,
                                                                               epoch_loss / num_train_all,
                                                                               ok_all / num_train_all))
        train_acc = ok_all / num_train_all
        dev_acc,_ = self._develop()
        return train_acc, dev_acc


    def _tst(self, dataset_pre, msg="Evaluating"):

        self.net.eval()
        ok_all = 0
        pred = 0
        skipped = 0
        mean_mse = 0
        mean_rmse = 0
        all_mse_loss = 0.0
        all_dat_size = 0.0
        with tqdm(total=len(dataset_pre.batch_t), desc=msg) as pbar:
            for iteration in range(0, len(dataset_pre.batch_t)):
                batch_t = dataset_pre.batch_t[iteration]
                r_t = dataset_pre.r_t[iteration]
                doc_level_u_t = dataset_pre.doc_level_u_t[iteration]
                sen_level_u_t = dataset_pre.sen_level_u_t[iteration]
                sent_order = dataset_pre.sent_order[iteration]
                ls = dataset_pre.ls[iteration]
                lr = dataset_pre.lr[iteration]
                aspect_r_t_list = dataset_pre.asp_r_t[iteration]

                batch_t_tensor = Variable(batch_t, volatile =True).cuda()
                r_t_tensor = Variable(r_t).cuda()
                doc_u_t_tensor = Variable(doc_level_u_t).cuda()
                sen_u_t_tensor = Variable(sen_level_u_t).cuda()
                sent_order_tensor = Variable(sent_order).cuda()
                aspect_r_t_tensor_list = []
                for aspect_r_t in aspect_r_t_list:
                    aspect_r_t_tensor_list.append(Variable(aspect_r_t).cuda())

                out,wrd_att_res, sent_att_res = self.net(batch_t_tensor, sent_order_tensor, ls, lr, r_t_tensor, doc_u_t_tensor,sen_u_t_tensor)

                out_all = torch.cat(tuple(out), 0)
                aspect_r_t_tensor_all = torch.cat(tuple(aspect_r_t_tensor_list), 0)
                ok, per, val_i, truth_new = accuracy(out_all, aspect_r_t_tensor_all)
                ok_all += per.data[0]
                mseloss = F.mse_loss(val_i, truth_new.float())
                all_mse_loss += mseloss.data[0] * val_i.size()[0]
                all_dat_size += val_i.size()[0]
                mean_rmse += math.sqrt(mseloss.data[0])
                mean_mse += mseloss.data[0]
                pred += 1

                pbar.update(1)
                pbar.set_postfix({"acc": ok_all / pred, "skipped": skipped, "mseloss": mean_mse / (iteration + 1),
                                  "rmseloss": mean_rmse / (iteration + 1)})


        logging.info("===> {} Complete:  {}% accuracy".format(msg, ok_all / pred))
        all_acc = ok_all / pred
        tst_mse_loss = all_mse_loss / all_dat_size
        return all_acc, tst_mse_loss

    def _get_attent(self, dataset_pre, msg="Evaluating"):
        wrd_attent_res_list = []
        sent_attent_res_list = []
        self.net.eval()
        with tqdm(total=len(dataset_pre.batch_t), desc=msg) as pbar:
            for iteration in range(0, len(dataset_pre.batch_t)):
                batch_t = dataset_pre.batch_t[iteration]
                r_t = dataset_pre.r_t[iteration]
                doc_level_u_t = dataset_pre.doc_level_u_t[iteration]
                sen_level_u_t = dataset_pre.sen_level_u_t[iteration]
                sent_order = dataset_pre.sent_order[iteration]
                ls = dataset_pre.ls[iteration]
                lr = dataset_pre.lr[iteration]
                aspect_r_t_list = dataset_pre.asp_r_t[iteration]

                batch_t_tensor = Variable(batch_t, volatile =True).cuda()
                #batch_t_tensor = Variable(batch_t).cuda()
                r_t_tensor = Variable(r_t).cuda()
                doc_u_t_tensor = Variable(doc_level_u_t).cuda()
                sen_u_t_tensor = Variable(sen_level_u_t).cuda()
                sent_order_tensor = Variable(sent_order).cuda()
                aspect_r_t_tensor_list = []
                for aspect_r_t in aspect_r_t_list:
                    aspect_r_t_tensor_list.append(Variable(aspect_r_t).cuda())

                out, wrd_att_res, sent_att_res = self.net(batch_t_tensor, sent_order_tensor, ls, lr, r_t_tensor, doc_u_t_tensor,sen_u_t_tensor)
                wrd_att_batch_list = []
                for wrd_att in wrd_att_res:
                    wrd_att_batch_list.append(wrd_att.data.cpu())
                sen_att_batch_list = []
                for sen_att in sent_att_res:
                    sen_att_batch_list.append(sen_att.data.cpu())
                wrd_attent_res_list.append(wrd_att_batch_list)
                sent_attent_res_list.append(sen_att_batch_list)

                pbar.update(1)
                # if iteration == 10: break
                # break
                # pbar.set_postfix({"acc": ok_all / pred, "skipped": skipped, "mseloss": mean_mse / (iteration + 1),
                #                   "rmseloss": mean_rmse / (iteration + 1)})


        # logging.info("===> {} Complete:  {}% accuracy".format(msg, ok_all / pred))
        # all_acc = ok_all / pred
        # tst_mse_loss = all_mse_loss / all_dat_size
        return wrd_attent_res_list, sent_attent_res_list


    def test_best_dev_model(self,is_need_run = True):
        model_file = self._get_best_dev_model()
        logging.info('Best Dev Model({})......'.format(model_file))
        if model_file != 'NULL':
            if is_need_run:
                word2index, usr2index, state,_ = self._load_model_para(model_file)
                self._load_dataset(mode = 'test',word2index=word2index, usr2index=usr2index)
                self._construct_net(token_size=len(word2index),n_users = len(usr2index))
                if self.cuda: self.net.cuda()
                # print(self.net.state_dict()['embed.weight'])
                # print(self.net.state_dict()['users.weight'])
                self.net.load_state_dict(state)
                # print(state.keys())
                # print(state['embed.weight'])
                # print(self.net.state_dict()['embed.weight'])
                # print(state['users.weight'])
                # print(self.net.state_dict()['users.weight'])
                # self.net.state_dict()['embed.weight'] = state['embed.weight']
                # print(self.net.state_dict()['embed.weight'])
                test_acc,test_mse = self._test()
                logging.info('test acc: {}, test_mse: {}'.format(test_acc,test_mse))
                return test_acc,test_mse
            else:
                _, _, state, acc_dict = self._load_model_para(model_file)
                # print(state.keys())
                # print(len(state.keys()))
                logging.info('test acc: {}, test_mse: {}'.format(acc_dict['test_acc'],acc_dict['test_mse']))
                return acc_dict['test_acc'],acc_dict['test_mse']
        else:
            logging.info('No Best Model......')
            return -1,-1

    def get_attent_res(self):
        model_file = self._get_best_dev_model()
        logging.info('Best Dev Model({})......'.format(model_file))
        if model_file != 'NULL':
            word2index, usr2index, state,_ = self._load_model_para(model_file)
            self._load_dataset(mode='train')
            self._construct_net(token_size=len(word2index),n_users = len(usr2index))
            if self.cuda: self.net.cuda()
            self.net.load_state_dict(state)
            wrd_attent_res_list, sent_attent_res_list = self._get_attent(self.dataset_train, msg='GetAttentRes')
            # _,_ = self._tst(self.dataset_train, msg="Testing")
            return wrd_attent_res_list,sent_attent_res_list,self.dataset_train,word2index
            # return [],[]
        else:
            logging.info('No Best Model......')
            return [],[]

    def construct_acc_dict(self,train_acc,dev_acc,test_acc,test_mse):
        temp_dict = {}
        temp_dict['train_acc'] = train_acc
        temp_dict['dev_acc'] = dev_acc
        temp_dict['test_acc'] = test_acc
        temp_dict['test_mse'] = test_mse
        return  temp_dict

    def run(self):

        is_better_than_last = True

        not_better_than_last_times = 0

        self._load_dataset(mode = 'train-test-dev')

        self._construct_net(token_size = self.dataset_train.token_size, n_users = self.dataset_train.usr_size)

        self.net.set_emb_tensor(self.dataset_train.wrd_embed_tensor)

        self._prepare_net_info()

        self.count_parameters()

        self._mkdir_model_filefold()

        if self.args['load_model']:

            bst_model_file = self._get_best_dev_model()

            logging.info("best_model_file : {}.\n-------------".format(bst_model_file))

            dev_acc_old = 1.0
            if bst_model_file != 'NULL':
                _,_, state,_ = self._load_model_para(bst_model_file)
                self.net.load_state_dict(state)
                dev_acc_old,_= self._develop()
        else:
            dev_acc_old = 0.0

        is_can_break = False

        for epoch in range(1, self.epoch_size + 1):
            logging.info("\n-------EPOCH {}-------".format(epoch))
            if self.args['eval_period'] == -1:
                eval_per_time_list = [-1]
            else:
                if len(self.dataset_train.batch_t) % self.args['eval_period'] == 0:
                    end_value = int(len(self.dataset_train.batch_t) / self.args['eval_period'])
                else:
                    end_value = int(len(self.dataset_train.batch_t) / self.args['eval_period']) + 1
                eval_per_time_list = range(0,end_value)

            for i,eval_per_time in enumerate(eval_per_time_list):
                logging.info('Training Phase: {} / {}'.format(i+1,len(eval_per_time_list)))
                train_acc, dev_acc = self._train(epoch,
                                                 eval_per_time=eval_per_time,
                                                 is_last = (i == (len(eval_per_time_list) - 1)))

                # self._save_model('model_' + str(epoch),self.construct_acc_dict(train_acc,dev_acc,test_acc))

                if dev_acc >= dev_acc_old:
                    is_better_than_last = True
                    not_better_than_last_times = 0
                    dev_acc_old = dev_acc
                    test_acc, test_mse = 0.0,0.0
                    self._save_model('best_dev_model',self.construct_acc_dict(train_acc,dev_acc,test_acc,test_mse))
                else:
                    is_better_than_last = False
                    not_better_than_last_times += 1

                logging.info('not_better_than_last_times: {}'.format(not_better_than_last_times))

                if not is_better_than_last:
                    self._adjust_learning_rate()
                if not_better_than_last_times >= self.args['early_stop']:
                    logging.info('not_better_than_last_times is bigger than early_stop_thres.')
                    logging.info('Training break......')
                    is_can_break = True
                    break
            if is_can_break:
                break



class RunMODELB(RunMODELA):

    def _construct_net(self,token_size,n_users):
        self.net = MODELB(ntoken=token_size,
                                n_rating=self.dataset_dict['class_num'],
                                n_users = n_users,
                                emb_size = self.args['embed_size'],
                                hid_size = self.args['hidden_size'],
                                num_class = self.dataset_dict['class_num'],
                                aspect_label_num = self.dataset_dict['aspect_num'],
                                mode = self.args['model_mode'],
                                rnn_type = self.args['rnn_type'],
                                drop_out_rate = self.args['drop_out_rate'],
                                model_init_needed = self.args['model_init_needed'])

    def run(self):

        is_better_than_last = True

        not_better_than_last_times = 0

        self._load_dataset(mode = 'train-test-dev')

        self._construct_net(token_size = self.dataset_train.token_size, n_users = self.dataset_train.usr_size)

        if self.args['load_init_aspect_emb']:
            self.net.set_asp_emb_tensor(self.dataset_train.aspect_embedding)

        self.net.set_emb_tensor(self.dataset_train.wrd_embed_tensor)

        self._prepare_net_info()

        self.count_parameters()

        self._mkdir_model_filefold()

        if self.args['load_model']:

            bst_model_file = self._get_best_dev_model()

            logging.info("best_model_file : {}.\n-------------".format(bst_model_file))

            dev_acc_old = 1.0
            if bst_model_file != 'NULL':
                _,_, state,_ = self._load_model_para(bst_model_file)
                self.net.load_state_dict(state)
                dev_acc_old,_= self._develop()
        else:
            dev_acc_old = 0.0
        is_can_break = False
        for epoch in range(1, self.epoch_size + 1):
            logging.info("\n-------EPOCH {}-------".format(epoch))

            if self.args['eval_period'] == -1:
                eval_per_time_list = [-1]
            else:
                if len(self.dataset_train.batch_t) % self.args['eval_period'] == 0:
                    end_value = int(len(self.dataset_train.batch_t) / self.args['eval_period'])
                else:
                    end_value = int(len(self.dataset_train.batch_t) / self.args['eval_period']) + 1
                eval_per_time_list = range(0,end_value)

            print('args[\'eval_period\']:{}'.format(self.args['eval_period']))

            for i,eval_per_time in enumerate(eval_per_time_list):
                logging.info('Training Phase: {} / {}'.format(i + 1, len(eval_per_time_list)))
                train_acc, dev_acc = self._train(epoch,eval_per_time=eval_per_time,
                                                 is_last = (i == (len(eval_per_time_list) - 1)))

                # self._save_model('model_' + str(epoch), self.construct_acc_dict(train_acc, dev_acc, test_acc))

                if dev_acc >= dev_acc_old:
                    is_better_than_last = True
                    not_better_than_last_times = 0
                    dev_acc_old = dev_acc
                    test_acc,test_mse = 0.0,0.0
                    self._save_model('best_dev_model', self.construct_acc_dict(train_acc, dev_acc, test_acc,test_mse))
                else:
                    is_better_than_last = False
                    not_better_than_last_times += 1

                logging.info('not_better_than_last_times: {}'.format(not_better_than_last_times))

                if not is_better_than_last:
                    self._adjust_learning_rate()

                if not_better_than_last_times >= self.args['early_stop']:
                    logging.info('not_better_than_last_times is bigger than early_stop_thres.')
                    logging.info('Training break......')
                    is_can_break = True
                    break
            if is_can_break:
                break


class RunMBGRU(RunMODELA):

    def _construct_net(self,token_size,n_users):
        self.net = MBGRU(ntoken=token_size,
                         emb_size = self.args['embed_size'],
                         hid_size = self.args['hidden_size'],
                         aspect_label_num = self.dataset_dict['aspect_num'],
                         num_class = self.dataset_dict['class_num'],
                         mode = self.args['model_mode'],
                         rnn_type = self.args['rnn_type'],
                         drop_out_rate = self.args['drop_out_rate'],
                         model_init_needed = self.args['model_init_needed'])
