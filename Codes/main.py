import argparse
import pickle as pkl
from tqdm import tqdm
import math

import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from Nets import *
import argparse
import torch
import dat_const
from tqdm import tqdm
from TrainTest import *
import logging
import numpy
import random
import os
import json

def load_config_para(config_file):
    try:
        config_json = json.load(open(config_file, 'r'))
        return config_json
    except Exception as e:
        logging.error(str(e))
        exit(0)

def config_logging(log_file):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y %b %d %H:%M:%S',
                        filename=log_file,
                        filemode='w+')

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def get_run_model(args):
    data_dict = dat_const.get_data_dict()
    if args['model_type'] == 'MODELA':
        return RunMODELA(args = args,
                       dataset_dict = data_dict[args['dataset']],
                       model_filefold = args['model_filefold'])
    elif args['model_type'] == 'MODELB':
        return RunMODELB(args = args,
                               dataset_dict = data_dict[args['dataset']],
                               model_filefold=args['model_filefold'])
    elif args['model_type'] == 'MBGRU':
        return RunMBGRU(args = args,
                        dataset_dict = data_dict[args['dataset']],
                        model_filefold=args['model_filefold'])

def train(args):
    torch.cuda.set_device(args['gpu_id'])
    run_NN = get_run_model(args)
    run_NN.run()

def test(args,is_need_run = True):
    torch.cuda.set_device(args['gpu_id'])
    run_NN = get_run_model(args)
    return run_NN.test_best_dev_model(is_need_run)

def set_log_file(args):

    def mk_dir(path):
        path = path.strip()
        path = path.rstrip("\\")
        is_exist = os.path.exists(path)
        if not is_exist:
            os.makedirs(path)

    mk_dir('./Log/')
    log_file = './Log/' + args['dataset'] + '_' + args['model_type'] + args['model_mode']
    log_file += '_hidden(' + str(args['hidden_size']) + ')'
    log_file += '_rnnType(' + str(args['rnn_type']) + ')'
    log_file += '_dropOutRate(' + str(args['drop_out_rate']) + ')'
    log_file += '_disLossAlpha(' + str(args['dis_loss_alpha']) + ')'
    if args['model_init_needed']: log_file += '_init(T)'
    else: log_file += '_init(F)'
    log_file += '.log'
    args['log_file'] = log_file

def get_argparse():

    parser = argparse.ArgumentParser(description='Model for Document Level Multi-Aspect Sentiment Classification')

    parser.add_argument("--config", type=str, default='config.json')

    args = parser.parse_args()

    para_json = load_config_para(args.config)

    if para_json['log_file'] == '': set_log_file(para_json)
    return para_json

def set_random_seed(args):
    random.seed(args['seed'])
    if args['cuda']:
        torch.cuda.manual_seed(args['seed'])
    else:
        torch.manual_seed(args['seed'])
    numpy.random.seed(args['seed'])

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Model for Document Level Multi-Aspect Sentiment Classification')
    parser.add_argument("--config", type=str, default='config.json')
    args = parser.parse_args()
    para_json = load_config_para(args.config)
    if para_json['log_file'] == '': set_log_file(para_json)
    set_random_seed(para_json)
    config_logging(para_json['log_file'])
    logging.info(str(para_json))
    logging.info('LOG File: {}'.format(para_json['log_file']))

    if para_json['code_mode'].find('train') != -1:
        train(para_json)
    if para_json['code_mode'].find('test') != -1:
        test(para_json,True)
