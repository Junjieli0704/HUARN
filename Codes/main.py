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
    if args['model_type'] == 'MBGRU':
        return RunMBGRU(args = args,
                       dataset_dict = data_dict[args['dataset']],
                       model_filefold = args['model_filefold'])
    elif args['model_type'] == 'MBGRUAsp':
        return RunMBGRUAsp(args = args,
                               dataset_dict = data_dict[args['dataset']],
                               model_filefold=args['model_filefold'])
    elif args['model_type'] == 'MHAN':
        return RunMHAN(args = args,
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

def print_test_out(args):
    torch.cuda.set_device(args['gpu_id'])
    run_NN = get_run_model(args)
    return run_NN.print_test_result(out_file='printtst.txt')


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

    parser.add_argument("--config", type=str, default='config.json',help = 'path to json config', required = True)
    parser.add_argument("--dataset", type=str, default='NULL',    help = 'TripOUR or TripDMS')
    parser.add_argument("--model_type", type=str, default='NULL', help = 'MBGRU, MBGRUAsp or MHAN')
    parser.add_argument("--model_mode", type=str, default='NULL',help = "'+U','+R','+U+R',''")
    parser.add_argument("--code_mode", type=str, default='NULL',help = "train, test")

    args = parser.parse_args()

    para_json = load_config_para(args.config)

    if para_json['log_file'] == '': set_log_file(para_json)
    set_random_seed(para_json)
    return para_json

def set_random_seed(args):
    random.seed(args['seed'])
    if args['cuda']:
        torch.cuda.manual_seed(args['seed'])
    else:
        torch.manual_seed(args['seed'])
    numpy.random.seed(args['seed'])

def test_models(out_file,args,is_need_run = False):

    def get_test_model_info_list():
        trip_model_list = ['TripOUR-MBGRU-150-',
                           'TripOUR-MHAN-150-',
                           'TripOUR-MBGRUAsp-150-',
                           'TripOUR-MBGRU-150-+R',
                           'TripOUR-MBGRUAsp-150-+R',
                           'TripOUR-MBGRU-150-+U',
                           'TripOUR-MBGRUAsp-150-+U',
                           'TripOUR-MBGRU-150-+U+R',
                           'TripOUR-MBGRUAsp-150-+U+R'
                           ]
        Trip_model_list = ['TripDMS-MBGRU-150-',
                           'TripDMS-MHAN-150-',
                           'TripDMS-MBGRUAsp-150-',
                           'TripDMS-MBGRU-150-+R',
                           'TripDMS-MBGRUAsp-150-+R',
                           'TripDMS-MBGRU-150-+U',
                           'TripDMS-MBGRUAsp-150-+U',
                           'TripDMS-MBGRU-150-+U+R',
                           'TripDMS-MBGRUAsp-150-+U+R'
                           ]

        model_info_list = []
        model_to_index = {}
        for k in range(0,len(trip_model_list)):
            trip_dat, trip_model, trip_batch, trip_mode = trip_model_list[k].split('-')
            Trip_dat, Trip_model, Trip_batch, Trip_mode = Trip_model_list[k].split('-')
            model_info_dict = {}
            model_info_dict['dataset'] = trip_dat
            model_info_dict['model_mode'] = trip_mode
            model_info_dict['batch_size'] = int(trip_batch)
            model_info_dict['model_type'] = trip_model
            model_info_list.append(model_info_dict)
            model_to_index[trip_model_list[k]] = len(model_info_list) - 1
            model_info_dict2 = {}
            model_info_dict2['dataset'] = Trip_dat
            model_info_dict2['model_mode'] = Trip_mode
            model_info_dict2['batch_size'] = int(Trip_batch)
            model_info_dict2['model_type'] = Trip_model
            model_info_list.append(model_info_dict2)
            model_to_index[Trip_model_list[k]] = len(model_info_list) - 1
        return (model_info_list, model_to_index,trip_model_list,Trip_model_list)

    model_info_list, model_to_index,trip_model_list,Trip_model_list = get_test_model_info_list()
    dataset_list = ['TripOUR','TripDMS']
    out_line_con_list = []
    out_line_con_list.append(' ' + '\t' + '\t'.join(dataset_list))
    tst_acc_ans_list = []
    tst_mse_ans_list = []
    for i, model_info in enumerate(model_info_list):
        logging.info('model_ID: {} / {}'.format(i,len(model_info_list)))
        args['dataset'] = model_info['dataset']
        args['model_mode'] = model_info['model_mode']
        args['batch_size'] = model_info['batch_size']
        args['model_type'] = model_info['model_type']
        test_acc, test_mse = test(args, is_need_run)
        tst_acc_ans_list.append(round(test_acc,4))
        tst_mse_ans_list.append(round(test_mse, 4))



    for i, trip_model_str in enumerate(trip_model_list):
        trip_dat, trip_model, trip_batch, trip_mode = trip_model_str.split('-')
        temp_list = []
        temp_list.append(trip_model + trip_mode)
        temp_list.append(str(tst_acc_ans_list[model_to_index[trip_model_str]]))
        temp_list.append(str(tst_mse_ans_list[model_to_index[trip_model_str]]))
        temp_list.append(str(tst_acc_ans_list[model_to_index[Trip_model_list[i]]]))
        temp_list.append(str(tst_mse_ans_list[model_to_index[Trip_model_list[i]]]))
        out_line_con_list.append('\t'.join(temp_list))

    print(model_to_index)
    print(model_info_list)
    print(Trip_model_list[0])
    print(trip_model_list[0])
    print(tst_acc_ans_list)

    open(out_file,'w+').write('\n'.join(out_line_con_list))



if __name__ == '__main__':
    para_json = get_argparse()
    config_logging(para_json['log_file'])
    logging.info(str(para_json))
    logging.info('LOG File: {}'.format(para_json['log_file']))

    if para_json['code_mode'].find('train') != -1:
        train(para_json)
    if para_json['code_mode'].find('test') != -1:
        test(para_json,True)
    if para_json['code_mode'].find('printst') != -1:
        print_test_out(para_json)
