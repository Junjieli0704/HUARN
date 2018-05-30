#-*- coding: UTF-8 -*-

def get_model_type_dict():
    model_type_dict = {}
    model_type_dict['MODELA'] = {'batch_size': 32}
    model_type_dict['MODELB'] = {'batch_size': 10}
    return model_type_dict

def get_data_dict():
    def get_data_info(dataname_dict):
        for key, dict in dataname_dict.items():
            dict['train'] = '../Data/' + key + '/train'
            dict['test'] = '../Data/' + key + '/test'
            dict['dev'] = '../Data/' + key + '/dev'
            dict['emb'] = '../Data/' + key + '/glove_6B_max_3_embs'
            #dict['emb'] = '../Data/' + key + '/embs'
            dict['word_dict'] = '../Data/' + key + '/word_dict'
            dict['aspect_words'] = '../Data/' + key + '/aspect.words'

    data_dict = {}
    data_dict['TripDMS'] = {'class_num': 5, 'aspect_num': 7}
    data_dict['TripOUR'] = {'class_num': 5, 'aspect_num': 7}

    get_data_info(data_dict)

    return data_dict



if __name__ == '__main__':
    print(get_data_dict())
