# Copyright 2018 Owkin, inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script to generate data to be registered to Substra
Titanic example
"""

import logging
import json
import os

from contextlib import contextmanager
from types import SimpleNamespace

import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm


DATA_OWNER_NUM = 2
DATA_SAMPLE_NUM = 50 # Per data owner.
# Each such Substra sample contains multiple MNIST samples
DATA_BATCH_NUM = 16
DEFAULT_TRAIN_CLIENTS_NUM = 3400
DEFAULT_TEST_CLIENTS_NUM = 3400
DEFAULT_BATCH_SIZE = DATA_BATCH_NUM
DEFAULT_TRAIN_FILE = 'fed_emnist_train.h5'
DEFAULT_TEST_FILE = 'fed_emnist_test.h5'

# group name defined by tff in h5 file
_EXAMPLE = 'examples'
_IMGAE = 'pixels'
_LABEL = 'label'

client_ids_train = None
client_ids_test = None

default_stream_handler = logging.StreamHandler()
substra_logger = logging.getLogger('substra')
substra_logger.addHandler(default_stream_handler)

@contextmanager
def progress_bar(length):
    """Provide progress bar for for loops"""

    pg = tqdm(total=length)
    progress_handler = logging.StreamHandler(
        SimpleNamespace(write=lambda x: pg.write(x, end='')))
    substra_logger.removeHandler(default_stream_handler)
    substra_logger.addHandler(progress_handler)
    try:
        yield pg
    finally:
        pg.close()
        substra_logger.removeHandler(progress_handler)
        substra_logger.addHandler(default_stream_handler)

def batch_data(dataset, data_dir, train_bs, test_bs, client_idx=None):

    train_h5 = h5py.File(os.path.join(data_dir, DEFAULT_TRAIN_FILE), 'r')
    test_h5 = h5py.File(os.path.join(data_dir, DEFAULT_TEST_FILE), 'r')

    # load data
    if client_idx is None:
        # get ids of all clients
        train_ids = client_ids_train
        test_ids = client_ids_test
    else:
        # get ids of single client
        train_ids = [client_ids_train[client_idx]]
        test_ids = [client_ids_test[client_idx]]

    # load data in numpy format from h5 file
    train_x = np.vstack([train_h5[_EXAMPLE][client_id][_IMGAE][()] for client_id in train_ids])
    train_y = np.vstack([train_h5[_EXAMPLE][client_id][_LABEL][()] for client_id in train_ids]).squeeze()
    test_x = np.vstack([test_h5[_EXAMPLE][client_id][_IMGAE][()] for client_id in test_ids])
    test_y = np.vstack([test_h5[_EXAMPLE][client_id][_LABEL][()] for client_id in test_ids]).squeeze()

    # randomly shuffle train and test data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(train_x)
    np.random.set_state(rng_state)
    np.random.shuffle(train_y)
    np.random.set_state(rng_state)
    np.random.shuffle(test_x)
    np.random.set_state(rng_state)
    np.random.shuffle(test_y)

    # loop through mini-batches
    train_batch_data = list()
    for i in range(0, len(train_x), train_bs):
        batched_x = train_x[i:i + train_bs]
        batched_x = np.reshape(batched_x, (batched_x.shape[0], -1)) # (BS, 28, 28) -> (BS, 784)
        batched_x = pd.DataFrame(batched_x)
        batched_y = pd.DataFrame(train_y[i:i + train_bs])
        train_batch_data.append((batched_x, batched_y))

    test_batch_data = list()
    for i in range(0, len(test_x), test_bs):
        batched_x = test_x[i:i + test_bs]
        batched_x = np.reshape(batched_x, (batched_x.shape[0], -1)) # (BS, 28, 28) -> (BS, 784)
        batched_x = pd.DataFrame(batched_x)
        batched_y = pd.DataFrame(test_y[i:i + test_bs])
        test_batch_data.append((batched_x, batched_y))

    train_h5.close()
    test_h5.close()

    return train_batch_data, test_batch_data

def load_partition_data_federated_emnist(dataset, data_dir, batch_size=DEFAULT_BATCH_SIZE):

    # client ids
    train_file_path = os.path.join(data_dir, DEFAULT_TRAIN_FILE)
    test_file_path = os.path.join(data_dir, DEFAULT_TEST_FILE)
    with h5py.File(train_file_path, 'r') as train_h5, h5py.File(test_file_path, 'r') as test_h5:
        global client_ids_train, client_ids_test
        client_ids_train = list(train_h5[_EXAMPLE].keys())
        client_ids_test = list(test_h5[_EXAMPLE].keys())

    # global dataset
    train_data_num = 0
    test_data_num = 0
    train_data_global = list()
    test_data_global = list()

    # local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    print('Building the client batches...')
    with progress_bar(DEFAULT_TRAIN_CLIENTS_NUM) as progress:
        for client_idx in range(DEFAULT_TRAIN_CLIENTS_NUM):
            train_data_local, test_data_local = batch_data(dataset, data_dir, batch_size, batch_size, client_idx)
            local_data_num = len(train_data_local) + len(test_data_local)
            data_local_num_dict[client_idx] = local_data_num
            # logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))
            # logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            #     client_idx, len(train_data_local), len(test_data_local)))
            train_data_local_dict[client_idx] = train_data_local
            test_data_local_dict[client_idx] = test_data_local
            train_data_global += train_data_local
            test_data_global += test_data_local
            user_train_data_num = len(train_data_local) # FIX
            user_test_data_num = len(test_data_local) # FIX
            train_data_num += user_train_data_num
            test_data_num += user_test_data_num
            progress.update()
    
    # class number
    train_file_path = os.path.join(data_dir, DEFAULT_TRAIN_FILE)
    with h5py.File(train_file_path, 'r') as train_h5:
        class_num = len(np.unique([train_h5[_EXAMPLE][client_ids_train[idx]][_LABEL][0] for idx in range(DEFAULT_TRAIN_CLIENTS_NUM)]))
        logging.info("class_num = %d" % class_num)

    return DEFAULT_TRAIN_CLIENTS_NUM, train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num

def convert_dict(input_dict: dict, input_type: str, num_owners: int, num_samples: int) -> list:
    input_len = len(input_dict)
    input_list = list(input_dict.values())
    owner_sample_len = input_len // num_owners
    owner_list = []
    for owner_idx in range(num_owners):
        print('Converting %s data for data owner %d...' % (input_type, owner_idx))
        with progress_bar(num_samples) as progress:
            sample_list = []
            owner_start = owner_idx * input_len // num_owners
            for sample_idx in range(num_samples):
                sample_start = owner_start + sample_idx * owner_sample_len // num_samples
                sample_end = owner_start + (sample_idx + 1) * owner_sample_len // num_samples
                batch_x = [batch[0] for sample in input_list[sample_start : sample_end] for batch in sample]
                batch_y = [batch[1] for sample in input_list[sample_start : sample_end] for batch in sample]
                sample_list.append((pd.concat(batch_x), pd.concat(batch_y)))
                progress.update()
            owner_list.append(sample_list)

    return owner_list

def save_data(configs):
    for conf in configs:
        print('Saving %s data for data owner %d...' %
              (conf['data_type'], conf['data_owner']))
        with progress_bar(len(conf['data'])) as progress:
            for i, data_sample in enumerate(conf['data']):
                filename_x = os.path.join(conf['data_samples_root'],
                                          f'data_sample_{i}/data_sample_{i}_x.csv')
                filename_y = os.path.join(conf['data_samples_root'],
                                          f'data_sample_{i}/data_sample_{i}_y.csv')
                os.makedirs(os.path.dirname(filename_x))
                with open(filename_x, 'w') as f:
                    data_sample[0].to_csv(f, header=False, index=False)
                with open(filename_y, 'w') as f:
                    data_sample[1].to_csv(f, header=False, index=False)
                progress.update()


if __name__ == "__main__":

    root_path = os.path.dirname(__file__)
    asset_path = os.path.join(root_path, '../assets')

    client_num,\
    train_data_num,\
    test_data_num,\
    train_data_global,\
    test_data_global, \
    train_data_local_num_dict,\
    train_data_local_dict,\
    test_data_local_dict, \
    class_num = load_partition_data_federated_emnist("femnist", "./data/FederatedEMNIST/datasets")

    train_data = convert_dict(train_data_local_dict,
                              'train',
                              DATA_OWNER_NUM,
                              DATA_SAMPLE_NUM)

    test_data = convert_dict(test_data_local_dict,
                             'test',
                             DATA_OWNER_NUM,
                             DATA_SAMPLE_NUM)

    # Save train and test data samples for every data owner
    train_test_configs = []
    for idx in range(DATA_OWNER_NUM):
        train_test_configs.append({
            'data_owner' : idx + 1,
            'data_type' : 'train',
            'data': train_data[idx],
            'data_samples_root': os.path.join(asset_path,
                                              'data/femnist/train_data%s_samples' % str(idx + 1)),
        })

        train_test_configs.append({
            'data_owner': idx + 1,
            'data_type': 'test',
            'data': test_data[idx],
            'data_samples_root': os.path.join(asset_path,
                                              'data/femnist/test_data%s_samples' % str(idx + 1)),
        })

    save_data(train_test_configs)
