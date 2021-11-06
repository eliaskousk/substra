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

import pandas as pd
import numpy as np
from tqdm import tqdm


DATA_OWNER_NUM = 2
DATA_SAMPLE_NUM = 50 # Per data owner.
# Each such Substra sample contains multiple MNIST samples
DATA_BATCH_NUM = 128

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

def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of non-unique client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''

    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]

    print('Reading train data...')
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]

    print('Reading test data...')
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = sorted(cdata['users'])

    return clients, groups, train_data, test_data

def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''

    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    batch_data = list()
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_y = data_y[i:i + batch_size]
        batched_x = pd.DataFrame(batched_x)
        batched_y = pd.DataFrame(batched_y)
        batch_data.append((batched_x, batched_y))

    return batch_data

def load_partition_data_mnist(batch_size,
                              train_path="./data/MNIST/train",
                              test_path="./data/MNIST/test"):

    users, groups, train_data, test_data = read_data(train_path, test_path)

    if len(groups) == 0:
        groups = [None for _ in users]

    train_data_num = 0
    test_data_num = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_global = list()
    test_data_global = list()
    client_idx = 0

    print('Building the client batches...')
    with progress_bar(len(users)) as progress:
        for u, g in zip(users, groups):
            user_train_data_num = len(train_data[u]['x'])
            user_test_data_num = len(test_data[u]['x'])
            train_data_num += user_train_data_num
            test_data_num += user_test_data_num
            train_data_local_num_dict[client_idx] = user_train_data_num

            # transform to batches
            train_batch = batch_data(train_data[u], batch_size)
            test_batch = batch_data(test_data[u], batch_size)

            # index using client index
            train_data_local_dict[client_idx] = train_batch
            test_data_local_dict[client_idx] = test_batch
            train_data_global += train_batch
            test_data_global += test_batch
            client_idx += 1
            progress.update()

    print('Finished building the client batches...')
    client_num = client_idx
    class_num = 10

    return client_num,\
           train_data_num,\
           test_data_num,\
           train_data_global,\
           test_data_global, \
           train_data_local_num_dict,\
           train_data_local_dict,\
           test_data_local_dict,\
           class_num

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
                sample_x = [sample[0][0] for sample in input_list[sample_start : sample_end]]
                sample_y = [sample[0][1] for sample in input_list[sample_start : sample_end]]
                sample_list.append((pd.concat(sample_x), pd.concat(sample_y)))
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
    class_num = load_partition_data_mnist(DATA_BATCH_NUM)

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
                                              'train_data%s_samples' % str(idx + 1)),
        })

        train_test_configs.append({
            'data_owner': idx + 1,
            'data_type': 'test',
            'data': test_data[idx],
            'data_samples_root': os.path.join(asset_path,
                                              'test_data%s_samples' % str(idx + 1)),
        })

    save_data(train_test_configs)
