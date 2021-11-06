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

import json
import logging
import os
from contextlib import contextmanager
from types import SimpleNamespace

from tqdm import tqdm

import substra


DATA_OWNER_NUM = 2

default_stream_handler = logging.StreamHandler()
substra_logger = logging.getLogger('substra')
substra_logger.addHandler(default_stream_handler)

@contextmanager
def progress_bar(length):
    """Provide progress bar for for loops"""
    pg = tqdm(total=length)
    progress_handler = logging.StreamHandler(SimpleNamespace(write=lambda x: pg.write(x, end='')))
    substra_logger.removeHandler(default_stream_handler)
    substra_logger.addHandler(progress_handler)
    try:
        yield pg
    finally:
        pg.close()
        substra_logger.removeHandler(progress_handler)
        substra_logger.addHandler(default_stream_handler)


current_directory = os.path.dirname(__file__)
assets_directory = os.path.join(current_directory, '../assets')

client1 = substra.Client.from_config_file(profile_name="node-1")
client2 = substra.Client.from_config_file(profile_name="node-2")

DATASET1 = {
    'name': 'MNIST Dataset 1 - Private [Auth: MyOrg2MSP]',
    'type': 'csv',
    'data_opener': os.path.join(assets_directory, 'dataset/opener.py'),
    'description': os.path.join(assets_directory, 'dataset/description.md'),
    'permissions': {
        'public': False,
        'authorized_ids': ["MyOrg2MSP"]
    },
}

DATASET2 = {
    'name': 'MNIST Dataset 2 - Private [Auth: MyOrg1MSP]',
    'type': 'csv',
    'data_opener': os.path.join(assets_directory, 'dataset/opener.py'),
    'description': os.path.join(assets_directory, 'dataset/description.md'),
    'permissions': {
        'public': False,
        'authorized_ids': ["MyOrg1MSP"]
    },
}

TRAIN_DATA1_SAMPLES_PATHS = [
    os.path.join(assets_directory, 'train_data1_samples', path)
    for path in os.listdir(os.path.join(assets_directory, 'train_data1_samples'))
]

TRAIN_DATA2_SAMPLES_PATHS = [
    os.path.join(assets_directory, 'train_data2_samples', path)
    for path in os.listdir(os.path.join(assets_directory, 'train_data2_samples'))
]

TEST_DATA1_SAMPLES_PATHS = [
    os.path.join(assets_directory, 'test_data1_samples', path)
    for path in os.listdir(os.path.join(assets_directory, 'test_data1_samples'))
]

TEST_DATA2_SAMPLES_PATHS = [
    os.path.join(assets_directory, 'test_data2_samples', path)
    for path in os.listdir(os.path.join(assets_directory, 'test_data2_samples'))
]

print('Adding datasets...')
dataset1_key = client1.add_dataset(DATASET1)
assert dataset1_key, 'Missing data manager key 1'
dataset2_key = client2.add_dataset(DATASET2)
assert dataset1_key, 'Missing data manager key 2'

train_data1_sample_keys = []
train_data2_sample_keys = []
test_data1_sample_keys = []
test_data2_sample_keys = []
data_samples_configs = (
    {
        'message': 'Adding train data1 samples...',
        'paths': TRAIN_DATA1_SAMPLES_PATHS,
        'test_only': False,
        'data_sample_keys': train_data1_sample_keys,
        'missing_message': 'Missing train data1 samples keys',
        'dataset_key': dataset1_key,
        'client' : client1
    },
    {
        'message': 'Adding train data2 samples...',
        'paths': TRAIN_DATA2_SAMPLES_PATHS,
        'test_only': False,
        'data_sample_keys': train_data2_sample_keys,
        'missing_message': 'Missing train data2 samples keys',
        'dataset_key': dataset2_key,
        'client' : client2
    },
    {
        'message': 'Adding test data1 samples...',
        'paths': TEST_DATA1_SAMPLES_PATHS,
        'test_only': True,
        'data_sample_keys': test_data1_sample_keys,
        'missing_message': 'Missing test data1 samples keys',
        'dataset_key': dataset1_key,
        'client' : client1
    },
    {
        'message': 'Adding test data2 samples...',
        'paths': TEST_DATA2_SAMPLES_PATHS,
        'test_only': True,
        'data_sample_keys': test_data2_sample_keys,
        'missing_message': 'Missing test data2 samples keys',
        'dataset_key': dataset2_key,
        'client' : client2
    },
)
for conf in data_samples_configs:
    print(conf['message'])
    with progress_bar(len(conf['paths'])) as progress:
        for path in conf['paths']:
            data_sample_key = conf['client'].add_data_sample({
                'data_manager_keys': [conf['dataset_key']],
                'test_only': conf['test_only'],
                'path': path,
            }, local=True)
            conf['data_sample_keys'].append(data_sample_key)
            progress.update()
    assert len(conf['data_sample_keys']), conf['missing_message']

print('Associating data samples with datasets...')
client1.link_dataset_with_data_samples(
    dataset1_key,
    train_data1_sample_keys + test_data1_sample_keys,
)
client2.link_dataset_with_data_samples(
    dataset2_key,
    train_data2_sample_keys + test_data2_sample_keys,
)

# Save assets keys
assets1_keys = {
    'dataset_key': dataset1_key,
    'train_data_sample_keys': train_data1_sample_keys,
    'test_data_sample_keys': test_data1_sample_keys,
}
assets1_keys_path = os.path.join(current_directory, '../assets1_keys.json')
with open(assets1_keys_path, 'w') as f:
    json.dump(assets1_keys, f, indent=2)

assets2_keys = {
    'dataset_key': dataset2_key,
    'train_data_sample_keys': train_data2_sample_keys,
    'test_data_sample_keys': test_data2_sample_keys,
}
assets2_keys_path = os.path.join(current_directory, '../assets2_keys.json')
with open(assets2_keys_path, 'w') as f:
    json.dump(assets2_keys, f, indent=2)

print(f'Assets keys have been saved to {os.path.abspath(assets1_keys_path)} and {os.path.abspath(assets2_keys_path)} ')
