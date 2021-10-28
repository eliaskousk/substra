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
import zipfile

import substra


current_directory = os.path.dirname(__file__)
assets_directory = os.path.join(current_directory, '../assets')

client = substra.Client.from_config_file(profile_name="node-1")

OBJECTIVE = {
    'name1': 'MNIST Objective - Private [Auth: MyOrg2MSP]',
    'name2': 'MNIST Objective - Private [Auth: MyOrg2MSP]',
    'description': os.path.join(assets_directory, 'objective/description.md'),
    'metrics_name': 'accuracy',
    'metrics': os.path.join(assets_directory, 'objective/metrics.zip'),
    'permissions1': {
        'public': False,
        'authorized_ids': ["MyOrg2MSP"]
    },
    'permissions2': {
        'public': False,
        'authorized_ids': ["MyOrg2MSP"]
    },
}

METRICS_DOCKERFILE_FILES = [
    os.path.join(assets_directory, 'objective/metrics.py'),
    os.path.join(assets_directory, 'objective/Dockerfile')
]

archive_path = OBJECTIVE['metrics']
with zipfile.ZipFile(archive_path, 'w') as z:
    for filepath in METRICS_DOCKERFILE_FILES:
        z.write(filepath, arcname=os.path.basename(filepath))

assets1_keys_path = os.path.join(current_directory, '../assets1_keys.json')
assets1_keys = {}
with open(assets1_keys_path, 'r') as f:
    assets1_keys = json.load(f)

assets2_keys_path = os.path.join(current_directory, '../assets2_keys.json')
assets2_keys = {}
with open(assets2_keys_path, 'r') as f:
    assets2_keys = json.load(f)

print('Adding objective1...')
objective1_key = client.add_objective({
    'name': OBJECTIVE['name1'],
    'description': OBJECTIVE['description'],
    'metrics_name': OBJECTIVE['metrics_name'],
    'metrics': OBJECTIVE['metrics'],
    'test_data_sample_keys': assets1_keys['test_data_sample_keys'],
    'test_data_manager_key': assets1_keys['dataset_key'],
    'permissions': OBJECTIVE['permissions1'],
})
assert objective1_key, 'Missing objective1 key'

print('Adding objective2...')
objective2_key = client.add_objective({
    'name': OBJECTIVE['name2'],
    'description': OBJECTIVE['description'],
    'metrics_name': OBJECTIVE['metrics_name'],
    'metrics': OBJECTIVE['metrics'],
    'test_data_sample_keys': assets2_keys['test_data_sample_keys'],
    'test_data_manager_key': assets2_keys['dataset_key'],
    'permissions': OBJECTIVE['permissions2'],
})
assert objective2_key, 'Missing objective2 key'

# Save assets keys
assets1_keys['objective_key'] = objective1_key
assets2_keys['objective_key'] = objective2_key

with open(assets1_keys_path, 'w') as f:
    json.dump(assets1_keys, f, indent=2)

with open(assets2_keys_path, 'w') as f:
    json.dump(assets2_keys, f, indent=2)

print(f'Assets keys have been saved to {os.path.abspath(assets1_keys_path)} and {os.path.abspath(assets2_keys_path)}')
