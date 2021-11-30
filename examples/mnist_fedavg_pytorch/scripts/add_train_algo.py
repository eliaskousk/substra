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
import os
import zipfile

import substra


current_directory = os.path.dirname(__file__)
assets_directory = os.path.join(current_directory, '../assets')

client = substra.Client.from_config_file(profile_name="node-1")

ALGO_LOCAL = {
    'name': 'MNIST Local Algo Linear Regression - Private [Auth: MyOrg2MSP]',
    'description': os.path.join(assets_directory, 'algo/description.md'),
    'permissions': {
        'public': False,
        'authorized_ids': ["MyOrg2MSP"]
    },
}
ALGO_LOCAL_DOCKERFILE_FILES = [
        os.path.join(assets_directory, 'algo/local/algo_local.py'),
        os.path.join(assets_directory, 'algo/local/Dockerfile'),
]

ALGO_GLOBAL = {
    'name': 'MNIST Global Algo Linear Regression - Private [Auth: MyOrg2MSP]',
    'description': os.path.join(assets_directory, 'algo/description.md'),
    'permissions': {
        'public': False,
        'authorized_ids': ["MyOrg2MSP"]
    },
}
ALGO_GLOBAL_DOCKERFILE_FILES = [
        os.path.join(assets_directory, 'algo/global/algo_global.py'),
        os.path.join(assets_directory, 'algo/global/Dockerfile'),
]

#################
# Build archives
#################

archive_path = os.path.join(current_directory, 'algo_local.zip')
with zipfile.ZipFile(archive_path, 'w') as z:
    for filepath in ALGO_LOCAL_DOCKERFILE_FILES:
        z.write(filepath, arcname=os.path.basename(filepath))
ALGO_LOCAL['file'] = archive_path

archive_path = os.path.join(current_directory, 'algo_global.zip')
with zipfile.ZipFile(archive_path, 'w') as z:
    for filepath in ALGO_GLOBAL_DOCKERFILE_FILES:
        z.write(filepath, arcname=os.path.basename(filepath))
ALGO_GLOBAL['file'] = archive_path

#################
# Add local algo
#################

print('Adding local algo...')
algo_local_key = client.add_algo({
    'name': ALGO_LOCAL['name'],
    'file': ALGO_LOCAL['file'],
    'description': ALGO_LOCAL['description'],
    'permissions': ALGO_LOCAL['permissions'],
})

##############################
# Add global algo (aggregate)
##############################

print('Adding global algo...')
algo_global_key = client.add_aggregate_algo({
    'name': ALGO_GLOBAL['name'],
    'file': ALGO_GLOBAL['file'],
    'description': ALGO_GLOBAL['description'],
    'permissions': ALGO_GLOBAL['permissions'],
})

##################################################################################
# Load previous assets keys from file, add algo assets keys and save back to file
##################################################################################

# Load assets keys
assets1_keys_path = os.path.join(current_directory, '../assets1_keys.json')
with open(assets1_keys_path, 'r') as f:
    assets1_keys = json.load(f)

assets2_keys_path = os.path.join(current_directory, '../assets2_keys.json')
with open(assets2_keys_path, 'r') as f:
    assets2_keys = json.load(f)

# Add algo assets keys
assets1_keys['algo_local_key'] = algo_local_key
assets1_keys['algo_global_key'] = algo_global_key

assets2_keys['algo_local_key'] = algo_local_key
assets2_keys['algo_global_key'] = algo_global_key

# Save assets keys
with open(assets1_keys_path, 'w') as f:
    json.dump(assets1_keys, f, indent=2)

with open(assets2_keys_path, 'w') as f:
    json.dump(assets2_keys, f, indent=2)

print(f'Assets keys have been saved to {os.path.abspath(assets1_keys_path)} and {os.path.abspath(assets2_keys_path)}')
