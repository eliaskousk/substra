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
import uuid

import substra


ROUNDS = 5

current_directory = os.path.dirname(__file__)
assets_directory = os.path.join(current_directory, '../assets')

client = substra.Client.from_config_file(profile_name="node-1")

###################
# Load assets keys
###################

assets1_keys_path = os.path.join(current_directory, '../assets1_keys.json')
with open(assets1_keys_path, 'r') as f:
    assets1_keys = json.load(f)

assets2_keys_path = os.path.join(current_directory, '../assets2_keys.json')
with open(assets2_keys_path, 'r') as f:
    assets2_keys = json.load(f)

###################
# Add compute plan
###################

print('Generating compute plan...')
local_traintuples = []
local_testtuples = []
global_aggregatetuples = []

for i in range(ROUNDS):
    local_traintuple_ids = []
    for asset in [assets1_keys, assets2_keys]:
        print(f'Adding local traintuple for dataset with key {asset["dataset_key"]}')
        local_traintuple = {
            'algo_key': asset['algo_local_key'],
            'data_manager_key': asset['dataset_key'],
            'train_data_sample_keys': asset['train_data_sample_keys'],
            'traintuple_id': uuid.uuid4().hex,
        }
        if global_aggregatetuples:
            local_traintuple['in_models_ids'] = [global_aggregatetuples[-1]['aggregatetuple_id']]
        print(f'Adding local testtuple for dataset with key {asset["dataset_key"]}')
        local_testtuple = {
            'objective_key': asset['objective_key'],
            'traintuple_id': local_traintuple['traintuple_id']
        }
        local_traintuples.append(local_traintuple)
        local_testtuples.append(local_testtuple)
        local_traintuple_ids.append(local_traintuple['traintuple_id'])

    print(f'Adding global aggregate tuple')
    global_aggregatetuple = {
        'aggregatetuple_id': uuid.uuid4().hex,
        'algo_key': asset['algo_global_key'],
        'worker': 'MyOrg1MSP',
        'in_models_ids': local_traintuple_ids,
    }
    global_aggregatetuples.append(global_aggregatetuple)

print(f'Adding final traintuple for dataset with key {assets1_keys["dataset_key"]}')
final_traintuple = {
    'algo_key': assets1_keys['algo_local_key'],
    'data_manager_key': assets1_keys['dataset_key'],
    'train_data_sample_keys': assets1_keys['train_data_sample_keys'],
    'traintuple_id': uuid.uuid4().hex,
    'in_models_ids': [global_aggregatetuples[-1]['aggregatetuple_id']]
}

print(f'Adding final testtuple for dataset with key {assets1_keys["dataset_key"]}')
final_testtuple = {
    'objective_key': assets1_keys['objective_key'],
    'traintuple_id': final_traintuple['traintuple_id']
}

print('Adding compute plan...')
compute_plan = client.add_compute_plan({
    'traintuples': local_traintuples + [final_traintuple],
    'testtuples': local_testtuples + [final_testtuple],
    'composite_traintuples': [],
    'aggregatetuples': global_aggregatetuples,
})

#########################
# Save compute plan keys
#########################

compute_plan_keys_path = os.path.join(current_directory, '../compute_plan_keys.json')
with open(compute_plan_keys_path, 'w') as f:
    json.dump(compute_plan.dict(exclude_none=False, by_alias=True), f, indent=2)

print(f'Compute plan keys have been saved to {os.path.abspath(compute_plan_keys_path)}')
print('\nRun the following commands to track the status of the tuples:')

for key in compute_plan.traintuple_keys:
    print(f'    substra get traintuple {key} --profile node-1')
for key in compute_plan.aggregatetuple_keys:
    print(f'    substra get aggregatetuple {key} --profile node-1')
for key in compute_plan.testtuple_keys:
    print(f'    substra get testtuple {key} --profile node-1')
