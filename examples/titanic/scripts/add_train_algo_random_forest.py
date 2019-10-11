import json
import os
import zipfile
import yaml

import substra

current_directory = os.path.dirname(__file__)
assets_directory = os.path.join(current_directory, '../assets')

with open(os.path.join(current_directory, "../../config.yaml"), 'r') as yaml_config:
    config = yaml.safe_load(yaml_config)

client = substra.Client()
client.add_profile(config['profile_name'], config['url'], '0.0',
                   user=config['user'], password=config['password'])

ALGO_KEYS_JSON_FILENAME = 'algo_random_forest_keys.json'

ALGO = {
    'name': 'Titanic: Random Forest',
    'description': os.path.join(assets_directory, 'algo_random_forest/description.md')
}
ALGO_DOCKERFILE_FILES = [
        os.path.join(assets_directory, 'algo_random_forest/algo.py'),
        os.path.join(assets_directory, 'algo_random_forest/Dockerfile'),
]

########################################################
#       Build archive
########################################################

archive_path = 'algo_random_forest.zip'
with zipfile.ZipFile(archive_path, 'w') as z:
    for filepath in ALGO_DOCKERFILE_FILES:
        z.write(filepath, arcname=os.path.basename(filepath))
ALGO['file'] = archive_path

########################################################
#       Load keys for dataset and objective
########################################################

assets_keys_path = os.path.join(current_directory, '../assets_keys.json')
with open(assets_keys_path, 'r') as f:
    assets_keys = json.load(f)

########################################################
#         Add algo
########################################################

print('Adding algo...')
algo_key = client.add_algo({
    'name': ALGO['name'],
    'file': ALGO['file'],
    'description': ALGO['description'],
}, exist_ok=True)['pkhash']

########################################################
#         Add traintuple
########################################################

print('Registering traintuple...')
traintuple = client.add_traintuple({
    'algo_key': algo_key,
    'objective_key': assets_keys['objective_key'],
    'data_manager_key': assets_keys['dataset_key'],
    'train_data_sample_keys': assets_keys['train_data_sample_keys']
}, exist_ok=True)
traintuple_key = traintuple.get('key') or traintuple.get('pkhash')
assert traintuple_key, 'Missing traintuple key'

########################################################
#         Add testtuple
########################################################
print('Registering testtuple...')
testtuple = client.add_testtuple({
    'traintuple_key': traintuple_key
}, exist_ok=True)
testtuple_key = testtuple.get('key') or testtuple.get('pkhash')
assert testtuple_key, 'Missing testtuple key'

########################################################
#         Save keys in json
########################################################

assets_keys['algo_random_forest'] = {
    'algo_key': algo_key,
    'traintuple_key': traintuple_key,
    'testtuple_key': testtuple_key,
}
with open(assets_keys_path, 'w') as f:
    json.dump(assets_keys, f, indent=2)

print(f'Assets keys have been saved to {os.path.abspath(assets_keys_path)}')
print('\nRun the following commands to track the status of the tuples:')
print(f'    substra get traintuple {traintuple_key}')
print(f'    substra get testtuple {testtuple_key}')
