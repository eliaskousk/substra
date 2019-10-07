import json
import re

import click
from click.testing import CliRunner
import pytest

import substra
from substra.cli.interface import cli, click_option_output_format

from . import datastore


@pytest.fixture
def workdir(tmp_path):
    d = tmp_path / "substra-cli"
    d.mkdir()
    return d


def execute(command, exit_code=0):
    runner = CliRunner()
    result = runner.invoke(cli, command)
    assert result.exit_code == exit_code
    return result.output


def client_execute(tmpdir, command, exit_code=0):
    # force using a new config file and a new profile
    if '--config' not in command:
        cfgpath = tmpdir / 'substra.cfg'
        substra.sdk.config.Manager(str(cfgpath)).add_profile(
            'default', url='http://foo')
        command.extend(['--config', str(cfgpath)])
    return execute(command, exit_code=exit_code)


def test_command_help():
    output = execute(['--help'])
    assert 'Usage:' in output


def test_command_version():
    output = execute(['--version'])
    assert substra.__version__ in output


def test_command_config(workdir):
    cfgfile = workdir / "cli.cfg"

    assert cfgfile.exists() is False

    new_url = 'http://foo'
    new_profile = 'foo'
    execute([
        'config',
        new_url,
        '--profile', new_profile,
        '--config', str(cfgfile),
    ])
    assert cfgfile.exists()

    # check new profile has been created, check also that default profile
    # has been created
    with cfgfile.open() as fp:
        cfg = json.load(fp)
    expected_profiles = ['default', 'foo']
    assert list(cfg.keys()) == expected_profiles


def mock_client_call(mocker, method_name, response=""):
    return mocker.patch(f'substra.cli.interface.Client.{method_name}',
                        return_value=response)


@pytest.mark.parametrize(
    'asset_name', ['objective', 'dataset', 'algo', 'testtuple', 'traintuple']
)
def test_command_list(asset_name, workdir, mocker):
    item = getattr(datastore, asset_name.upper())
    method_name = f'list_{asset_name}'
    with mock_client_call(mocker, method_name, [item]) as m:
        output = client_execute(workdir, ['list', asset_name])
    assert m.is_called()
    assert item['key'] in output


def test_command_list_node(workdir, mocker):
    with mock_client_call(mocker, 'list_node', datastore.NODES):
        output = client_execute(workdir, ['list', 'node'])
    assert output == ('NODE ID                     \n'
                      'foo                         \n'
                      'bar         (current)       \n')


@pytest.mark.parametrize('asset_name,params', [
    ('dataset', []),
    ('algo', []),
    ('traintuple', ['--objective-key', 'foo', '--algo-key', 'foo', '--dataset-key', 'foo',
                    '--data-samples-path']),
    ('testtuple', ['--traintuple-key', 'foo', '--data-samples-path'])]
)
def test_command_add(asset_name, params, workdir, mocker):
    method_name = f'add_{asset_name}'

    file = workdir / "non_existing_file.json"
    json_data = {"keys": []}

    with open(str(file), 'w') as fp:
        json.dump(json_data, fp)

    md_file = workdir / "non_existing_file.md"
    md_file.write_text('foo')

    with mock_client_call(mocker, method_name, response={}) as m:
        client_execute(workdir, ['add', asset_name] + params + [str(file)])
    assert m.is_called()

    res = client_execute(workdir, ['add', asset_name] + params + ['test.txt'], exit_code=2)
    assert re.search(r'File ".*" does not exist\.', res)

    res = client_execute(workdir, ['add', asset_name] + params + [str(md_file)], exit_code=2)
    assert re.search(r'File ".*" is not a valid JSON file\.', res)


def test_command_add_objective(workdir, mocker):
    file = workdir / "non_existing_file.json"
    json_data = {"keys": []}

    with open(str(file), 'w') as fp:
        json.dump(json_data, fp)

    md_file = workdir / "non_existing_file.md"
    md_file.write_text('foo')

    with mock_client_call(mocker, 'add_objective', response={}) as m:
        client_execute(workdir, ['add', 'objective', str(file), '--dataset-key', 'foo',
                                 '--data-samples-path', str(file)])
    assert m.is_called()

    res = client_execute(workdir, ['add', 'objective', 'non_existing_file.txt', '--dataset-key',
                                   'foo', '--data-samples-path', str(file)], exit_code=2)
    assert re.search(r'File ".*" does not exist\.', res)

    res = client_execute(workdir, ['add', 'objective', str(md_file), '--dataset-key', 'foo',
                                   '--data-samples-path', str(file)], exit_code=2)
    assert re.search(r'File ".*" is not a valid JSON file\.', res)

    res = client_execute(workdir, ['add', 'objective', str(file), '--dataset-key', 'foo',
                                   '--data-samples-path', 'non_existing_file.txt'], exit_code=2)
    assert re.search(r'File ".*" does not exist\.', res)

    res = client_execute(workdir, ['add', 'objective', str(file), '--dataset-key', 'foo',
                                   '--data-samples-path', str(md_file)], exit_code=2)
    assert re.search(r'File ".*" is not a valid JSON file\.', res)


def test_command_add_data_sample(tmp_path, workdir, mocker):
    temp_dir = tmp_path / "test_dir"
    temp_dir.mkdir()

    with mock_client_call(mocker, 'add_data_samples') as m:
        client_execute(workdir, ['add', 'data_sample', str(temp_dir), '--dataset-key', 'foo',
                                 '--test-only'])
    assert m.is_called()

    res = client_execute(workdir, ['add', 'data_sample', 'dir', '--dataset-key', 'foo'],
                         exit_code=2)
    assert re.search(r'Directory ".*" does not exist\.', res)


@pytest.mark.parametrize(
    'asset_name', ['objective', 'dataset', 'algo', 'testtuple', 'traintuple']
)
def test_command_get(asset_name, workdir, mocker):
    item = getattr(datastore, asset_name.upper())
    method_name = f'get_{asset_name}'
    with mock_client_call(mocker, method_name, item) as m:
        output = client_execute(workdir, ['get', asset_name, 'fakekey'])
    assert m.is_called()
    assert item['key'] in output


def test_command_describe(workdir, mocker):
    response = "My description."
    with mock_client_call(mocker, 'describe_objective', response) as m:
        output = client_execute(workdir, ['describe', 'objective', 'fakekey'])
    assert m.is_called()
    assert response in output


def test_command_download(workdir, mocker):
    with mock_client_call(mocker, 'download_objective') as m:
        client_execute(workdir, ['download', 'objective', 'fakekey'])
    assert m.is_called()


def test_command_update_dataset(workdir, mocker):
    with mock_client_call(mocker, 'update_dataset') as m:
        client_execute(workdir, ['update', 'dataset', 'key1', 'key2'])
    assert m.is_called()


def test_command_update_data_sample(workdir, mocker):
    data_samples = {
        'keys': ['key1', 'key2'],
    }

    data_samples_path = workdir / 'non_existing_file.json'
    data_samples_path_content = workdir / 'invalid_content.json'
    data_samples_path_invalid = workdir / 'non_existing_file.md'

    with data_samples_path.open(mode='w') as fp:
        json.dump(data_samples, fp)

    data_samples_path_content.write_text('test')

    with mock_client_call(mocker, 'link_dataset_with_data_samples') as m:
        client_execute(
            workdir, ['update', 'data_sample', str(data_samples_path), '--dataset-key', 'foo'])
    assert m.is_called()

    res = client_execute(workdir, ['update', 'data_sample', str(data_samples_path_invalid),
                                   '--dataset-key', 'foo'], exit_code=2)
    assert re.search(r'File ".*" does not exist\.', res)

    res = client_execute(workdir, ['update', 'data_sample', str(data_samples_path_content),
                                   '--dataset-key', 'foo'], exit_code=2)
    assert re.search(r'File ".*" is not a valid JSON file\.', res)


@pytest.mark.parametrize('params,output', [
    ([], 'pretty\n'),
    (['--pretty'], 'pretty\n'),
    (['--json'], 'json\n'),
    (['--yaml'], 'yaml\n'),
])
def test_option_output_format(params, output):
    @click.command()
    @click_option_output_format
    def foo(output_format):
        click.echo(output_format)

    runner = CliRunner()

    res = runner.invoke(foo, params)
    assert res.output == output
