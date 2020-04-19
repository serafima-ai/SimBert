import argparse
from simbert.kernel import Kernel
from simbert.configs.process import process_json_config
from simbert.models.model import Model

parser = argparse.ArgumentParser()

parser.add_argument("mode", help="select a mode, train or interact", type=str,
                    choices={'train', 'evaluate', 'interact', 'predict', 'download', 'install'})
parser.add_argument("config_path", help="path to a pipeline json config", type=str)


def main():
    args = parser.parse_args()

    K = Kernel()


def build(json_configs: str = '', configs_file: str = '') -> dict:
    configs = process_json_config(content_json=json_configs, json_file=configs_file)

    models = configs.get('models', [])

    M = Model

    models_dict = {}

    for m in models:

        model = M.build_model(m)

        if model is not None:
            if models_dict.get(model.label, None) is None:
                models_dict[model.label] = model
            else:
                models_dict[model.label + '_#2'] = model

    return models_dict


def train(json_configs: str = '', configs_file: str = '') -> dict:
    models = build(json_configs=json_configs, configs_file=configs_file)

    for _, model in models.items():

        if model is not None:
            model.fit()

    return models


def test(json_configs: str = '', configs_file: str = '') -> dict:
    models = build(json_configs=json_configs, configs_file=configs_file)

    results = {}

    for label, model in models.items():

        if model is not None:
            results[label] = model.test()

    return results
