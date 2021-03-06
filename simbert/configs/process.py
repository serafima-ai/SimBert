import json
from dotmap import DotMap


def process_json_config(content_json: str = '', json_file: str = '') -> DotMap:
    content = DotMap()

    if content_json:
        content = parse_json(content_json)
    elif json_file:
        content = parse_json(parse_configs_file(json_file))
    else:
        return content

    content = DotMap(content)

    packages = []

    for model in content.models:
        packages = find_packages(packages, model.toDict(), key='models')

    import_packages(packages)

    return content


def find_packages(packages, content, key='', package=''):
    if type(content) is dict:
        package = key if package == '' else package + '.' + key

        for key, entity in content.items():

            if key == 'metrics' and type(entity) == list:
                entity = find_metric_packages(entity)

            packages = find_packages(packages, entity, key, package)

    elif type(content) is list:
        if key == 'metrics':
            package = key

        for entity in content:
            packages = find_packages(packages, entity, key, package)

    else:
        if type(content) == str and ('name' in key or key in ['metrics']):
            package_name = 'simbert.{}.{}'.format(package, content)

            packages.append(clean_package_name(package_name))

    return packages


def find_metric_packages(content):
    metric_packages = []

    from simbert.metrics.metric import Metric

    for metric in content:

        name = Metric.get_class_name(metric)

        if name is not '':
            metric_packages.append(name)

    return metric_packages


package_replacer = (
    ("models.dataset", "datasets"), ("processor.features", "features"), ("datasets.features", "datasets"),
    ("datasets.processor", "datasets"), ("models.optimizer", "optimizers"), ("models.loss", "losses"),
    ("models.trainer", "trainers"))


def clean_package_name(name: str) -> str:
    for r in package_replacer:
        name = name.replace(*r)
    return name


def parse_configs_file(filepath) -> str:
    with open(filepath, 'r') as f:
        content = f.read()

    f.close()

    return content


def parse_json(content: str) -> dict:
    return json.loads(content)


def import_packages(packages: list) -> None:
    print(packages)
    """Import packages from list to execute their code."""
    for package in packages:
        __import__(package)
