import json
from dotmap import DotMap


def process_json_config(content_json: str = '', file: str = '') -> DotMap:
    content = DotMap()

    if content_json:
        content = DotMap(parse_json(content_json))

    packages = []

    print(content)

    for model in content.models:
        packages = find_packages(packages, model.toDict(), key='models')
    print(packages)
    import_packages(packages)

    return content


def find_packages(packages, content, key='', package=''):
    if type(content) is dict:
        package = key if package == '' else package + '.' + key

        for key, entity in content.items():
            packages = find_packages(packages, entity, key, package)

    else:
        if type(content) == str:
            if 'name' in key:
                package_name = 'simbert.{}.{}'.format(package, content)

            else:
                return packages

        else:
            package_name = 'simbert.{}.{}'.format(package, content['name'])

        packages.append(clean_package_name(package_name))

    return packages


package_replacer = (
("models.dataset", "datasets"), ("processor.features", "features"), ("datasets.features", "datasets"),
("datasets.processor", "datasets"))


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