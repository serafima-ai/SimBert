import json


def process_json_config(content_json: str = '', file: str = '') -> dict:

    content = {}

    if content_json:
        content = parse_json(content_json)

    packages = []

    print(content)

    for key, entity in content.items():
        for item in entity:
            packages.append('simbert.{}.{}'.format(key, item['name']))

    import_packages(packages)

    return content


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