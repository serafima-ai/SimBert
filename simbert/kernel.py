from simbert.models import *
from simbert.datasets import *
from simbert.configs import process


class Kernel(object):
    items = {}

    get_key_error = "{}.get(): no class named {} was found"

    def __init_subclass__(cls, **kwargs):
        print(cls.__name__)
        super().__init_subclass__(**kwargs)
        cls.items.update({cls.__name__: cls})

    def get(self, item):
        item = prepare_name(item)
        try:
            return self.items[item]
        except KeyError:
            print(self.get_key_error.format(self.__class__.__name__, item))
            return None


def prepare_name(name: str) -> str:
    print("Kernel", name)
    return name.split('.')[-1]
