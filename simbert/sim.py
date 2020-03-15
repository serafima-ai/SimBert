import argparse
from simbert.kernel import Kernel


parser = argparse.ArgumentParser()

parser.add_argument("mode", help="select a mode, train or interact", type=str,
                    choices={'train', 'evaluate', 'interact', 'predict', 'download', 'install', 'crossval'})
parser.add_argument("config_path", help="path to a pipeline json config", type=str)


def main():

    args = parser.parse_args()



    K = Kernel()





