import inspect
import warnings
from abc import ABC
from argparse import Namespace

import pytorch_lightning as pl
import torch
from dotmap import DotMap
from pytorch_lightning.core.saving import load_hparams_from_tags_csv
from pytorch_lightning.utilities.debugging import MisconfigurationException
from typing import Any, Callable, Dict, Optional, Union


class SimbertLightningModule(pl.LightningModule, ABC):

    def __init__(self, *args, **kwargs):
        super(SimbertLightningModule, self).__init__(*args, **kwargs)

        #: Current dtype
        self.dtype = torch.FloatTensor

        self.exp_save_path = None

        #: The current epoch
        self.current_epoch = 0

        #: Total training batches seen across all epochs
        self.global_step = 0

        self.loaded_optimizer_states_dict = {}

        #: Pointer to the trainer object
        self.trainer = None

        #: Pointer to the logger object
        self.logger = None
        self.example_input_array = None

        #: True if your model is currently running on GPUs.
        #: Useful to set flags around the LightningModule for different CPU vs GPU behavior.
        self.on_gpu = False

        #: True if using dp
        self.use_dp = False

        #: True if using ddp
        self.use_ddp = False

        #: True if using ddp2
        self.use_ddp2 = False

        #: True if using amp
        self.use_amp = False

        self.hparams = None

    @classmethod
    def load_from_checkpoint(
            cls,
            checkpoint_path: str,
            map_location: Optional[Union[Dict[str, str], str, torch.device, int, Callable]] = None,
            tags_csv: Optional[str] = None,
            configs: DotMap = DotMap()
    ) -> 'LightningModule':
        r"""
        Primary way of loading model from a checkpoint. When Lightning saves a checkpoint
        it stores the hyperparameters in the checkpoint if you initialized your LightningModule
        with an argument called `hparams` which is a Namespace (output of using argparse
        to parse command line arguments).
        Example:
            .. code-block:: python
                from argparse import Namespace
                hparams = Namespace(**{'learning_rate': 0.1})
                model = MyModel(hparams)
                class MyModel(LightningModule):
                    def __init__(self, hparams):
                        self.learning_rate = hparams.learning_rate
        Args:
            checkpoint_path: Path to checkpoint.
            map_location:
                If your checkpoint saved a GPU model and you now load on CPUs
                or a different number of GPUs, use this to map to the new setup.
                The behaviour is the same as in
                `torch.load <https://pytorch.org/docs/stable/torch.html#torch.load>`_.
            tags_csv: Optional path to a .csv file with two columns (key, value)
                as in this example::
                    key,value
                    drop_prob,0.2
                    batch_size,32
                You most likely won't need this since Lightning will always save the hyperparameters
                to the checkpoint.
                However, if your checkpoint weights don't have the hyperparameters saved,
                use this method to pass in a .csv file with the hparams you'd like to use.
                These will be converted into a argparse.Namespace and passed into your
                LightningModule for use.
        Return:
            LightningModule with loaded weights and hyperparameters (if available).
        Example:
            .. code-block:: python
                # load weights without mapping ...
                MyLightningModule.load_from_checkpoint('path/to/checkpoint.ckpt')
                # or load weights mapping all weights from GPU 1 to GPU 0 ...
                map_location = {'cuda:1':'cuda:0'}
                MyLightningModule.load_from_checkpoint(
                    'path/to/checkpoint.ckpt',
                    map_location=map_location
                )
                # or load weights and hyperparameters from separate files.
                MyLightningModule.load_from_checkpoint(
                    'path/to/checkpoint.ckpt',
                    tags_csv='/path/to/hparams_file.csv'
                )
                # predict
                pretrained_model.eval()
                pretrained_model.freeze()
                y_hat = pretrained_model(x)
        """
        if map_location is not None:
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

        if tags_csv is not None:
            # add the hparams from csv file to checkpoint
            hparams = load_hparams_from_tags_csv(tags_csv)
            hparams.__setattr__('on_gpu', False)
            checkpoint['hparams'] = vars(hparams)

        model = cls._load_model_state(checkpoint, configs)
        return model

    @classmethod
    def _load_model_state(cls, checkpoint: Dict[str, Any], configs: DotMap) -> 'LightningModule':
        cls_takes_hparams = 'hparams' in inspect.signature(cls.__init__).parameters
        ckpt_hparams = checkpoint.get('hparams')

        if cls_takes_hparams:
            if ckpt_hparams is not None:
                is_namespace = checkpoint.get('hparams_type', 'namespace') == 'namespace'
                hparams = Namespace(**ckpt_hparams) if is_namespace else ckpt_hparams
            else:
                warnings.warn(
                    f"Checkpoint does not contain hyperparameters but {cls.__name__}'s __init__ "
                    f"contains argument 'hparams'. Will pass in an empty Namespace instead."
                    " Did you forget to store your model hyperparameters in self.hparams?"
                )
                hparams = Namespace()
        else:  # The user's LightningModule does not define a hparams argument
            if ckpt_hparams is None:
                hparams = None
            else:
                raise MisconfigurationException(
                    f"Checkpoint contains hyperparameters but {cls.__name__}'s __init__ "
                    f"is missing the argument 'hparams'. Are you loading the correct checkpoint?"
                )

        # load the state_dict on the model automatically
        model_args = [hparams] if hparams else []
        model = cls(configs=configs, *model_args)
        model.load_state_dict(checkpoint['state_dict'])

        # give model a chance to load something
        model.on_load_checkpoint(checkpoint)

        return model
