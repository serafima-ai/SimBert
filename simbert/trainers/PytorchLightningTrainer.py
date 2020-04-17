from typing import Optional, Union, Dict, List

from simbert.trainers.trainer import Trainer
import pytorch_lightning as pl


class PytorchLightningTrainer(Trainer):

    def trainer(self,
                logger=None,
                gradient_clip_val: float = 0,
                process_position: int = 0,
                num_nodes: int = 1,
                gpus: Optional[Union[List[int], str, int]] = None,
                auto_select_gpus: bool = False,
                log_gpu_memory: Optional[str] = None,
                progress_bar_refresh_rate: int = 1,
                track_grad_norm: int = -1,
                check_val_every_n_epoch: int = 1,
                fast_dev_run: bool = False,
                accumulate_grad_batches: Union[int, Dict[int, int], List[list]] = 1,
                max_epochs: int = 1000,
                min_epochs: int = 1,
                log_save_interval: int = 100,
                precision: int = 32,
                weights_save_path: Optional[str] = None,
                amp_level: str = 'O1',
                num_sanity_val_steps: int = 5,
                resume_from_checkpoint: Optional[str] = None,
                benchmark: bool = False,
                reload_dataloaders_every_epoch: bool = False,
                auto_lr_find: Union[bool, str] = False,
                use_amp=None
                ):
        logger = logger if logger is not None \
            else pl.loggers.TensorBoardLogger('tb_logs', name=self.configs.get('logs_dir', './logs/'))

        gradient_clip_val = gradient_clip_val if gradient_clip_val is not 0 \
            else self.configs.get('gradient_clipping_value', 0)

        process_position = process_position if process_position is not 0 \
            else self.configs.get('process_position', 0)

        num_nodes = num_nodes if num_nodes is not 1 \
            else self.configs.get('num_nodes', 1)

        gpus = gpus if gpus is not None \
            else self.configs.get('gpus', None)

        auto_select_gpus = auto_select_gpus if auto_select_gpus \
            else self.configs.get('auto_select_gpus', False)

        log_gpu_memory = log_gpu_memory if log_gpu_memory is not None \
            else self.configs.get('log_gpu_memory', None)

        progress_bar_refresh_rate = progress_bar_refresh_rate if progress_bar_refresh_rate != 1 \
            else self.configs.get('progress_bar_refresh_rate', 1)

        track_grad_norm = track_grad_norm if track_grad_norm != -1 \
            else self.configs.get('track_grad_norm', -1)

        check_val_every_n_epoch = check_val_every_n_epoch if check_val_every_n_epoch != 1 \
            else self.configs.get('check_val_every_n_epoch', 1)

        fast_dev_run = fast_dev_run if fast_dev_run \
            else self.configs.get('fast_dev_run', False)

        accumulate_grad_batches = accumulate_grad_batches if accumulate_grad_batches != 1 \
            else self.configs.get('accumulate_gradient_batches', 1)

        accumulate_grad_batches = accumulate_grad_batches if accumulate_grad_batches != 1 \
            else self.configs.get('accumulate_gradient_batches', 1)

        max_epochs = max_epochs if max_epochs != 1000 \
            else self.configs.get('max_epochs', 1000)

        min_epochs = min_epochs if min_epochs > 0 \
            else self.configs.get('min_epochs', 1)

        log_save_interval = log_save_interval if log_save_interval != 100 \
            else self.configs.get('log_save_interval', 100)

        precision = precision if precision != 32 \
            else self.configs.get('precision', 32)

        weights_save_path = weights_save_path if weights_save_path is not None \
            else self.configs.get('weights_save_path', './weights/')

        amp_level = amp_level if amp_level is not 'O1' \
            else self.configs.get('amp_level', 'O1')

        num_sanity_val_steps = num_sanity_val_steps if num_sanity_val_steps != 5 \
            else self.configs.get('num_sanity_val_steps', 5)

        resume_from_checkpoint = resume_from_checkpoint if resume_from_checkpoint is not None \
            else self.configs.get('resume_from_checkpoint', None)

        benchmark = benchmark if benchmark \
            else self.configs.get('benchmark', False)

        reload_dataloaders_every_epoch = reload_dataloaders_every_epoch if reload_dataloaders_every_epoch \
            else self.configs.get('reload_dataloaders_every_epoch', False)

        auto_lr_find = auto_lr_find if auto_lr_find \
            else self.configs.get('auto_lr_find', False)

        use_amp = use_amp if use_amp is not None \
            else self.configs.get('use_amp', None)

        return pl.Trainer(logger=logger,
                          gradient_clip_val=gradient_clip_val,
                          process_position=process_position,
                          num_nodes=num_nodes,
                          gpus=gpus,
                          auto_select_gpus=auto_select_gpus,
                          log_gpu_memory=log_gpu_memory,
                          progress_bar_refresh_rate=progress_bar_refresh_rate,
                          track_grad_norm=track_grad_norm,
                          check_val_every_n_epoch=check_val_every_n_epoch,
                          fast_dev_run=fast_dev_run,
                          accumulate_grad_batches=accumulate_grad_batches,
                          max_epochs=max_epochs,
                          min_epochs=min_epochs,
                          log_save_interval=log_save_interval,
                          precision=precision,
                          weights_save_path=weights_save_path,
                          amp_level=amp_level,
                          num_sanity_val_steps=num_sanity_val_steps,
                          resume_from_checkpoint=resume_from_checkpoint,
                          benchmark=benchmark,
                          reload_dataloaders_every_epoch=reload_dataloaders_every_epoch,
                          auto_lr_find=auto_lr_find,
                          use_amp=use_amp)
