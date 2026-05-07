import os
import functools
import torch
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from omegaconf import OmegaConf

_orig_load = torch.load
@functools.wraps(_orig_load)
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

from src.models.huggingface_gan.main import SAR2OPTLightningModule
from src.data.sen12_full.datamodule import SEN12FullDataModule
from src.utils.cleanup_memory import full_cleanup
from src.utils.callbacks import EMAWeightAveraging

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

CONFIG_PATH = 'src/models/huggingface_gan/config.yaml'


def main():
    cfg = OmegaConf.load(CONFIG_PATH)

    dm = SEN12FullDataModule(
        data_dir=cfg.data.data_dir.sen12_full,
        batch_size=cfg.data.batch_size,
        image_size=cfg.data.image_size,
        num_workers=cfg.data.num_workers,
        persistent_workers=cfg.data.persistent_workers,
        prefetch_factor=cfg.data.prefetch_factor,
        train_val_split_ratio=cfg.data.train_val_split_ratio,
        seed=cfg.data.seed,
        sar_channels=cfg.data.sar_channels,
        use_augmentation=cfg.data.use_train_common_transform,
        scenes=list(cfg.data.scenes),
    )
    model = SAR2OPTLightningModule(cfg)

    checkpoints = ModelCheckpoint(
        dirpath=f"{cfg.system.checkpoints_dir}/{cfg.system.tb_version}",
        filename='epoch={epoch:03d}-psnr={val/psnr:.4f}',
        monitor='val/psnr',
        mode='max',
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks = [checkpoints]
    if cfg.ema.use_ema:
        callbacks.append(EMAWeightAveraging(
            decay=cfg.ema.decay,
            update_starting_at_epoch=cfg.ema.start_epoch,
        ))

    tb_logger  = TensorBoardLogger(cfg.system.output_dir + '/tb_logs',  name=cfg.system.tb_version)
    csv_logger = CSVLogger(cfg.system.output_dir + '/csv_logs', name=cfg.system.tb_version)

    trainer = Trainer(
        logger=[tb_logger, csv_logger],
        callbacks=callbacks,
        accelerator=cfg.system.device,
        devices=1,
        precision=cfg.system.precision,
        max_epochs=cfg.system.max_epochs,
        num_sanity_val_steps=0,
        deterministic=cfg.system.deterministic,
        benchmark=cfg.system.benchmark,
        limit_train_batches=cfg.system.limit_train_batches,
        limit_val_batches=cfg.system.limit_val_batches,
        log_every_n_steps=50,
    )

    try:
        weights_ckpt = cfg.system.get('weights_ckpt', None)

        if weights_ckpt:
            print(f'Используются веса из чекпоинта {weights_ckpt}')
            ckpt = torch.load(weights_ckpt, map_location='cpu')
            raw_sd = ckpt.get('state_dict', ckpt)
            gen_sd = {}
            for k, v in raw_sd.items():
                if not k.startswith('netG.'):
                    continue
                new_k = k.replace('netG.', '')
                gen_sd[new_k] = v
            dis_sd = {k.replace('netD.', ''): v for k, v in raw_sd.items() if k.startswith('netD.')}
            miss_g, unex_g = model.netG.load_state_dict(gen_sd, strict=False)
            miss_d, unex_d = model.netD.load_state_dict(dis_sd, strict=False)
            print(f'[weights_ckpt] G missing={len(miss_g)} unexpected={len(unex_g)}')
            print(f'[weights_ckpt] D missing={len(miss_d)} unexpected={len(unex_d)}')
            ckpt_path = None
        else:
            ckpt_path = cfg.system.resume_ckpt or None
            if (ckpt_path): print(f'Продолжается обучение с чекпоинта {ckpt_path}')
        trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
    except KeyboardInterrupt:
        pass
    finally:
        full_cleanup(trainer=trainer, model=model, datamodule=dm, log=True)


if __name__ == '__main__':
    main()
