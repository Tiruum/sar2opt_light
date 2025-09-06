# configs/schema.py
from dataclasses import dataclass

@dataclass
class DataConfig:
    data_dir:               str
    batch_size:             int
    image_size:             int
    num_workers:            int = 4
    train_val_split_ratio:  float = 0.8
    persistent_workers:     bool = True
    prefetch_factor:        int = 2
    seed:                   int = 42

@dataclass
class ModelConfig:
    input_nc:  int
    output_nc: int
    ngf:       int
    ndf:       int
    n_blocks:  int = 8
    n_layers:  int = 3
    num_D:     int = 4

@dataclass
class OptimizerConfig:
    lr_g:         float
    lr_d:         float
    beta1:        float
    beta2:        float

@dataclass
class SchedulerConfig:
    eta_min: float = 0.0

@dataclass
class LossConfig:
    gan_weight:      float
    l1_weight:       float

@dataclass
class SystemConfig:
    max_epochs:                 int
    checkpoints_dir:            str
    log_dir:                    str
    output_dir:                 str
    device:                     str
    precision:                  int
    deterministic:              bool = False
    benchmark:                  bool = False
    use_amp:                    bool = False
    limit_train_batches:        float = 1.0
    limit_val_batches:          float = 1.0
    accumulate_grad_batches:    int = 1

@dataclass
class Config:
    data:      DataConfig
    model:     ModelConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    loss:      LossConfig
    system:    SystemConfig
