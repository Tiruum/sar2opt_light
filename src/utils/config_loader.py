from omegaconf import OmegaConf
from configs.schema import Config

def load_config(path: str) -> Config:
    # 1) загружаем raw YAML
    raw = OmegaConf.load(path)
    # 2) создаём пустой структурированный объект с default-значениями
    structured = OmegaConf.structured(Config)
    # 3) мёржим: значения из raw перепишут defaults, и проверится схема
    cfg: Config = OmegaConf.merge(structured, raw)
    return cfg