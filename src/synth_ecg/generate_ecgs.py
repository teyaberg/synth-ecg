import os

import hydra
import rootutils
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf

os.environ["PROJECT_ROOT"] = str(rootutils.setup_root(search_from=__file__, indicator="pyproject.toml"))
logger.info(f"Project root inferred to be {os.environ['PROJECT_ROOT']}")


@hydra.main(version_base=None, config_path="configs", config_name="generate_ecgs")
def main(cfg: DictConfig):
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # instantiate the generator
    generator = instantiate(cfg)
    ecgs = generator.generate_ecgs()
    logger.info("ECGs generated successfully.")
    save_fp = generator.save_ecgs(ecgs)
    logger.info(f"ECGs saved to {save_fp}")

    return 0


if __name__ == "__main__":
    main()
