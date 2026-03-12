import logging
import yaml
from box.exceptions import BoxValueError
from base_config import BaseConfig
from box import ConfigBox
from pathlib import Path
from ensure import ensure_annotations

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e


class Configuration_Manager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH) :
        self.config = read_yaml(config_filepath)


    def get_base_config(self) -> BaseConfig:
        config = self.config.config

        base_config = BaseConfig(
            RAG_MODEL = config.RAG_MODEL,
            EMBEDD_MODEL = config.EMBEDD_MODEL,
            CHROMA_HOST = config.CHROMA_HOST,
            CHROMA_PORT = config.CHROMA_PORT,
            COLLECTION_NAME = config.COLLECTION_NAME,
            MONGODB_URI=config.MONGODB_URI,
            DB_NAME=config.DB_NAME,
            HISTORY_COLLECTION_NAME=config.HISTORY_COLLECTION_NAME,
            COLLECTION_USER=config.COLLECTION_USER,
            POSTGRES_HOST=config.POSTGRES_HOST,
            POSTGRES_PORT=config.POSTGRES_PORT,
            POSTGRES_DB=config.POSTGRES_DB,
            POSTGRES_USER=config.POSTGRES_USER,
            POSTGRES_PASSWORD=config.POSTGRES_PASSWORD,
            REDIS_HOST=config.REDIS_HOST,
            REDIS_PORT=config.REDIS_PORT,
            REDIS_DB=config.REDIS_DB,
            REDIS_USERNAME=config.REDIS_USERNAME,
            REDIS_PASSWORD=config.REDIS_PASSWORD,
            CACHE_TTL=config.CACHE_TTL,
        )
        return base_config