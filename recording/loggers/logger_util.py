"""
Utility file that implements methods used by the recorder and loggers
"""
import os
from util.Types import *


def process_logger_name(logger_name: str):
    logger_name = logger_name.lower()
    if logger_name.endswith("logger"):
        logger_name = logger_name[:-6]
    return logger_name


def get_trial_directory_path():
    from cw2.cw_config import cw_conf_keys as cw2_keys
    trial_directory_path = cw2_keys.i_REP_LOG_PATH  # filepath for the current cw2 trial
    return trial_directory_path


def process_logger_message(entity_key: str, entity_value: Any, indent: int = 30):
    message_string_template = "{"+":<{}".format(indent)+"}: {}"
    if isinstance(entity_value, float):
        entity_value = round(entity_value, ndigits=3)
    return message_string_template.format(entity_key.title(), entity_value)


def save_to_yaml(dictionary: Dict, save_name: str, recording_directory: str) -> None:
    """
    Save the current dictionary as an input_type.yaml

    Args:
        dictionary: The dictionary to save
        save_name: Name of the file to save to
        recording_directory: The directory to record (or in this case save the .yaml) to

    Returns:

    """
    import yaml
    filename = os.path.join(recording_directory, save_name + ".yaml")
    with open(filename, "w") as file:
        if isinstance(dictionary, dict):
            yaml.dump(dictionary, file, sort_keys=True, indent=2)
        else:
            yaml.dump(dictionary.__dict__, file, sort_keys=True, indent=2)
