import yaml
from box import ConfigBox
from ensure import ensure_annotations
import yaml
from box.exceptions import BoxValueError
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path

@ensure_annotations 
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e

def show_config(cfg):
    for k,v in cfg.__dict__.items():
        print(f"{k}: {v}")
        
def execute_pipeline(stage_name, pipeline_class):
    try:
        print(f">>>>>>> Processing Stage: {stage_name} <<<<<<<<")
        pipeline = pipeline_class()
        pipeline.main()
        print(f">>>>>>> Finished {stage_name} <<<<<<<\n\nx==========x")
    except Exception as e:
        print(f'Failed to execute pipeline for stage: {stage_name}')
        print(e)
        raise e
