from . import PARAMS_FILEPATH
from .utils.common import read_yaml
from pathlib import Path
from dataclasses import dataclass
from typing import List
import os
import subprocess
import pandas as pd 
from sklearn.model_selection import train_test_split

# Configuration
@dataclass(frozen=True)
class Config: 
    doi: str
    out_dir: Path
    train_test_val_split: List
    metadata_file: Path
    random_seed: int

class ConfigManager:
    def __init__(self, config_path = PARAMS_FILEPATH) -> None:
        self.config = read_yaml(config_path)

    def get_config(self) -> Config:
        Path(self.config["out_dir"]).mkdir(parents=True, exist_ok=True)
        cfg = Config(
            doi = self.config["doi"],
            out_dir = Path(self.config["out_dir"]),
            train_test_val_split = self.config["train_test_val_split"],
            metadata_file = Path(self.config["metadata_file"]),
            random_seed = self.config["random_seed"]
        )
        return cfg

cfg = ConfigManager().get_config()

# Components
class Components:
    def __init__(self, cfg: Config) -> None:
        self.config = cfg
    
    def download_data(self, unzip: bool = True, remove_zip: bool = False) -> None:
        print(f">>>>>>>>>>>> Downloading data from Zenodo <<<<<<<<<<<<")
        if not Path(self.config.out_dir).joinpath("OpenEarthMap_wo_xBD").exists() and not Path(self.config.out_dir).joinpath("OpenEarthMap.zip").exists():
            cmd = f"zenodo_get -e -d {self.config.doi} -o {self.config.out_dir}"
            subprocess.run(cmd, shell=True)
            if unzip:
                print(f"--> Unzipping data")
                cmd = f"unzip -q {self.config.out_dir}/OpenEarthMap.zip -d {self.config.out_dir}"
                subprocess.run(cmd, shell=True)
                if remove_zip:
                    os.remove(f"{self.config.out_dir}/OpenEarthMap.zip")
        else:
            print("--> Data already downloaded. Skipping.")
            
    def aggregate_data(self) -> None:
        print(f">>>>>>>>>>>> Aggregating data <<<<<<<<<<<<")
        # Aggregate Metadata
        imgs = sorted(list(Path(self.config.out_dir).joinpath("OpenEarthMap_wo_xBD").rglob('*/images/*.tif')))
        labels = sorted(list(Path(self.config.out_dir).joinpath("OpenEarthMap_wo_xBD").rglob('*/labels/*.tif')))

        self.images = []
        self.masks = []
        # Check if labels exist
        for img in imgs:
            for label in labels:
                if img.stem == label.stem:
                    self.images.append(img)
                    self.masks.append(label)
                    
        print(f"--> Number of images with labels: {len(self.images)}")
        
        # Delete images without labels
        print("--> Deleting images without labels")
        for img in imgs:
            if img not in self.images:
                os.remove(img)
        for label in labels:
            if label not in self.masks:
                os.remove(label)
    
    def split_data(self) -> None:
        print(f">>>>>>>>>>>> Splitting data into train/val/test sets <<<<<<<<<<<<") 
        meta = pd.DataFrame({"image": self.images, "mask": self.masks})

        # Assuming cfg.train_test_val_split is a list of three values summing to 1, e.g. [0.7, 0.2, 0.1]
        train_ratio, val_ratio, test_ratio = self.config.train_test_val_split

        # First split: separate train from the rest
        train_df, temp_df = train_test_split(meta, train_size=train_ratio, random_state=self.config.random_seed)

        # Second split: divide the rest into val and test
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(temp_df, train_size=val_ratio_adjusted, random_state=self.config.random_seed)

        # Assign groups
        train_df["group"] = "train"
        val_df["group"] = "val"
        test_df["group"] = "test"
        
        print(f"--> Number of images in train/val/test sets: {len(train_df)}, {len(val_df)}, {len(test_df)}")

        # Combine
        self.metadata = pd.concat([train_df, val_df, test_df], axis=0)

        # Save
        self.metadata.to_csv(self.config.metadata_file, index=False)
        print(f"-->Metadata saved to {self.config.metadata_file}")
        
# Pipeline
STAGE_NAME = 'Data Download and Preparation'

class DataIngestionPipeline:
    def __init__(self) -> None:
        pass
    
    def main(self):
        pipeline = Components(ConfigManager().get_config())
        pipeline.download_data(unzip=True, remove_zip=True)
        pipeline.aggregate_data()
        pipeline.split_data()
        
if __name__ == "__main__":
    try:
        print(f">>>>>>> Processing Stage: {STAGE_NAME} <<<<<<<<")
        obj = DataIngestionPipeline()
        obj.main()
        print(f">>>>>>> Finished {STAGE_NAME} <<<<<<<\n\nx==========x")
    except Exception as e:
        print(e)
        raise e