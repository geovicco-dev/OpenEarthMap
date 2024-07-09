from . import PARAMS_FILEPATH
from dataclasses import dataclass
from pathlib import Path
from src.utils.common import read_yaml, save_json
# Components
from src.utils.dataloader import OEMDataModule, get_training_augmentation, get_preprocessing, get_preprocessing_function
from src.utils.model import SegmentationModel
from src.utils.common import save_json
from src.utils.plotting import plot_test_batch
import pytorch_lightning as pl
import torch
import os
import warnings; warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('medium')

@dataclass(frozen=True)
class Config:
    metadata_csv: Path
    models_dir: Path
    results_dir: Path
    logs_dir: Path
    architecture: str
    encoder: str
    encoder_weights: str
    n_classes: int
    n_channels: int
    epochs: int
    lr: float
    batch_size: int
    device: str
    num_workers: int
    resize_dimension: int
    apply_preprocessing: bool
    dev_run: bool
    tune_lr: bool
    checkpoint_path: Path
    encoder: str
    optimizer: str
    loss: str
    save_best_checkpoint: bool
    metric_threshold: float
    evaluate: bool
    save_predictions: bool

class ConfigManager:
    def __init__(self, params_filepath: Path = PARAMS_FILEPATH) -> None:
        self.config = read_yaml(params_filepath)
        if not Path(self.config.models_dir).exists(): Path(self.config.models_dir).mkdir(parents=True, exist_ok=True)
        if not Path(self.config.logs_dir).exists(): Path(self.config.logs_dir).mkdir(parents=True, exist_ok=True)
        if not Path(self.config.results_dir).exists(): Path(self.config.results_dir).mkdir(parents=True, exist_ok=True)
        
    def get_config(self) -> Config:
        params = self.config
        cfg = Config(
            models_dir=Path(params.models_dir),
            results_dir=Path(params.results_dir),
            metadata_csv=Path(params.metadata_file),
            logs_dir=Path(params.logs_dir),
            architecture=params.architecture,
            encoder=params.encoder,
            encoder_weights=params.encoder_weights,
            n_classes=params.n_classes,
            n_channels=params.n_channels,
            epochs=params.epochs,
            lr=params.lr,
            batch_size=params.batch_size,
            device=params.device,
            num_workers=params.num_workers,
            resize_dimension=params.resize_dimension,
            dev_run=params.dev_run,
            tune_lr=params.tune_lr,
            apply_preprocessing=params.apply_preprocessing,
            checkpoint_path=None if params.checkpoint_path == 'None' else Path(params.checkpoint_path),
            optimizer=params.optimizer,
            loss=params.loss,
            save_best_checkpoint=params.save_best_checkpoint,
            metric_threshold=params.metric_threshold,
            evaluate=params.evaluate,
            save_predictions=params.save_predictions
        )
        return cfg

class Components:
    def __init__(self, config: Config) -> None:
        self.config = config
        
    def create_dataloaders(self):
        print(f'------------- Creating Dataloaders -------------')
        if self.config.apply_preprocessing:
            print('------------->>> Applying Preprocessing <<<-------------')
            self.dm = OEMDataModule(
                metadata_csv=self.config.metadata_csv,
                augmentation=get_training_augmentation(),
                preprocessing=get_preprocessing(get_preprocessing_function(self.config.encoder, self.config.encoder_weights)),
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                resize_dimensions=self.config.resize_dimension
            )
        else:
            print('------------->>> Skipping: Applying Preprocessing <<<-------------')
            self.dm = OEMDataModule(
                metadata_csv=self.config.metadata_csv,
                augmentation=get_training_augmentation(),
                preprocessing=None,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                resize_dimensions=self.config.resize_dimension
            )
    
    def initialise_model(self):
        print(f'------------- Inistialising Model: Architecture: {self.config.architecture} | Encoder: {self.config.encoder} | Encoder Weights: {self.config.encoder_weights} -------------')
        self.model = SegmentationModel(
            architecture=self.config.architecture,
            n_channels=self.config.n_channels,
            n_classes=self.config.n_classes,
            lr=self.config.lr,
            encoder=self.config.encoder,
            encoder_weights=self.config.encoder_weights,
            loss=self.config.loss,
            optimizer=self.config.optimizer,
        )
    
    def load_checkpoint(self):
        print('------------- Loading Checkpoint -------------')
        print(f'Loading checkpoint from {self.config.checkpoint_path}')
        try:
            self.model = SegmentationModel.load_from_checkpoint(self.config.checkpoint_path, hparams_file='params.yaml')
            print('Checkpoint loaded successfully')
        except Exception as e:
            print(f'Failed to load checkpoint: {e}')

    
    def create_callbacks(self):
        print('------------- Creating Callbacks -------------')
        ### Define Checkpoints for Early Stopping, Tensorboard Summary Writer, and Best Checkpoint Saving
        from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
        from pytorch_lightning.loggers import TensorBoardLogger

        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',  # Metric to monitor
            patience=10,          # Number of epochs with no improvement after which training will be stopped
            verbose=True,
            mode='min'           # Mode can be 'min' for minimizing the monitored metric or 'max' for maximizing it
        )

        # Model checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            monitor='val_f1',   # Metric to monitor
            filename='{epoch:02d}-{val_f1:.2f}',  # Filename format
            save_top_k=1,         # Save the top k models
            mode='max',           # Mode can be 'min' or 'max'
            verbose=True,
            dirpath=self.config.logs_dir.joinpath(f'{self.config.architecture}_{self.config.encoder}/checkpoints')
        )

        # TensorBoard logger
        self.tensorboard_logger = TensorBoardLogger(
            save_dir=self.config.logs_dir,     # Directory to save the logs
            name=f"{self.config.architecture}_{self.config.encoder}"       # Experiment name
        )

        from pytorch_lightning.callbacks import LearningRateMonitor
        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        
        self.callbacks = [early_stopping, checkpoint_callback, lr_monitor]
        
    def tune_lr(self):
        from pytorch_lightning.tuner.tuning import Tuner
        print('------------- Tunning Learning Rate -------------')
        # Define a separate trainer for hyperparameter tuning
        self.tuning_trainer = pl.Trainer(
            accelerator=self.config.device,
            precision="16-mixed",
            logger=self.tensorboard_logger,
            callbacks=None,
            max_epochs=5  # Set this to a low number for faster tuning
        )
        self.dm.setup('fit')

        # Hyperparameter tuning
        self.tuner = Tuner(self.tuning_trainer)
        self.new_lr = self.tuner.lr_find(self.model, train_dataloaders=self.dm.train_dataloader(), val_dataloaders=self.dm.val_dataloader()).suggestion()
        print(f'Suggested learning rate: {self.new_lr}')
        
    def create_trainer(self):
        print(f'------------- Creating Trainer -------------')
        self.trainer = pl.Trainer(
            accelerator=self.config.device,
            max_epochs=self.config.epochs,
            precision="16-mixed",
            logger= self.tensorboard_logger if hasattr(self, 'tensorboard_logger') else None,
            callbacks=self.callbacks if hasattr(self, 'callbacks') else None,
            enable_progress_bar=True,
            fast_dev_run=self.config.dev_run,
        )
    
    def train(self):
        print('------------- Training Started -------------')
        self.dm.setup('fit')
        if self.config.checkpoint_path is not None and os.path.exists(self.config.checkpoint_path):
            print(f'Resuming training from checkpoint: {self.config.checkpoint_path}')
            self.trainer.fit(model=self.model, train_dataloaders=self.dm.train_dataloader(), val_dataloaders=self.dm.val_dataloader(), ckpt_path=self.config.checkpoint_path)
        else:
            self.trainer.fit(model=self.model, train_dataloaders=self.dm.train_dataloader(), val_dataloaders=self.dm.val_dataloader())
        print('------------- Training Completed -------------')
        
        # Save Model
        if self.config.save_best_checkpoint and self.metrics['val_f1'] > self.config.metric_threshold:
            print('------------- Saving Model -------------')
            self.model.save(Path(self.config.models_dir).joinpath('best_model.ckpt')) 
                
    def evaluate(self):
        print('------------- Evaluating Model -------------')
        self.dm.setup('test')
        try:
            test_results = self.trainer.test(model=self.model, dataloaders=self.dm.test_dataloader())
            
            # Extract relevant metrics from test_results
            # Assuming test_results is a list of dictionaries and contains y_true and y_pred
            self.metrics = test_results[0] if test_results else {}
            self.metrics_filepath = Path(self.config.results_dir).joinpath('metrics.json')
            # Save metrics to a file
            save_json(Path(self.metrics_filepath), self.metrics)  # Save the metrics to a file
            
            # Save Model as ONNX
            if self.metrics['test_f1'] > self.config.metric_threshold:
                print(f"Test F1-Score ({self.metrics['test_f1']:.3f}) above threshold ({self.config.metric_threshold}), saving model as ONNX...")
                save_path = Path(self.config.models_dir).joinpath(f"{Path(self.config.checkpoint_path).parent.parent.name}.onnx")
                print(f'Saving model to: {save_path}')
                test_dataloader = self.dm.test_dataloader()
                input_sample, _ = next(iter(test_dataloader))
                input_sample = input_sample[0].unsqueeze(0)
                self.model.to_onnx(save_path, input_sample=input_sample, export_params=True)
            
        except Exception as e:
            print(f"An error occurred: {e}")
            
    def save_predictions(self):
        if self.config.save_predictions:
            print('------------- Predicting on Test Batch and Saving Predictions -------------')
            plot_test_batch(
                pipeline=self, 
                randomised=False, 
                savefig_path=Path(self.config.results_dir).joinpath("test_predictions.png")
            )
            
# Pipeline
STAGE_NAME = 'Model Training and Evaluation'

class TrainEvaluatePipeline:
    def __init__(self) -> None:
        pass
    
    def main(self):
        pipeline = Components(ConfigManager().get_config())
        pipeline.create_dataloaders()
        if pipeline.config.evaluate and Path(pipeline.config.checkpoint_path).exists():
            # Evaluate the model
            try:
                print(f'------------- Evaluating Model using Checkpoint: {pipeline.config.checkpoint_path} -------------')
                pipeline.load_checkpoint()
                pipeline.create_trainer()
                pipeline.evaluate()
                if pipeline.config.save_predictions:
                    try:
                        pipeline.save_predictions()
                    except Exception as e:
                        print(e)
            except Exception as e:
                print(e)
        else:
            # Train the model
            try:
                print(f'------------- Training Model: {pipeline.config.architecture} with {pipeline.config.encoder} Encoder -------------')
                pipeline.initialise_model()
                if pipeline.config.checkpoint_path is not None and Path(pipeline.config.checkpoint_path).exists():
                    pipeline.load_checkpoint()
                pipeline.create_callbacks()
                if pipeline.config.tune_lr:
                    pipeline.tune_lr()
                pipeline.create_trainer()
                pipeline.train()
            except Exception as e:
                print(e)
        
if __name__ == "__main__":
    try:
        print(f">>>>>>> Processing Stage: {STAGE_NAME} <<<<<<<<")
        obj = TrainEvaluatePipeline()
        obj.main()
        print(f">>>>>>> Finished {STAGE_NAME} <<<<<<<\n\nx==========x")
    except Exception as e:
        print(e)
        raise e