import pytorch_lightning as pl
import segmentation_models_pytorch as sm_torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.optim as optim
import torch.nn.functional as F

def get_optimizer(optimizer_name, parameters, lr, momentum=0.9):
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(parameters, lr=lr)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(parameters, lr=lr)
    elif optimizer_name == 'adadelta':
        optimizer = optim.Adadelta(parameters, lr=lr)
    elif optimizer_name == 'adamax':
        optimizer = optim.Adamax(parameters, lr=lr)
    elif optimizer_name == 'asgd':
        optimizer = optim.ASGD(parameters, lr=lr)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(parameters, lr=lr, momentum=momentum)
    elif optimizer_name == 'adagrad':
        optimizer = optim.Adagrad(parameters, lr=lr)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(parameters, lr=lr, momentum=momentum)
    else:
        raise ValueError(f'Unsupported optimizer: {optimizer_name}')
    
    return optimizer

class SegmentationModel(pl.LightningModule):
    def __init__(self, architecture, n_channels, n_classes, lr, encoder, encoder_weights, loss, optimizer):
        super().__init__()
        self.save_hyperparameters() # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.lr = lr
        self.architecture = architecture
        self.encoder = encoder
        self.encoder_weights = encoder_weights
        self.loss = loss
        self.mode = 'binary' if self.n_classes == 1 else 'multiclass'
        self.optimizer = optimizer
        
        self.model_dict = {
            'Unet': sm_torch.Unet,
            'DeepLabV3Plus': sm_torch.DeepLabV3Plus,
            'DeepLabV3': sm_torch.DeepLabV3,
            'UnetPlusPlus': sm_torch.UnetPlusPlus,
            'Linknet': sm_torch.Linknet,
            'PSPNet': sm_torch.PSPNet,
            'FPN': sm_torch.FPN,
            'MAnet': sm_torch.MAnet,
            'PAN': sm_torch.PAN
        }
        
        self.loss_dict = {
            'DiceLoss': sm_torch.losses.DiceLoss,
            'FocalLoss': sm_torch.losses.FocalLoss,
            'TverskyLoss': sm_torch.losses.TverskyLoss,
            'JaccardLoss': sm_torch.losses.JaccardLoss,
            'LovaszLoss': sm_torch.losses.LovaszLoss,
            'SoftBCEWithLogitsLoss': sm_torch.losses.SoftBCEWithLogitsLoss,
            'SoftCrossEntropyLoss': sm_torch.losses.SoftCrossEntropyLoss,
            'MCCLoss': sm_torch.losses.MCCLoss
        }
        
        # Initialize model
        self.model = self.model_dict[self.architecture](encoder_name=self.encoder, encoder_weights=self.encoder_weights, in_channels=self.n_channels, classes=self.n_classes)
        
        # Loss function
        if self.loss in ['DiceLoss', 'FocalLoss', 'TverskyLoss', 'JaccardLoss', 'LovaszLoss']:
            self.loss_fn = self.loss_dict[self.loss](mode=self.mode, from_logits=True)
        else:
            self.loss_fn = self.loss_dict[self.loss]()

    def forward(self, x):
        return self.model(x)
    
    def shared_step(self, batch, stage):
        image, mask = batch
        out = self.forward(image)
        loss = self.loss_fn(out, mask.long())
        tp, fp, fn, tn = sm_torch.metrics.get_stats(torch.argmax(out, 1).unsqueeze(1), mask.long(), mode=self.mode, num_classes = self.n_classes)        
        iou_score = sm_torch.metrics.iou_score(tp, fp, fn, tn, reduction="macro-imagewise")
        f1_score = sm_torch.metrics.f1_score(tp, fp, fn, tn, reduction="macro-imagewise")
        f2_score = sm_torch.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="macro-imagewise")
        recall = sm_torch.metrics.recall(tp, fp, fn, tn, reduction="macro-imagewise")
        precision = sm_torch.metrics.precision(tp, fp, fn, tn, reduction="macro-imagewise")

        self.log_dict({
            f"{stage}_loss": loss,
            f"{stage}_iou": iou_score,
            f"{stage}_f1": f1_score,
            f"{stage}_f2": f2_score,
            f"{stage}_precision": precision,
            f"{stage}_recall": recall,
        }, prog_bar=True, on_step=False, on_epoch=True)
        
        # log values
        if self.logger:
            self.logger.experiment.add_scalar(f'{stage}/F1Score', f1_score, self.global_step)
            self.logger.experiment.add_scalar(f'{stage}/F2Score', f2_score, self.global_step)
            self.logger.experiment.add_scalar(f'{stage}/IoU', iou_score, self.global_step)
            self.logger.experiment.add_scalar(f'{stage}/Recall', recall, self.global_step)
            self.logger.experiment.add_scalar(f'{stage}/Precision', precision, self.global_step)
            self.logger.experiment.add_scalar(f'{stage}/Loss', loss, self.global_step)
            self.logger.experiment.add_scalar(f'{stage}/LearningRate', self.lr, self.global_step)
            
        # return self.log_dict

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")
        
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")
    
    def on_validation_epoch_end(self):
        metrics = self.trainer.logged_metrics

        # Ensure metrics are iterable before attempting to stack them
        mean_outputs = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor) and v.dim() == 0:
                mean_outputs[k] = v  # Use the scalar value directly
            elif isinstance(v, list) and all(isinstance(i, torch.Tensor) for i in v):
                mean_outputs[k] = torch.stack(v).mean()  # Calculate the mean if it's a list of tensors
            else:
                mean_outputs[k] = torch.tensor(v).mean()  # Default case, convert to tensor and calculate mean

        # Log the mean metrics
        self.log_dict(mean_outputs, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = get_optimizer(self.optimizer, self.parameters(), self.lr)
        lr_scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5),
            'monitor': 'val_loss'
        }
        return [optimizer], [lr_scheduler]