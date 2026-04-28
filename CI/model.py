import segmentation_models_pytorch as smp
import torch.nn as nn
import numpy as np

class CI_unet(nn.Module):
    
    def __init__(
        self,
        activation = None, # softmax should be applied later in inference from raw logits
        encoder_name = "resnet34",
        encoder_weights = None, # should be loaded later from the weights for each leadtime
        encoder_depth = 5, # should be set from the config file for each leadtime
        decoder_channels = [256, 128, 64, 32, 16], # should be set from the config file for each leadtime
        in_channels = 10,
        classes = 4,
        decoder_attention_type = "scse",
        decoder_interpolation = "bilinear",
        decoder_use_norm = "batchnorm",
        **kwargs
    ):
        
        super().__init__()
        
        self.activation = activation
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.encoder_depth = encoder_depth
        self.decoder_channels = decoder_channels
        self.in_channels = in_channels
        self.classes = classes
        self.decoder_attention_type = decoder_attention_type
        self.decoder_interpolation = decoder_interpolation
        self.decoder_use_norm = decoder_use_norm

        self.model = smp.Unet(
            encoder_name = encoder_name,        
            encoder_weights = encoder_weights,    
            in_channels = in_channels,                
            encoder_depth = encoder_depth,
            classes = classes, 
            activation= activation, 
            decoder_channels = decoder_channels,
            decoder_attention_type = decoder_attention_type,
            decoder_interpolation = decoder_interpolation,
            decoder_use_norm = decoder_use_norm
        )
        
    def forward(self, x):
        
        y = self.model(x)

        return y
