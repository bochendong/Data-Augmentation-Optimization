import torch
import torch.nn as nn
import torch.nn.functional as F

def perceptual_loss(vgg_model, input_images, output_images):
    feature_layers = [vgg_model.features[i] for i in range(len(vgg_model.features))]
    feature_extractor = nn.Sequential(*feature_layers[:-1]).cuda()
    
    input_features = feature_extractor(input_images)
    output_features = feature_extractor(output_images)
    
    return nn.functional.mse_loss(input_features, output_features)

