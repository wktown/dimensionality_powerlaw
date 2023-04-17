import os

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import torch
from torch import nn
from torchvision.models import resnet18, resnet50, alexnet, vgg16
from torchvision.models import convnext_tiny, maxvit_t, mnasnet1_3, regnet_x_400mf,resnext50_32x4d, swin_t
from candidate_models.base_models.unsupervised_vvs import ModelBuilder
from model_tools.activations.pytorch import PytorchWrapper, load_preprocess_images
from visualpriors.taskonomy_network import TASKONOMY_PRETRAINED_URLS, TaskonomyEncoder
from functools import partial
from utils import properties_to_id
#from counter_example.train_imagenet import LitResnet

resnet18_pt_layers = [f'layer1.{i}.relu' for i in range(2)] + \
                     [f'layer2.{i}.relu' for i in range(2)] + \
                     [f'layer3.{i}.relu' for i in range(2)] + \
                     [f'layer4.{i}.relu' for i in range(2)]

resnet50_pt_layers = [f'layer1.{i}.relu' for i in range(3)] + \
                     [f'layer2.{i}.relu' for i in range(4)] + \
                     [f'layer3.{i}.relu' for i in range(6)] + \
                     [f'layer4.{i}.relu' for i in range(3)]

resnet18_tf_layers = [f'encode_{i}' for i in range(2, 10)]

alexnet_layers = [f"features.{i}" for i in [1, 4, 7, 9, 11]]

convnext_layers = [f"features.{j}.{i}.block.4" for i in range(3) for j in [1, 3, 7]] + \
    [f"features.5.{i}.block.4" for i in range(9)]
    
maxvit_layers = [f"blocks.{i[0]}.layers.{i[1]}.layers.{j}_attention.mlp_layer.2" for i in [
    (0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (3, 0), (3, 1)] for j in["window", "grid"]]

mnasnet_layers = [f"layers.{k}.{j}.layers.5" for k in range (8, 11) for j in range(3)] + \
    [f"layers.11.{j}.layers.5" for j in range(2)] + \
        [f"layers.12.{j}.layers.5" for j in range(4)] + \
            [f"layers.13.0.layers.5", "layers.16"]

regnet_layers = [f"trunk_output.block1.block1-0.activation"] + \
    [f"trunk_output.block2.block2-{i}.activation" for i in range(2)] + \
        [f"trunk_output.block3.block3-{i}.activation" for i in range(7)] + \
            [f"trunk_output.block4.block4-{i}.activation" for i in range(12)]

swin_layers = [f"features.{i}.{j}.mlp.1" for i in [1, 3, 7] for j in range(2)] + \
    [f"features.5.{i}.mlp.1" for i in range(6)]
            
vgg_layers = [f"features.{i}" for i in [1, 3, 6, 8, 11, 13, 15, 18, 20, 22, 25, 27, 29]]



def get_activation_models(pytorch=False, untrained=True, transformers=False, vgg=False,
                          vvs=False, taskonomy=False, pytorch_hub=False):
    
    torch.manual_seed(seed=0)
    
    if pytorch:
        for model, layers in pytorch_models():
            yield model, layers
    if untrained:
        for model, layers in untrained_models():
            yield model, layers
    if transformers:
        for model, layers in transformer_models():
            yield model, layers
    if vgg:
        for model, layers in vgg_test():
            yield model, layers
    
    if vvs:
        for model, layers in vvs_models():
            yield model, layers
    if taskonomy:
        for model, layers in taskonomy_models():
            yield model, layers
    if pytorch_hub:
        for model, layers in pytorch_hub_models():
            yield model, layers
    

def kai_uniform(m):
    # - don't change BatchNorm
    #if isinstance(m, nn.BatchNorm2d):
    #    nn.init.ones_(m.weight)
    #    nn.init.zeros_(m.bias)
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_out', nonlinearity='relu')
        #  - keep bias = 0??
        #if m.bias is not None:
        #    nn.init.kaiming_uniform_(m.bias, a=0, mode='fan_out', nonlinearity='relu')
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_out', nonlinearity='relu')
        #mnasnet = sigmoid instead of relu
        
        #if m.bias is not None:
        #    nn.init.kaiming_uniform_(m.bias, a=0, mode='fan_out', nonlinearity='relu')

def uniform(m):
    if isinstance(m, nn.Conv2d):
        nn.init.uniform_(m.weight, a= -0.1, b=0.1)
        if m.bias is not None:
            nn.init.uniform_(m.bias, a= -0.1, b= 0.1)
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, a= -0.1, b= 0.1)
        if m.bias is not None:
            nn.init.uniform_(m.bias, a= -0.1, b= 0.1)

def normal(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0.0, std=0.05)
        if m.bias is not None:
            nn.init.normal_(m.bias, mean=0.0, std=0.05)
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.05)
        if m.bias is not None:
            nn.init.normal_(m.bias, mean=0.0, std=0.05)
            
def sparse(m):
    if isinstance(m, nn.Conv2d):
        print(m.size)
        s = 0.1
        nn.init.sparse_(m.weight, sparsity=s, std=0.05)
        if m.bias is not None:
            nn.init.sparse_(m.weight, sparsity=s, std=0.05)
    if isinstance(m, nn.Linear):
        nn.init.sparse_(m.weight, sparsity=s, std=0.05)
        if m.bias is not None:
            nn.init.sparse_(m.bias, sparsity=s, std=0.05)

def orthogonal(m):
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.orthogonal_(m.bias)
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.orthogonal_(m.bias)


def untrained_models():
    
    new_init = True
    
    model = resnet18(weights=None)
    if new_init:
        model.apply(orthogonal)
    identifier = properties_to_id('ResNet18', 'O', 'Untrained', 'PyTorch')
    model = wrap_pt(model, identifier)
    yield model, resnet18_pt_layers
    
    model = resnet50(weights=None)
    if new_init:
        model.apply(orthogonal)
    identifier = properties_to_id('ResNet50', 'O', 'Untrained', 'PyTorch')
    model = wrap_pt(model, identifier)
    yield model, resnet50_pt_layers
    
    model = alexnet(weights=None)
    if new_init:
       model.apply(orthogonal)
    identifier = properties_to_id('AlexNet', 'O', 'Untrained', 'PyTorch')
    model = wrap_pt(model, identifier)
    yield model, alexnet_layers
    
    #model = convnext_tiny(weights=None)
    #identifier = properties_to_id('ConvNeXt_Tiny', 'None', 'Untrained', 'PyTorch')
    #model = wrap_pt(model, identifier)
    #yield model, convnext_layers
    
    model = mnasnet1_3(weights=None)
    if new_init:
        model.apply(orthogonal)
    identifier = properties_to_id('MNASNet13', 'O', 'Untrained', 'PyTorch')
    model = wrap_pt(model, identifier)
    yield model, mnasnet_layers
    
    model = regnet_x_400mf(weights=None)
    if new_init:
        model.apply(orthogonal)
    identifier = properties_to_id('RegNet_400mf', 'O', 'Untrained', 'PyTorch')
    model = wrap_pt(model, identifier)
    yield model, regnet_layers
    
    model = resnext50_32x4d(weights=None)
    if new_init:
        model.apply(orthogonal)
    identifier = properties_to_id('ResNeXt50', 'O', 'Untrained', 'PyTorch')
    model = wrap_pt(model, identifier)
    yield model, resnet50_pt_layers
    
    #model = maxvit_t(weights=None)
    #identifier = properties_to_id('MaxViT', 'Kai_Uni', 'Untr.', 'Py')
    #if new_init:
    #    model.apply(kai_uniform)
    #model = wrap_pt(model, identifier)
    #yield model, maxvit_layers
    
    #model = swin_t(weights=None)
    #if new_init:
    #    model.apply(kai_uniform)
    #identifier = properties_to_id('Swin_t', 'Kai_Uni', 'Untrained', 'PyTorch')
    #model = wrap_pt(model, identifier)
    #yield model, swin_layers

def pytorch_models():

    model = resnet18(weights="IMAGENET1K_V1")
    identifier = properties_to_id('ResNet18', 'Object Classification', 'Supervised', 'PyTorch')
    model = wrap_pt(model, identifier)
    yield model, resnet18_pt_layers

    model = resnet50(weights="IMAGENET1K_V1") #also V2
    identifier = properties_to_id('ResNet50', 'Object Classification', 'Supervised', 'PyTorch')
    model = wrap_pt(model, identifier)
    yield model, resnet50_pt_layers
    
    model = alexnet(weights="IMAGENET1K_V1")
    identifier = properties_to_id('AlexNet', 'Object Classification', 'Supervised', 'PyTorch')
    model = wrap_pt(model, identifier)
    yield model, alexnet_layers
    
    #model = convnext_tiny(weights="IMAGENET1K_V1")
    #identifier = properties_to_id('ConvNeXt_Tiny', 'Object Classification', 'Supervised', 'PyTorch')
    #model = wrap_pt(model, identifier)
    #yield model, convnext_layers
    
    model = mnasnet1_3(weights="IMAGENET1K_V1")
    identifier = properties_to_id('MNASNet13', 'Object Classification', 'Supervised', 'PyTorch')
    model = wrap_pt(model, identifier)
    yield model, mnasnet_layers
    
    model = regnet_x_400mf(weights="IMAGENET1K_V1") #also V2
    identifier = properties_to_id('RegNet400mf', 'Obj.Class.', 'Sprvsd', 'Py')
    model = wrap_pt(model, identifier)
    yield model, regnet_layers
    
    model = resnext50_32x4d(weights="IMAGENET1K_V1") #also V2
    identifier = properties_to_id('ResNeXt50', 'Object Classification', 'Supervised', 'PyTorch')
    model = wrap_pt(model, identifier)
    yield model, resnet50_pt_layers
    
    model = maxvit_t(weights="IMAGENET1K_V1")
    identifier = properties_to_id('MaxViT', 'ObjClass.', 'Sprvsd', 'Py')
    model = wrap_pt(model, identifier)
    yield model, maxvit_layers
    
    model = swin_t(weights="IMAGENET1K_V1")
    identifier = properties_to_id('Swin_t', 'Obj.Class.', 'Sprvsd', 'Py')
    model = wrap_pt(model, identifier)
    yield model, swin_layers


def transformer_models():
    
    model = maxvit_t(weights=None)
    model.apply(kai_uniform)
    identifier = properties_to_id('MaxViT', 'ObjClass.', 'Untr.', 'Py')
    model = wrap_pt(model, identifier)
    yield model, maxvit_layers
    
    #model = maxvit_t(weights="IMAGENET1K_V1")
    #identifier = properties_to_id('MaxViT', 'ObjClass.', 'Sprvsd', 'Py')
    #model = wrap_pt(model, identifier)
    #yield model, maxvit_layers
    
    model = swin_t(weights=None)
    model.apply(kai_uniform)
    identifier = properties_to_id('Swin_t', 'None', 'Untrained', 'PyTorch')
    model = wrap_pt(model, identifier)
    yield model, swin_layers
    
    #model = swin_t(weights="IMAGENET1K_V1")
    #identifier = properties_to_id('Swin_t', 'Obj.Class.', 'Sprvsd', 'Py')
    #model = wrap_pt(model, identifier)
    #yield model, swin_layers
    
    
def vgg_test():
    model = vgg16(weights=None)
    identifier = properties_to_id('VGG16', 'None', 'Untrained', 'PyTorch')
    model = wrap_pt(model, identifier)
    yield model, vgg_layers
    
    model = vgg16(weights="IMAGENET1K_V1")
    identifier = properties_to_id('VGG16', 'Object Classification', 'Supervised', 'PyTorch')
    model = wrap_pt(model, identifier)
    yield model, vgg_layers



def pytorch_hub_models():
    model = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
    identifier = properties_to_id('ResNet50', 'Barlow-Twins', 'Self-Supervised', 'Pytorch Hub')
    model = wrap_pt(model, identifier)
    yield model, resnet50_pt_layers

def vvs_models():
    configs = [('resnet18-simclr', 'SimCLR', 'Self-Supervised'),
               ('resnet18-supervised', 'Object Classification', 'Supervised'),
               ('resnet18-la', 'Local Aggregation', 'Self-Supervised'),
               ('resnet18-ir', 'Instance Recognition', 'Self-Supervised'),
               ('resnet18-ae', 'Auto-Encoder', 'Self-Supervised'),
               ('resnet18-cpc', 'Contrastive Predictive Coding', 'Self-Supervised'),
               ('resnet18-color', 'Colorization', 'Self-Supervised'),
               ('resnet18-rp', 'Relative Position', 'Self-Supervised'),
               ('resnet18-depth', 'Depth Prediction', 'Supervised'),
               ('resnet18-deepcluster', 'Deep Cluster', 'Self-Supervised'),
               ('resnet18-cmc', 'Contrastive Multiview Coding', 'Self-Supervised')]

    for vvs_identifier, task, kind in configs:
        tf.reset_default_graph()

        model = ModelBuilder()(vvs_identifier)
        identifier = properties_to_id('ResNet18', task, kind, 'VVS')
        model.identifier = identifier

        if vvs_identifier in ModelBuilder.PT_MODELS:
            layers = resnet18_pt_layers
        else:
            layers = resnet18_tf_layers

        yield model, layers


def taskonomy_models():
    configs = [('autoencoding', 'Auto-Encoder', 'Self-Supervised'),
               ('curvature', 'Curvature Estimation', 'Supervised'),
               ('denoising', 'Denoising', 'Self-Supervised'),
               ('edge_texture', 'Edge Detection (2D)', 'Supervised'),
               ('edge_occlusion', 'Edge Detection (3D)', 'Supervised'),
               ('egomotion', 'Egomotion', 'Supervised'),
               ('fixated_pose', 'Fixated Pose Estimation', 'Supervised'),
               ('jigsaw', 'Jigsaw', 'Self-Supervised'),
               ('keypoints2d', 'Keypoint Detection (2D)', 'Supervised'),
               ('keypoints3d', 'Keypoint Detection (3D)', 'Supervised'),
               ('nonfixated_pose', 'Non-Fixated Pose Estimation', 'Supervised'),
               ('point_matching', 'Point Matching', 'Supervised'),
               ('reshading', 'Reshading', 'Supervised'),
               ('depth_zbuffer', 'Depth Estimation (Z-Buffer)', 'Supervised'),
               ('depth_euclidean', 'Depth Estimation', 'Supervised'),
               ('normal', 'Surface Normals Estimation', 'Supervised'),
               ('room_layout', 'Room Layout', 'Supervised'),
               ('segment_unsup25d', 'Unsupervised Segmentation (25D)', 'Self-Supervised'),
               ('segment_unsup2d', 'Unsupervised Segmentation (2D)', 'Self-Supervised'),
               ('segment_semantic', 'Semantic Segmentation', 'Supervised'),
               ('class_object', 'Object Classification', 'Supervised'),
               ('class_scene', 'Scene Classification', 'Supervised'),
               ('inpainting', 'Inpainting', 'Self-Supervised'),
               ('vanishing_point', 'Vanishing Point Estimation', 'Supervised')]

    for taskonomy_identifier, task, kind in configs:
        model = TaskonomyEncoder()
        model.eval()
        pretrained_url = TASKONOMY_PRETRAINED_URLS[taskonomy_identifier + '_encoder']
        checkpoint = torch.hub.load_state_dict_from_url(pretrained_url)
        model.load_state_dict(checkpoint['state_dict'])

        identifier = properties_to_id('ResNet50', task, kind, 'Taskonomy')
        model = wrap_pt(model, identifier, res=256)

        yield model, resnet50_pt_layers


def counterexample_models():
    def most_recent_ckpt(run_name):
        ckpt_path = f'counter_example/saved_runs/{run_name}/lightning_logs'
        latest_version = sorted([int(f.split('_')[1]) for f in os.listdir(ckpt_path)])[-1]
        ckpt_path = f'{ckpt_path}/version_{latest_version}/checkpoints/best.ckpt'
        return ckpt_path

    model = LitResnet.load_from_checkpoint(most_recent_ckpt('imagenet_resnet18')).model
    identifier = properties_to_id('ResNet18', 'Object Classification', 'Supervised', 'Counter-Example')
    model = wrap_pt(model, identifier)
    yield model, resnet18_pt_layers

    model = LitResnet.load_from_checkpoint(most_recent_ckpt('imagenet_resnet18_scrambled_labels')).model
    identifier = properties_to_id('ResNet18', 'Object Classification', 'Supervised Random', 'Counter-Example')
    model = wrap_pt(model, identifier)
    yield model, resnet18_pt_layers


def wrap_pt(model, identifier, res=224, norm=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))):
    preprocess = partial(load_preprocess_images, image_size=res,
                         normalize_mean=norm[0], normalize_std=norm[1])
    return PytorchWrapper(model=model,
                          preprocessing=preprocess,
                          identifier=identifier)
