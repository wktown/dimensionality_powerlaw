#compute the effective dimensionality of the models
#calls generators.py's get_activation_models function, custom_tools.eigenspectrum & image_transform

import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from torchvision.transforms import Grayscale
from activation_models.generators import get_activation_models
from custom_model_tools.eigenspectrum import EigenspectrumImageNet, EigenspectrumImageNet21k, \
    EigenspectrumObject2Vec, EigenspectrumMajajHong2015
from custom_model_tools.zscore_eigenspectrum import ZScore_EigenspectrumImageNet
from custom_model_tools.image_transform import ImageDatasetTransformer
from utils import timed
import logging

logging.basicConfig(level=logging.INFO)


@timed
def main(dataset, data_dir, pooling, seed, grayscale, debug=False):
    image_transform = ImageDatasetTransformer('grayscale', Grayscale()) if grayscale else None
    eigspec_df = pd.DataFrame()
    eigmetrics_df = pd.DataFrame()
    if pooling == 'layerPCA' or 'PCA_zscore':
        n_pcs = 1000
    else:
        n_pcs = 'NA'
    
    for model, layers in get_activation_models(seed, n_pcs):
        eigspec = get_eigenspectrum(dataset, data_dir, model, pooling, image_transform)
        eigspec.fit(layers)
        eigspec_df = eigspec_df.append(eigspec.as_df())
        eigmetrics_df = eigmetrics_df.append(eigspec.metrics_as_df())
        if debug:
            break

    if not debug:
        if dataset == 'imagenet_zscore_acts':
            dataset_csv = 'imagenet'
        else:
            dataset_csv = dataset
        eigspec_df.to_csv(f'results/variance_SVD/eigspectra_None|seed:{seed}|dataset:{dataset_csv}|pooling:{pooling}|grayscale:{grayscale}.csv', index=False)
        eigmetrics_df.to_csv(f'results/variance_SVD/eigmetrics_None|seed:{seed}|dataset:{dataset_csv}|pooling:{pooling}|grayscale:{grayscale}.csv', index=False)
        

def get_eigenspectrum(dataset, data_dir, activations_extractor, pooling, image_transform):
    if dataset == 'imagenet':
        return EigenspectrumImageNet(activations_extractor=activations_extractor,
                                     pooling=pooling,
                                     image_transform=image_transform)
        
    elif dataset == 'imagenet_zscore_acts':
        return ZScore_EigenspectrumImageNet(activations_extractor=activations_extractor,
                                            pooling=pooling,
                                            image_transform=image_transform)
        
    elif dataset == 'imagenet21k':
        return EigenspectrumImageNet21k(data_dir=data_dir,
                                        activations_extractor=activations_extractor,
                                        pooling=pooling,
                                        image_transform=image_transform)
    elif dataset == 'object2vec':
        return EigenspectrumObject2Vec(data_dir=data_dir,
                                       activations_extractor=activations_extractor,
                                       pooling=pooling,
                                       image_transform=image_transform)
    elif dataset == 'majajhong2015':
        return EigenspectrumMajajHong2015(activations_extractor=activations_extractor,
                                          pooling=pooling,
                                          image_transform=image_transform)
    else:
        raise ValueError(f'Unknown eigenspectrum dataset: {dataset}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute and store eigenspectra of models')
    parser.add_argument('--dataset', type=str,
                        choices=['imagenet', 'imagenet21k', 'object2vec', 'majajhong2015', 'imagenet_zscore_acts'],
                        help='Dataset of concepts for which to compute the eigenspectrum')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory containing stimuli')
    parser.add_argument('--pooling', dest='pooling', type=str, default=None,
                        choices=['max', 'avg', 'projections', 'spatial_pca', 'random_spatial', 'layerPCA', 'PCA_zscore'], #zscore
                        help='Choose global max pooling, avg pooling, no pooling, to select one random spatial position, or to compute the eigenspectrum at each spatial position in the final layer(s) of the model prior to computing the eigenspectrum')
    parser.add_argument('--seed', dest='seed', type=int, default=0,
                        help='Choose a random seed for analysis (torch and numpy)')
    parser.add_argument('--grayscale', action='store_true',
                        help='Compute the eigenspectrum on grayscale inputs')
    parser.add_argument('--debug', action='store_true',
                        help='Just run a single model to make sure there are no errors')
    args = parser.parse_args()

    main(dataset=args.dataset, data_dir=args.data_dir, pooling=args.pooling, seed=args.seed, grayscale=args.grayscale, debug=args.debug)
