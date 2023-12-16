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
def main(seed, debug=False):
    n_pcs = 'NA'
    eigmetrics_df = pd.DataFrame()
    
    for model, layers in get_activation_models(seed, n_pcs):
        model.
        
        
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
        eigspec_df.to_csv(f'results/fall2023/eigspectra_SVD_nwpc|seed:{seed}|dataset:{dataset_csv}|pooling:{pooling}|grayscale:{grayscale}.csv', index=False)
        eigmetrics_df.to_csv(f'results/fall2023/eigmetrics_SVD_nwpc|seed:{seed}|dataset:{dataset_csv}|pooling:{pooling}|grayscale:{grayscale}.csv', index=False)
        
        

@timed
def main(benchmark, seed, pooling, debug=False):
    if pooling=='max' or pooling=='avg': #projections, spatial_PCA, random_spatial
        n_pcs = 'NA'
    elif pooling=='layerPCA' or pooling=='maxpool_PCA' or pooling=='zscore_PCA' or pooling=='PCAtrans_zscore':
        n_pcs = 1000 #default = 1000
    
    save_path = f'results/fall2023/encoding_SVD_nwpc|seed:{seed}|pooling:{pooling}|nPCs:{n_pcs}|benchmark:{benchmark._identifier}.csv'
    if os.path.exists(save_path):
        print(f'Results already exists: {save_path}')
        return
    
    scores = pd.DataFrame()
    for model, layers in get_activation_models(seed, n_pcs):
        #get_activation models = call_2: frame(0)=generators, fback/frame(1)=fit_encoding, frame(1).fback/frame(2)=__main__
        layer_scores = fit_encoder(benchmark, model, layers, pooling, n_pcs)
        scores = scores.append(layer_scores)
        if debug:
            break
    if not debug:
        scores.to_csv(save_path, index=False)