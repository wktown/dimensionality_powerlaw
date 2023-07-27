import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from torchvision.transforms import Grayscale
from activation_models.generators import get_activation_models
from utils import timed
from brainscore.metrics.regression import linear_regression, ridge_regression
import brainscore.benchmarks as bench
from utils import id_to_properties
from custom_model_tools.hooks import GlobalMaxPool2d, N_PCs_Max, N_PCs_None
from model_tools.activations.pca import LayerPCA
import logging

from typing import Optional, List
from model_tools.utils import fullname
from result_caching import store_dict
from custom_model_tools.eigenspectrum import EigenspectrumImageNet, EigenspectrumImageNet21k, EigenspectrumMajajHong2015
from sklearn.decomposition import PCA
from model_tools.activations.core import flatten
from tqdm import tqdm
import os
from custom_model_tools.image_transform import ImageDatasetTransformer

import numpy as np

logging.basicConfig(level=logging.INFO)

#get model activations from majaj images (multiple powerlaw slopes in eigenspectrum)
#get brain data

# - project model activations onto PCs (transformed)
#     - do RSA & encoding model predictions with various numbers of PCs (top 10, 50, 100, etc)
#     - plot rsa/encoding performance vs. number of PCs
# - same as above but z-score pc data first (could be similar to variance scale in eig script)
# - create/plot RDMs



@timed
def main(dataset, data_dir, pooling, grayscale, debug=False):
    encoding_df = pd.DataFrame()
    rsa_df = pd.DataFrame()
    for model, layers in get_activation_models():
        encoding = get_encoding()
        rsa = get_rsa()
        #eigspec = get_eigenspectrum(dataset, data_dir, model, pooling, image_transform)
        #eigspec.fit(layers)
        encoding_df = encoding_df.append(encoding.as_df())
        rsa_df = rsa_df.append(rsa.as_df())
        if debug:
            break

    if not debug:
        encoding_df.to_csv(f'results/variablePC_encoding|dataset:{dataset}|pooling:{pooling}|grayscale:{grayscale}.csv', index=False)
        rsa_df.to_csv(f'results/variablePC_RSA|dataset:{dataset}|pooling:{pooling}|grayscale:{grayscale}.csv', index=False)
        
        


#eigenspectrum:
# - handle for pooling or projection (self._extractor)
# - activations = self._extractor(image_paths, layers=[layer])
# - activations = activations.sel(layer=layer).values
# - activations = flatten(activations)
# - pca = PCA(random_state=0)
# - pca.fit(activations)

#encoding:
# - handle for pooling (model)
#    - layer scores
# - handle for layerPCA
#    - imagenet_activations = self._extractor(imagenet_paths, layers=layers)
#    - imagenet_activations = {layer: imagenet_activations.sel(layer=layer).values for layer in np.unique(imagenet_activations['layer'])}
#    - activations = flatten(activations)
#    - if activations.shape[1] <= n_components: pca = None    #num conv filters?
#    - else pca = PCA(n_components=n_components, random_state=0)
#    - pca.fit(activations)
#    - pca.transform(activations)
# - layer scores





def get_encoding(benchmark, model, layers, pooling, hooks=None):
    """Fit layers one at a time to save on memory"""

    layer_scores = pd.DataFrame()
    model_identifier = model.identifier
    model_properties = id_to_properties(model_identifier)
    
    for layer in layers:
        if pooling == 'max_npcs':
            handle = N_PCs_Max.hook(model)
            model.identifier = model_identifier + f'|layer:{layer}|pooling:max'
        if pooling == 'none_npcs':
            handle = N_PCs_None.hook(model)

        

        handles = []
        if hooks is not None:
            handles = [cls.hook(model) for cls in hooks]
            
        logging.info(model.identifier)
        logging.info(layer)
        model_scores = LayerScores(model_identifier=model.identifier,
                                   activations_model=model,
                                   visual_degrees=8)
        score = model_scores(benchmark=benchmark, layers=[layer], prerun=True)
        handle.remove()

        for h in handles:
            h.remove()

        if 'aggregation' in score.dims:
            score = score.to_dataframe(name='').unstack(level='aggregation').reset_index()
            score.columns = ['layer', 'score', 'score_error']
        else:
            score = score.to_dataframe(name='').reset_index()
            score.columns = ['layer', 'score']

        layer_scores = layer_scores.append(score)

    layer_scores = layer_scores.assign(**model_properties)
    return layer_scores


def get_benchmark(benchmark, region, regression, data_dir):
    if benchmark == 'majajhong2015':
        assert region in ['IT', 'V4']
        identifier = f'dicarlo.MajajHong2015public.{region}-pls'
        benchmark = bench.load(identifier)
        if regression == 'lin':
            benchmark._identifier = benchmark.identifier.replace('pls', 'lin')
            benchmark._similarity_metric.regression = linear_regression()
            benchmark._similarity_metric.regression._regression.alpha = 0.1
        elif regression == 'l2':
            alpha = 1000
            benchmark._identifier = benchmark.identifier.replace('pls', f'ridge_alpha={alpha}')
            benchmark._similarity_metric.regression = ridge_regression(regression_kwargs= {'alpha':alpha})
    elif benchmark == 'freeman2013':
        assert region == 'V1'
        identifier = f'movshon.FreemanZiemba2013public.{region}-pls'
        benchmark = bench.load(identifier)
        if regression == 'lin':
            benchmark._identifier = benchmark.identifier.replace('pls', 'lin')
            benchmark._similarity_metric.regression = linear_regression()
            benchmark._similarity_metric.regression._regression.alpha = 0.1
        elif regression == 'l2':
            benchmark._identifier = benchmark.identifier.replace('pls', 'l2')
            benchmark._similarity_metric.regression = ridge_regression()
    elif benchmark == 'object2vec':
        if region == 'all':
            region = None
        regions = region if region is None or ',' not in region else region.split(',')
        #benchmark = Object2VecEncoderBenchmark(data_dir=data_dir, regions=regions, regression=regression)
    else:
        raise ValueError(f'Unknown benchmark: {benchmark}')
    return benchmark












if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit encoding models to a neural dataset')
    parser.add_argument('--bench', type=str, default='majajhong2015',
                        choices=['majajhong2015', 'freeman2013', 'object2vec'],
                        help='Neural benchmark dataset to fit')
    parser.add_argument('--region', type=str, default='IT',
                        help='Region(s) to fit. Valid region(s) depend on the neural benchmark')
    parser.add_argument('--regression', type=str, default='pls',
                        choices=['pls', 'lin', 'l2'],
                        help='Partial-least-squares or ordinary-least-squares for fitting')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory for neural benchmark (only required for "object2vec")')
    parser.add_argument('--pooling', dest='pooling', type=str,
                        choices=['max','avg','none', 'random_spatial'],
                        help='Choose global max-pooling, global avg-pooling, no pooling, or random spatial positions prior to fitting')
    parser.add_argument('--debug', action='store_true',
                        help='Just run a single model to make sure there are no errors')
    args = parser.parse_args()

    benchmark = get_benchmark(benchmark=args.bench, region=args.region,
                              regression=args.regression, data_dir=args.data_dir)
    main(benchmark=benchmark, pooling=args.pooling, debug=args.debug)











