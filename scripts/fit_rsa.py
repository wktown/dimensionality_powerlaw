import os
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from model_tools.activations.pca import LayerPCA
from model_tools.brain_transformation.neural import LayerScores
from activation_models.generators import get_activation_models
from custom_model_tools.hooks import GlobalMaxPool2d
from custom_model_tools.layerPCA_modified import LayerPCA_Modified
from benchmarks.majajhong2015_rsa import DicarloMajajHong2015V4RSA, DicarloMajajHong2015ITRSA
from utils import timed, id_to_properties

import logging
logging.basicConfig(level=logging.INFO)


@timed
def main(benchmark, seed, pooling, debug=False):
    if pooling=='max' or pooling=='avg': #projections, spatial_PCA, random_spatial
        n_pcs = 'NA'
    elif pooling=='layerPCA' or pooling=='PCA_maxpool' or pooling=='PCA_zscore':
        n_pcs = 1000
        
    save_path = f'results/variance_SVD/rsa_None|seed:{seed}|pooling:{pooling}|nPCs:{n_pcs}|benchmark:{benchmark._identifier}.csv'
    print(save_path)
    if os.path.exists(save_path):
        print(f'Results already exists: {save_path}')
        return
    
    scores = pd.DataFrame()
    for model, layers in get_activation_models(seed, n_pcs):
        layer_scores = fit_rsa(benchmark, model, layers, pooling, n_pcs)
        scores = scores.append(layer_scores)
        if debug:
            break
    if not debug:
        scores.to_csv(save_path, index=False)


def fit_rsa(benchmark, model, layers, pooling, n_pcs, hooks=None):
    """Fit layers one at a time to save on memory"""

    layer_scores = pd.DataFrame()
    model_identifier = model.identifier
    model_properties = id_to_properties(model_identifier)

    for layer in layers:
        if pooling == 'max':
            handle = GlobalMaxPool2d.hook(model)
            model.identifier = model_identifier + f'|layer:{layer}|pooling:{pooling}'
        elif pooling == 'layerPCA':
            handle = LayerPCA.hook(model, n_components=n_pcs)
            model.identifier = model_identifier + f'|layer:{layer}|pooling:{pooling}|n_components:{n_pcs}'
        elif pooling == 'PCA_maxpool':
            handle = LayerPCA_Modified.hook(model, n_components=n_pcs, mod='max_pool')
            model.identifier = model_identifier + f'|layer:{layer}|pooling:{pooling}|n_components:{n_pcs}'
        elif pooling == 'PCA_zscore':
            handle = LayerPCA_Modified.hook(model, n_components=n_pcs, mod='z_score')
            model.identifier = model_identifier + f'|layer:{layer}|pooling:{pooling}|n_components:{n_pcs}'

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


def get_benchmark(benchmark, region):
    if benchmark == 'majajhong2015':
        assert region in ['IT', 'V4']
        benchmark = DicarloMajajHong2015ITRSA() if region == 'IT' else DicarloMajajHong2015V4RSA()
    else:
        raise ValueError(f'Unknown benchmark: {benchmark}')
    return benchmark


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit rsa to a neural dataset')
    parser.add_argument('--bench', type=str, default='majajhong2015',
                        choices=['majajhong2015'],
                        help='Neural benchmark dataset to fit')
    parser.add_argument('--region', type=str, default='IT',
                        help='Region(s) to fit. Valid region(s) depend on the neural benchmark')
    parser.add_argument('--pooling', dest='pooling', type=str,
                    choices=['max', 'avg', 'layerPCA', 'PCA_maxpool', 'PCA_zscore'],
                    #projections, spatial_PCA, random_spatial
                    help='Choose global max-pooling, avg-pooling, or no pooling (layerPCA) prior to fitting')
    #parser.add_argument('--no_pooling', dest='pooling', action='store_false',
    #                    help='Do not perform global max-pooling prior to fitting')
    parser.add_argument('--debug', action='store_true',
                        help='Just run a single model to make sure there are no errors')
    parser.add_argument('--seed', dest='seed', type=int, default=0,
                        help='Choose a random seed for analysis (torch and numpy)')
    args = parser.parse_args()

    benchmark = get_benchmark(benchmark=args.bench, region=args.region)
    
    main(benchmark=benchmark, seed=args.seed, pooling=args.pooling, debug=args.debug)
