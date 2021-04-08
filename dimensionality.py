import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.decomposition import PCA
import pandas as pd
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from candidate_models.base_models.unsupervised_vvs import ModelBuilder
from model_tools.activations.pca import LayerPCA
from model_tools.brain_transformation.neural import LayerScores
from result_caching import store_dict, store_xarray

tf_res18_layers = ['encode_1.conv'] + ['encode_%i' % i for i in range(1, 10)]
pt_resnet18_layers = ['relu', 'maxpool'] +\
                     ['layer1.0.relu', 'layer1.1.relu'] +\
                     ['layer2.0.relu', 'layer2.1.relu'] +\
                     ['layer3.0.relu', 'layer3.1.relu'] +\
                     ['layer4.0.relu', 'layer4.1.relu']
prednet_layers = ['A_%i' % i for i in range(1, 4)] \
                 + ['Ahat_%i' % i for i in range(1, 4)] \
                 + ['E_%i' % i for i in range(1, 4)] \
                 + ['R_%i' % i for i in range(1, 4)]
vvs_models = ['resnet18-supervised', 'resnet18-la', 'resnet18-ir', 'resnet18-ae',
              'resnet18-cpc', 'resnet18-color', 'resnet18-rp', 'resnet18-depth',
              'resnet18-simclr', 'resnet18-deepcluster', 'resnet18-cmc']


def get_results(benchmark, n_components, models=None):
    if models is None:
        models = vvs_models

    try:
        stimulus_identifier = benchmark._assembly.stimulus_set_identifier
    except AttributeError:
        stimulus_identifier = benchmark.stimulus_set.identifier

    results = pd.DataFrame()

    for model_identifier in models:
        if model_identifier in ModelBuilder.PT_MODELS:
            layers = pt_resnet18_layers
        elif model_identifier == 'prednet':
            layers = prednet_layers
        elif model_identifier == 'resnet18-simclr':
            layers = tf_res18_layers[1:]
        else:
            layers = tf_res18_layers

        result = get_model_scores(model_identifier, benchmark, layers, n_components)
        effdims_dataset = get_model_effdims_dataset(model_identifier, n_components, stimulus_identifier)
        result = pd.merge(result, effdims_dataset, on=['model', 'layer'])
        if n_components is not None:
            effdims_imagenet = get_model_effdims_imagenet(model_identifier, n_components)
            result = pd.merge(result, effdims_imagenet, on=['model', 'layer'])

        results = results.append(result)

    return results


def get_model_scores(model_identifier, benchmark, layers, n_components):
    tf.reset_default_graph()

    activations_model = ModelBuilder()(model_identifier)
    if n_components is not None:
        activations_model.identifier += f'-{n_components}components'
        _ = LayerPCA.hook(activations_model, n_components=n_components)

    model_scores = LayerScores(model_identifier=activations_model.identifier,
                               activations_model=activations_model,
                               visual_degrees=8)
    score = model_scores(benchmark=benchmark, layers=layers, prerun=True)

    score = score.to_dataframe(name='').unstack(level='aggregation').reset_index()
    score.columns = ['layer', 'score', 'score_error']
    score = score.assign(model=model_identifier)

    return score


def get_model_effdims_imagenet(model_identifier, n_components):
    activations_model_identifier = model_identifier
    if n_components is not None:
        activations_model_identifier = activations_model_identifier + f'-{n_components}components'
    function_identifier = f'{LayerPCA.__module__}.{LayerPCA.__name__}._pcas/' \
                          f'identifier={activations_model_identifier},n_components={n_components}'
    store = store_dict(dict_key='layers', identifier_ignore=['layers'])
    pcas = store.load(function_identifier)
    effdims = {layer: effective_dimensionality(pca) for layer, pca in pcas.items()}

    effdims = [{'layer': layer, 'effective_dimensionality_imagenet': dim} for layer, dim in effdims.items()]
    effdims = pd.DataFrame(effdims)
    effdims = effdims.assign(model=model_identifier)

    return effdims


def get_model_effdims_dataset(model_identifier, n_components, stimuli_identifier):
    activations_model_identifier = model_identifier
    if n_components is not None:
        activations_model_identifier = activations_model_identifier + f'-{n_components}components'
    function_identifier = 'model_tools.activations.core.ActivationsExtractorHelper._from_paths_stored/' \
                          f'identifier={activations_model_identifier},stimuli_identifier={stimuli_identifier}'
    store = store_xarray(identifier_ignore=['model', 'benchmark', 'layers', 'prerun'],
                         combine_fields={'layers': 'layer'})
    regressors = store.load(function_identifier)
    pcas = {layer: PCA().fit(x) for layer, x in regressors.groupby('layer')}

    effdims = {layer: effective_dimensionality(pca) for layer, pca in pcas.items()}
    effdims = [{'layer': layer, 'effective_dimensionality_dataset': dim} for layer, dim in effdims.items()]
    effdims = pd.DataFrame(effdims)
    effdims = effdims.assign(model=model_identifier)

    return effdims


def effective_dimensionality(pca):
    eigen_values = pca.singular_values_ ** 2 / (pca.n_components_ - 1)
    effective_dim = eigen_values.sum() ** 2 / (eigen_values ** 2).sum()
    return effective_dim