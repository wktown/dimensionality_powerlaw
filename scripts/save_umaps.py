import os
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
#import torchlens as tl
from utils import timed#, id_to_properties
from torchvision.transforms import Grayscale
from custom_model_tools.hooks import GlobalMaxPool2d,  RandomProjection
from model_tools.activations.pca import LayerPCA
from custom_model_tools.layerPCA_modified import LayerPCA_Modified
from activation_models.generators import get_activation_models
from custom_model_tools.image_transform import ImageDatasetTransformer
from typing import Optional, List
from model_tools.utils import fullname
from tqdm import tqdm
from model_tools.activations.core import flatten
from result_caching import store_dict
from utils import id_to_properties, get_imagenet_val
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns
from umap import UMAP
import xarray as xr
from PIL import Image

import logging
logging.basicConfig(level=logging.INFO)



#get model activations from majaj images (multiple powerlaw slopes in eigenspectrum)
#get brain data
# - create/plot RDMs

# - project model activations onto PCs (transformed)
#     - do RSA & encoding model predictions with various numbers of PCs (top 10, 50, 100, etc)
#     - plot rsa/encoding performance vs. number of PCs
# - same as above but z-score pc data first (could be similar to variance scale in eig script)

@timed
def main(dataset, data_dir, seed, pooling, grayscale, debug=False):
    image_transform = ImageDatasetTransformer('grayscale', Grayscale()) if grayscale else None
    
    if pooling=='max' or pooling=='none': #projections, spatial_PCA, random_spatial
        n_pcs = 'NA'
    elif pooling=='layerPCA' or pooling=='PCA_maxpool' or pooling=='PCA_zscore':
        n_pcs = 1000
    
    model_umaps = {}
    for model, layers in get_activation_models(seed, n_pcs):
        properties = id_to_properties(model.identifier)
        
        umaps = get_umaps(dataset, data_dir, model, pooling, n_pcs, image_transform)
        umaps.get(layers)
        
        key = properties['architecture']+'|'+properties['kind']
        model_umaps[key] = umaps._umaps
        
        if debug:
            break

    if not debug:
        
        sns.set(style = 'white', context='paper',
        rc={'axes.titlesize': 12,
            'lines.linewidth': 2.5,
            'axes.linewidth': 0.4})
        #sns.set(palette="colorblind")
        
        method = properties['task'].split('_')[0]
        if method == 'Eig':
            m = 'X_transformed'
        elif method == 'SVD':
            m = 'SVD'
        
        if pooling == 'layerPCA':
            pool = f'layerPCA:{n_pcs}'
        elif pooling == 'max':
            pool = f'pooling:{pooling}'
        elif pooling == 'PCA_maxpool':
           pool = f'maxpool_PCA:{n_pcs}'
        elif pooling == 'PCA_zscore':
            pool = f'Zscore_PCA:{n_pcs}'
            
        if dataset == 'majajhong2015':
            majaj_dir = '/data/shared/brainio/brain-score/dicarlo.hvm-public'
            majaj_csv = '/data/shared/brainio/brain-score/image_dicarlo_hvm-public.csv'
            majaj_column = 'image_file_name'
            majaj_paths = paths_from_csv(majaj_dir, majaj_csv, column=majaj_column)
            stimuli = images_from_path(majaj_paths, return_da=True)
    #on rockfish cannnot use images_from_dir for mjh images (csv file is in the directory), on server can use images_from_dir
    #rockfish = '/home/wtownle1/data-mbonner5/shared/brainio/bonner-datasets/image_dicarlo_hvm-public'
    #rockfish csv: '/home/wtownle1/data-mbonner5/shared/brainio/bonner-datasets/image_dicarlo_hvm-public/image_dicarlo_hvm-public.csv'
    
        
        alphas = ['-0.2', '-0.6', '-1.0', '-2', '-3']
        for model in model_umaps.keys():
            a = model.split('|')[1].split('_')[1]
            if a in alphas:
                architecture = model.split('|')[0]
                path = f'/home/wtownle1/dimensionality_powerlaw/figures/keaton/UMAPs/{architecture}'
                if not os.path.exists(path):
                    os.makedirs(path)
                
                for layer_umaps in model_umaps[model].items():
                    layer = layer_umaps[0]
                    data_umap = layer_umaps[1]
                    
                    fig, ax = plt.subplots(figsize=(20, 20))
                    for i_stimulus in range(len(stimuli)):
                        image_box = OffsetImage(stimuli[i_stimulus].values, zoom=0.5)
                        image_box.image.axes = ax

                        ab = AnnotationBbox(
                            image_box,
                            xy=(data_umap[i_stimulus, 0], data_umap[i_stimulus, 1]),
                            xycoords="data",
                            frameon=False,
                            pad=0,
                        )
                        ax.add_artist(ab)

                    ax.set_xlim([data_umap[:, 0].min(), data_umap[:, 0].max()])
                    ax.set_ylim([data_umap[:, 1].min(), data_umap[:, 1].max()])
                    ax.axis("off")
                    #fig.suptitle(f"UMAP (alpha={a}|{pool})")
                    ax.set(title=f'UMAP (alpha={a}|{pool})')
                    
                    plt.savefig(f'{path}/UMAP|seed:{seed}|alpha={a}|layer:{layer}|{pool}|{m}.png')#, dpi=300)
                    #plt.savefig(f'{path}/UMAP|alpha={a}|{pool}.png')
                    #fig.show()

        #*np.save(f'results/RDM_{method}_pearson|dataset:{dataset}|pooling:{pooling}|grayscale:{grayscale}', pearson_rdms)
        #*np.save(f'results/RDM_{method}_euclidean|dataset:{dataset}|pooling:{pooling}|grayscale:{grayscale}', euclidean_rdms)



def paths_from_csv(image_dir, csv_path, column):
    file_names = pd.read_csv(csv_path)[column].values.tolist()
    image_paths = [os.path.join(image_dir, file) for file in file_names]
    return image_paths

def images_from_path(image_paths, return_da=False):
    images = np.stack(Image.open(i) for i in image_paths)
    if return_da:
        images = xr.DataArray(data=images)
    return images

#def paths_from_dir(dir_path):
#    image_paths = [os.path.join(dir_path, file) for file in os.listdir(dir_path)]
#    return image_paths

#def images_from_dir(dir_path, return_da=False):
#    images = np.stack([Image(i) for i in os.listdir(dir_path)])
#    if return_da:
#        images = xr.DataArray(data=images)
#    return images
        

def get_umaps(dataset, data_dir, activations_extractor, pooling, n_pcs, image_transform):
    if dataset == 'imagenet':
        return EigenspectrumImageNet(activations_extractor=activations_extractor,
                                     dataset=dataset,
                                     pooling=pooling,
                                     n_pcs=n_pcs,
                                     image_transform=image_transform)
        
    elif dataset == 'majajhong2015':
        return EigenspectrumMajajHong2015(activations_extractor=activations_extractor,
                                          dataset=dataset,
                                          pooling=pooling,
                                          n_pcs=n_pcs,
                                          image_transform=image_transform)
       
    elif dataset == 'imagenet21k':
        data_dir = data_dir
        #return EigenspectrumImageNet21k(data_dir=data_dir,
        #                                activations_extractor=activations_extractor,
        #                                pooling=pooling,
        #                                image_transform=image_transform)
        
    else:
        raise ValueError(f'Unknown eigenspectrum dataset: {dataset}')
    
        
class EigenspectrumBase:

    def __init__(self, activations_extractor, dataset, pooling, n_pcs, stimuli_identifier=None,
                 image_transform: Optional[ImageDatasetTransformer] = None,
                 hooks: Optional[List] = None):
        self._logger = logging.getLogger(fullname(self))
        self._extractor = activations_extractor
        self._pooling = pooling
        self._n_pcs = n_pcs
        self._hooks = hooks
        self._stimuli_identifier = stimuli_identifier
        self._image_transform = image_transform
        self._dataset = dataset
        self._uamps = {}
        
    def get(self, layers):
        transform_name = None if self._image_transform is None else self._image_transform.name
        
        self._umaps = self.umaps(identifier=self._extractor.identifier,
                                                        stimuli_identifier=self._stimuli_identifier,
                                                        layers=layers,
                                                        pooling=self._pooling,
                                                        n_pcs = self._n_pcs,
                                                        image_transform_name=transform_name)
        
        #np.save(f'results/modelRDM_{self._metric}|dataset:{self._dataset}|pooling:{self._pooling}|grayscale:{self._image_transform}', self._rdms)

    @store_dict(dict_key='layers', identifier_ignore=['layers'])
    def umaps(self, identifier, stimuli_identifier, layers, pooling, n_pcs, image_transform_name):
        image_paths = self.get_image_paths()
        if self._image_transform is not None:
            image_paths = self._image_transform.transform_dataset(self._stimuli_identifier, image_paths)

        # Compute activations and PCA for every layer individually to save on memory.
        # This is more inefficient because we run images through the network several times,
        # but it is a more scalable approach when using many images and large layers.
        
        umaps = {}

        for layer in layers:
            if pooling == 'max':
                handle = GlobalMaxPool2d.hook(self._extractor)
                self._extractor.identifier = self._extractor.identifier + f'|layer:{layer}|pooling:max'
            elif pooling == 'none':
                handle = RandomProjection.hook(self._extractor)
                self._extractor.identifier = self._extractor.identifier + f'|layer:{layer}|pooling:none'
            elif pooling == 'layerPCA':
                handle = LayerPCA.hook(self._extractor, n_components=n_pcs)
                self._extractor.identifier = self._extractor.identifier + f'|layer:{layer}|pooling:layerPCA|n_components:{n_pcs}'
            elif pooling == 'PCA_maxpool':
                handle = LayerPCA_Modified.hook(self._extractor, n_components=n_pcs, mod='max_pool')
                self._extractor.identifier = self._extractor.identifier + f'|layer:{layer}|pooling:{pooling}|n_components:{n_pcs}'
            elif pooling == 'PCA_zscore':
                handle = LayerPCA_Modified.hook(self._extractor, n_components=n_pcs, mod='z_score')
                self._extractor.identifier = self._extractor.identifier + f'|layer:{layer}|pooling:{pooling}|n_components:{n_pcs}'
                
            handles = []
            if self._hooks is not None:
                handles = [cls.hook(self._extractor) for cls in self._hooks]
                
            logging.info(identifier)
            logging.info(layer)
                
            self._logger.debug('Retrieving stimulus activations')
            activations = self._extractor(image_paths, layers=[layer])
            activations = activations.sel(layer=layer).values
            print(activations.shape)

            self._logger.debug('Computing UMAPs')
            progress = tqdm(total=1, desc="UMAPs")
            
            activations = flatten(activations)
            
            umap = UMAP()
            data_umap = umap.fit_transform(activations)
            
            progress.update(1)
            progress.close()
            
            #key = identifier+layer
            umaps[layer] = data_umap
            #euclidean_rdms[layer] = edistance_model

            handle.remove()

            for h in handles:
                h.remove()

        return umaps
    
    
    def as_df(self):
        df = pd.DataFrame()
        for layer, eigspec in self._rdms.items():
            layer_df = pd.DataFrame({'n': range(1, len(eigspec) + 1), 'variance': eigspec})
            layer_df = layer_df.assign(layer=layer)
            df = df.append(layer_df)
        properties = id_to_properties(self._extractor.identifier)
        df = df.assign(**properties)
        return df

    def get_image_paths(self) -> List[str]:
            raise NotImplementedError()
    
    
class EigenspectrumImageNet(EigenspectrumBase):

    def __init__(self, *args, num_classes=1000, num_per_class=10, **kwargs):
        super(EigenspectrumImageNet, self).__init__(*args, **kwargs,
                                                    stimuli_identifier='imagenet')
        assert 1 <= num_classes <= 1000 and 1 <= num_per_class <= 100
        self.num_classes = num_classes
        self.num_per_class = num_per_class
        self.image_paths = get_imagenet_val(num_classes=num_classes, num_per_class=num_per_class)

    def get_image_paths(self) -> List[str]:
        return self.image_paths
    
class EigenspectrumImageFolder(EigenspectrumBase):

    def __init__(self, data_dir, *args, num_images=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_images = num_images

        assert os.path.exists(data_dir)
        self.data_dir = data_dir

        image_paths = os.listdir(data_dir)
        if num_images is not None:
            assert len(image_paths) >= num_images
            image_paths = image_paths[:num_images]
        image_paths = [os.path.join(data_dir, file) for file in image_paths]
        self.image_paths = image_paths

    def get_image_paths(self) -> List[str]:
        return self.image_paths
    
class EigenspectrumMajajHong2015(EigenspectrumImageFolder):

    def __init__(self, *args, **kwargs):
        data_dir = os.getenv('BRAINIO_HOME', os.path.expanduser('~/.brainio'))
        data_dir = os.path.join(data_dir, 'image_dicarlo_hvm-public')
        assert os.path.exists(data_dir)
        super().__init__(data_dir, *args, **kwargs,
                         stimuli_identifier='dicarlo.hvm-public')

        self.image_paths = [p for p in self.image_paths if p[-4:] == '.png']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute and store eigenspectra of models')
    parser.add_argument('--dataset', type=str,
                        choices=['imagenet', 'imagenet21k', 'object2vec', 'majajhong2015'],
                        help='Dataset of concepts for which to compute the eigenspectrum')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory containing stimuli')
    parser.add_argument('--pooling', dest='pooling', type=str, default=None,
                        choices=['max', 'none', 'layerPCA', 'PCA_zscore', 'PCA_maxpool'],
                        help='Choose global max pooling, random projection, or layer PCA prior to computing RDMs')
    parser.add_argument('--seed', dest='seed', type=int, default=0,
                        help='Choose a random seed for analysis (torch and numpy)')
    parser.add_argument('--grayscale', action='store_true',
                        help='Compute the eigenspectrum on grayscale inputs')
    parser.add_argument('--debug', action='store_true',
                        help='Just run a single model to make sure there are no errors')
    args = parser.parse_args()

    main(dataset=args.dataset, data_dir=args.data_dir, seed=args.seed, pooling=args.pooling, grayscale=args.grayscale, debug=args.debug)