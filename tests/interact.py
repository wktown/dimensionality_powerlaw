#random projection
#def apply(layer, activations):
n_components = 1024
activations = activations.reshape(activations.shape[0], -1)
if layer not in self.layer_ws:
    w = np.random.normal(size=(activations.shape[-1], n_components)) / np.sqrt(n_components)
    self.layer_ws[layer] = w
else:
    w = self.layer_ws[layer]
    activations = activations @ w
    return activations


testacts = np.ones((64,64,56,56))
testw = np.random.normal(size = testacts.shape)
testproj = testacts @ testw

test_pca = {}
components = {}
for i in range(testacts.shape[-1]):
    for k in range(testacts.shape[-2]):
        local = testproj[:,:,i,k]
        pca = PCA(random_state=0)
        pca.fit(local)
        test_pca[i,k] = pca.explained_variance_
        components[i,k] = pca.components_
        
        
        
resnet50_pt_layers = [f'layer1.{i}.relu' for i in range(3)] + \
                     [f'layer2.{i}.relu' for i in range(4)] + \
                     [f'layer3.{i}.relu' for i in range(6)] + \
                     [f'layer4.{i}.relu' for i in range(3)]
                     
                     

    
@store_dict(dict_key='layers', identifier_ignore=['layers'])
    def _fit(self, identifier, stimuli_identifier, layers, pooling, image_transform_name):
        image_paths = self.get_image_paths()
        if self._image_transform is not None:
            image_paths = self._image_transform.transform_dataset(self._stimuli_identifier, image_paths)
    
    layer_eigenspectra = {}
    for layer in layers:
        if pooling == 'max':
            handle = GlobalMaxPool2d.hook(self._extractor)
        elif pooling == 'avg':
            handle = GlobalAvgPool2d.hook(self._extractor)
        else:
            handle = RandomProjection.hook(self._extractor)

        handles = []
        if self._hooks is not None:
            handles = [cls.hook(self._extractor) for cls in self._hooks]

        self._logger.debug('Retrieving stimulus activations')
        activations = self._extractor(image_paths, layers=[layer])
        activations = activations.sel(layer=layer).values









class EigenspectrumBase:

    def __init__(self, activations_extractor, pooling=True, stimuli_identifier=None,
                 image_transform: Optional[ImageDatasetTransformer] = None,
                 hooks: Optional[List] = None):
        self._logger = logging.getLogger(fullname(self))
        self._extractor = activations_extractor
        self._pooling = pooling
        self._hooks = hooks
        self._stimuli_identifier = stimuli_identifier
        self._image_transform = image_transform
        self._layer_eigenspectra = {}

    def fit(self, layers):
        transform_name = None if self._image_transform is None else self._image_transform.name
        self._layer_eigenspectra = self._fit(identifier=self._extractor.identifier,
                                             stimuli_identifier=self._stimuli_identifier,
                                             layers=layers,
                                             pooling=self._pooling,
                                             image_transform_name=transform_name)

    @store_dict(dict_key='layers', identifier_ignore=['layers'])
    def _fit(self, identifier, stimuli_identifier, layers, pooling, image_transform_name):
        image_paths = self.get_image_paths()
        if self._image_transform is not None:
            image_paths = self._image_transform.transform_dataset(self._stimuli_identifier, image_paths)

        # Compute activations and PCA for every layer individually to save on memory.
        # This is more inefficient because we run images through the network several times,
        # but it is a more scalable approach when using many images and large layers.
        layer_eigenspectra = {}
        for layer in layers:
            if pooling == 'max':
                handle = GlobalMaxPool2d.hook(self._extractor)
            elif pooling == 'avg':
                handle = GlobalAvgPool2d.hook(self._extractor)
            else:
                handle = RandomProjection.hook(self._extractor)

            handles = []
            if self._hooks is not None:
                handles = [cls.hook(self._extractor) for cls in self._hooks]

            self._logger.debug('Retrieving stimulus activations')
            activations = self._extractor(image_paths, layers=[layer])
            activations = activations.sel(layer=layer).values

            self._logger.debug('Computing principal components')
            progress = tqdm(total=1, desc="layer principal components")
            
            print(activations.shape())
            activations = flatten(activations)
            print(activations.shape())
            pca = PCA(random_state=0)
            pca.fit(activations)
            eigenspectrum = pca.explained_variance_
            progress.update(1)
            progress.close()

            layer_eigenspectra[layer] = eigenspectrum

            handle.remove()

            for h in handles:
                h.remove()

        return layer_eigenspectra
    
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
    



tf_to_pt_layer_map = {'encode_2': 'layer1.0.relu', 'encode_3': 'layer1.1.relu', 
                      'encode_4': 'layer2.0.relu', 'encode_5': 'layer2.1.relu', 
                      'encode_6': 'layer3.0.relu', 'encode_7': 'layer3.1.relu', 
                      'encode_8': 'layer4.0.relu', 'encode_9': 'layer4.1.relu'}

data.loc[:, 'layer'] = data['layer'].replace(tf_to_pt_layer_map)

#new tf_to_pt_layer_map for figs
for l in data['layer']:
    if l > 'encode_7':
        split = l.split('.position',1)
        tf_layer = split[0]
        if tf_layer == 'encode_8':
            pytorch_layer = 'layer4.0.relu'
        elif tf_layer == 'encode_9':
            pytorch_layer = 'layer4.1.relu'
        else:
            break
        new_layer = pytorch_layer + '.position' + split[1]
        data.loc[:,'layer'] = data['layer'].replace(l,new_layer)
        

    else:
        data.loc[:, 'layer'] = data['layer'].replace(tf_to_pt_layer_map)

