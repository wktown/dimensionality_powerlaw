






@store_dict(dict_key='layers', identifier_ignore=['layers'])
    def _fit(self, identifier, stimuli_identifier, layers, pooling, image_transform_name):
        image_paths = self.get_image_paths()
        if self._image_transform is not None:
            image_paths = self._image_transform.transform_dataset(self._stimuli_identifier, image_paths)

        # Compute activations and PCA for every layer individually to save on memory.
        # This is more inefficient because we run images through the network several times,
        # but it is a more scalable approach when using many images and large layers.
        
        layer_eigenspectra = {}
        #spatial_eigenspectra = {}
        for layer in layers:
            if pooling == 'max':
                handle = GlobalMaxPool2d.hook(self._extractor)
            elif pooling == 'avg':
                handle = GlobalAvgPool2d.hook(self._extractor)
            elif pooling == 'projections':
                handle = RandomProjection.hook(self._extractor)
            elif pooling == 'random_spatial':
                handle = RandomSpatial.hook(self._extractor)
            
            elif pooling == 'spatial_pca':
                split_position = layer.split('.position',1)
                if len(split_position) < 1:
                    handle = RandomProjection.hook(self._extractor)
                    print('hook = random (from spatial)')
                else:
                    handle = SpatialPCA.hook(self._extractor)
                    print('hook = spatial')
                    
                
            handles = []
            if self._hooks is not None:
                handles = [cls.hook(self._extractor) for cls in self._hooks]
                
            self._logger.debug('Retrieving stimulus activations')
            activations = self._extractor(image_paths, layers=[layer])
            activations = activations.sel(layer=layer).values

            logging.info(identifier)
            logging.info(layer)
            print(layer)
            print(activations.shape)
            
            self._logger.debug('Computing principal components')
            progress = tqdm(total=1, desc="layer principal components")
            
            activations = flatten(activations)
            pca = PCA(random_state=0)
            pca.fit(activations)
            eigenspectrum = pca.explained_variance_
            progress.update(1)
            progress.close()

            layer_eigenspectra[layer] = eigenspectrum