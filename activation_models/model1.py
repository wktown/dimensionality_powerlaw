

#general:
# - import model
# - wrap (*wrapper includes preprocessing (defined elsewhere))
#   - preprocessing = loading & transforms.Compose()
#   - preprocess and store images, or preprocess before running forward pass

#where does Eric pass a image loader/parser?

#Ray:
# - use model tools PytorchWrapper
# - (can use own pre-processing or pytorch)
# - need to create my own hooks ()
# - seletcs a random set of 200k images from imagenet in batches of 1000 by default
# - stimuli = list of file names & preprocess = load_images(filenames), but where is stimuli passed to the model?
#   - probably where ever the forward pass is actually called?
#   - self._extractor(stimuli, layers, stim_identifier) but PytorchWrapper def _extractor(identifier, preprocessing, get_activations=self.get_activations,
#           **kwargs)
#   - PytorchWrapper def self.get_activations(images, layer_name) -> preprocessed images, stack, model.eval()
#       - model.eval() sets model to eval mode (instead of training)
#       - Atlas then selects layers, hooks, and calls model(images) (with torch.no_grad)
#   - PytorchWrapper build_extractor(identifier, get_activations, preprocessing): 
#       - __call__(): from_paths (or from_stimulus_set)
#       - _from_paths()
#       - _get_batch_activations()
#           - batching
#           - preprocess
#           - activations = get_activations(pre-processed-batch)


#activations:
# - max_pooled: imagesxchannels
# - else: imagesx(channels*h*w)
#num images:
# - 10k (in batches of 64) for Eric's eigenspectrum
# - n_images = n_components when training PCA fit in brainscore LayerPCA
# - 