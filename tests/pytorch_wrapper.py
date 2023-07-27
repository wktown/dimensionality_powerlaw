#__brainscore__
class PytorchWrapper:
    def __init__(self, model, preprocessing, identifier=None, forward_kwargs=None, *args, **kwargs):
        import torch
        #logger = logging.getLogger(fullname(self))
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #logger.debug(f"Using device {self._device}")
        self._model = model
        self._model = self._model.to(self._device)
        
        identifier = identifier or model.__class__.__name__
        self._extractor = self._build_extractor(
            identifier=identifier, preprocessing=preprocessing, get_activations=self.get_activations, *args, **kwargs)
        self._extractor.insert_attrs(self)
        self._forward_kwargs = forward_kwargs or {}
        
#__atlas__  
class PytorchWrapper:
    def __init__(self, model,forward_kwargs=None): 
        
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model
        self._model = self._model.to(self._device)
        self._forward_kwargs = forward_kwargs or {}
        
#__brainscore
def _build_extractor(self, identifier, preprocessing, get_activations, *args, **kwargs):
        return ActivationsExtractorHelper(
            identifier=identifier, get_activations=get_activations, preprocessing=preprocessing,
            *args, **kwargs)
        
        