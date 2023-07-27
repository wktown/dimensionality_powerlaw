import logging
from model_tools.utils import fullname
import inspect

class LogTest2:
    def __init__(self):
        self._logger = logging.getLogger(fullname(self))
        print(self.__module__) #__main__ or class_logger
        #print(self.__name__) #function name
        print(self.__class__.__name__) #LogTest2
        module = self.__module__
        name = self.__name__ if inspect.isfunction(self) else self.__class__.__name__
        print('fullname:')
        print(module + "." + name) #main/script.class/func
        
    def test(self):
        print(__name__)
        
    def __str__(self):
        print(__name__)
        return self.__class__.__name__

