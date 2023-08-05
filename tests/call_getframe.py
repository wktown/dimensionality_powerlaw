import sys
p = '/home/wtownle1/dimensionality_powerlaw/tests'
sys.path.append(p)

from tests.test_getframe import call2

print('imported script home')
print('f_0:')
f0 = sys._getframe(0)
print(f0.f_globals['__name__']) 
#call_getframe

print('f_0, back:')
print(f0.f_back.f_globals['__name__'])
#_frozen_importlib


def call_func():
    print('call_1')
    
    print('f_0:')
    f0 = sys._getframe(0)
    print(f0.f_globals['__name__'])
    #call_getframe
    
    print('f_0, fback:')
    print(f0.f_back.f_globals['__name__'])
    #__main__
    
    print('f_1:')
    f1 = sys._getframe(1)
    print(f1.f_globals['__name__'])
    #__main__
    
    call2()