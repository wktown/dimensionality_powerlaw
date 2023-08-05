import sys

def call2(call=True):
    print('call_2')
    
    print('f_0:')
    f0 = sys._getframe(0)
    print(f0.f_globals['__name__'])
    #test_getframe
    
    print('f_0, fback:')
    print(f0.f_back.f_globals['__name__'])
    #call_getframe
    
    print('f_1:')
    f1 = sys._getframe(1)
    print(f1.f_globals['__name__'])
    #call_getframe
    
    print('f_1, fback:')
    print(f1.f_back.f_globals['__name__'])
    #__main__
    
    print('f_2:')
    f2 = sys._getframe(2)
    print(f2.f_globals['__name__'])
    #__main__
    
    if call:
        my_func()


def my_func():
    print('my_func')
    
    print('getframe_0:')
    f0 = sys._getframe(0)
    print(f0.f_globals['__name__'])
    #test_getframe
    
    print('getframe_0, fback:')
    print(f0.f_back.f_globals['__name__'])
    #test_getframe
    
    print('getframe_1:')
    f1 = sys._getframe(1)
    print(f1.f_globals['__name__'])
    #test_getframe
    
    print('getframe_1, fback:')
    print(f1.f_back.f_globals['__name__'])
    #call_getframe
    
    print('getframe_2:')
    f2 = sys._getframe(2)
    print(f2.f_globals['__name__'])
    #call_getframe
    
    #---------------------------
    print('getframe_2, fback:')
    print(f2.f_back.f_globals['__name__'])
    #__main__
    
    print('getframe_3:')
    f3 = sys._getframe(3)
    print(f3.f_globals['__name__'])
    #__main__
    
    