
kernel =True
#other = False
if kernel:
    kernel_sizes = [1, 3, 5, 9, 12, 15, 22, 31] #input=32, max kernelsize=31
    # - conv2 padding=0, stride=1 #nope..
    #output = ((in_height + padding*2 - kernel_height)/stride) + 1
    #inputsize=32, max kernelsize=15 (but this depends on padding..)
    #Output height = (Input height + padding height top + padding height bottom - kernel height) / (stride height) + 1
                #  = ((96 + p+p - 15) / (2+1) = 31
                # p = 6
                # - weigh.shape[-1] = 12 (or 13)?
                