import os
import cv2
import numpy as np

def getbbox(mask,border=1):
    img = 1-mask[:,:,0]
    h,w = img.shape[:2]
    hsum = img.sum(0)
    wsum = img.sum(1)
    xmin,ymin,xmax,ymax = 0,0,w,h
    for i in range(w):
        if hsum[i]<h and xmin == 0:
            xmin = i-border
        if hsum[w-i-1]<h and xmax == w:
            xmax = w-i-1+border
        if xmin!=0 and xmax!=h:
            break
    for i in range(h):
        if wsum[i]<w and ymin == 0:
            ymin = i-border
        if wsum[h-i-1]<w and ymax == h:
            ymax = h-i-1+border
        if ymin!=0 and ymax!=h:
            break
    return xmin,ymin,xmax,ymax
    
def check_size(image,layers=5):
    h,w = image.shape[:2]
    base_size = 2**(1+layers)
    nh = (h//base_size+1)*base_size
    nw = (w//base_size+1)*base_size
    img = cv2.resize(image,(nw,nh),cv2.INTER_NEAREST)
    return img
def check_loc(loc,layers=5):
    x,y = loc
    base_size = 2**(1+layers)
    nx = (x//base_size+1)*base_size
    ny = (y//base_size+1)*base_size
    return nx,ny

def gaussian_pyramid(img,layers=5):
    pyrs = [img]
    for _ in range(layers-1):
        x = pyrs[-1]
        pyrs.append(cv2.pyrDown(x))
    return pyrs

def laplacian_pyramid(gau_pyramid):
    n = len(gau_pyramid)
    pyrs = []
    pyrs.append(gau_pyramid[-1])
    for i in range(n-1,0,-1):
        x = cv2.pyrUp(gau_pyramid[i])
        dif = gau_pyramid[i-1]-x
        pyrs.append(dif)
    return pyrs

def customPyrUP(image):
    h,w = image.shape[:2]
    result = np.zeros(shape=(h*2,w*2,3),dtype=image.dtype)
    result[::2,::2] = image
    result[0,1:-1:2] = (image[0,:-1]+image[0,1:])/2 # first row
    result[h-1,1:-1:2] = (image[h-1,:-1]+image[h-1,1:])/2 # last row
    result[1:-1:2,0] = (image[:-1,0]+image[1:,0])/2 # first col
    x = image[:,:-1]+image[:,1:]
    result[1:-1:2,1:-1:2] = (x[:-1,:]+x[1:,:])/4 # even row even col
    x = (image[:-1,:]+image[1:,:])/2 # even row odd col
    result[1:-1:2,:-1:2] = x
    x = (image[:,:-1]+image[:,1:])/2 # odd row even col
    result[:-1:2,1:-1:2] = x
    result[:,-1] = result[:,-2]
    result[-1,:] = result[-2,:]
    result = result.astype(np.uint8)
    return result 

def alpha_blend(src_img,des_img,src_mask,center):
    cx,cy = center
    A = src_img*src_mask
    xmin,ymin,xmax,ymax = getbbox(src_mask)
    
    sw = xmax-xmin
    sh = ymax-ymin
    dshape = des_img.shape
    x = min(max(0,cx - sw//2),dshape[1]-sw)
    y = min(max(cy - sh//2,0),dshape[0]-sh)
    mask = np.zeros(shape=dshape)
    try:
        mask[y:y+sh,x:x+sw] = src_mask[ymin:ymax,xmin:xmax]
    except Exception as e:
        return des_img

    blending = des_img*(1-mask)
    blending[y:y+sh,x:x+sw] += A[ymin:ymax,xmin:xmax]
    return blending

def laplacian_blend(src_img,des_img,src_mask,loc,layers=5):
    h,w = des_img.shape[:2]
    src_img = check_size(src_img,layers=5)/255.0
    des_img = check_size(des_img,layers=5)/255.0
    src_mask = check_size(src_mask,layers=5)/255.0
    loc = check_loc(loc,layers)

    src_gau = gaussian_pyramid(src_img,layers)
    des_gau = gaussian_pyramid(des_img,layers)
    mask_gau = gaussian_pyramid(src_mask,layers)
    src_lap = laplacian_pyramid(src_gau)
    des_lap = laplacian_pyramid(des_gau)
    mask_gau.reverse() 
    locs = [(loc[0]//2**i,loc[1]//2**i) for i in range(layers)]
    locs.reverse()

    blending = alpha_blend(src_lap[0],des_lap[0],mask_gau[0],locs[0])
    for i in range(1,layers):
        blending = cv2.pyrUp(blending)+alpha_blend(src_lap[i],des_lap[i],mask_gau[i],locs[i])
    blending = cv2.resize(blending,(w,h),cv2.INTER_NEAREST)
    return blending

