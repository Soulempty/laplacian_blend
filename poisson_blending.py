#coding:utf-8
import os
import cv2
import numpy as np
from utils import check_size,alpha_blend,laplacian_blend

flag = False
flag2 = False
src_img = None
des_img = None
temp = None
temp2 = None
xc = 100
yc = 100
points = []

#将一张图像上的前景裁剪下来，放到另一张背景图像上的指定位置，同时进行视觉融合。

def FastColorTransfer(img_src,img_des,sp='bgr'):
    if sp == 'lab':
        img_src = cv2.cvtColor(img_src,cv2.COLOR_BGR2LAB)
        img_des = cv2.cvtColor(img_des,cv2.COLOR_BGR2LAB)
    src_stat = cv2.meanStdDev(img_src[139:531,45:196])
    des_stat = cv2.meanStdDev(img_des[139:531,45:196])
    src_mean,src_std = np.expand_dims(src_stat[0],-1).transpose((1,2,0)),np.expand_dims(src_stat[1],-1).transpose((1,2,0))
    des_mean,des_std = np.expand_dims(des_stat[0],-1).transpose((1,2,0)),np.expand_dims(des_stat[1],-1).transpose((1,2,0))
    print(src_mean,des_mean,src_std,des_std)
    result = (img_src-src_mean)*(src_std/src_std)+des_mean
    result = np.clip(result,0,255).astype(np.uint8)
    if sp == 'lab':
        result = cv2.cvtColor(result,cv2.COLOR_LAB2BGR)
    
    return result

def blending(src,dest,src_mask,center): #laplacian_blend(src_img,des_img,src_mask,loc,4) 
    # 选择泊松融合，或者拉普拉斯金字塔
    #result = cv2.seamlessClone(src,dest,src_mask,center,cv2.cv::NORMAL_CLONE) 
    result = laplacian_blend(src,dest,src_mask,center)
    return result

def assignLoc(event, x, y, flags, param):
    global xc,yc,flag2,des_img
    if event == cv2.EVENT_LBUTTONUP:
        des_img = temp2.copy()
        flag2 = True
        xc = x
        yc = y

def onMouse(event, x, y, flags, param):
    global xmin,ymin,src_img,temp,points,flag
    if event == cv2.EVENT_LBUTTONUP:
        if flag:
            flag = False
            points = []
        if len(points)>=1:
            cv2.line(src_img,points[-1],(x,y),(0,0,255),1)
            if abs(x-points[0][0])+abs(y-points[0][1]) < 8:
                flag = True
                src_img = temp.copy()        
        points.append((x,y))  
    elif event == cv2.EVENT_RBUTTONDOWN: 
        points.pop()

def draw_mask(src_path,des_path,save_path):
    global src_img,des_img,temp,flag2,temp2,points,flag
    src_img = 255-cv2.imread(src_path)
    temp = src_img.copy()
    des_img = check_size(cv2.imread(des_path))
    temp2 = des_img.copy()
    src_mask = np.zeros(src_img.shape,src_img.dtype)
    cv2.namedWindow("selectROI",0)
    cv2.namedWindow("blending",0)
    cv2.imshow("blending",des_img)
    cv2.setMouseCallback('selectROI', onMouse)
    while True:
        cv2.imshow("selectROI",src_img)
        cv2.imshow("blending",des_img)
        if flag:
            poly = np.array(points)
            cv2.fillPoly(src_mask,[poly],(255,255,255))
            
            cv2.setMouseCallback('blending', assignLoc)
            if flag2:
                des_img = blending(src_img,des_img,src_mask,(xc,yc))
                flag2 = False
        k = cv2.waitKey(1)&0xFF
        if k==ord('q'):
            cv2.destroyAllWindows()
            points = []
            flag = False
            break

        if k==ord('s'):
            cv2.imwrite(os.path.join(save_path,os.path.basename(des_path)),des_img*255)
            cv2.destroyAllWindows()
            points = []
            flag = False
            break
        if k==ord('e'):
            cv2.destroyAllWindows()
            return 1
    return 0

def use_mask(src_path,des_path,mask_path,save_path):
    global src_img,des_img,temp,flag2,temp2
    src_img = 255-cv2.imread(src_path)
    src_mask = cv2.imread(mask_path)
    des_img = cv2.imread(des_path)
    src_img = FastColorTransfer(src_img,des_img)
    temp2 = des_img.copy()
    cv2.namedWindow("blending",0)
    cv2.imshow("blending",des_img)
    while True:
        cv2.imshow("blending",des_img)
        cv2.setMouseCallback('blending', assignLoc)
        if flag2:
            des_img = blending(src_img,des_img,src_mask,(xc,yc))
            flag2 = False
        k = cv2.waitKey(1)&0xFF
        if k==ord('q'):
            cv2.destroyAllWindows()
            break

        if k==ord('s'):
            cv2.imwrite(os.path.join(save_path,os.path.basename(des_path)),des_img*255)
            cv2.destroyAllWindows()
            break
        if k==ord('e'):
            cv2.destroyAllWindows()
            return 1
    return 0

if __name__ == "__main__":
    ok_images = list(os.scandir("ok"))
    src_images = list(os.scandir("test/image"))
    save_path = "blend"
    for i in range(len(src_images)):  
        src_path = src_images[i].path
        des_path =  ok_images[i].path
        mask_path = src_path.replace('image','mask')
        if os.path.exists(mask_path):
            k = draw_mask(src_path,des_path,save_path)  #手画mask
            #k = use_mask(src_path,des_path,mask_path,save_path)   #现成mask
            if k:
                break
