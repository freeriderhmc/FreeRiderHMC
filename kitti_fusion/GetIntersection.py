import numpy as np


def intersect(arr1, arr2):
    arr1_view = arr1.view([('',arr1.dtype)]*arr1.shape[1])
    arr2_view = arr2.view([('',arr2.dtype)]*arr2.shape[1])
    # print(arr1_view)
    # print(arr2_view)
    intersected = np.intersect1d(arr1_view, arr2_view)
    # print(intersected)
    return intersected.view(arr1.dtype).reshape(-1, arr1.shape[1])


def GetIntersection(lidarpoint2D,semanticMap,label):
    mask_h = np.where(semanticMap == label)[0].reshape(-1,1)
    mask_w = np.where(semanticMap == label)[1].reshape(-1,1)
    # seg_index = np.array([np.where(semanticMap == label)[0],  np.where(semanticMap == label)[1]] ).T
    # print(seg_index)
    seg_index = np.append(mask_h,mask_w,axis = 1)
    intersection = intersect(lidarpoint2D,seg_index)
    return intersection
