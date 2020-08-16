import numpy as np

def checkCoin(lidarpoint2D,semanticMap,label):

    seg_index = np.array([np.where(semanticMap == label)[0],  np.where(semanticMap == label)[1]] ).T
    print(seg_index)
    mask_h = np.isin(lidarpoint2D[:][:,1],seg_index[:][:,0]).reshape(-1,1)
    mask_w = np.isin(lidarpoint2D[:][:,0],seg_index[:][:,1]).reshape(-1,1)
    mask = np.append(mask_h,mask_w, axis =1)
    mask = 1*mask
    mask_sum = mask.sum(axis = 1)
    # mask_sum = mask.sum(axis = 1).reshape([-1,1])
    fusion_index = np.array(np.where(mask_sum == 2)[0])
    # fusion_index = np.array(np.where(mask[:][:] == [True,  True])[0])
    print(fusion_index)
    return fusion_index
