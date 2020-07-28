import numpy as np
import open3d as o3d
import time
from sklearn.cluster import MeanShift
from sklearn.neighbors import KDTree
import clusteringModule as clu
import planeSegmentation as seg
import copy
import loadData

# Set Visualizer class
vis = o3d.visualization.Visualizer()
vis.create_window()
pcd_processed = o3d.geometry.PointCloud()

path = '../../media/jinwj1996/Samsung_T5/v1.0-trainval01_blobs/v1.0-trainval01_blobs/samples/LIDAR_TOP/'
file_list = loadData.load_data(path)


for files in file_list:
    data = np.fromfile(path+files, dtype = np.float32)
    data = data.reshape(-1,5)
    data = data[:,0:3]

    

    pcd_load = o3d.geometry.PointCloud()
    pcd_load.points = o3d.utility.Vector3dVector(data)

    pcd_downsample = pcd_load.voxel_down_sample(voxel_size=0.1)
    #print(type(pcd_load.points[0]))
    #print(pcd_load.points[0].shape)
    #print(pcd_load.points[0])

    outerBox = o3d.geometry.PointCloud()
    outerBox.points = o3d.utility.Vector3dVector([[20,-10,-1.8],[20,-10,1.8],[20,10,-1.8],[20,10,1.8],[-20,-10,-1.8],[-20,-10,1.8],[-20,10,-1.8],[-20,10,1.8]])
    # Convert pcd to numpy array
    cloud_downsample = np.asarray(pcd_downsample.points)
    #innerBox = o3d.geometry.PointCloud()
    #innerBox.points = o3d.utility.Vector3dVector([[1.4,-0.8,-0.15],[1.4,-0.8,0],[1.4,0.8,-0.15],[1.4,0.8,0],[-1.4,-0.8,-0.15],[-1.4,-0.8,0],[-1.4,0.8,-0.15],[-1.4,0.8,0]])
    #pcd_downsample.crop([outerBox],[innerBox])
    #print(type(pcd_downsample))
    

    # Crop Pointcloud -20m < x < 20m && -20m < y < 20m && z > -1.80m
    # cut the road by height(z) threshold
    cloud_downsample = cloud_downsample[((cloud_downsample[:, 0] <= 15))]
    cloud_downsample = cloud_downsample[((cloud_downsample[:, 0] >= -15))]
    cloud_downsample = cloud_downsample[((cloud_downsample[:, 1] <= 10))]
    cloud_downsample = cloud_downsample[((cloud_downsample[:, 1] >= -10))]
    cloud_downsample = cloud_downsample[((cloud_downsample[:, 2] >= -1.30))]
    #print(len(cloud_downsample))

    
    
    pcd_processed.points = o3d.utility.Vector3dVector(cloud_downsample)

    
    start = time.time()
    labels = np.asanyarray(pcd_processed.cluster_dbscan(0.45,3))
    print(time.time() - start)
    print(np.max(labels))
    for i in range(np.max(labels)):
        cluster = pcd_processed.select_by_index(np.where(labels == i)[0])
        if len(cluster.points) <= 10:
            continue

        if i % 6 == 0:
            cluster.paint_uniform_color([1,0,0])
        elif i % 6 == 1:
            cluster.paint_uniform_color([1,1,0])
        elif i % 6 == 2:
            cluster.paint_uniform_color([0,1,0])
        elif i % 6 == 3:
            cluster.paint_uniform_color([0,1,1])
        elif i % 6 == 4:
            cluster.paint_uniform_color([0,0,1])
        else:
            cluster.paint_uniform_color([1,0,1])

        vis.add_geometry(cluster)
    #vis.add_geometry(pcd_processed)
    vis.run()

    '''
    # Non Blocking Visualization (Sequence)
    vis.add_geometry(pcd_processed)
    vis.run()
    if i==0:
        vis.add_geometry(pcd_viewer)
    else:
        vis.update_geometry(pcd_viewer)
        vis.poll_events()
        vis.update_renderer()
    '''
    
    input("Press Enter to continue...")
    vis.clear_geometries()
    
    

    

vis.destroy_window()