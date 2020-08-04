    x, y, u, v = point[1][0], point[1][1], cos(yaw), sin(yaw)
    [x,y] = (np.array([ [0,-1], [1,0]]) @ np.asarray([x,y]).T).T 
    [u,v] = (np.array([ [0,-1], [1,0]]) @ np.asarray([u,v]).T).T 
    plt.quiver(x, y, u, v, scale= 2, scale_units = 'inches', color = 'red')
