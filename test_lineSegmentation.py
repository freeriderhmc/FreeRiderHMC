# make data
import random
import math
import numpy as np
from matplotlib import pyplot as plt
import lineSegmentation as lsg

random.seed()

# 101 data
x_0 = y_0 = 0
points = np.empty([0, 2])
points = np.append(points, [[x_0,y_0]], axis = 0)

points_x = []
points_y = []
points_x.append(x_0)
points_y.append(y_0)

for i in range(1,101):
    temp_x = 0.1 * i
    temp_y = temp_x * math.tan(math.pi/6)
    temp_x = temp_x + random.gauss(0, 0.05)
    temp_y = temp_y + random.gauss(0, 0.05)

    points = np.append(points, [[temp_x, temp_y]], axis = 0)
    points_x.append(temp_x)
    points_y.append(temp_y)


for i in range(1,51):
    temp_x = 10
    temp_y = 10/math.sqrt(3) + 0.1 * i
    temp_x = temp_x + random.gauss(0, 0.05)
    temp_y = temp_y + random.gauss(0, 0.05)

    points = np.append(points, [[temp_x, temp_y]], axis = 0)
    points_x.append(temp_x)
    points_y.append(temp_y)

for i in range(0,21):
    temp_x = 3+0.1*i + random.gauss(0,0.1)
    temp_y = 5+ random.gauss(0,0.1)

    points = np.append(points, [[temp_x, temp_y]], axis = 0)
    points_x.append(temp_x)
    points_y.append(temp_y)


outliers = np.empty(shape = [0,2])

outliers_list, long_a, long_b, long_c = lsg.RansacLine(points, 20, 0.1)
outliers_list = list(outliers_list)
print(outliers_list)

outliers = points[outliers_list[:], :]

print(outliers)
print("Longer line's line equation parameter: ")
print(long_a, long_b, long_c)

temp, short_a, short_b, short_c = lsg.RansacLine(outliers, 20, 0.1)

print("Shorter line's line equation parameter: ")
print(short_a, short_b, short_c)



# clusterCloud = cloudoutliers[clusters[i][:],:]

plt.plot(points_x, points_y)
plt.show()