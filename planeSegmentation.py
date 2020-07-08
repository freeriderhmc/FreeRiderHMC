# planeSegmentation Module
# Using RANSAC algorithm
# author Aaron Brown
# modified by W.Jin

import numpy as np
import random
import math

def RansacPlane(points, maxIterations, distanceTol):
    # Initialize unordered set inliersResult
    inliersResult = {}
    inliersResult = set()
    
    while maxIterations:
        # Initialize unordered set inliers
        inliers = {}
        inliers = set()
        
        # Pick 3 Random Samples
        while len(inliers) < 3 :
            inliers.add(random.randint(0,len(points)-1))

        inliers_iter = inliers.__iter__()
        itr = next(inliers_iter)
        x1 = points[itr][0]
        y1 = points[itr][1]
        z1 = points[itr][2]
        itr = next(inliers_iter)
        x2 = points[itr][0]
        y2 = points[itr][1]
        z2 = points[itr][2]
        itr = next(inliers_iter)
        x3 = points[itr][0]
        y3 = points[itr][1]
        z3 = points[itr][2]

        # Get Normal Vector by Cross Product
        a = ((y2-y1)*(z3-z1) - (y3-y1)*(z2-z1))
        b = ((z2-z1)*(x3-x1) - (x2-x1)*(z3-z1))
        c = ((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
        d = -(a*x1 + b*y1 + c*z1)

        for i in range(len(points)):
            # Not consider three points already picked
            if i in inliers:
                continue
            
            x4 = points[i][0]
            y4 = points[i][1]
            z4 = points[i][2]

            # Distance between picked point and the plane
            dist = math.fabs(a*x4 + b*y4 + c*z4 +d) / math.sqrt(a*a + b*b + c*c)

            if dist <= distanceTol:
                inliers.add(i)
    
        if len(inliers) > len(inliersResult):
            inliersResult = inliers
        
        maxIterations -= 1

    return inliersResult

if __name__ == "__main__":
    print("Error.. Why PlaneSegmentation execute")
