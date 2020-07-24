
# lineSegmentation Module
# Using RANSAC algorithm
# author Aaron Brown
# modified by W.Jin

####################################################################################
############# Return Line's equation & Picked Point & Outliers in R^2 ##############
####################################################################################

import numpy as np
import random
import math

def RansacLine(points, maxIterations, distanceTol):
    if(len(points)<2):
        return None
    # Initialize unordered set inliersResult
    inliersResult = set({})
    outliersResult = set({})
    #Line_a = Line_b = Line_c = 0
    while maxIterations:
        # Initialize unordered set inliers
        inliers = set({})
        outliers = set({})
        
        # Pick 2 Random Samples
        while len(inliers) < 2 :
            inliers.add(random.randint(0,len(points)-1))
        
        inliers_iter = inliers.__iter__()

        itr = next(inliers_iter)
        x1 = points[itr][0]
        y1 = points[itr][1]
        itr = next(inliers_iter)
        x2 = points[itr][0]
        y2 = points[itr][1]

        # Get Line Equation : ax+by+c = 0
        a = y1-y2
        b = -x1+x2
        c = (x1-x2)*y1-(y1-y2)*x1

        for i in range(len(points)):
            # Not consider three points already picked
            if i in inliers:
                continue
            
            x3 = points[i][0]
            y3 = points[i][1]

            # Distance between picked point and the plane
            dist = math.fabs(a*x3 + b*y3 + c) / math.sqrt(a*a + b*b)

            if dist <= distanceTol:
                inliers.add(i)
            else:
                outliers.add(i)

        
        if len(inliers) > len(inliersResult):
            inliersResult = inliers
            outliersResult = outliers
            #Line_a = a
            #Line_b = b
            #Line_c = c
        
        maxIterations -= 1

    if(len(outliersResult)==0):
        return list(inliersResult), []

    return list(inliersResult), list(outliersResult)

if __name__ == "__main__":
    print("Error.. Why PlaneSegmentation execute")
