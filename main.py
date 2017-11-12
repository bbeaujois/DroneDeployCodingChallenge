##
#   main.py
#   Author: Benjamin Beaujois
#   Date: 11/12/2017
#   Description: This file contains the source code for the option 1 of the coding challenge.
#   The code generate a graphic to visualize where the camera was when each image was taken and how it was posed, relative to a pattern.

import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

##
#   fn: getTransformMatrix(rvec, tvec)
#   params:
#       - rvec: rotation vector
#       - tvec: translation vector
#   return: matrix4x4
#   description: this function transforms a rotation and translation vector to a matrix4x4
def getTransformMatrix(rvec, tvec):
    mat, jac = cv2.Rodrigues(rvec)
    return np.matrix([[mat[0][0],mat[0][1],mat[0][2],tvec[0]],
                     [mat[1][0],mat[1][1],mat[1][2],tvec[1]],
                     [mat[2][0],mat[2][1],mat[2][2],tvec[2]],
                     [0        ,0        ,0        ,1      ]])

##
#   fn: translate(mat, tvec)
#   params:
#       - mat: matrix4x4
#       - tvec: translation vector
#   return: matrix4x4
#   description: This function translates a transformation matrix 
def translate(mat, tvec):
    x,y,z = tvec
    mat1 = np.matrix([[1,0,0,x],
                      [0,1,0,y],
                      [0,0,1,z],
                      [0,0,0,1]])
    return mat*mat1

##
#   fn: getVector(mat, direction)
#   params:
#       - mat: matrix4x4
#       - direction: 3D vector representing the direction in local coordinates system
#   return: tuple with the position and direction of the vector in global coordinates system
#   description: This function get a vector from a transformation matrix and a direction to be able to draw the vector on a 3D graph.
def getVector(mat, direction):
    #position of the vector
    a = mat.getA()
    x = a[0][3]
    y = a[1][3]
    z = a[2][3]
    mat = translate(mat,direction)
    #direction of the vector
    a = mat.getA()
    u = x-a[0][3]
    v = y-a[1][3]
    w = z-a[2][3]
    return (x,y,z,u,v,w)

##
#   fn: visualizePose(ax,mat,name)
#   params:
#       - ax: Axes3D object. It represents the 3D graph
#       - mat: matrix4x4
#       - name: string which contains index of an image
#   description: This function draws 3 arrows an a 3D graph which represent the pose of a camera.
def visualizePose(ax,mat,name):
    x,y,z,u,v,w = getVector(mat, (0,0,-1))
    ax.quiver(x,y,z,u,v,w,length=10.0, normalize=True, label='z')
    x,y,z,u,v,w = getVector(mat, (0,1,0))
    ax.quiver(x,y,z,u,v,w,length=3.0, color='r', normalize=True, label='y')
    x,y,z,u,v,w = getVector(mat, (-1,0,0))
    ax.quiver(x,y,z,u,v,w,length=3.0, color='g', normalize=True, label='x')
    ax.text(x,y,z,name)
    
##
#   fn: drawPattern(ax)
#   params:
#       - ax: Axes3D object. It represents the 3D graph
#   description: This function draws a square on a 3D graph which represents the pattern on the floor.
def drawPattern(ax):
    X = np.array([-4.4, 4.4])
    Y = np.array([-4.4, 4.4])
    X, Y = np.meshgrid(X, Y)
    Z = np.array([[0.0,0.0],[0.0,0.0]])
    ax.plot_surface(X, Y, Z)

##
#   fn: printPose(mat, name)
#   params:
#       - mat: matrix4x4
#       - name: string which contains index of an image
#   description: This function prints on the console the position and Euler angles of a camera from a matrix.
def printPose(mat, name):
    mat = mat.getA()
    x,y,z = (round(mat[0][3],2),round(mat[1][3],2),round(mat[2][3],2))
    mat = np.matrix([[mat[0][0],mat[0][1],mat[0][2]],
                     [mat[1][0],mat[1][1],mat[1][2]],
                     [mat[2][0],mat[2][1],mat[2][2]]])
    #get Euler angles
    retval, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(mat)
    pitch, yaw, roll = retval
    txt = name+" [x: "+str(x)+" cm, y: "+str(y)+" cm, z: "+str(z)+" cm, roll: "+str(round(roll,2))+"°, pitch: "+str(round(pitch,2))+"°, yaw: "+str(round(yaw,2))+"°]"
    print(txt)
    
##
#   fn: findPattern(pattern, images)
#   params:
#       - pattern: image representing the pattern
#       - images: array of images representing the views of the pattern on a flat surface
#   return: array of 2D points representing the position of the pattern corners on the picture.
#   description: This function finds the position of the pattern corners on the picture.
def findPattern(pattern, images):
    orb= cv2.ORB_create() #FAST keypoint detector and BRIEF descriptor
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) #Match keypoints of two images.
    
    h,w = pattern.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)# the 4 corners of the pattern
    ret = []
    # find the keypoints and descriptors with ORB
    kpp, desp = orb.detectAndCompute(pattern,None)
    for img in images:
        kp, des = orb.detectAndCompute(img,None)  
        
        # Match the keypoints and descriptors with a brute-force matcher
        matches = bf.match(desp,des)
        src_pts = np.float32([ kpp[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

        #Find the perspective transformation
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        ret.append(cv2.perspectiveTransform(pts,M))
    return ret

##
#   fn: posePattern(imgPts)
#   params:
#       - imgPts: array of arrays of 2D points
#   return: tuple whith an array of translation and rotation vectors 
#   description: This function compares the 2D points with a 3D object (the pattern with the real measures) to find the position and rotation of each object.
def posePattern(imgPts):
    objPts = []
    shape = imgs[0].shape[::-1]
    for p in imgPts:
        objPts.append([[-4.4,4.4,0],[-4.4,-4.4,0],[4.4,-4.4,0],[4.4,4.4,0]])
    objPts = np.float32(objPts)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPts, imgPts, shape,None,None)
    return(rvecs, tvecs)

if __name__ == '__main__':
    #Extract all images
    pattern = cv2.imread('images/pattern.png',0)
    imgs = []
    imgs.append(cv2.imread('images/IMG_6719.JPG',0)) #0
    imgs.append(cv2.imread('images/IMG_6720.JPG',0)) #1
    imgs.append(cv2.imread('images/IMG_6721.JPG',0)) #2
    imgs.append(cv2.imread('images/IMG_6722.JPG',0)) #3
    imgs.append(cv2.imread('images/IMG_6723.JPG',0)) #4
    imgs.append(cv2.imread('images/IMG_6724.JPG',0)) #5
    imgs.append(cv2.imread('images/IMG_6725.JPG',0)) #6
    imgs.append(cv2.imread('images/IMG_6726.JPG',0)) #7
    imgs.append(cv2.imread('images/IMG_6727.JPG',0)) #8

    #Find the pose of each pattern on the scene.
    pts = findPattern(pattern,imgs)
    rvecs, tvecs = posePattern(pts)

    #Create the 3D graph
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    drawPattern(ax)

    for i in range(0,9):
        mat = getTransformMatrix(rvecs[i], tvecs[i]).getI() #The matrix have to be inverted to have the pose of the camera
        visualizePose(ax,mat, str(i))
        printPose(mat, str(i))
        #Construct the legend
        if(i == 0):
            plt.legend(loc=2, borderaxespad=0.)
    plt.show()

