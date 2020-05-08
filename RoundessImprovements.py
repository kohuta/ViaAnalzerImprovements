import cv2 
import numpy as np 
import os
import math
import matplotlib.pyplot as plt
import matplotlib
import scipy
import circle_fit
import csv
from os import path
import errno
from scipy.interpolate import splprep, splev



from skimage.draw import ellipse
from skimage.measure import label, regionprops, label#, regionprops_table
from skimage.transform import rotate

import fitellipse_RANSAC_edit




def processImage(img,threshold,noiseReduction,method="d+s",debug=False):
    gain=10
    gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    gray = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    # Blur using 3 * 3 kernel. 
    gray_blurred =  gray#cv2.blur(gray, (5,5)) 

    gray_blurred = cv2.bilateralFilter(gray, 3, 30, 175)
    gray_gaus =  cv2.blur(gray, (3,3)) 


    #

    #iMapEdge = cv2.Canny(gray_blurred, 15, 80)
    
    #
    #Sobel Derivative
    #
    grad_x = np.abs(cv2.Sobel(gray_blurred,cv2.CV_32F,1,0,ksize=1))
    grad_y = np.abs(cv2.Sobel(gray_blurred,cv2.CV_32F,0,1,ksize=1))
    SobelEdge = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)

    #
    #Canny 
    #
    CannyEdge=cv2.Canny(gray,60,200)

    #
    #Difference Of Gaussian
    #
    gray_gaus =  cv2.GaussianBlur(gray, (9,9),0.5) 
    DoG=(gray-gray_gaus)


    #
    #LaPlacian
    #
    laplace=cv2.Laplacian(gray_blurred,cv2.CV_64F)
    iMapEdge=SobelEdge

    if (method=="d+s"):
        iMapEdge=DoG+SobelEdge
        iMapEdge=np.divide(iMapEdge-np.min(iMapEdge),np.max(iMapEdge)-np.min(iMapEdge))*255
        iMapEdge=iMapEdge*gain
        iMapEdge[np.where(iMapEdge>255)]=255
    elif (method=="sobel"):
        iMapEdge=SobelEdge
        iMapEdge=np.divide(iMapEdge-np.min(iMapEdge),np.max(iMapEdge)-np.min(iMapEdge))*255
        iMapEdge=iMapEdge*gain
        iMapEdge[np.where(iMapEdge>255)]=255
    elif (method=="dog"):
        iMapEdge=DoG
        iMapEdge=np.divide(iMapEdge-np.min(iMapEdge),np.max(iMapEdge)-np.min(iMapEdge))*255
        iMapEdge=iMapEdge*gain
        iMapEdge[np.where(iMapEdge>255)]=255
    elif (method=="canny"):
        iMapEdge=CannyEdge
        iMapEdge=np.divide(iMapEdge-np.min(iMapEdge),np.max(iMapEdge)-np.min(iMapEdge))*255
        iMapEdge=iMapEdge*gain
        iMapEdge[np.where(iMapEdge>255)]=255
    elif (method=="intensity"):
        iMapEdge=gray


    if(debug):
        cv2.namedWindow('Contours',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Contours', 800,600)
        cv2.imshow('Contours', iMapEdge) 
        cv2.waitKey(0) 
        cv2.destroyAllWindows()





    thresh,binaryImage  = cv2.threshold(iMapEdge,threshold,255,cv2.THRESH_BINARY_INV )

    if (method=="adaptive"):
        binaryImage = cv2.adaptiveThreshold(gray_blurred.astype(np.uint8),255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,79,2)

    kernel = np.ones((noiseReduction,noiseReduction),np.uint8)

    binaryImage=cv2.medianBlur(binaryImage.astype(np.uint8),noiseReduction)
    binaryImage=cv2.morphologyEx(binaryImage.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    #print(binaryImage.dtype)

    if(debug):
        cv2.namedWindow('Contours',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Contours', 800,600)
        cv2.imshow('Contours', binaryImage) 
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
        



    return binaryImage

def findVias(binaryImage):
    contours, hierarchy = cv2.findContours(binaryImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    holes = [contours[i] for i in range(len(contours)) if (hierarchy[0,i,2] == -1)]
    vias=[]
    
    for cnt in contours:
        #properties to use
        #Equivilant Diameter
        area = cv2.contourArea(cnt)
        equi_diameter = np.sqrt(4*area/np.pi)
        if(equi_diameter<(minDiameter/pixelSize)):
            continue
        if(equi_diameter>(maxDiameter/pixelSize)):
            continue
        #aspect Ratio
        x,y,w,h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h
        if(aspect_ratio<.65):
            continue
        if(x<2):
            continue
        if(y<2):
            continue
        if(x+w>binaryImage.shape[1]-2):
            continue
        if(y+h>binaryImage.shape[0]-2):
            continue
        #extent
        area = cv2.contourArea(cnt)
        x,y,w,h = cv2.boundingRect(cnt)
        rect_area = w*h
        extent = float(area)/rect_area
        if(extent<.6):
            continue
        #Solidity
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area)/hull_area
        #Orientation
        #(x,y),(MA,ma),angle = cv2.fitEllipse(cnt)
        vias.append(cnt)
   
    return vias

def circleFit(outputImage,vias,method):
    circles=[]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for cnt in vias:
        #xc,yc,r,car=circle_fit.hyper_fit((cnt[:,0,:]),1000)
        #cv2.circle(outputImage, (int(xc),int(yc)), int(r), (0, 0, 255), 2)
        cv2.drawContours(outputImage, [cnt], 0, (0, 255, 0), 1) 
        #circles.append((xc,yc,r,car))
        #(x,y),(MA,ma),angle = cv2.fitEllipse(cnt)

        if(method=="ransac"):
            print("ransac fit")
            best_ellipse,inlier_pnts=fitellipse_RANSAC_edit.FitVia_RANSAC2(cnt.reshape((cnt.shape[0],2)),input_pts=5,max_itts=3000, max_refines=1,max_perc_inliers=99.0)
        else: best_ellipse=cv2.fitEllipse(cnt)
        #print(best_ellipse)
        cv2.ellipse(outputImage, (int(best_ellipse[0][0]),int(best_ellipse[0][1])), (int(best_ellipse[1][0]/2),int(best_ellipse[1][1]/2)),best_ellipse[2],0,359, (0, 0, 255), 1)
        ((x,y),(ma,MA),angle)=best_ellipse
        mD, outx,outy, outang=getMinDiameter(cnt)
        mr=mD/2
        startpt=rotateLine((x,y), (x-mr,y), -math.radians(outang))#(xc-mr,yc)#(int(xc-mr/np.cos(outang)),int(yc-mr/np.cos(outang)))
        endpt=rotateLine((x,y), (x+mr,y), -math.radians(outang))#(xc+mr,yc)#(int(xc+mr/np.cos(outang)),int(yc+mr/np.cos(outang)))
        MAstartpt=rotateLine((x,y), (x-MA/2,y), math.radians(angle)-math.pi/2)
        MAendpt=rotateLine((x,y), (x+MA/2,y), math.radians(angle)-math.pi/2)
        mastartpt=rotateLine((x,y), (x-ma/2,y), math.radians(angle))
        maendpt=rotateLine((x,y), (x+ma/2,y),math.radians(angle))

        edgeDeviation=calcED(best_ellipse,cnt)
        inlier_percent=len(inlier_pnts)/len(cnt)#.shape(0)
        print(inlier_percent)
        r=np.average(np.array([ma,MA]))
        cv2.putText(outputImage, str(100*np.average(edgeDeviation))[:5], (int(endpt[0]),int(endpt[1])), font, 1, (226, 173, 93), 2, cv2.LINE_AA)


        #print(startpt,endpt)
        cv2.line(outputImage,(int(startpt[0]),int(startpt[1])),(int(endpt[0]),int(endpt[1])), (255, 0, 0), 1)
        cv2.line(outputImage,(int(MAstartpt[0]),int(MAstartpt[1])),(int(MAendpt[0]),int(MAendpt[1])), (0, 255, 255), 1)
        cv2.line(outputImage,(int(mastartpt[0]),int(mastartpt[1])),(int(maendpt[0]),int(maendpt[1])), (0, 0, 255), 1)
        


        resultText.append(filename+", "+str(x*pixelSize)+", "+str(y*pixelSize)+", "+str(r*pixelSize)+", "+str(ma*pixelSize)+", "+str(MA*pixelSize)+", "+str(angle)+ ", "+str(mD*pixelSize)+ ", "+str(ma/MA)+ ", "+str(mD/MA)+ ", "+str(np.average(edgeDeviation))+", "+str(np.median(edgeDeviation))+", "+str(np.amax(edgeDeviation))+", "+str(np.sum(edgeDeviation))+", "+str(np.std(edgeDeviation))+ ", "+str(inlier_percent))
    cv2.putText(outputImage, 'Major', (10,50), font, 2, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(outputImage, 'Minor', (10,100), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(outputImage, 'Min', (10,150), font, 2, (255, 0,0), 2, cv2.LINE_AA)


    return outputImage, circles


def calcED(best_ellipse,cnt):
    ((x,y),(ma,MA),angle)=best_ellipse

    angle=math.radians(angle)

    ellipse_pnts=[]
    cnt_pnts=[]

    ed=[]
    ed.clear()
    # calculate moments of binary image

    #M = cv2.moments(cnt)

# calculate x,y coordinate of center

    #cx = int(M["m10"] / M["m00"])

    #cy = int(M["m01"] / M["m00"])
    #centered_cnt=cnt.reshape((cnt.shape[0],2))
    #centered_cnt[:,0]=centered_cnt[:,0]-cX
    #centered_cnt[:,1]=centered_cnt[:,1]-cY
    centered_cnt = cnt - [x, y]

    centered_cnt=rotate_centered_contour(centered_cnt,(90-math.degrees(angle))).reshape((cnt.shape[0],2))
    #centered_cnt=centered_cnt.reshape((centered_cnt.shape[0],2))
    #RAH_cnt=cart2pol(centered_cnt[:,0],centered_cnt[:,1])
    #print("RAH ",RAH_cnt)j

    #testimg=np.zeros((1400,1800,3), np.uint8)
    # cv2.ellipse(testimg, (int(best_ellipse[0][0]),int(best_ellipse[0][1])), (int(best_ellipse[1][0]/2),int(best_ellipse[1][1]/2)),best_ellipse[2],0,359, (0, 0, 255), 1)
    # #cv2.drawContours(testimg, [cnt], 0, (0, 255, 0), 1) 
    #cv2.ellipse(testimg, (int(500),int(500)), (int(best_ellipse[1][0]/2),int(best_ellipse[1][1]/2)),best_ellipse[2],0,359, (0, 0, 255), 1)
    #cv2.drawContours(testimg, [centered_cnt], 0, (0, 255, 0), 1) 
    #((x,y),(ma,MA),angle)=cv2.fitEllipse(centered_cnt)

    a=MA/2
    b=ma/2
    
    for i in range(centered_cnt.shape[0]):
        theta_temp = np.arctan2(centered_cnt[i,1],centered_cnt[i,0])
        r_temp = np.hypot(centered_cnt[i,0],centered_cnt[i,1])

        
        #print("orig", theta_temp)

        #xR=np.multiply(np.divide(MA,2),math.cos(theta_temp))
        #yR=np.multiply(np.divide(ma,2),math.sin(theta_temp))
        if(np.abs(theta_temp)<=math.pi/2):
            xR=(a*b)/math.sqrt(b**2+np.tan(theta_temp)**2*a**2)
            yR=(a*b*np.tan(theta_temp))/math.sqrt(b**2+np.tan(theta_temp)**2*a**2)
        else:
            xR=-(a*b)/math.sqrt(b**2+np.tan(theta_temp)**2*a**2)
            yR=-(a*b*np.tan(theta_temp))/math.sqrt(b**2+np.tan(theta_temp)**2*a**2)

        r=np.hypot(xR,yR)
        theta = np.arctan2(yR, xR)
        ellipse_pnts.append((theta, r))
        #print("th", theta)
        #print(type(r),type(r_temp))
        ed.append(np.abs(np.abs(r_temp)-np.abs(r))/np.abs(r))
        #testimg[int(centered_cnt[i,1]+500),int(centered_cnt[i,0])+500]=(255,255,255)
        #testimg[int(yR+500),int(xR+500)]=(255,0,255)


        # cv2.namedWindow('Contours',cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Contours', 900,700)
        # cv2.imshow('Contours', testimg) 
        # cv2.waitKey(0) 




    #print(ed)
    #ed=np.abs(np.abs(cnt_pnts[1][:])-np.abs(first_ellipse_elements ))/np.abs(first_ellipse_elements)
    return ed#np.abs(np.abs(first_cnt_elements)-np.abs(first_ellipse_elements ))/np.abs(first_ellipse_elements)




def NewcalcED(best_ellipse,cnt):
    ((x,y),(ma,MA),angle)=best_ellipse



    ed=[]
    ed.clear()
 
    centered_cnt=cnt.reshape((cnt.shape[0],2))
    centered_cnt[:,0]=centered_cnt[:,0]-x
    centered_cnt[:,1]=centered_cnt[:,1]-y


    for i in range(centered_cnt.shape[0]):


        err=fitellipse_RANSAC_edit.EllipseEDError(cnt.reshape((cnt.shape[0],2)), best_ellipse)
        ed.append(1)

    return err

def getMinDiameter(contour):
    diam=99999
    outx=-1
    outy=-1
    outang=-1
    angle=0
    
    while angle < 359:
        cnt=rotate_contour(contour, angle)
        x,y,w,h = cv2.boundingRect(cnt)
        #w=np.amax(cnt[np.where])
        #image=np.zeros((4000,4000))
        #cv2.drawContours(image,[cnt],-1,(255,0,0),5)
        #cv2.namedWindow('Contours',cv2.WINDOW_NORMAL)
        #cv2.resizeWindow('Contours', 800,600)
        #cv2.imshow('Contours', image) 
        #cv2.waitKey(0) 
        if(w<diam):
            diam=w
            outx=x
            outy=y 
            outang=angle  
        angle+=0.25



    return diam, outx, outy, outang

def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def rotate_contour(cnt, angle):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    
    coordinates = cnt_norm[:, 0, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)
    
    thetas = np.rad2deg(thetas)
    thetas = (thetas + angle) % 360
    thetas = np.deg2rad(thetas)
    
    xs, ys = pol2cart(thetas, rhos)
    
    cnt_norm[:, 0, 0] = xs
    cnt_norm[:, 0, 1] = ys

    cnt_rotated = cnt_norm + [cx, cy]
    cnt_rotated = cnt_rotated.astype(np.int32)

    return cnt_rotated

def rotate_centered_contour(cnt, angle):
   # M = cv2.moments(cnt)
    #cx = int(M['m10']/M['m00'])
    #cy = int(M['m01']/M['m00'])

    cnt_norm = cnt# - [cx, cy]
    
    coordinates = cnt_norm[:, 0, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)
    
    thetas = np.rad2deg(thetas)
    thetas = (thetas + angle) % 360
    thetas = np.deg2rad(thetas)
    
    xs, ys = pol2cart(thetas, rhos)
    
    cnt_norm[:, 0, 0] = xs
    cnt_norm[:, 0, 1] = ys

    cnt_rotated = cnt_norm #+ [cx, cy]
    cnt_rotated = cnt_rotated.astype(np.int32)

    return cnt_rotated

def smoothContours(contours):
    smoothened = []
    for contour in contours:
        x,y = contour.T
        # Convert from numpy arrays to normal arrays
        x = x.tolist()[0]
        y = y.tolist()[0]
        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
        tck, u = splprep([x,y], u=None, s=1.0, per=1)
        # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
        u_new = np.linspace(u.min(), u.max(), 25)
        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
        x_new, y_new = splev(u_new, tck, der=0)
        # Convert it back to numpy format for opencv to be able to display it
        res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new,y_new)]
        smoothened.append(np.asarray(res_array, dtype=np.int32))
    
    return smoothened


def rotateLine(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy
# The following section will run this file, save the three difference matrices
# as images, and complete the video frame extraction into the output folder.
# You will need to modify the alpha value in order to achieve good results.
if __name__ == "__main__":

    directory = r'C:\Users\kohuta\Documents\Edge Detection Improvements\images\DNP\Seperated\55\H0\fail'#DNP\78 - Artificial'#\DNP\Seperated\55\H0\fail'
    OUT_FOLDER = '\output'
    out_dir=directory+OUT_FOLDER
    print(out_dir)
    pixelSize=0.3235#0.112#0.3235#
    maxDiameter=65
    minDiameter=45
    replicate=False
    resultText=[]
    resultText.append("Image, Xloc, Yloc, Radius, Minor Axis, Major Axis, Angle, Min Diameter, Roundess, Circularity, Edge Deviation Mean, Edge Deviation Median, Edge Deviation Max, Edge Deviation Sum,  Edge Deviation Std, Inlier%")
    try:
        os.makedirs(out_dir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    for filename in os.listdir(directory):
        if filename.endswith(".bmp") or filename.endswith(".jpg") or filename.endswith(".png"):
            # Read image. 
            img = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_COLOR) 
            debug = False
            # Convert to grayscale.
            #cropX=img.shape[0]//25
            #cropY=img.shape[1]//25
            #img=img[cropX:-cropX,cropY:-cropY]
            #available methods: "d+s"   "canny" "sobel" "dog" "intensity" "adaptive"
            binaryImage=processImage(img,70,15,method="sobel",debug=debug)
            vias=findVias(binaryImage)
            smoothVias = vias#smoothContours(vias)
            outputImg,circles=circleFit(img,smoothVias,"ransac")
            cv2.imwrite(path.join(out_dir,filename),outputImg)
            print(path.join(out_dir,filename))#os.path.splitext(os.path.basename(filename))[0]+"_output.jpg"))
            if(debug):
                cv2.namedWindow('Contours',cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Contours', 800,600)
                cv2.imshow('Contours', outputImg) 
                cv2.waitKey(0) 
                cv2.destroyAllWindows()
                cv2.namedWindow('Contours',cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Contours', 800,600)
                cv2.imshow('Contours', binaryImage) 
                cv2.waitKey(0) 
                cv2.destroyAllWindows()


    with open(out_dir+r'\results.csv', 'w') as f:
        for item in resultText:
            f.write("%s\n" % item)


#https://ojskrede.github.io/inf4300/exercises/week_05/
#https://stackoverflow.com/questions/26222525/opencv-detect-partial-circle-with-noise
#https://www.sciencedirect.com/science/article/pii/S0167947310004809?via%3Dihub
#https://pypi.org/project/circle-fit/
#https://stackoverflow.com/questions/29850157/least-squares-for-circle-detection
#https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html
#https://github.com/jmtyszka/mrgaze/blob/master/mrgaze/fitellipse.py
#https://github.com/jmtyszka/mrgaze/blob/master/mrgaze/fitellipse.py
#https://stackoverflow.com/questions/37160143/how-can-i-extract-internal-contours-holes-with-python-opencv
#https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=drawcontours
#https://www.pyimagesearch.com/2014/04/21/building-pokedex-python-finding-game-boy-screen-step-4-6/
#https://www.programcreek.com/python/example/88831/skimage.measure.regionprops
#https://docs.opencv.org/master/d9/d8b/tutorial_py_contours_hierarchy.html
#https://math.stackexchange.com/questions/2645689/what-is-the-parametric-equation-of-a-rotated-ellipse-given-the-angle-of-rotatio
#https://math.stackexchange.com/questions/22064/calculating-a-point-that-lies-on-an-ellipse-given-an-angle
#https://agniva.me/scipy/2016/10/25/contour-smoothing.html