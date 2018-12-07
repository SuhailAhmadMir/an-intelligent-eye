##############################################################################
#This Project is Developed by :
#  1. SUHAIL AHMAD MIR
#  2. AFIQ BILAL
#  UNIVERSITY OF KASHMIR
#  DEPARTMENT OF COMPUTER SCIENCE
#############################################################################

#***********************
#**** Object Detection and Stereo Vision ****
#***********************


# Package importation
import numpy as np
import cv2
import win32com.client
import keyboard
from sklearn.preprocessing import normalize


###################################################################################################################
######################### Object detection var ####################################

MIN_MATCH_COUNT=30
Distance=0
detector=cv2.xfeatures2d.SIFT_create()
######################## Creating Feature Matcher ############################
FLANN_INDEX_KDITREE=0  #flag
flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
flann=cv2.FlannBasedMatcher(flannParam,{})  #{}empty braces is actually cv2 bug



# Loading Training Data
trainImg=cv2.imread("TrainingData/TrainImg2.jpg",0)

###################### Extracting the features from this image ##################################
# detector will return key point 'kp' and descriptor 'decs'
# key points are the coordinates where it find the feature and description is the description of that key point
trainKP,trainDesc=detector.detectAndCompute(trainImg,None) #second parameter is MASK=None


###################################################################################################################



# Filtering
kernel= np.ones((3,3),np.uint8)



##################################################################################################################
#########################################  MOTION DETECTION  #####################################################
##################################################################################################################
#def motion():
#    cv2.ocl.setUseOpenCL(False)
#    version = cv2.__version__.split('.')[0]
#    print (version)
#    cap = cv2.VideoCapture(0)
#    if version == '2' :
#        fgbg = cv2.BackgroundSubtractorMOG2()
#    if version == '3':
#        fgbg = cv2.createBackgroundSubtractorMOG2()
#    while (cap.isOpened):
#        ret,frame=cap.read()
#        if ret==True:
#            fgmask = fgbg.apply(frame)
#            if version == '2' : 
#                (contours, hierarchy) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#            if version == '3' : 
#                (im2, contours, hierarchy) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#            for c in contours:
#                if cv2.contourArea(c) < 2000:
#                    continue
#                (x, y, w, h) = cv2.boundingRect(c)
#                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#            cv2.imshow('foreground and background',fgmask)
#            cv2.imshow('rgb',frame)
#            if cv2.waitKey(1) & 0xFF == ord("q"):
#                break
#    cap.release()
#    cv2.destroyAllWindows()

###################################################################################################################
############################################## DISTANCE CALCULATION ###############################################
###################################################################################################################
def coords_mouse_disp(event,x,y,flags,param):
    global Distance
    if event == cv2.EVENT_LBUTTONDBLCLK:
        #print (x,y,disp[y,x],filteredImg[y,x])
        average=0
        for u in range (-1,2):
            for v in range (-1,2):
                average += disp[y+u,x+v]
        average=average/9
        Distance= -593.97*average**(3) + 1506.8*average**(2) - 1373.1*average + 522.06 #polynomial regression =n of order 3
        Distance= np.around(Distance*0.01,decimals=2)       #rounding of array to 2 decimal places
        print('Distance: '+ str(Distance)+' m')
        
        #speaker = win32com.client.Dispatch("SAPI.SpVoice")
        #speaker.Speak("Distance" +str(Distance)+ 'metres')
        




#*************************************************
#***** Parameters for Distortion Calibration *****
#*************************************************

# Termination criteria
criteria =(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all images
objpoints= []   # 3d points in real world space
imgpointsR= []   # 2d points in image plane
imgpointsL= []

# Start calibration from the camera
print('Starting calibration for the 2 cameras... ')

# Call all saved images
for i in range(0,50):   # Put the amount of pictures you have taken for the calibration inbetween range(0,?) wenn starting from the image number 0
    t= str(i)           # string conversion
    ChessImaR= cv2.imread('chessboard-R'+t+'.png',0)    # Right side
    ChessImaL= cv2.imread('chessboard-L'+t+'.png',0)    # Left side
    retR, cornersR = cv2.findChessboardCorners(ChessImaR,(9,6),None)  # Define the number of chees corners we are looking for
    retL, cornersL = cv2.findChessboardCorners(ChessImaL,(9,6),None)  # Left side
    if (True == retR) & (True == retL):
        objpoints.append(objp)
        cv2.cornerSubPix(ChessImaR,cornersR,(11,11),(-1,-1),criteria)
        cv2.cornerSubPix(ChessImaL,cornersL,(11,11),(-1,-1),criteria)
        imgpointsR.append(cornersR)
        imgpointsL.append(cornersL)


################################# Calibration of distortion ################################################
# Determine the new values for different parameters
#   Right Side
# distortion coefficient, rotation and translation vectors
#It returns the camera matrix, distortion coefficients, rotation and translation vectors etc.
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints,
                                                        imgpointsR,
                                                        ChessImaR.shape[::-1],None,None)
hR,wR= ChessImaR.shape[:2]
OmtxR, roiR= cv2.getOptimalNewCameraMatrix(mtxR,distR,
                                                   (wR,hR),1,(wR,hR))

#   Left Side
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints,
                                                        imgpointsL,
                                                        ChessImaL.shape[::-1],None,None)
hL,wL= ChessImaL.shape[:2]
OmtxL, roiL= cv2.getOptimalNewCameraMatrix(mtxL,distL,(wL,hL),1,(wL,hL))

print('Cameras Ready to use')


#################################################################################################################3
#********************************************
#***** Calibrate the Cameras for Stereo Vision*****
#********************************************

# StereoCalibrate function
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC

retS, MLS, dLS, MRS, dRS, R, T, E, F= cv2.stereoCalibrate(objpoints,
                                                          imgpointsL,
                                                          imgpointsR,
                                                          mtxL,
                                                          distL,
                                                          mtxR,
                                                          distR,
                                                          ChessImaR.shape[::-1],
                                                          criteria_stereo,
                                                          flags)

# StereoRectify function
rectify_scale= 0 # if 0 image croped, if 1 image nor croped
#rectification transform left , rectification transform right, projection right, output 4x4 disparity depth mapping matrix
RL, RR, PL, PR, Q, roiL, roiR= cv2.stereoRectify(MLS, dLS, MRS, dRS,
                                                 ChessImaR.shape[::-1], R, T,
                                                 rectify_scale,(0,0))  # last paramater is alpha, if 0= croped, if 1= not croped
# initUndistortRectifyMap function
Left_Stereo_Map= cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                             ChessImaR.shape[::-1], cv2.CV_16SC2)   # cv2.CV_16SC2 this format enables us the programme to work faster
Right_Stereo_Map= cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                              ChessImaR.shape[::-1], cv2.CV_16SC2)
#*******************************************
#***** Parameters for the StereoVision *****
#*******************************************

# Create StereoSGBM and prepare all parameters
window_size = 3
min_disp = 2
num_disp = 130-min_disp
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = window_size,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32,
    disp12MaxDiff = 5,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2)

# Used for the filtered image
stereoR=cv2.ximgproc.createRightMatcher(stereo) # Create another stereo for right this time

# WLS FILTER Parameters
lmbda = 80000
sigma = 1.8
visual_multiplier = 1.0
 
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

#*************************************
#***** Starting the StereoVision *****
#*************************************
# Call the two cameras
CamR= cv2.VideoCapture(0)   # When 0 then Right Cam and when 2 Left Cam
CamL= cv2.VideoCapture(1)

##################################################################################################################
################################ OBJECT DETECTION ################################################################
##################################################################################################################
while True:
    ret, QueryImgBGR=CamR.read()
    QueryImg=cv2.cvtColor(QueryImgBGR,0)
    # Extracting the key points from this image
    queryKP,queryDesc=detector.detectAndCompute(QueryImg,None)
    # MATCHING the features if their are any
    matches=flann.knnMatch(queryDesc,trainDesc,k=2)
    # Filtering the matches for more clean results. Filtering method is called as Ratio Test
    # If distance between trainDesc and queryDesc is less than 70% then it is not a good match
    goodMatch=[]
    # where 'm' is queryMatch and 'n' is trainMatch
    for m,n in matches:
        if(m.distance<0.75*n.distance):
            goodMatch.append(m)
    if(len(goodMatch)>MIN_MATCH_COUNT):     # if goodMatch > MIN_MATCH_COUNT then OBJECT DETECTION is applied
        tp=[]
        qp=[]
        for m in goodMatch:
            tp.append(trainKP[m.trainIdx].pt) # gives index of particular ID and .pt gives the coordinates
            qp.append(queryKP[m.queryIdx].pt)
        tp,qp=np.float32((tp,qp))  #converting it to numpy array
        
        # To detect borders we use HOMOGRAPHY
        H,status=cv2.findHomography(tp,qp,cv2.RANSAC,3.0) # 'H' is transformation variable
        h,w=trainImg.shape   # gives height and width 
        # Calculating border
        trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
        # Converting into its corresponding query image
        queryBorder=cv2.perspectiveTransform(trainBorder,H)
        cv2.polylines(QueryImgBGR,[np.int32(queryBorder)],True,(0,255,0),5)
        cv2.putText(QueryImgBGR, '%s'%('Distance: '+ str(Distance)+' m'),(200,50),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,0),2)

    cv2.imshow('result',QueryImgBGR)
    
#################################################################################################################
#################################################################################################################    
    
    
    
    # Start Reading Camera images
    retR, frameR= CamR.read()
    retL, frameL= CamL.read()

    # Rectify the images on rotation and alignement
    Left_nice= cv2.remap(frameL,Left_Stereo_Map[0],Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)  # Rectify the image using the calibration parameters found during the initialisation
    Right_nice= cv2.remap(frameR,Right_Stereo_Map[0],Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    # Draw Red lines
    for line in range(0, int(Right_nice.shape[0]/20)): # Draw the Lines on the images Then numer of line is defines by the image Size/20
        Left_nice[line*20,:]= (0,0,255)
        Right_nice[line*20,:]= (0,0,255)

    for line in range(0, int(frameR.shape[0]/20)): # Draw the Lines on the images Then numer of line is defines by the image Size/20
        frameL[line*20,:]= (0,255,0)
        frameR[line*20,:]= (0,255,0)    
        
    # Show the Undistorted images
    cv2.imshow('Both Images', np.hstack([Left_nice, Right_nice]))
    cv2.imshow('Normal', np.hstack([frameL, frameR]))
    

    # Convert from color(BGR) to gray
    grayR= cv2.cvtColor(Right_nice,cv2.COLOR_BGR2GRAY)
    grayL= cv2.cvtColor(Left_nice,cv2.COLOR_BGR2GRAY)

    # Compute the 2 images for the Depth_image
    disp= stereo.compute(grayL,grayR)#.astype(np.float32)/ 16
    dispL= disp
    dispR= stereoR.compute(grayR,grayL)
    dispL= np.int16(dispL)
    dispR= np.int16(dispR)

    # Using the WLS filter
    filteredImg= wls_filter.filter(dispL,grayL,None,dispR)
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)
    #cv2.imshow('Disparity Map', filteredImg)
    disp= ((disp.astype(np.float32)/ 16)-min_disp)/num_disp # Calculation allowing us to have 0 for the most distant object able to detect



    # Filtering the Results with a closing filter
    closing= cv2.morphologyEx(disp,cv2.MORPH_CLOSE, kernel) # Apply an morphological filter for closing little "black" holes in the picture(Remove noise) 

    # Colors map
    dispc= (closing-closing.min())*255
    dispC= dispc.astype(np.uint8)                                   # Convert the type of the matrix from float32 to uint8, this way you can show the results with the function cv2.imshow()
    disp_Color= cv2.applyColorMap(dispC,cv2.COLORMAP_OCEAN)         # Change the Color of the Picture into an Ocean Color_Map
    filt_Color= cv2.applyColorMap(filteredImg,cv2.COLORMAP_OCEAN) 
    # Show the result for the Depth_image
    cv2.imshow('Disparity', disp)
    #cv2.imshow('Closing',closing)
    cv2.imshow('Color Depth',disp_Color)
    cv2.imshow('Filtered Color Depth',filt_Color)
    #cv2.imshow('boundboxL',frameL)
    #cv2.imshow('boundboxR',frameR)

    # Mouse click
    cv2.setMouseCallback("Filtered Color Depth",coords_mouse_disp,filt_Color)
    
    # Keyevent press for motion detection
#    if keyboard.is_pressed('m'):                               
#        motion()
    # End the Programme
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break
    


# Release the Cameras
CamR.release()
CamL.release()
cv2.destroyAllWindows()
