import logging
import math
from turtle import color
from venv import logger
import numpy as np
from numpy import angle, uint8 
import pydicom
import matplotlib.pyplot as plt
from typing import Dict
from scipy.ndimage import rotate
import Helpers.MathHelper as helper
import cv2
import scipy.ndimage
from sklearn.cluster import DBSCAN
from matplotlib.patches import Polygon, Rectangle
from scipy.spatial import distance
from scipy.spatial.distance import cdist


class DicomImageParser:
    def __init__(self, dicomPath, logger):
        #Initialize with the path to the DICOM file
        self.dicomPath=dicomPath
        self.dataset=None
        self.logger=logger
        
    def loadDicom(self):#Captures DICOM file 
        #load DICOM file into a dataset
        try:
            self.dataset=pydicom.dcmread(self.dicomPath)
            self.logger.info(f'Successfully loaded DICOM file:{self.dicomPath}')
        except Exception as ex:
            self.logger.error(f'Error loading DICOM file:{ex}') 
            
    def findMetadata(self): #Quick check of metadata 
        #Retrieve metadata from the DICOM file
        if not self.dataset:
            self.logger.warning('DICOM file is not loaded')
            return None
        metadata={
            "Patient Name":self.dataset.get("PatientName", "N/A"),
            "Patient ID":self.dataset.get("PatientID", "N/A"),
            "Acquisition Date":self.dataset.get("AcquisitionDate", "N/A"),
            "Acquisition Time":self.dataset.get("AcquisitionTime", "N/A"),
            "Modality":self.dataset.get("Modality", "N/A"),
            "Manufacturer":self.dataset.get("Manufacturer", "N/A"),
            "Operators' Name":self.dataset.get("OperatorsName", "N/A"),
            "Radiation Machine Name":self.dataset.get("RadiationMachineName", "N/A"),
            "Device Serial Number":self.dataset.get("DeviceSerialNumber", "N/A"),
            "Gantry Angle":self.dataset.get("GantryAngle", "N/A"),
            "Beam Limiting Device Angle":self.dataset.get("BeamLimitingDeviceAngle", "N/A"),
            "RT image SID":self.dataset.get("RTImageSID", "N/A")}
        self.logger.info(f"Captured metadata: {metadata}") 
        
    def findAbsoluteJawPositions(self): #Capture jaw positions from DICOM 
        jaw_positions = {"X1": None, "X2": None, "Y1": None, "Y2": None}

        if not self.dataset:
            self.logger.warning("No DICOM file loaded")
            return jaw_positions

    # Check for the Exposure Sequence
        if "ExposureSequence" in self.dataset:
            for exposure_item in self.dataset.ExposureSequence:
                if "BeamLimitingDeviceSequence" in exposure_item:
                    for deviceItem in exposure_item.BeamLimitingDeviceSequence:
                        # Check for X and Y jaw positions
                        if deviceItem.RTBeamLimitingDeviceType == "ASYMX":
                            jaw_positions["X1"], jaw_positions["X2"] = deviceItem.LeafJawPositions
                            self.logger.info(f"Captured X jaw positions: X1={jaw_positions['X1']}, X2={jaw_positions['X2']}")
                        elif deviceItem.RTBeamLimitingDeviceType == "ASYMY":
                            jaw_positions["Y1"], jaw_positions["Y2"] = deviceItem.LeafJawPositions
                            self.logger.info(f"Captured Y jaw positions: Y1={jaw_positions['Y1']}, Y2={jaw_positions['Y2']}")

    # Check if any jaw positions are still None and notify the user
        if None in jaw_positions.values():
            self.logger.warning("No jaw positions fully captured")

        return jaw_positions 

    def findISOcoordinate(self): #Capture iso coordinate from DICOM 
        isoCoordinate=float()
        if not self.dataset:
            self.logger.warning("No DICOM file loaded")
            return isoCoordinate
        if "IsocenterPosition" in self.dataset:
            isoCoordinate = self.dataset.IsocenterPosition
            self.logger.info(f'ISO coordinate captured: {isoCoordinate}')
            return isoCoordinate 
        
    def findCollimatorAngle(self): #Capture collimator angle from DICOM 
        if not self.dataset:
            self.logger.warning("No DICOM file loaded")
            return
        else:
            collimatorAngle=self.dataset.get("BeamLimitingDeviceAngle",None)
            if collimatorAngle is not None:
                self.logger.info(f"Collimator angle captured: {collimatorAngle} deg")
                return collimatorAngle
            else:
                self.logger.error('No collimator angle found in the DICOM file')
                return 
            
    def findImagePlanePixelSpacing(self): #Capture how many mm in one pixel 
        imagePlanePixelSpacing=float()
        if not self.dataset:
            self.logger.warning("No DICOM file loaded")
            return 
        
        imagePlanePixelSpacing=self.dataset.get("ImagePlanePixelSpacing", None)
        if imagePlanePixelSpacing is not None:
            self.logger.info(f'Image plane pixel spacing captured: {imagePlanePixelSpacing} pixes')
            
            collimatorAngle=self.findCollimatorAngle()
            self.logger.info(f'Collimator angle captured: {collimatorAngle} deg')
            
            theta=np.radians(collimatorAngle)
            cosTheta=np.cos(theta)
            sinTheta=np.sin(theta)
            xSpacing=imagePlanePixelSpacing[0]
            ySpacing=imagePlanePixelSpacing[1]
            
            xEffectiveSpacing=np.sqrt((xSpacing*cosTheta)**2+(ySpacing*sinTheta)**2)
            yEffectiveSpacing=np.sqrt((xSpacing*sinTheta)**2+(ySpacing*cosTheta)**2)
            imagePlanePixelSpacingValue=(xEffectiveSpacing,yEffectiveSpacing)
            self.logger.info(f'Image plane pixel spacing converted from: {imagePlanePixelSpacing} pixes to {imagePlanePixelSpacingValue} mm')
            return imagePlanePixelSpacingValue
        else:
            self.logger.warning('No pixel spacing captured')
            return 

    def findImagerPosition(self): #Capture imager position 
        imagerPosition=float()
        if not self.dataset:
            self.logger.warning("No DICOM file loaded")
            return
        
        imagerPosition=None
        for item in self.dataset:
            if item.tag==('0x3002','0x000D'): #I use tag as there is no descripption available
                imagerPosition=item.value
                
        if imagerPosition is not None:
            self.logger.info(f'Imager position captured: {imagerPosition}')  
        else:
            self.logger.warning('No imager postion found')
        return imagerPosition  

    def findRowsColumns(self): #Returns a tuple with (number of Rows,number of Columns) 
        if not self.dataset:
            self.logger.warning("No DICOM file loaded")
        if "Columns" in self.dataset and "Rows" in self.dataset:
            imageColumns=self.dataset.get('Columns', None) 
            imageRows=self.dataset.get('Rows', None)
            self.logger.info(f'Image Rows, Columns captured: {imageRows},{imageColumns}')
            return imageRows,imageColumns
        else:
            self.logger.warning('No rows or columns data found in the DICOM file')
            return 
                    
    def findSIDandSAD(self): #Returns a tuple: (image SID, linac SAD)
        if not self.dataset:
            self.logger.warning("No DICOM file loaded")
            return
        sid=self.dataset.get('RTImageSID',None)
        sad=self.dataset.get('RadiationMachineSAD',None)
        if sid is not None and sad is not None:
            self.logger.info(f"SID captured: {sid} mm; SAD captured: {sad} mm;")
            return sid, sad
        else:
            self.logger.error("No SID or SAD data found in the DICOM file")
            return
            
    def convertJawsMmToRotatedRectangleDictionary(self,
                                                  isoXcoordinate,isoYcoordinate,
                                                  jawX1mm,jawX2mm,jawY1mm,jawY2mm,
                                                  collimatorAngle,magnificationFactor,pixelSampling):
        try:
            rectangleJawPositions={'point1':(0,0),'point2':(0,0),'point3':(0,0),'point4':(0,0)}
            
            x1JawPixels=float((isoXcoordinate)+(jawX1mm/pixelSampling[0])*magnificationFactor)
            x2JawPixels=float((isoXcoordinate)+(jawX2mm/pixelSampling[0])*magnificationFactor)
            y1JawPixels=float((isoYcoordinate)-(jawY1mm/pixelSampling[1])*magnificationFactor)
            y2JawPixels=float((isoYcoordinate)-(jawY2mm/pixelSampling[1])*magnificationFactor)
            
            
            rectangleJawPositions["point1"]=helper.MathHelper.rotateCoordinate(self,
                x1JawPixels,y2JawPixels,collimatorAngle,isoXcoordinate,isoYcoordinate)
            
            rectangleJawPositions["point2"]=helper.MathHelper.rotateCoordinate(self,
                x2JawPixels,y2JawPixels,collimatorAngle,isoXcoordinate,isoYcoordinate)
            
            rectangleJawPositions["point3"]=helper.MathHelper.rotateCoordinate(self,
                x1JawPixels,y1JawPixels,collimatorAngle,isoXcoordinate,isoYcoordinate)
            
            rectangleJawPositions["point4"]=helper.MathHelper.rotateCoordinate(self,
                x2JawPixels,y1JawPixels,collimatorAngle,isoXcoordinate,isoYcoordinate)

            self.logger.info(f'Jaw positions converted from mm to pixels...\n'+
                             f'from:X1={jawX1mm},X2={jawX2mm},Y1={jawY1mm},Y2={jawY2mm}\n'+
                             f'to:  X1={x1JawPixels},X2={x2JawPixels},Y1={y1JawPixels},Y2={y2JawPixels}'+
                             f'{collimatorAngle}\n rotated rectangle coordinates calculated as: {rectangleJawPositions}')
            
            return rectangleJawPositions
        except Exception as ex:
            self.logger.error(f'Fatal error: {ex}')
            return
        
    def findPointWithIntensity(self,image,pixelSampling, desiredIntensityPercent,
                               searchRadiusMm, magnificationCoef, 
                               centerPoint,
                               tolerancePercent):
        pixelSpacing=min(pixelSampling[0],pixelSampling[1])
        searchRadius=(searchRadiusMm*magnificationCoef)/pixelSpacing
        targetIntensity=np.max(image)*(desiredIntensityPercent/100)
        tolerance=targetIntensity*(tolerancePercent/100)
        farthestPoint=None
        maxDistance=0
        #2 functions to ensure that the pixel is somwhere in the corner
        def checkCrossNeighbors(image,x,y):
            neighbors=[
                (y-1,x),(y+1,x),
                (y,x-1),(y,x+1)]
            maxIntensity=np.max(image)*(desiredIntensityPercent/100)
            hotter=0
            cooler=0
            for ny,nx in neighbors:
                if 0<=ny<image.shape[0] and 0<=nx<image.shape[1]:
                    pointIntesity=image[ny,nx]
                    if pointIntesity>maxIntensity*1.02:
                        hotter+=1
                    elif pointIntesity<maxIntensity*0.98:
                        cooler+=1
            return hotter>=2 and cooler>=2
        def checkDiagonalNeighbors(image,x,y):
            neighbors=[
                (y-1,x-1),(y-1,x+1),
                (y+1,x-1),(y+1,x+1)]
            maxIntensity=np.max(image)*(desiredIntensityPercent/100)
            same=0
            cooler=0
            hotter=0
            for ny,nx in neighbors:
                if 0<=ny<image.shape[0] and 0<=nx<image.shape[1]:
                    pointIntesity=image[ny,nx]
                    if pointIntesity<=maxIntensity*0.9:
                        cooler+=1
                    elif pointIntesity>=maxIntensity*1.10:
                        hotter+=1
                    if (1-pointIntesity/maxIntensity)<=0.1:
                        same+=1
            return cooler>=1 and hotter>=1 and same>=2
        centerXcoordinate=int(image.shape[1]/2)
        centerYcoordinate=int(image.shape[0]/2)
        
        x_min=max(int(centerPoint[0]-searchRadius),0)
        x_max=min(int(centerPoint[0]+searchRadius),centerXcoordinate*2)
        y_min=max(0, int(centerPoint[1]-searchRadius))
        y_max=min(centerYcoordinate*2, int(centerPoint[1]+searchRadius))
           
        for y in range(y_min,y_max+1):
            for x in range(x_min,x_max+1):
                if abs(image[y,x]-targetIntensity)<=tolerance:
                    distance=np.sqrt((centerXcoordinate-x)**2+(centerYcoordinate-y)**2)
                    if distance>maxDistance:
                        if checkCrossNeighbors(image,x,y)==True and checkDiagonalNeighbors(image,x,y)==True:
                            farthestPoint=(x,y)
                            maxDistance=distance
        if farthestPoint is not None:
            return farthestPoint
        else:
            self.logger.warning("No matching point found within the specified area.")       
            return None
        
    def findJawEdgesOnTheImage(self, image,pixelSampling, 
                               desiredIntensityPercent,
                               searchRadiusMm,
                               tolerancePercent,magnificationFactor,
                               rectangleCoordinates):
        self.logger.info(f'Detecting jaw edges on the image. Intensity={desiredIntensityPercent}% with the tolerance of {tolerancePercent}%; '+
                         f'The SID/SAD  magnification is {magnificationFactor}; expected corner points: {rectangleCoordinates}; radius={searchRadiusMm}mm;')
        point1coordinate=self.findPointWithIntensity(image,
                                                        pixelSampling,
                                                        desiredIntensityPercent,
                                                        searchRadiusMm, magnificationFactor,
                                                        rectangleCoordinates['point1'],
                                                        tolerancePercent)
        point2coordinate=self.findPointWithIntensity(image,
                                                        pixelSampling,
                                                        desiredIntensityPercent,
                                                        searchRadiusMm, magnificationFactor,
                                                        rectangleCoordinates['point2'],
                                                        tolerancePercent)
        point3coordinate=self.findPointWithIntensity(image,
                                                        pixelSampling,
                                                        desiredIntensityPercent,
                                                        searchRadiusMm, magnificationFactor,
                                                        rectangleCoordinates['point3'],
                                                        tolerancePercent)
        point4coordinate=self.findPointWithIntensity(image,
                                                        pixelSampling,
                                                        desiredIntensityPercent,
                                                        searchRadiusMm, magnificationFactor,
                                                        rectangleCoordinates['point4'],
                                                        tolerancePercent)
        detectedJawCoordinatesDictionary={
            'point1':point1coordinate,
            'point2':point2coordinate,
            'point3':point3coordinate,
            'point4':point4coordinate}
        if detectedJawCoordinatesDictionary['point1'] is not None and detectedJawCoordinatesDictionary['point2'] is not None and detectedJawCoordinatesDictionary['point3'] is not None and detectedJawCoordinatesDictionary['point4'] is not None:
                        self.logger.info(f'Coordinates detected: \npoint1 {detectedJawCoordinatesDictionary['point1']}; '+
                         f'point2 {detectedJawCoordinatesDictionary['point2']}; '+
                         f'point3 {detectedJawCoordinatesDictionary['point3']}; '+
                         f'point4 {detectedJawCoordinatesDictionary['point4']}; ')
        else:
           self.logger.warning(f'No jaw edjes detected... Check that the image has at least {searchRadiusMm}mm gap to the visible jaw edge from each side')
        return detectedJawCoordinatesDictionary
      
    def plotJawCoordinatesWithFourPoints(self,ax,
                                         rectangleCoordinates, 
                                         X1lineColor,
                                         X2lineColor,
                                         Y1lineColor,
                                         Y2lineColor,
                                         X1lineWidth, 
                                         X2lineWidth,
                                         Y1lineWidth,
                                         Y2lineWidth,
                                         X1Legendlabel,
                                         X2Legendlabel,
                                         Y1Legendlabel,
                                         Y2Legendlabel):
        self.logger.info('Called function to draw jaws with four points')
        if ax==None:
            fig,ax = plt.subplots()
            self.logger.info('No view detected. created new axis and view')
        #draw X1 jaw
        self.logger.info(f'Drawing line for: {X1Legendlabel}')
        plt.plot([rectangleCoordinates['point1'][0], rectangleCoordinates['point3'][0]], 
                 [rectangleCoordinates['point1'][1], rectangleCoordinates['point3'][1]], 
                 color=X1lineColor, linewidth=X1lineWidth, label=X1Legendlabel)
        #draw X2 jaw
        self.logger.info(f'Drawing line for: {X2Legendlabel}')
        plt.plot([rectangleCoordinates['point2'][0], rectangleCoordinates['point4'][0]], 
                 [rectangleCoordinates['point2'][1], rectangleCoordinates['point4'][1]], 
                 color=X2lineColor, linewidth=X2lineWidth, label=X2Legendlabel)
        
        #draw Y1 jaw
        self.logger.info(f'Drawing line for: {Y1Legendlabel}')
        plt.plot([rectangleCoordinates['point3'][0], rectangleCoordinates['point4'][0]], 
                 [rectangleCoordinates['point3'][1], rectangleCoordinates['point4'][1]], 
                 color=Y1lineColor, linewidth=Y1lineWidth, label=Y1Legendlabel)
        
        #draw Y2 jaw
        self.logger.info(f'Drawing line for: {Y2Legendlabel}')
        plt.plot([rectangleCoordinates['point1'][0], rectangleCoordinates['point2'][0]], 
                 [rectangleCoordinates['point1'][1], rectangleCoordinates['point2'][1]], 
                 color=Y2lineColor, linewidth=Y2lineWidth, label=Y2Legendlabel)
        ax.legend()
        return ax
    
    def calculateAngleBetweenLinesWithDifferentLength(self, line1endPoint1,line1ednPoint2,line2endPoint1,line2endPoint2):
            self.logger.info('Called a function to calculate angle between two lines with different length: '
                             f'{line1endPoint1}{line1ednPoint2} '+ 
                             f'and {line2endPoint1}{line2endPoint2}')  
            try:
                #Calculate fector for lines and normalize them    
                line1Vector=np.array([line1ednPoint2[0]-line1endPoint1[0],line1ednPoint2[1]-line1endPoint1[1]])
                line1VectorNorm=line1Vector/np.linalg.norm(line1Vector)
            
                line2Vector=np.array([line2endPoint2[0]-line2endPoint1[0],line2endPoint2[1]-line2endPoint1[1]])
                line2VectorNorm=line2Vector/np.linalg.norm(line2Vector)
            
                #Calculate dot product
                dotProduct=np.dot(line1VectorNorm,line2VectorNorm)
                dotProduct=np.clip(dotProduct,-1,1)
                #Calculate angle and convert to deg
                angleRad=np.arccos(dotProduct)
                angleDeg=np.degrees(angleRad)
                if angleDeg>90:
                    angleDeg=angleDeg-180
                elif angleDeg<-90:
                    angleDeg=angleDeg+180
                return angleDeg
            
            except Exception as ex:
                self.logger.error(f'{ex}')
                return None

    def calculateAngleAndDistance(self, expectedCoordinates, 
                                  detectedCoordinates,pixelSampling,
                                  magnificationFactor):
        # Prepare lines from coordinates
        expected_lines = {
            'X1jaw': (expectedCoordinates['point1'], expectedCoordinates['point3']),
            'X2jaw': (expectedCoordinates['point2'], expectedCoordinates['point4']),
            'Y1jaw': (expectedCoordinates['point3'], expectedCoordinates['point4']),
            'Y2jaw': (expectedCoordinates['point1'], expectedCoordinates['point2'])
        }
        detected_lines = {
            'X1jaw': (detectedCoordinates['point1'], detectedCoordinates['point3']),
            'X2jaw': (detectedCoordinates['point2'], detectedCoordinates['point4']),
            'Y1jaw': (detectedCoordinates['point3'], detectedCoordinates['point4']),
            'Y2jaw': (detectedCoordinates['point1'], detectedCoordinates['point2'])
        }
        def calculateAngle(line1ednPoint1,line1ednPoint2,line2ednPoint1,line2ednPoint2):
            angle=0
            x1_1=line1ednPoint1[0]
            y1_1=line1ednPoint1[1]
            x2_1=line1ednPoint2[0]
            y2_1=line1ednPoint2[1]
            
            x1_2=line2ednPoint1[0]
            y1_2=line2ednPoint1[1]
            x2_2=line2ednPoint2[0]
            y2_2=line2ednPoint2[1]
            
            try:
                slope1=(y2_1-y1_1)/(x2_1-x1_1)
            except ZeroDivisionError:
                slope1=float('inf')
                
            try:
                slope2=(y2_2-y1_2)/(x2_2-x1_2)
            except ZeroDivisionError:
                slope2=float('inf')
                
            if slope1==float('inf') and slope2==float('inf'):
                angle=0
            elif slope1==float('inf'):
                slope1=90
            elif slope2==float('inf'):
                slope2=90
            else:    
                angle=np.degrees(np.arctan(abs((slope2-slope1))/(1+slope2*slope1)))
            return angle
        def calculateDistance(line1ednPoint1,line1ednPoint2,line2ednPoint1,line2ednPoint2):
            distanceInPixels=0
            x1_1=line1ednPoint1[0]
            y1_1=line1ednPoint1[1]
            x2_1=line1ednPoint2[0]
            y2_1=line1ednPoint2[1]
            
            x1_2=line2ednPoint1[0]
            y1_2=line2ednPoint1[1]
            x2_2=line2ednPoint2[0]
            y2_2=line2ednPoint2[1]
            if abs(x1_1-x2_1)>abs(y1_1-y2_1):
                #the line is horisontal, looking for the endpoint difference for y axis only
                distanceInPixels=max(abs(y1_1-y1_2),abs(y2_1-y2_2))
            else:
                #the line is vertical
                distanceInPixels=max(abs(x1_1-x1_2),abs(x2_1-x2_2))    
            return distanceInPixels
        
        angleX1=round((calculateAngle(expected_lines['X1jaw'][0],expected_lines['X1jaw'][1],
                             detected_lines['X1jaw'][0],detected_lines['X1jaw'][1])),4)       
        angleX2=round((calculateAngle(expected_lines['X2jaw'][0],expected_lines['X2jaw'][1],
                             detected_lines['X2jaw'][0],detected_lines['X2jaw'][1])),4)       
        angleY1=round((calculateAngle(expected_lines['Y1jaw'][0],expected_lines['Y1jaw'][1],
                             detected_lines['Y1jaw'][0],detected_lines['Y1jaw'][1])),4)       
        angleY2=round((calculateAngle(expected_lines['Y2jaw'][0],expected_lines['Y2jaw'][1],
                             detected_lines['Y2jaw'][0],detected_lines['Y2jaw'][1])),4)       

        distanceX1=round(((pixelSampling[0]/magnificationFactor)*calculateDistance(expected_lines['X1jaw'][0],expected_lines['X1jaw'][1],
                             detected_lines['X1jaw'][0],detected_lines['X1jaw'][1])),2)
        distanceX2=round(((pixelSampling[0]/magnificationFactor)*calculateDistance(expected_lines['X2jaw'][0],expected_lines['X2jaw'][1],
                             detected_lines['X2jaw'][0],detected_lines['X2jaw'][1])),2)
        distanceY1=round(((pixelSampling[1]/magnificationFactor)*calculateDistance(expected_lines['Y1jaw'][0],expected_lines['Y1jaw'][1],
                             detected_lines['Y1jaw'][0],detected_lines['Y1jaw'][1])),2)
        distanceY2=round(((pixelSampling[1]/magnificationFactor)*calculateDistance(expected_lines['Y2jaw'][0],expected_lines['Y2jaw'][1],
                             detected_lines['Y2jaw'][0],detected_lines['Y2jaw'][1])),2)
        angleDistanceDictionary={
            "X1angleDeviation(deg)":angleX1,
            "X2angleDeviation(deg)":angleX2,
            "Y1angleDeviation(deg)":angleY1,
            "Y2angleDeviation(deg)":angleY2,
            "X1distanceDeviation(mm)":distanceX1,
            "X2distanceDeviation(mm)":distanceX2,
            "Y1distanceDeviation(mm)":distanceY1,
            "Y2distanceDeviation(mm)":distanceY2}
        return angleDistanceDictionary
    
    def findPointLineDistance(self,point,pointLine1,pointLine2,pixelSampling):
        self.logger.info(f'Calculating shortest distance to the point {point} to the line:{pointLine1}, {pointLine2}...')
        x0,y0=point
        x1,y1=pointLine1
        x2,y2=pointLine2
        
        A=y2-y1
        B=x1-x2
        C=x2*y1-x1*y2
        
        distance=abs(A*x0+B*y0+C)/np.sqrt(A**2+B**2)
        distance=distance*pixelSampling
        self.logger.info(f'Distance={distance} mm')
        return distance
    
    def findLeafPixels(self,image,ax,cornerCoordinates,rectangleCoordinates, thresholdPercent,pixelSampling,magnificationFactor,reportLeaves):
        maxIntensity=np.max(image)
        thresholdValue=maxIntensity*thresholdPercent/100
        
        #Capture jaw corners coordinates
        point1,point2=cornerCoordinates['point1'],cornerCoordinates['point2']
        point3,point4=cornerCoordinates['point3'],cornerCoordinates['point4']
        
        point1rectangle,point2rectangle=rectangleCoordinates['point1'],rectangleCoordinates['point2']
        point3rectangle,point4rectangle=rectangleCoordinates['point3'],rectangleCoordinates['point4']
        
        points=np.array([
                         [int(round(point1[0])),int(round(point1[1]))],
                         [int(round(point2[0])),int(round(point2[1]))],
                         [int(round(point4[0])),int(round(point4[1]))],
                         [int(round(point3[0])),int(round(point3[1]))]
                         ])

        #Create a mask for the polygon to improve efficiency
        mask=np.zeros(image.shape,dtype=uint8)
        cv2.fillPoly(mask,[points],1)
        
        #Debugging
        self.logger.info(f'Rectangle points: {points}')
        self.logger.info(f'Mask created with shape: {mask.shape}')
        

        #Find low-intensity points within the mask
        lowIntensityMask=(image<thresholdValue) & (mask==1)
        lowIntensityPoints=np.column_stack(np.where(lowIntensityMask))
        
        self.logger.info(f'Number of low intensity points: {len(lowIntensityPoints)}')
        

        #Apply Density-based spatial clustering of applications with noise
        if len(lowIntensityPoints)>0:
            db=DBSCAN(eps=5,min_samples=5).fit(lowIntensityPoints)  
            labels=db.labels_
            uniqueLabels=set(labels)
            for k in uniqueLabels:
                if k==-1:
                    continue
                classMemberMask=(labels)==k
                xy=lowIntensityPoints[classMemberMask]

                #Use convex hull to capture the sophysticated shae of MLCs
                hull=cv2.convexHull(xy)
                
                #Due to the difference in coordinate plotting between libraries cv2 and plt ((x,y) vs (y,x))
                hullPoints=[(y,x) for x,y in hull.squeeze()]
                self.logger.info(f'hull points: {hullPoints}')
                #Plot bounding polygon
                polygon=Polygon(hullPoints,linewidth=3,edgecolor='black',facecolor='none')
                ax.add_patch(polygon)   
                
                #Calculate center points for the leaf edges in order to determine leaf number and leaf bank
                moments=cv2.moments(hull)
                if moments['m00']!=0:
                    centerY=int(moments['m10']/moments['m00'])
                    centerX=int(moments['m01']/moments['m00'])
                    
                else:
                    centerY,centerX=hull.mean(axis=0)[0]
                    
                #Find leaf number and leaf bank
                
                #Determine lef bank
                centerX1jaw=((point1[0]+point3[0]))/2,((point1[1]+point3[1]))/2
                centerX2jaw=((point2[0]+point4[0]))/2,((point2[1]+point4[1]))/2
                
                distanceToX1=np.sqrt((centerX1jaw[0]-centerX)**2+(centerX1jaw[1]-centerY)**2)
                distanceToX2=np.sqrt((centerX2jaw[0]-centerX)**2+(centerX2jaw[1]-centerY)**2)
                
                if distanceToX1>distanceToX2:
                    leafBank='A'
                else:
                    leafBank='B'  
                self.logger.info(f'Leaf bank captured:{leafBank}')

                distance=self.findPointLineDistance((centerX,centerY),centerX1jaw,centerX2jaw,pixelSampling[1])
                
                #Determine leaf number 
                y1centerPoint=(point4[0]+point3[0])/2, (point4[1]+point3[1])/2
                y2centerPoint=(point1[0]+point1[0])/2, (point2[1]+point2[1])/2
                
                distanceToY1=np.sqrt((y1centerPoint[1]-centerY)**2+(y1centerPoint[0]-centerX)**2)
                distanceToY2=np.sqrt((y2centerPoint[1]-centerY)**2+(y2centerPoint[0]-centerX)**2)  

                if distanceToY2>distanceToY1:
                    leafNumber=31
                    while distance>0:
                        leafNumber=leafNumber-1
                        if distance>100*magnificationFactor:
                              distance-=10*magnificationFactor
                        else:
                              distance-=5*magnificationFactor  
                else:
                    leafNumber=30
                    while distance>0:
                        leafNumber=leafNumber+1
                        if distance>100*magnificationFactor:
                              distance-=10*magnificationFactor
                        else:
                              distance-=5*magnificationFactor
                self.logger.info(f'Leaf number captured:{leafBank}-{leafNumber}')

                #Plot found leaf center+leaf index
                ax.plot(centerX,centerY,'ko',markersize=3)
                ax.text(centerX,centerY+2,f'{leafBank}-{leafNumber}',color='white',fontsize=12,ha='left')
                
                #Plot leaf centerline
                hullPointsNp=np.array(hullPoints,dtype=np.int32)
                rect=cv2.minAreaRect(hullPointsNp)
                box=cv2.boxPoints(rect)
                box=np.int0(box)
                polygonPatch=Polygon(box,closed=True,edgecolor='red',fill=None,linewidth=2)
                ax.add_patch(polygonPatch)
                
                sideLength=[np.linalg.norm(box[i]-box[(i+1)%4]) for i in range(4)]
                shortestSides=sorted(range(4),key=lambda i: sideLength[i])[:2]
                midPoints=[]
                for i in shortestSides:
                    p1,p2=box[i],box[(i+1)%4]
                    midpoint=((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
                    midPoints.append(midpoint)
                ax.plot([midPoints[0][0],midPoints[1][0]], [midPoints[0][1],midPoints[1][1]], 'r--', linewidth=1)

                angle=round(self.calculateAngleBetweenLinesWithDifferentLength(
                    (point1rectangle[0],point1rectangle[1]),
                    (point2rectangle[0],point2rectangle[1]),
                    [midPoints[0][0],midPoints[0][1]], 
                    [midPoints[1][0],midPoints[1][1]]),2)
                
                self.logger.info(f'Leaf number {leafNumber}, angle deviation is {angle} deg')
                reportLeaves[f'{leafBank}-{leafNumber}']=angle

    def showImageWithJawLeafPositionsAndCalculateDeviations(self):
        #Generate list for reporting
        report={}
        #Display the image stored in the DICOM file with jaw positions shown
        if not self.dataset or 'PixelData' not in self.dataset:
            self.logger.error("Unable to isplay image. No DICOM file loaded of no pixel data found")
            return
        #Generate view
        fig,ax=plt.subplots()
        
        #Log selected metadata
        getMetadata=self.findMetadata()
        
        #Find the image
        image=self.dataset.pixel_array
        
        #Draw iso position and lines
        dicomImageISOcoordinate=image.shape[1]/2,image.shape[0]/2
        plt.scatter(dicomImageISOcoordinate[0],
                    dicomImageISOcoordinate[1],
                    color='black',s=150,marker='x',
                    label='DICOM iso')
        plt.axvline(x=dicomImageISOcoordinate[0],color='black')
        plt.axhline(y=dicomImageISOcoordinate[1],color='black')

        #Draw jaw positions
        jawInMm=self.findAbsoluteJawPositions()
        report['JawSize']=jawInMm
        
        collimatorAngle=self.findCollimatorAngle()
        report['COL']=collimatorAngle

        sidSadTuple=self.findSIDandSAD()
        report['SID,SAD']=sidSadTuple
        
        magnificationFactor=sidSadTuple[0]/sidSadTuple[1]
        
        pixelSampling=self.findImagePlanePixelSpacing()
        
        rectangleCoordinates=self.convertJawsMmToRotatedRectangleDictionary(
            dicomImageISOcoordinate[0],dicomImageISOcoordinate[1],
            jawInMm['X1'],jawInMm['X2'],jawInMm['Y1'],jawInMm['Y2'],
            -collimatorAngle,magnificationFactor,pixelSampling)
        
        #draw expected jaw positions
        self.plotJawCoordinatesWithFourPoints(ax,rectangleCoordinates,
                                              'red','blue','purple','pink',
                                              1,1,1,1,
                                              'X1jaw','X2jaw','Y1jaw','Y2jaw')
        
        #Logic for looking into jaw edges
        detectedJawPositionsDictionary=self.findJawEdgesOnTheImage(image,pixelSampling,
                                                                   50,10,10,
                                                                   magnificationFactor,
                                                                   rectangleCoordinates)
        
        self.plotJawCoordinatesWithFourPoints(ax,detectedJawPositionsDictionary,
                                              'black','black','black','black',
                                              1,1,1,1,
                                              'X1jawDet','X2jawDet','Y1jawDet','Y2jawDet')
        
        test=self.calculateAngleAndDistance(rectangleCoordinates,detectedJawPositionsDictionary,pixelSampling,magnificationFactor)  
        self.logger.info(f'Angle and distance deviations captured: {test}')
        report['JawDiscrepancies:']=test
        
        #Show leaves detected on the image
        reportLeaves={}
        self.findLeafPixels(image,ax,detectedJawPositionsDictionary, rectangleCoordinates, 47,pixelSampling,magnificationFactor,reportLeaves)
        reportLeavesSorted=dict(sorted(reportLeaves.items(),key=lambda item: item[1], reverse=True))
        self.logger.info('**********Report*********')
        self.logger.info(report)
        self.logger.info(reportLeaves)
        
        
        ax.imshow(image,cmap='jet')
        plt.axis('on')        
        plt.legend(loc='upper left',bbox_to_anchor=(1,1))
        plt.show()
        
    
        