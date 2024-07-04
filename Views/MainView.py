import io
import sys
import threading
import tkinter as tk
from tkinter import RIGHT, messagebox, Text
from turtle import position
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import logging
import numpy as np
from PIL import Image, ImageTk
import pydicom
from Helpers.DICOMimageParser import DicomImageParser


class TextHandler(logging.Handler):
    def __init__(self,text_widget):
        super().__init__()
        self.text_widget=text_widget
        
    def emit(self,record):
        msg=self.format(record)
        self.text_widget.config(state='normal')
        self.text_widget.insert(tk.END,msg+'\n')
        self.text_widget.see(tk.END)
        self.text_widget.config(state='disabled')
    
class View:
    
    def __init__(self,controller):
        #self.root=root
        self.root=tk.Tk()
        self.root.title('PD Annual QA Tool')
        self.root.geometry('900x600')
        
        icon_image=tk.PhotoImage(file=r'C:\Users\Danie\Desktop\Python\AnnualQAProject\AnnualQA\PDtoolIcon.png')
        self.root.iconphoto(False, icon_image)
        self.controller=controller
        self.createWidgets()
        self.currentImageIndex=0
        

        
    def createWidgets(self):
            PADX = 0
            PADY = 0
            
            firstTwoButtonsFrame=tk.Frame(self.root)
            firstTwoButtonsFrame.pack(side=tk.TOP,fill=tk.X,padx=PADX,pady=1)
            
            testButtonsFrame=tk.Frame(self.root)
            testButtonsFrame.pack(side=tk.TOP,fill=tk.X,padx=PADX,pady=0)
            
            nextPrevImageButtonsFrame=tk.Frame(self.root)
            nextPrevImageButtonsFrame.pack(side=tk.TOP,fill=tk.BOTH,padx=PADX,pady=PADY)
            
            loggerFrame=tk.Frame(self.root)
            loggerFrame.pack(side=tk.BOTTOM,fill=tk.X,expand=True,padx=PADX,pady=PADY)
            
            
            self.selectFolderButton = tk.Button(firstTwoButtonsFrame, text='Select DICOM Folder', command=self.controller.selectFolder)
            self.selectFolderButton.pack(side=tk.LEFT, expand=True, fill=tk.X)
    
            self.selectFileButton = tk.Button(firstTwoButtonsFrame, text='Select DICOM File     ', command=self.controller.selectFile)
            self.selectFileButton.pack(side=tk.LEFT, expand=True, fill=tk.X)
            '''
            self.leafAlingmentButton = tk.Button(testButtonsFrame, text='Run Leaf-Jaw Alignment Test', command=self.controller.processDicomFiles)
            self.leafAlingmentButton.pack(side=tk.LEFT, expand=True, fill=tk.X)
            
            self.jawAlingmentButton = tk.Button(testButtonsFrame, text='Run Jaw Alignment Test', command=self.controller.performJawAlignmentTest)
            self.jawAlingmentButton.pack(side=tk.LEFT, expand=True, fill=tk.X)

            self.mlcJawCenterButton = tk.Button(testButtonsFrame,text='Run Jaw/MLC center Test', command=self.controller.performJawMlcCenterTest)
            self.mlcJawCenterButton.pack(side=tk.LEFT, expand=True, fill=tk.X)
            
            self.mlcJawCenterButton = tk.Button(testButtonsFrame,text='Run MLC leakage Test', command=self.controller.performMlcLeakageTest)
            self.mlcJawCenterButton.pack(side=tk.LEFT, expand=True, fill=tk.X)
            '''
            
            buttonInfo={
                'Run Leaf-Jaw Alignment Test':self.controller.processDicomFiles,
                'Run Jaw Alignment Test':self.controller.performJawAlignmentTest,
                'Run Jaw/MLC center Test':self.controller.performJawMlcCenterTest,
                'Run MLC leakage Test': self.controller.performMlcLeakageTest}
            
            maxWidth=max(len(text) for text in buttonInfo.keys())
            self.buttons=[]
            for text, command in buttonInfo.items():
                button=tk.Button(testButtonsFrame, text=text,width=maxWidth,command=command)
                button.pack(side=tk.LEFT,expand=True,fill=tk.X,padx=0,pady=0)
                self.buttons.append(button)
            
            self.nextImageButton=tk.Button(nextPrevImageButtonsFrame, text='>>', command=self.showNextImage)
            self.nextImageButton.pack(side=tk.RIGHT, expand=True, fill=tk.X)
            
            self.prevImageButton=tk.Button(nextPrevImageButtonsFrame, text='<<', command=self.showPrevImage)
            self.prevImageButton.pack(side=tk.LEFT, expand=True, fill=tk.X)

            self.figure, self.ax = plt.subplots()
            plt.tight_layout()
            self.figure.subplots_adjust(top=0.9)
            
            self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.root)
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.X, expand=True, padx=PADX, pady=PADY)
            self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
            self.canvas.draw()
            
            self.logOutput = Text(loggerFrame, height=10, wrap='word', state='disabled')
            self.logOutput.pack(side=tk.BOTTOM, fill=tk.X, expand=True)

    def showNextImage(self):
        if self.controller.model.dicomFileModels:
            self.currentImageIndex=(self.currentImageIndex+1)%len(self.controller.model.dicomFileModels)
            self.displayImage(self.controller.model.dicomFileModels[self.currentImageIndex],
                              self.controller.model.dicomImageParser, self.controller.logger)
            
    def showPrevImage(self):
        if self.controller.model.dicomFileModels:
            self.currentImageIndex=(self.currentImageIndex-1)%len(self.controller.model.dicomFileModels)
            self.displayImage(self.controller.model.dicomFileModels[self.currentImageIndex], 
                              self.controller.model.dicomImageParser, self.controller.logger)       
        
    def setupLogging(self,logger):
        textHandler=TextHandler(self.logOutput)
        formatter=logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
        textHandler.setFormatter(formatter)
        logger.addHandler(textHandler)
        logger.setLevel(logging.INFO)
        
    def displayImageFromDICOMfile(self, dicomPath):
        try:
            dataset=pydicom.dcmread(dicomPath)
            if not dataset:
                self.logger.error(f'No DICOM data found in the selected file: {dicomPath}')
                return
            if hasattr(dataset,'pixel_array'):   
                image=dataset.pixel_array
            else:
                self.logger.error(f'No pixel data found in the selected file: {dicomPath}')
                return
            ax=self.ax
            ax.clear()
            ax.imshow(image,cmap='jet') 
            self.canvas.draw()
        except Exception as ex:
            self.logger.error(f'Error processing file: {dicomPath}\nError: {ex}') 
        
    def displayImage(self, dicomModel, imageParser, logger):
        
        ax=self.ax
        ax.clear()
        ax.set_title(f'Image {self.currentImageIndex+1}')
        if not dicomModel:
            return
        reportPositioningDeviations=dicomModel.leafPositionDeviations  
        reportAngleDeviations=dicomModel.leafAngleDeviations
        
        #Logic for leaf shape delineation
        #Find the image
        image=dicomModel.image
        
        #Draw iso position and centerlines
        dicomImageISOcoordinate=image.shape[1]/2,image.shape[0]/2
        ax.scatter(dicomImageISOcoordinate[0],
                    dicomImageISOcoordinate[1],
                    color='black',s=150,marker='x',
                    label='DICOM iso')
        ax.axvline(x=dicomImageISOcoordinate[0],color='black')
        ax.axhline(y=dicomImageISOcoordinate[1],color='black')
        
        if not dicomModel.metadata:
            return

        #Draw jaw positions
        jawInMm=dicomModel.jawPositions
        collimatorAngle=dicomModel.metadata['Beam Limiting Device Angle']
        sidSadTuple=dicomModel.sidSadTuple
        magnificationFactor=sidSadTuple[0]/sidSadTuple[1]
        pixelSampling=dicomModel.pixelSampling
        rectangleCoordinates=dicomModel.rectangleCoordinates
        detectedJawPositionsDictionary=dicomModel.detectedJawPositionsDictionary
        jawDiscrepancies=dicomModel.jawDiscrepancies
        
        #Show leaves and jaws detected on the image
        #Draw leaf positions
        imageParser._findLeafPixels(image,ax,
                            detectedJawPositionsDictionary, rectangleCoordinates, 
                            47,
                            pixelSampling,magnificationFactor,
                            collimatorAngle,reportPositioningDeviations,reportAngleDeviations)
        
        #Draw expected jaw positions
        imageParser._plotJawCoordinatesWithFourPoints(ax,rectangleCoordinates,
                                              'red','blue','purple','pink',
                                              1,1,1,1,
                                              'X1jaw','X2jaw','Y1jaw','Y2jaw')
        imageParser._plotJawCoordinatesWithFourPoints(ax,detectedJawPositionsDictionary,
                                              'black','black','black','black',
                                              1,1,1,1,
                                              'X1jawDet','X2jawDet','Y1jawDet','Y2jawDet')
        

        
        reportPositioningDeviationsSorted=dict(sorted(reportPositioningDeviations.items(),key=lambda item: item[1], reverse=True))
        reportAngleDeviationsSorted=dict(sorted(reportAngleDeviations.items(),key=lambda item: item[1], reverse=True))
        

        logger.info('\n\n\n\n\n**********Report*********\n')
        report={}
        report['JawSize']=jawInMm
        report['COL']=collimatorAngle
        report['SID,SAD']=sidSadTuple
        logger.info(report)
        logger.info(f'Leaves distance deviations from iso centerline:\n{reportPositioningDeviationsSorted}')
        logger.info(f'Leaves angle deviations from iso centerline:\n{reportAngleDeviationsSorted}')
        logger.info(f'Jaws detected descrepancies vs expected positions:\n{jawDiscrepancies}')
        logger.info('\n**********END*********\n')
        
        ax.imshow(image,cmap='jet')      
        ax.legend(loc='upper left',bbox_to_anchor=(1,1))
        
        self.canvas.draw()
           
    def displayCenterlTestImage(self, dicomModel, imageParser, logger):
        
        ax=self.ax
        ax.clear()
        ax.set_title(f'Image {self.currentImageIndex+1}')
        
        #reportPositioningDeviations=dicomModel.leafPositionDeviations  
        #reportAngleDeviations=dicomModel.leafAngleDeviations
        
        #Logic for leaf shape delineation
        #Find the image
        if dicomModel:
            image=dicomModel.image
        else:
            return
        
        #Draw iso position and centerlines
        
        dicomReferenceImageCenter=dicomModel.referenceImageCenter
        dicomComparisonImageCenter=dicomModel.comparisonImageCenter
        
        ax.scatter(dicomReferenceImageCenter[0],
                    dicomReferenceImageCenter[1],
                    color='black',s=150,marker='x',
                    label='reference iso')
        ax.scatter(dicomComparisonImageCenter[0],
                    dicomComparisonImageCenter[1],
                    color='blue',s=150,marker='x',
                    label='comparison iso')

        #Draw jaw positions
        jawInMm=dicomModel.jawPositions
        collimatorAngle=dicomModel.metadata['Beam Limiting Device Angle']
        sidSadTuple=dicomModel.sidSadTuple
        magnificationFactor=sidSadTuple[0]/sidSadTuple[1]
        pixelSampling=dicomModel.pixelSampling
        rectangleCoordinates=dicomModel.rectangleCoordinates
        detectedJawPositionsDictionary=dicomModel.detectedJawPositionsDictionary
        jawDiscrepancies=dicomModel.jawDiscrepancies
        
        imageParser._plotJawCoordinatesWithFourPoints(ax,rectangleCoordinates,
                                              'red','blue','purple','pink',
                                              1,1,1,1,
                                              'X1edge com','X2edge com','Y1edge com','Y2edge com')
        imageParser._plotJawCoordinatesWithFourPoints(ax,detectedJawPositionsDictionary,
                                              'black','black','black','black',
                                              1,1,1,1,
                                              'X1edge ref','X2edge ref','Y1edge ref','Y2edge ref')

        ax.imshow(image,cmap='jet')      
        ax.legend(loc='upper left',bbox_to_anchor=(1,1))
        
        self.canvas.draw()
        
    def displayMlcLeakageTestImage(self, dicomModel, imageParser, logger):
        ax=self.ax
        ax.clear()
        ax.set_title(f'Image {self.currentImageIndex+1}')
        
        if dicomModel:
            image=dicomModel.image
        else:
            return
        
        mlcDeviationPixel=dicomModel.mlcDeviationPixel
        ax.scatter(mlcDeviationPixel[1],
                    mlcDeviationPixel[0],
                    color='black',s=150,marker='x',
                    label='Max deviation')

        ax.imshow(image,cmap='jet')      
        ax.legend(loc='upper left',bbox_to_anchor=(1,1))
        
        self.canvas.draw()
        return
   
    def displayDeviationsReport(self,deviationsReport):
        self.logOutput.config(state='normal')
        self.logOutput.delete(1.0,tk.END)
        for leaf,deviation in deviationsReport.items():
            self.logOutput.insert(tk.END, f'{leaf}:{deviation}\n')
        self.logOutput.config(state='disabled')
        
    def displayCenterlTestImageReport(self, distanceDiscrepancyMm):
        self.logOutput.config(state='normal')
        self.logOutput.delete(1.0,tk.END)
        self.logOutput.insert(tk.END, f'Distance discrepancy between centers: {distanceDiscrepancyMm} mm\n')
        self.logOutput.config(state='disabled')
        return
    
    def displayMlcLeakageTestReport(self, pixelCoordinate, pixelDeviationPercentage, meanDeviation):
        self.logOutput.config(state='normal')
        self.logOutput.delete(1.0,tk.END)
        self.logOutput.insert(tk.END, f'Maximim MLC leakage: {round(pixelDeviationPercentage,2)}% at {pixelCoordinate[1],pixelCoordinate[0]}\n')
        self.logOutput.insert(tk.END, f'Mean MLC leakage: {round(meanDeviation,2)}% \n')
        self.logOutput.config(state='disabled')
        return
            
    def showMessage(self,message):
        messagebox.showinfo("Info",message)
        
    def main(self):
        self.root.mainloop()