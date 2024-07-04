import threading
import pydicom
from Helpers.loggerSetup import setupLogging
from Models.DICOMmodel import Model
from Views.MainView import View
import re

class Controller:
    def __init__(self):
        #self.root=root
        self.view=View(self)
        self.logger=setupLogging()
        self.model=Model(self.logger)
        self.view.setupLogging(self.logger)
        
    def selectFolder(self):
        folderPath=self.model.selectFolder()
        if folderPath:
            self.view.showMessage(f'Selected folder: {folderPath}')
            
    def selectFile(self):
        filePath=self.model.selectFile()
        if filePath:
            self.view.showMessage(f'Selected file: {filePath}')
            self.view.displayImageFromDICOMfile(filePath)
            
     
    def processDicomFiles(self):
        threading.Thread(target=self._processDicomFiles).start()
        
    def _processDicomFiles(self):
        self.model.dicomFileModels.clear()
        result=self.model.runLeafAligmentTest()
        if result:
            self.view.showMessage('Jaw-leaf alignment test competed')
            modelToDisplay=self.model.getCurrentDicomImageModel()
            imageParser=self.model.dicomImageParser    
            self.view.displayImage(modelToDisplay,imageParser,self.logger)
            self.view.displayDeviationsReport(self.collectLeafDeviations())
        else:
            self.view.showMessage('No folder selected or no DICOM files found')
    
    def performJawAlignmentTest(self):
        threading.Thread(target=self._performJawAlignmentTest).start()    
        
    def _performJawAlignmentTest(self):
        self.model.dicomFileModels.clear()
        result=self.model.runJawAligmentTest()
        if result:
            self.view.showMessage('Jaw alignment test competed')
            modelToDisplay=self.model.getCurrentDicomImageModel()
            imageParser=self.model.dicomImageParser    
            self.view.displayImage(modelToDisplay,imageParser,self.logger)
            self.view.displayDeviationsReport(self.collectJawDeviations())
        else:
            self.view.showMessage('No folder selected or no DICOM files found')
            
    def performJawMlcCenterTest(self):
        threading.Thread(target=self._performJawMlcCenterTest).start()
            
    def _performJawMlcCenterTest(self):
        self.model.dicomFileModels.clear()
        result=self.model.runJawMlcCenterTest()
        if result:
            self.view.showMessage('Jaw/MLC center alignment test competed')
            modelToDisplay=self.model.getCurrentDicomImageModel()
            imageParser=self.model.dicomImageParser    
            self.view.displayCenterlTestImage(modelToDisplay,imageParser,self.logger)
            if modelToDisplay:
                distanceDiscrepancyMm=modelToDisplay.centerDeviation
            else:
                return
            self.view.displayCenterlTestImageReport(distanceDiscrepancyMm)
        else:
            self.view.showMessage('No DICOM files found')
            
    def performMlcLeakageTest(self):
        threading.Thread(target=self._performMlcLeakageTest).start()
        
    def _performMlcLeakageTest(self):
        self.model.dicomFileModels.clear()
        result=self.model.runMlcLeakageTest()
        if result:
            self.view.showMessage('MLC leakage test completed')
            modelToDisplay=self.model.getCurrentDicomImageModel()
            imageParser=self.model.dicomImageParser
            self.view.displayMlcLeakageTestImage(modelToDisplay,imageParser,self.logger)
            if modelToDisplay:
                pixelCoordinate=modelToDisplay.mlcDeviationPixel
                pixelDeviationPercentage=modelToDisplay.pixelDeviationPercentage
                meanDeviation=modelToDisplay.meanDeviation
            else:
                return
            self.view.displayMlcLeakageTestReport(pixelCoordinate, pixelDeviationPercentage,meanDeviation)
        else:
            self.view.showMessage('No DICOM files found')
            
    def collectLeafDeviations(self):
        deviations={}
        i=0
        dicomFileModels=self.model.getDicomFileModels()
        for dicomModel in dicomFileModels:
            i+=1
            for key,value in dicomModel.leafAngleDeviations.items():
                if key in deviations:
                    key+=' DUPLICATED LEAF DETECTED, image# '+str(i)
                    deviations[key]=value
                else:
                    deviations[key]=value
        sortedDeviations=dict(sorted(deviations.items(), key=lambda item:item[1],reverse=True))
        
        max_key, max_deviation = max(sortedDeviations.items(), key=lambda item: abs(float(item[1])))
        
        sortedDeviations['\nMax deviation detected for ']=f'{key} {max_deviation}deg'
        return sortedDeviations
    
    def collectJawDeviations(self):
        deviations={}
        i=0
        dicomFileModels=self.model.getDicomFileModels()
        for dicomModel in dicomFileModels:
            i+=1
            for key,value in dicomModel.jawDiscrepancies.items():
                if key in deviations:
                    #key+=' DUPLICATED JAW DETECTED, image# '+str(i)
                    deviations[key]=value
                else:
                    deviations[key]=value
        sortedDeviations=dict(sorted(deviations.items(), key=lambda item:item[0],reverse=True))
        
        max_key, max_deviation = max(sortedDeviations.items(), key=lambda item: abs(float(item[1])))
        
        sortedDeviations['\nMax deviation detected for ']=f'{key} {max_deviation}mm'
        
        return sortedDeviations
    
    def main(self):
        self.view.main()

