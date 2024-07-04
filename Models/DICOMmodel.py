import os
from io import BytesIO
from tkinter import messagebox
from Helpers.DICOMfileSelectorClass import DICOMfileSelectorClass
from Helpers.DICOMimageParser import DicomImageParser
from Models.DICOMfileModel import DicomFileModel



class Model:
    def __init__(self,logger):
        self.logger=logger
        self.fileSelector=DICOMfileSelectorClass(self.logger,'Please select a folder containing DICOM files')
        self.folderPath=None
        self.filePath=None
        self.currentIndex=0
        self.dicomFileModels=[]
        self.dicomImageParser=None
        self.jawDeviationsSummary={}
        self.leafDeviationsSummary={}
        
    def selectFolder(self):
        self.folderPath=self.fileSelector.openFolderDialog()
        self.filePath=None
        self.dicomFileModels.clear()
        if self.folderPath:
            self.logger.info(f'Selected forlder: {self.folderPath}')
        else:
            self.logger.info('No folder selected')
        return self.folderPath
    
    def selectFile(self):
        self.filePath=self.fileSelector.openFileDialog()
        self.folderPath=None
        self.dicomFileModels.clear()
        if self.filePath:
            self.logger.info(f'Selected file: {self.filePath}')
        else:
            self.logger.info('No DICOM file selected')
        return self.filePath

    def initializeParser(self, filePath):
        try:
            self.dicomImageParser=DicomImageParser(filePath, self.logger)
            self.dicomImageParser._loadDicom()
            self.logger.info(f'DICOM image parser initialize for the file: {filePath}')
        except Exception as ex:
            self.logger.error(f'Error initializing image parser for the file: {filePath}')
    
    def runLeafAligmentTest(self):
        if self.folderPath:
            self.runLeafAlingmentTestForImagesInFolder(self.folderPath)
            return 'folder'
        if self.filePath:
            self.runLeafAlingmentTestForAsingleImage(self.filePath)
            return 'file'
        else:
            self.logger.warning('No files selected for processing')
            return None

    def runJawAligmentTest(self):
        if self.folderPath:
            self.runJawAlingmentTestForImagesInFolder(self.folderPath)
            return 'folder'
        if self.filePath:
            self.runJawAlingmentTestForAsingleImage(self.filePath)
            return 'file'
        else:
            self.logger.warning('No files selected for processing')
            return None
        
    def runLeafAlingmentTestForImagesInFolder(self,folderPath):
        for file in os.listdir(folderPath):
            filePath=os.path.join(folderPath,file)
            if file.endswith('.dcm') and os.path.isfile(filePath):
                self.runLeafAlingmentTestForAsingleImage(filePath)
                    
    def runLeafAlingmentTestForAsingleImage(self,filePath):
        try:
            self.initializeParser(filePath)
            dicomModel=self.dicomImageParser.runJawLeafAlingmentTest()
            self.dicomFileModels.append(dicomModel)
        except Exception as ex:
            self.logger.error(f'Error processing file: {filePath},\n{str(ex)}')
            
    def runJawAlingmentTestForImagesInFolder(self,folderPath):
        for file in os.listdir(folderPath):
            filePath=os.path.join(folderPath,file)
            if file.endswith('.dcm') and os.path.isfile(filePath):
                    self.runJawAlingmentTestForAsingleImage(filePath)
                    
    def runJawAlingmentTestForAsingleImage(self,filePath):
        try:
            self.initializeParser(filePath)
            dicomModel=self.dicomImageParser.runJawAlingmentTest()
            self.dicomFileModels.append(dicomModel)
        except Exception as ex:
            self.logger.error(f'Error processing file: {filePath},\n{str(ex)}')
            
    def runJawMlcCenterTest(self):
        if self.folderPath:
            self.logger.warning('Select a reference DICOM file')
            return None
        if self.filePath:
            self.runJawMlcCenterTestForSingleImage(self.filePath)
            return 'file'
        else:
            self.logger.warning('No files selected for processing, Jaw/MLC center test aborted')
            return None
        
    def runJawMlcCenterTestForSingleImage(self,filePath):
        try:
            self.initializeParser(filePath)
            dicomModel=self.dicomImageParser.runJawMlcCenterTest()
            self.dicomFileModels.append(dicomModel)
        except Exception as ex:
            self.logger.error(f'Error processing file: {filePath},\n{str(ex)}')
            
    def runMlcLeakageTest(self):
        if self.folderPath:
            self.logger.warning('Select DICOM file for MLC field')
            return None
        if self.filePath:
            self.runMlcLeakageTestForSingleImage(self.filePath)
            return 'file'
        else:
            self.logger.warning('No files selected for processing, MLC leakage test aborted')
            return None
        
    def runMlcLeakageTestForSingleImage(self,filePath):
        try:
            self.initializeParser(filePath)
            dicomModel=self.dicomImageParser.runMlcLeakageTest()
            self.dicomFileModels.append(dicomModel)
        except Exception as ex:
            self.logger.error(f'Error processing file: {filePath},\n{str(ex)}')
            
    def getCurrentDicomImageModel(self):
        test=len(self.getDicomFileModels())
        if not self.dicomFileModels:
            self.logger.warning('No DICOM files captured in DICOM file models')
            return None
        else:
            return self.dicomFileModels[self.currentIndex]
            
    def displayNextImage(self):
        if self.currentIndex<len(self.dicomFileModels)-1:
            self.currentIndex+=1
        return self.getCurrentDicomImageModel()
    
    def displayPreviousImage(self):
        if self.currentIndex>0:
            self.currentIndex-=1
        return self.getCurrentDicomImageModel()
    
    def getDicomFileModels(self):
        return self.dicomFileModels
    
    def getImageFromModel(self,index):
        if index<len(self.dicomFileModels):
            return self.dicomFileModels[index]._image
        return None
