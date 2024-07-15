from datetime import datetime
import os
from io import BytesIO
from tkinter import messagebox
import re
import pydicom
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
        
    def organizeDICOMfilesFunction(self):
        if self.folderPath:
            self.runOrganizeDICOMfilesFunctionForImagesInFolder(self.folderPath)
            return 'folder'
        if self.filePath:
            self.logger.error('Please select folder with DICOM files containing pixel data')
            return None
        else:
            self.logger.warning('No files selected for processing')
            return None
    def extractNumbers(self,s):
        match=re.search(r'-(d\+)\.dcm$',s)
        return int(match.group(1)) if match else float('inf')
    
    def runOrganizeDICOMfilesFunctionForImagesInFolder(self, folderPath):
        try:
            #Get a sorted list of DICOM files in the folder
            self.logger.info(f'Called a function to extract pixel data for each leaf pair')
            dicomFiles=sorted([f for f in os.listdir(folderPath) if f.endswith('.dcm')], key=self.extractNumbers)
        
            #Drop the first file as it does not contain relevant pixel data
            dicomFiles=dicomFiles[1:]
        
            #Create output folder with a timestamp
            self.logger.info('Creating a folder for organized DICOM files')
            timeStamp=datetime.now().strftime("%d-%b-%Y-%H_%M_%S")
            outputFolder=os.path.join(folderPath, f'ProcessedDICOM_{timeStamp}')
            os.makedirs(outputFolder, exist_ok=True)
        
            #Create DICOM files with pixel data for each leaf pair separately
            for i in range(len(dicomFiles)-1):
                file1Path=os.path.join(folderPath,dicomFiles[i])
                file2Path=os.path.join(folderPath,dicomFiles[i+1])
            
                self.logger.info(f'Reading DICOM file {file1Path}')
                dicom1=pydicom.dcmread(file1Path)
            
                self.logger.info(f'Reading DICOM file {file2Path}')
                dicom2=pydicom.dcmread(file2Path)
            
                self.logger.info(f'Extracting pixel array for the file: {file1Path}')
                pixelData1=dicom1.pixel_array
            
                self.logger.info(f'Extracting pixel array for the file: {file2Path}')
                pixelData2=dicom2.pixel_array
            
                self.logger.info('Substracting pixel data')
                substractedPixelData=pixelData2-pixelData1

                #Fix negative values if any
                substractedPixelData[substractedPixelData<0]=0
            
                self.logger.info(f'Copying DICOM file: {file2Path}')
                newDicom=dicom2.copy()
            
                self.logger.info('Inserting pixel data')
                newDicom.PixelData=substractedPixelData
            
                newDicomFileName=f'DICOM_PDtool_{dicomFiles[i+1]}'
                newFilePath=os.path.join(outputFolder,newDicomFileName)
            
                self.logger.info(f'Saving new dicom file: {newFilePath}')
                newDicom.save_as(newFilePath)
            
                dicomModel=self.runLeafAlingmentTestForAsingleImage(newFilePath)
                self.dicomFileModels.append(dicomModel)
                
        except Exception as ex:
            self.logger.error(f'Error organizing files: {ex}')
            return


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
