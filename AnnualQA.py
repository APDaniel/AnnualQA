from Helpers.loggerSetup import setupLogging                 #Hook up the logger
from DICOMparsers.DICOMimageParser import DicomImageParser   #Hook up the DICOM image handler custon class
from Helpers.DICOMfileSelectorClass import DICOMfileSelector #Hook up the custom file selector
import pydicom

#Initialize logger
logger=setupLogging()
  
#Prompt file explorer
fileSelector=DICOMfileSelector(logger,
    title="Select a DICOM file", 
    filetypes=[("DICOM files","*dcm"),("All files","*")])
dicomPath=fileSelector.openFileDialog()

#Open image with jaw positions
dicomImageParser=DicomImageParser(dicomPath,logger)
dicomImageParser.loadDicom()
dicomImageParser.showImageWithJawLeafPositionsAndCalculateDeviations()

