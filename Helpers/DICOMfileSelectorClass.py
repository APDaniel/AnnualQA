import tkinter as tk
from tkinter import filedialog as fd
import logging

class DICOMfileSelector:
    def __init__(self,logger,title="Select a DICOM file", filetypes=None):
        #Initialize the FileSelector
        self.logger=logger
        self.title=title
        self.filetypes=filetypes if filetypes else [("All files","*")]
    def openFileDialog(self):
        #Prompt user to select a file through the file dialog
        root=tk.Tk()
        root.withdraw()
        filePath=fd.askopenfilename(title=self.title,filetypes=self.filetypes)
        
        if filePath:
            self.logger.info(f"Selected file:{filePath}")
        else:
            self.logger.warning(f"No file selected...")
        return filePath

