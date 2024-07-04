import tkinter as tk
from tkinter import filedialog

class DICOMfileSelectorClass:
    def __init__(self,logger,title='Please select a file', filetypes=[("All files","*")]):
        self.logger=logger
        self.title=title
        self.filetypes=filetypes
        
    def openFileDialog(self):
        root=tk.Tk()
        root.withdraw()
        filePath=filedialog.askopenfilename(title=self.title,filetypes=self.filetypes)
        root.destroy()
        return filePath
    
    def openFolderDialog(self):
        root=tk.Tk()
        root.withdraw()
        folderPath=filedialog.askdirectory(title='Please select a folder')
        root.destroy()
        return folderPath
    
