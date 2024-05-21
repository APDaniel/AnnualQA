import os
import logging
import logging.config
from datetime import datetime
import getpass

#Function to create/find a log directory
def createLogDirectory():
    logDir=os.path.join(os.getcwd(),'Logs')
    os.makedirs(logDir,exist_ok=True)
    return logDir

#Define logger cofiguration through a dictionary (tried a separate file, did not work well, decided to use dictionary)
def setupLogging():
    logDirectory=createLogDirectory() #Create/find direcroty: "Logs"
    #Define timestamp
    username=getpass.getuser()
    timestamp=datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    logFilePath=os.path.join(logDirectory,f'annualQAtool_{username}_{timestamp}.txt')
    #Define dictionary with configuration for the logger
    loggingConfig={
        'version':1,
        'disable_existing_loggers':False,
        'formatters':{
            'standardFile':{
                'format':'---%(levelname)s---%(asctime)s---\"%(message)s\"',
                'datefmt':'%Y-%m-%d %H:%M:%S'
            },
            'standardConsole':{
                'format':'%(asctime)s...Logger reports %(levelname)s: \"%(message)s\"',
                'datefmt':'%Y-%m-%d %H:%M:%S'}
        },
        'handlers':{
            'console':{
                'class':'logging.StreamHandler',
                'level':'DEBUG',
                'formatter':'standardConsole',
                'stream':'ext://sys.stdout'},
            'file':{
                'class':'logging.FileHandler',
                'level':'DEBUG',
                'formatter':'standardFile',
                'filename':logFilePath,
                'mode':'w'}
            },
        'loggers':{
            'annualQAtoolLogger':{
                'level':'DEBUG',
                'handlers':['console','file'],
                'propogate':False}    
            }
    }
    logging.config.dictConfig(loggingConfig)
    return logging.getLogger('annualQAtoolLogger')
    
        
    