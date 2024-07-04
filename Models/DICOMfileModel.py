class DicomFileModel:
    def __init__(self,_metadata,
                 _jawPositions,_rectangleCoordinates,_detectedJawPositionsDictionary,
                 _jawDiscrepancies,_pixelSampling,
                 _leafPositionDeviations, _leafAngleDeviations,
                 _image,_sidSadTuple,
                 _referenceImageCenter, _comparisonImageCenter,_centerDeviation,
                 _mlcDeviationPixel, _pixelDeviationPercentage, _meanDeviation):
        
        self.metadata=_metadata
        self.jawPositions=_jawPositions
        self.rectangleCoordinates=_rectangleCoordinates
        self.detectedJawPositionsDictionary=_detectedJawPositionsDictionary
        self.jawDiscrepancies=_jawDiscrepancies
        self.pixelSampling=_pixelSampling
        self.leafPositionDeviations=_leafPositionDeviations
        self.leafAngleDeviations=_leafAngleDeviations
        self.image=_image
        self.sidSadTuple=_sidSadTuple
        self.referenceImageCenter=_referenceImageCenter
        self.comparisonImageCenter=_comparisonImageCenter
        self.centerDeviation=_centerDeviation
        self.mlcDeviationPixel=_mlcDeviationPixel
        self.pixelDeviationPercentage=_pixelDeviationPercentage
        self.meanDeviation=_meanDeviation
 
