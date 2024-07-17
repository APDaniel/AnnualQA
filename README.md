PD Annual QA Tool
The “PD Annual QA Tool” (PD Tool) script has been designed using the Python programming language. It is specifically tailored to meet the needs of the annual QA program at the CN Center for the North, covering functionality gaps in available software solutions. 
 
To run the PD Tool, locate the “.exe” file in the folder: …\AnnualQA\dist\AnnualQA. It can be run from a network folder, although processing times may vary. For best efficiency, run the executable directly on the workstation. 

The PD Tool can perform 4 different tests:
•	Leaf/Jaw alignment Test: Utilizes the DBSCAN algorithm (Density-Based Spatial Clustering of Applications with Noise), a machine learning algorithm used to identify clusters in spatial data (DICOM-extracted pixel data) by grouping together points that are closely packed together while marking points in low-density regions as outliers. The algorithm is effective in finding clusters of arbitrary shapes and robust to noise, without needing to specify the number of clusters a priori. 
Custom algorithms are designed for leaf number detection, angle, and positioning deviations calculations. Due to the pixel nature of analyzed images, the expected precision for leaf deviation detection is within 2 pixels, providing a sustainable result below the CPQR recommended action level of 1 degrees angle deviation. 
Image acquisition on the linear accelerator is automated in Service Mode (or Developer Mode, as it aligns better with the department’s workflow for custom image acquisitions) for each control point. The plan file was manually modified to instruct the system to acquire one high-resolution MV image for each control point. Each control point contains data for one leaf per leaf bank protruding well into the field, as per CPQR requirements. Due to imager size and limitations of the custom algorithms, 56 leaf pairs can be analyzed out of 60 existing in the Varian Millennium MLC. 
This test can be applied to either a folder or a separate file. 
•	Jaw alignment Test: An ad-hoc test that determines jaw deviations compared to the expected digital positions. It outputs angle and distance measurements. The algorithm accounts for different imager heights and collimator angles. Analysis of a single image may not be informative, as there is no built-in functionality to calculate imager skew and misalignment. However, running the test on multiple images taken at various collimator angles can provide reliable feedback on jaw calibration precision. 
This test can be applied to either a folder or a separate file. 
•	Jaw/MLC center Test: Calculates the Euclidean distance between the centers of two images: reference and comparison. It converts pixel distance to millimeters with respect to imager vertical position and collimator angle, determining uncertainties between MLC-defined and jaw-defined field. Unlike the Jaw Alignment Test, the centers of images are determined through custom calculations rather by capturing data from the DICOM isocenter position. 
This test can only be applied to a separate file.
•	MLC leakage Test: Analyzes pixel data between two images. It runs through pixels within a threshold value of 80% max intensity, capturing deviations above the tolerance of 0.5%. The test outputs maximum detected deviation and average discrepancy along the detected pixels. 
This test can be only applied to a separate file.


 
