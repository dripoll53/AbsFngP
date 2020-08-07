We used the Robust Convolutionary AutoEncoder (RCAE) introduced by 
 R. Chalapathy, A. K. Menon, and S. Chawla;
Preprints: arXiv:1802.06360v2 [cs.LG] 11 Jan 2019 and arXiv:1704.06743v3 [cs.LG] 30 Jul 2017
PDF files of these preprints are included in the directory ./PapersChalapathy.

The original code for RCAE was obtained from: 
https://github.com/raghavchalapathy/oc-nn
(Readers are encouraged to use the original code for their research).

We adapted the RCAE code for our particular application.
The ./OneClassRCAE-code/ directory contain a modified version of the RCAE code
that reads Ab-fingerprint images from ./ADI_DB/dataL1/, including a 2nd set of images for 
cross-validation (../ADI_DB/dataL1/MyTest/).

In the directory ./OneClassRCAE-code, we include a README-ADIAbs file with additional 
explanations, a python file (adi_RCAE_Class0.py) as an example of input file, and the 
output file (adi_RCAE_Class0-Testrun) generated after two epochs.

The file A32JPGs.tgz contains the complete set of Ab images used in our computations. 
The file can be decompressed using the command: tar xvfz A32JPGs.tgz

ADI_DB/dataL1 directory contains a large set of fingerprint images of multiple Abs that 
can be used for generating new input data for additional test runs.

The output images for the normal and anomalous categories are in:
./OneClassRCAE-code/reports/figures/Clss0ADI/L1_RCAE/

The png files start with:
"most_normal-*" and "most_abnormal-*" correspond to the testing set as defined by Chalapathy et al.
"t2-most_normal-*" and "t2-most_abnormal-*" correspond to the images in ./ADI_DB/data1/test 
"t3-most_normal-*" and "t3-most_abnormal-*" correspond to the images in ./ADI_DB/data1/Mytest 

The indexes 1,2,3 and 4 following these strings indicate that the images correspond to values of the 
Chalapathy's lamdba parameter 0, 0.5, 1.0 and 100, respectively.
 
