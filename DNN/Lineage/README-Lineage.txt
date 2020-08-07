Example of DNN for antibody classification. The DNN is trained to
recognize Abs binding  to one out of three possible epitopes on EBOV  

Scripts
-------
Scrp1.py;  script to generate and test the DNN model
           The file should be edited replacing /home_USER/ to point to the proper path
           To run type: python Scrp1.py >& Name_of_output (e.g., OUT-A10r1Ejb1)

DATAG/  ;  directory containing the fingerprints 
           The script setTVT.csh assigns the Ab fingerprints from each class to 
           the proper directories. It generates subdirectories, named data# 
           (with #, 1, ..., ) for each DNN model produced.
           The batchA.sh script runs setTVT.csh multiple times in a batch queue with using 
           different arguments to generate and fill with images all needed subdirectories.
           Explanation of the required arguments is provided at the top of the setTVT.csh script.  
           The format of the jpg images is 150x150x3

Output files
------------
OUT-A10r1Ejb1;         this is the main output; The last lines have a statistical summary 
                       for the predictions made for the fingerprints included in the test set.

results-A10r1Ejb1.csv; column 1 lists the images used for testing; 
                       column 2 list the predictions produced by the trained DNN model  

checkpoints/;          this directory is used for saving a large file with the best optimized 
                       DNN model. 
                       NOTE: DIR NEEDS TO BE CREATED BEFORE SUBMITTING THE JOB or the run will fail 
