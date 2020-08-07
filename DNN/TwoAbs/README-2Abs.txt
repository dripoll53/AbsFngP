Example of a DNN for recognition of two antibodies from fingerprints. 

Scripts
-------
Scrp1.py;  script to generate and test the DNN model; The file must be edited
            - replacing /home_USER/ so that it points to the correct paths for the train,
               valid and test directories
            - adding the number of EPOCH
            - adding the batch size passed to the optimizer
           To run type: python Scrp1.py >& Name_of_output (e.g., OUT-twoAbsj1)


DATAG/  ;  directory containing the fingerprint images 
           The script setTVt.csh assigns the Ab fingerprints from each Ab to 
           two classes BIND and NBND. It generates subdirectories, named data1 
           (with #, 1, ..., ) for each DNN model produced.
           The datrn.sh script runs setTVt.csh two times using 
           different arguments to generate and fill with images all needed subdirectories.
           Explanation of the required arguments is provided at the top of the setTVT.csh script.  
           The format of the jpg images is 150x150x3

Output files
------------
OUT-twoAbsj1;          this is the main output; The last lines have a statistical summary 
                        for the predictions made from the fingerprints included in the test set.

results-twoAbsj1.csv;  column 1 lists the images used for testing; 
                       column 2 list the predictions produced by the trained DNN model  

checkpoints/;          this directory is used for saving a large file with the best optimized 
                       DNN model. 
                       NOTE: DIR NEEDS TO BE CREATED BEFORE SUBMITTING THE JOB or the run will fail 
