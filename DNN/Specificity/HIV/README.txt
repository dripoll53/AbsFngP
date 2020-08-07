Example of DNN for antibody classification. The DNN is trained to recognize Abs binding 
       to one out of two possible epitopes on HIV-1 gp120/gp41 complex 

Scripts
-------
Scrp1.py;  Script to generate and test the DNN model; The file must be edited 
            - replacing /home_USER/ so that it points to the correct paths for the train, 
               valid and test directories  
            - adding the number of EPOCH
            - adding the batch size passed to the optimizer
           To run type: python Scrp1.py >& Name_of_output 

batch1.sh; Example of batch submission script

data1/  ;  Directory containing the Ab fingerprints; image format 150x150x3
           (NOTE: The two classes of antibodies are labelled SITE1 and SITE2)

Output files
------------
OUT-HIVb1;           This is the main output; The last linest the percentage of 
                     correct predictions made for the fingerprints on the test set.

results-HIVb1.csv; Column 1 lists the images used for testing (./data1/test/test_folder); 
                     column 2 list the preditions produced by the trained DNN model  

checkpoints/;        This directory is used for saving a large file with the best optimized 
                     DNN model 
                     NOTE: DIR NEEDS TO BE CREATED BEFORE SUBMITTING THE JOB or the run will fail
