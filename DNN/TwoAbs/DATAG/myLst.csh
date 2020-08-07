#!/bin/tcsh

# to run
#  csh  createData.sh
@ N = 10
@ i = 1
while  ( $i <= $N )
echo " "
echo "data$i"
echo "test-Bind"
ls -1 data$i/test/test_folder/BIND.* | wc -l
echo "test-NoBind"
ls -1 data$i/test/test_folder/NBND.* | wc -l
echo "train-NoBind"
ls -1 data$i/train/NBND//NBND.* | wc -l
echo "train-Bind"
ls -1 data$i/train/BIND//BIND.* | wc -l
echo "valid-Bind"
ls -1 data$i/valid//BIND/*g | wc -l
echo "valid-NoBind"
ls -1 data$i/valid//NBND/*g | wc -l

@ i++
end
