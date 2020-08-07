#!/bin/tcsh
# This is an example of a script to allocate the fingerprints for training, validation and test.
#
# Script fetches jpg files from a database of images (e.g., JPGDir) where the fingeprints 
# for a particular Ab (e.g., "AbId") are located. It expects that the image files sit under 
# a subdir "./JPGDir/JPG1-AbId". The label "AbId" is used for Ab identification.
# The fingerprints are used to fill the train, validation, andd test dirs under a main directory "dataN" (1, 2 ... N) that will be used to produce and test DNN model # N. Dir data N contains subdirectories train, valid & test. train and valid cotains the images separated by class (e.g., ./data1/train/SITE1/*jpg). images from all classes used for testing are stored in one folder (e.g., data1/test/test_folder/SITE1-file10.jpg)

# First, this script divides the fingerprints between two sets, train-valitation and test sets, 
# using the Abs names and numbers ("AbId") 
# Than, thee total number of fingerprints of a given class assigned to train-valitation set are 
# splitted into two parts ("train" and "valid" subdirectories)
#
#  The test dir contains fingerprints of Abs that were not seen during training.
# To run
#  csh  setTVT.sh   dataDir file-train-valid-test   perc-to-train-valid-Set   perc-to-train-Set  MxTVimgs   MxTSimgs   r(r= 1, 2) 
#
# ARG:
#   1- dataDir is a directory where training validation and test set are define for the specific run. 
#      If the directory does not exist it will be created 
#   2- file-train-valid-test file is a list of Ab decoys correspondint to a specific category (i.e., ListSITE1)
#      the name must contain "SITE" followed by a number (1, 2 ... or n) spcifying the category, with n 
#      the total # of categories  
#   3- %to-train/valid-Set   (integer 0 to 100) percentage of images of each Abs going to the training & 
#      validation set (rest goes to test)
#   4- %to-train-Set   (integer 0 to 100) percentage of images of each Abs going to the training set 
#      (rest goes to validation) 
#   5- MxTVimgs:    maximum no. of images per decoy to be used for Training and validation
#       (use to restrict the # of images of Abs for which we have a large # of images
#   6- MxTSimgs:  maximum no. of images per decoy to be used for test
#   7- r  (r= 1, 2) 
###
### example
#  csh  setTVd.sh dataDir ListDecoys    50   75   10    20   1  
#
# ******************* IMPORTANT **************************
# Set the variable JPGds to the directory containing the fingerprint images of the Abs

# ******************* IMPORTANT **************************
set st="SITE"
set DirStr=`pwd`
set datdir=$1
if (! -d ./$datdir ) then
  mkdir $datdir
endif
cd $datdir
set PWDir=`pwd`
#echo "Fulanito START"

# JPGDir Is the main directory containing all the fingerprints; The script expect
# subdirectories containing the fingerprints of the Abs.
# All the fingerprits of an Ab must be contained in a sudirectory labelled 
# JPG1-AbId.  Example if AbId = "ADI-15772", then, the name of the subdirectory 
# should be  JPG1-ADI-15772    
set JPGds="$PWDir/../../../JPGDir/"
#  echo "JPGds= " $JPGds 

if ( $2 =~ *$st* ) then

# Category Number can now be more than 1 digit 
   set stLng=`echo -n $st| wc -m` 
#  echo $stLng 
   set strlg=`echo -n $2 | wc -m` 
#  echo $strlg 
#
   @ nml =  -( $strlg - $stLng )
#  echo $nml 
    
  set lCatg=(`echo "$2" | sed "s/$st//"`)
# or
#  set lCatg=(`echo "$2" | rev | cut -c $nml | rev `)

#  echo $lCatg

   if (! $lCatg ) then
        echo "Bad input"
        exit
   else
        echo "Found Category= " $lCatg
   endif
else
        echo "Word <SITE> is missing in arg #2 " $2
    exit
endif
set catgr=$st$lCatg
echo $catgr

if (! -d $PWDir/train ) then
   mkdir $PWDir/train
endif
if (! -d $PWDir/train/$catgr ) then
   mkdir $PWDir/train/$catgr
else
   /bin/rm -f $PWDir/train/$catgr/*g
endif
if (! -d $PWDir/valid ) then
   mkdir $PWDir/valid
endif
if (! -d $PWDir/valid/$catgr ) then
   mkdir $PWDir/valid/$catgr
else
   /bin/rm -f $PWDir/valid/$catgr/*g
endif
if (! -d $PWDir/test ) then
   mkdir $PWDir/test
endif
if (! -d $PWDir/test/test_folder ) then
   mkdir $PWDir/test/test_folder
else
   /bin/rm -f $PWDir/test/$catgr*g
endif
set TRNdir="$PWDir/train/$catgr/"
set TRNNam="$catgr"
set VALdir="$PWDir/valid/$catgr/"
set VALNam="$catgr"
set TSTdir="$PWDir/test/test_folder/"
set TSTNam="$catgr"
set ListTV="./ListTRN$catgr.txt"
set ListTS="./ListTST$catgr.txt"
set NwFile="./NewTVTlist-$catgr"
touch  $NwFile
set FFile="./Failed_TVTlist-$catgr"
touch  $FFile
#######

set  p="100"
#echo "Fulanito AGAIN"
set listTV = ""
set listTS = ""
 
set lstdcy = (`cat  $DirStr/$2`)
set Totdcy=$#lstdcy

if ( $Totdcy < 2) then
   echo "ERROR: $catgr contains ONE Ab only"
   exit
endif

if ( $Totdcy > 2) then
   @ nttd = ( $Totdcy * $3  )
   @ nttd = ( $nttd / $p )
else
   echo "WARNING: $catgr contains TWO Abs "
   @ nttd = 1
endif

 echo "Total Abs=" $Totdcy  "train/valid Abs="  $nttd

#set lttd =(`shuf -i 0-$Totdcy -n $nttd`)
set lttd =(`shuf -i 1-$Totdcy -n $nttd`)
echo $lttd
@ i = 1
while  ($i <= $Totdcy )
   set fnd=0
   foreach m ($lttd)
      if (! $?m) then
         echo "m is undefined in lttd"
      else
         if ($m == $i) then
            set fnd=1
            break
         endif
      endif
   end
   if ($fnd > 0 ) then
#     echo "i= $i to TV set"
      set listTV=( $listTV $lstdcy[$i] )
   else
#      echo "i= $i to TS set"
      set listTS=( $listTS $lstdcy[$i] )
   endif
   
   @ i++
end #while

echo $listTV > $ListTV
echo "listTV=" $listTV 
#echo " "
echo $listTS > $ListTS
echo "listTS=" $listTS 


echo "type jpg= " $7
# 1 for JPG1 or 2 for JPG2
set tjpg=(`echo $7`)

#echo "tjpg= " $tjpg

set mxT=(`echo  $5`)
echo "Max (TRAIN/VALID) total imgs per Ab= " $mxT

@ maxtst = $6
echo "Max TEST total imgs per Ab= " $maxtst

@ imgtrn = 0
@ imgval = 0
foreach s ( V S )
   if ( $s == "V" ) then
      set crrlst=(`echo $listTV`) 
   else
      set crrlst=(`echo $listTS`) 
      @ imgtst = 0
   endif
   echo $crrlst 
   foreach k ($crrlst)
     echo aC-$k

#     echo "Fulanito is Here"
        echo "$k " >> $NwFile
#       echo 'set jpgs=(`ls -1' $jpgdr'*g`)'

        echo "$JPGds/JPG$tjpg*$k/*g"
        set jpgs=(`ls -1 $JPGds/JPG$tjpg*$k/*g`)

        if ( $s == "V" ) then
           set maxtrn=$#jpgs
           echo  "maxtrn= " $maxtrn

           if ( $maxtrn > $mxT ) then
              echo "maxtrn was " $maxtrn " New value=" $mxT 
              set maxtrn=(`echo $mxT`)
           endif
#     echo  "2 maxtrn $maxtrn"
#
#     set  p="100"
#      @ ntrn = ( $maxtrn * $4  )
           @ ntrn = ( $maxtrn * $4  )
           @ ntrn = ( $ntrn / $p )
#   set ntrn=(`printf '%.0f\n' $ntrn`)   # convet to integer
           echo "ntrn= " $ntrn
#
#         # generate a list of random numbers
#         # shuf -i MIN-MAX -n COUNT
#         # Where:
#         # MIN and MAX are the lower and upper limits of the range of numbers, respectively.
#         # COUNT is the number of lines (random numbers) to display.
#
#           set ltrn =(`shuf -i 0-$maxtrn -n $ntrn`)
           set ltrn =(`shuf -i 1-$maxtrn -n $ntrn`)
           echo $ltrn
           @ i = 1
           while  ($i <= $maxtrn )
              set fnd=0
              foreach m ($ltrn)
                 if (! $?m) then
                   echo "m is undefined"
                 else
                   if ($m == $i) then
                     echo "i= $i to train set"
                     set fnd=1
                   endif
                 endif
              end
              if ($fnd > 0 ) then
                  @ imgtrn++
                  echo 'ln -s '$jpgs[$i] $TRNdir'/'$TRNNam'.'$imgtrn'.jpg'
#                 echo "set oldf =  $jpgs[$i]"
                  eval "set oldf =  $jpgs[$i]"
#                 echo "set lnkf =  $TRNdir/$TRNNam.$imgtrn.jpg"
                  eval "set lnkf =  $TRNdir/$TRNNam.$imgtrn.jpg"
#           eval "set oldf =  "\$"$jpgs[$i]"
                  ln -s  $oldf $lnkf
              else
                  @ imgval++
                  echo 'ln -s  '$jpgs[$i] $VALdir'/'$VALNam'.'$imgval'.jpg'
#                 echo "set oldf =  $jpgs[$i]"
                  eval "set oldf =  $jpgs[$i]"
#                 echo "set lnkf =  $VALdir/$VALNam.$imgval.jpg"
                  eval "set lnkf =  $VALdir/$VALNam.$imgval.jpg"
                  ln -s  $oldf $lnkf
              endif
              @ i++
           end #while

        else # this for $s=S
           if ( $#jpgs < $maxtst) then
              @ MAXt =  $#jpgs
                echo  "MAXt = " $MAXt
           else
              @ MAXt =  $maxtst
                echo  "New MAXt = " $MAXt
           endif
#          echo  "jpgdr=" $jpgdr "; maxtst" $maxtst
           @ i = 1
           while  ($i <= $MAXt )
              @ imgtst++
              echo 'ln -s '$jpgs[$i] $TSTdir'/'$TSTNam'.'$imgtst'.jpg'
#         echo "set oldf =  $jpgs[$i]"
              eval "set oldf =  $jpgs[$i]"
#         echo "set lnkf =  $TSTdir/$TSTNam.$imgtst.jpg"
              eval "set lnkf =  $TSTdir/$TSTNam.$imgtst.jpg"
              ln -s  $oldf $lnkf

              @ i++
           end
        endif # $s

#    endif   # if ( $jpgdr == "" )

   end   # end foreach k
end   # end foreach s

cd ../
