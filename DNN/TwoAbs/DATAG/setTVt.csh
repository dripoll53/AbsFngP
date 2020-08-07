#!/bin/tcsh
# Script fetches jpg files from PNGsGP.... dirs  and regroup them into train,validation and  test
# Names of Dirs containing the BINDER and NONBINDER jpg file are hard-code
# We divide the train valid and test set for BINDER and NON-BINDERS using the ADI numbers
#
# The total number of BINDER (or NON-BINDERS) Abs are splitted into two parts: (1) (train and valid)
#  and (2) test, so that "test" contains Abs that were not seen during training.
#
#To run
# csh  setTVt.sh dataDir file-train-valid-test  MaxTVTimgs  perc-to-test-Set  perc-to-validation-Set  r(r= 1, 2) 
#
# ARG:
#   1- data dir to be created (eg, "data1" , "data2" )
#   2- file-train-valid-test file with the list of Ab decoys to split among training, validation, and test sets
#    **IMPORTANT** The name for list of non-binders should contain a substring "NonB"
#     otherwise BINDER will be assumed
#   3- MaxTVTimgs:    maximum no. of images per decoy to be used  
#   4- %to-test-Set   (integer 0 to 100)  percentage of images of each Abs going to the test, rest to train/val
#   5- %to-val-Set   (integer 0 to 100) fraction of TV set going to validation 
#   6- r  (r= 1, 2)
###
### example
#  csh  setTVd.sh dataDir ListDecoys    500   25   20   1
#
set DirStr=`pwd`
set datdir=$1
if (! -d ./$datdir ) then
  mkdir $datdir
endif
cd $datdir
set PWDir=`pwd`
echo "Fulanito START"

# determine if this list correspond to Binders
set JPGds="/p/work1/dripoll/NN/ALLPNGs"    # for mustang
#set JPGds="/p/work1/workspace/dripoll/NN/"    # for thunder
#   set JPGdir="/home/dripoll/Work/Antibodies/NN/TSTBNRD"
if ( $2 =~ *NonB* ) then
   if (! -d $PWDir/train ) then
      mkdir $PWDir/train
      mkdir $PWDir/train/NBND
      mkdir $PWDir/train/BIND
   else
       echo "rm  files in $PWDir/train/NBND/"
      /bin/rm -f $PWDir/train/NBND/*g
   endif
   if (! -d $PWDir/valid ) then
      mkdir $PWDir/valid
      mkdir $PWDir/valid/NBND
      mkdir $PWDir/valid/BIND
   else
       echo "rm  files in $PWDir/valid/NBND/"
      /bin/rm -f $PWDir/valid/NBND/*g
   endif
   if (! -d $PWDir/test ) then
      mkdir $PWDir/test
      mkdir $PWDir/test/test_folder
   else
       echo "rm  files in $PWDir/test/NBND/"
      /bin/rm -f $PWDir/test/NBND/*g
   endif
   set TRNdir="$PWDir/train/NBND/"
   set TRNNam="NBND"
   set VALdir="$PWDir/valid/NBND/"
   set VALNam="NBND"
   set TSTdir="$PWDir/test/test_folder/"
   set TSTNam="NBND"
   set ListTV="./ListTRNNBND.txt"
   set ListTS="./ListTSTNBND.txt"
   set NwFile="./NewTVTlist-NBND"
   touch  $NwFile
   set FFile="./Failed_TVTlist-NBND"
   touch  $FFile

else   # BIND set
   if (! -d $PWDir/train ) then
      mkdir $PWDir/train
      mkdir $PWDir/train/NBND
      mkdir $PWDir/train/BIND
   else
       echo "rm  files in $PWDir/train/BIND/"
      /bin/rm -f $PWDir/train/BIND/*g
   endif
   if (! -d $PWDir/valid ) then
      mkdir $PWDir/valid
      mkdir $PWDir/valid/NBND
      mkdir $PWDir/valid/BIND
   else
       echo "rm  files in $PWDir/valid/BIND/"
      /bin/rm -f $PWDir/valid/BIND/*g
   endif
   if (! -d $PWDir/test ) then
      mkdir $PWDir/test
      mkdir $PWDir/test/test_folder
   else
       echo "rm  files in $PWDir/test/BIND/"
      /bin/rm -f $PWDir/test/BIND/*g
   endif
   set TRNdir="$PWDir/train/BIND/"
   set TRNNam="BIND"
   set VALdir="$PWDir/valid/BIND/"
   set VALNam="BIND"
   set TSTdir="$PWDir/test/test_folder/"
   set TSTNam="BIND"
   set ListTV="./ListTRNBIND.txt"
   set ListTS="./ListTSTBIND.txt"
   set NwFile="./NewTVTlist-BND"
   touch  $NwFile
   set FFile="./Failed_TVTlist-BND"
   touch  $FFile
endif
#set  p = "100"
echo "Fulanito AGAIN"
@  p=100
set listTVT = ""
#set listTS = ""
set imgTV = ""
 
set lstdcy = (`cat  $DirStr/$2`)
set Totdcy=$#lstdcy
echo "Totdcy= " $Totdcy
@ i = 1
set listTVT=( $listTVT $lstdcy[$i] )
echo "type jpg= " $6
# 1 for JPG1 or 2 for JPG2
set tjpg=(`echo $6`)

#echo "tjpg= " $tjpg

set mxT=(`echo  $3`)
echo "Max total imgs= " $mxT


@ imgtrn = 0
@ imgval = 0
@ imgtst = 0
   foreach k ($listTVT)
     echo ADI-$k
     set allDrs=(`ls -d $JPGds/PNG*/JPG$tjpg-ADI-$k`)
     set jpgdr=""
     foreach j ($allDrs)
        echo "j= " $j
        if ( -d $j ) then

#        echo "$j found"
           set jpgdr=(`echo $j:q`)
#        echo " jpgdr = $jpgdr "
           break
        endif
     end
     if ( $jpgdr == "" ) then
        echo "$jpgdr does not exist"
        echo "$k " >> $FFile
        exit
     else

#     echo "Fulanito is Here"
        echo "$k " >> $NwFile
        echo 'set jpgs=(`ls -1' $jpgdr'*g`)'
        set jpgs=(`ls -1 $j*g`)

#       if ( $s == "V" ) then
           set maxtot=$#jpgs
#     echo  "1 maxtot $maxtot mxT $mxT"

           if ( $maxtot > $mxT ) then
              set maxtot=(`echo $mxT`)
              echo  "Reset maxtot to $maxtot"
           endif
#

# this % to test
           @ ntts = ( $maxtot * $4  )
           @ ntts = ( $ntts / $p )
           echo "to Test ntts= " $ntts
# remaining to Train/Valid
           @ nttv = ( $maxtot - $ntts  )
#   set ntrn=(`printf '%.0f\n' $ntrn`)   # convet to integer
           echo "to TR/VL nttv= " $nttv
#
#         # generate a list of random numbers
#         # shuf -i MIN-MAX -n COUNT
#         # Where:
#         # MIN and MAX are the lower and upper limits of the range of numbers, respectively.
#         # COUNT is the number of lines (random numbers) to display.
#
           set ltts =(`shuf -i 0-$maxtot -n $ntts`)
           @ i = 1
           @ z = 0
           while  ($i <= $maxtot )
              set fnd=0
              foreach m ($ltts)
                 if (! $?m) then
                   echo "m is undefined"
                 else
                   if ($m == $i) then
#                    echo "i= $i to test set"
                     set fnd=1
                   endif
                 endif
              end
              if ($fnd > 0 ) then
                  @ imgtst++
#                 echo 'ln -s '$jpgs[$i] $TSTdir'/'$TSTNam'.'$imgtst'.jpg'
#                 echo "set oldf =  $jpgs[$i]"
                  eval "set oldf =  $jpgs[$i]"
#                 echo "set lnkf =  $TSTdir/$TSTNam.$imgtst.jpg"
                  eval "set lnkf =  $TSTdir/$TSTNam.$imgtst.jpg"
                  ln -s  $oldf $lnkf

              else 
#this image to train/val
#                  echo "To train-val list  $jpgs[$i]"
                  set imgTV=($imgTV $jpgs[$i])
                  @ z++
              endif

              @ i++
           end

           set  nvvx=$#imgTV
           if ($nvvx  !=  $nttv ) then
                echo "z= $z nvvx= $nvvx is not nttv = $nttv  #imgtst = $imgtst"
                exit
           else 
                echo "nttv = $nttv #imgtst = $imgtst"
           endif 
           @ ntrn = ( $nttv * ( $p - $5)  )
           @ ntrn = ( $ntrn / $p )
           echo "ntrn= $ntrn"
           set ltrn =(`shuf -i 0-$nttv -n $ntrn`)
           echo " To training set ltrn= $ltrn"
#          exit

           @ i = 1
           while  ($i <= $nttv )
              set fnd=0
              foreach m ($ltrn)
                 if (! $?m) then
                   echo "m is undefined"
                 else
                   if ($m == $i) then
#                    echo "i= $i to train set"
                     set fnd=1
                   endif
                 endif
              end
              if ($fnd > 0 ) then
                  @ imgtrn++
#                 echo 'ln -s '$imgTV[$i] $TRNdir'/'$TRNNam'.'$imgtrn'.jpg'
                  echo "set oldf =  $imgTV[$i]"
                  eval "set oldf =  $imgTV[$i]"
                  echo "set lnkf =  $TRNdir/$TRNNam.$imgtrn.jpg"
                  eval "set lnkf =  $TRNdir/$TRNNam.$imgtrn.jpg"
#           eval "set oldf =  "\$"$jpgs[$i]"
                  ln -s  $oldf $lnkf
              else
                  @ imgval++
#                 echo 'ln -s  '$imgTV[$i] $VALdir'/'$VALNam'.'$imgval'.jpg'
                  echo "set oldf =  $imgTV[$i]"
                  eval "set oldf =  $imgTV[$i]"
                  echo "set lnkf =  $VALdir/$VALNam.$imgval.jpg"
                  eval "set lnkf =  $VALdir/$VALNam.$imgval.jpg"
                  ln -s  $oldf $lnkf
              endif
              @ i++
           end #while

           echo "to TR= $imgtrn  to VL = $imgval  To Test = $imgtst"

     endif   # if ( $jpgdr == "" )

   end   # end foreach k

cd ../
