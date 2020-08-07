#/bin/csh
# generate symbolic links to all models
#
# csh FngPset.sh ListAbId.txt  ADI-
#
# 1st argument  is a file listing the Abs
# 2nd argument  is the generic labe of the set of Abs
#
# decoy directories adi number 


set PWDir=(`pwd`)

set xdr =($PWDir/../Models3D)

if( !  -d "$PWDir/RosMod" ) then
    mkdir $PWDir/RosMod
endif
set dcy=(`cat  $1 `)

foreach k ($dcy)

   echo $k
   set kdir=("$PWDir/RosMod/$2"$k)
   if( -d "$kdir" ) then
      /bin/rm -r $kdir
     echo "$kdir deleted "
   endif
   echo " mkdir $kdir"
   mkdir $kdir

# total number of run dirs created 
#   set j = $2 
   @ j = 1 
   @ m  = 0
# while limit is j+1 
   @ j++ 

   @ n  = 1
   while ( $n < $j )
 
#    counter batch jobs

      set i =($xdr/*$k*/grafting/mod*pdb) 
      set a=(`ls $i`)
      foreach i ($a)
      echo  $i "\n" 

         @ m++ 
         ln -s  $i $kdir/model$m.pdb  

      end
      @ n++ 
   end
end

# Generate a list of the Abs in RosMod  

set frstL =("$2Abs 1\n")
ls  -1 -dt RosMod/$2* > ListADIs

sed -i 's|RosMod\/||g' ListADIs
sed -i 's|\/$||'       ListADIs
sed -i "1s|^|$frstL|"  ListADIs
sed  -i '2,$s|$| 2|'  ListADIs
