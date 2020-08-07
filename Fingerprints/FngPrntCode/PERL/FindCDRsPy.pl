# USAGE
#perl -w FindCDRsPy.pl filename 
# One argument: filename of fasta file

use File::Copy "copy";
use File::Copy qw(move);

sub ltrim { my $s = shift; $s =~ s/^\s+//;       return $s };
sub rtrim { my $s = shift; $s =~ s/\s+$//;       return $s };
sub  trim { my $s = shift; $s =~ s/^\s+|\s+$//g; return $s };


my $fasta_arg= shift;

$qryL="query_l.fasta";
$qryH="query_h.fasta";

my $pdb_file= $fasta_arg .'.pdb';
my $fasta_file= $fasta_arg .'.fasta';

copy $pdb_file, "fort.10";
$maindir= $ENV{'FNGRPDIR'}; 
$GETSEQ="$maindir/util/pdbAbseq";
system "$GETSEQ";
move "fort.12", $fasta_file;

my $RegexH1=  "C[A-Z]{1,16}(W)(I|V|F|Y|A|M|L|N|G|W)(R|K|Q|V|N|C|G)(Q|K|H|E|L|R|F|P|S)";      

my $RegexH3=  "C[A-Z]{1,35}(W)(G|A|C)[A-Z]{1,2}(Q|S|G|R)";   
my $RegexL1=  "C[A-Z]{1,17}(WYL|WLQ|WFQ|WYQ|WYH|WVQ|WVR|WWQ|WVK|WYR|WLL|WFL|WVF|WIQ|WYR|WNQ|WHL|WHQ|WYM|WYY|WYK|WYV|WFH|WYE|WCQ)";
my $RegexL3=  "C[A-Z]{1,15}(L|F|V|S)(G|V)[A-Z](G|A|Y)";

my $RegexL1b= "C[A-Z]{1,17}(WFL|WFQ|WHL|WHQ|WHR|WIQ|WLL|WLQ|WNQ|WVF|WVK|WVQ|WVR|WWQ|WYH|WYL|WYM|WYQ|WYR|WYS|WYY)";
my $RegexL3b= "C[A-Z]{1,15}(F|I|L|S|V|W)G[A-Z](G|Y)";

open (INP,$fasta_file) or die("Cannot open $fasta_file");

$icnt=0;
$readChn=0;
chomp (@lines =<INP>);
foreach (@lines) {
   $this_line = $_;
   if ( $this_line =~ /^>/ ) {
      $cha[$icnt]= trim($');
      print "found $cha[$icnt]\n";
      $readChn=1;
   }elsif ($readChn) {
      $chain{$cha[$icnt]}=uc $this_line;
      print "read $cha[$icnt]\n";
      $icnt++;
   }else{
      print "Chain not found\n";
      exit;
   }
}

$jchH=0;
$jchL=0;
$jangn=0;
$iheav=0;
$lght=0; 
for ($i=0;$i<$icnt;$i++){
   if (( $chain{$cha[$i]} =~ /$RegexH1/ ) && ( $chain{$cha[$i]} =~ /$RegexH3/ ) ){
      $match = $&;
      print "match(", $i," ) H Chain= ", $match ,"\n";
      $lgmatch = length $match;
      if ($lgmatch > 0) { 
         $iheav=$i;
         $iihv[$jchH]=$i;
         $jchH++;
      }
   }elsif (( $chain{$cha[$i]} =~ /$RegexL1/ ) && ( $chain{$cha[$i]} =~ /$RegexL3/ ) ){
      $match = $&;
      print "match(", $i," ) L Chain= ", $match ,"\n";
      $lgmatch = length $match;
      if ($lgmatch > 0) {
         $lght=$i; 
         $iilg[$jchL]=$i;
         $jchL++;
      }
      print "Light Chain(",$cha[$lght],")= \n", $chain{$cha[$lght]},"\n";
   }elsif ( ( $chain{$cha[$i]} =~ /$RegexL1b/ ) && ( $chain{$cha[$i]} =~ /$RegexL3/ )
           || ( $chain{$cha[$i]} =~ /$RegexL1/ ) && ( $chain{$cha[$i]} =~ /$RegexL3b/ ) 
           || ( $chain{$cha[$i]} =~ /$RegexL1b/ ) && ( $chain{$cha[$i]} =~ /$RegexL3b/ ) ){
      $match = $&;
      print "Alternative match L Chain= ", $match ,"\n";
      $lgmatch = length $match;
      if ($lgmatch > 0) {
         $lght=$i;
         $iilg[$jchL]=$i;
         $jchL++;
      }
      print "Light Chain(",$cha[$lght],")= \n", $chain{$cha[$lght]},"\n";
   }elsif( $icnt > 2 ){
      print "Not identified Chain ",$i+1,"\n";
      print "This seem to be the antigen, right? = \n", $chain{$cha[$i]},"\n";
      $iatgn[$jangn]=$i;
      $jangn++;
   }else{
      print "FAIL to identify Chain ",$i+1,"\n";
      print $chain{$cha[$i]},"\n";
      exit;
   }

}
if ( $jangn > 1){print "WARNING more than 1 antigen"; }

print "# of Heavy Chain FOUND",$jchH,"\n";
if ( ($jchH == 1 ) && ($jchL == 1) ) {
   print "Light Chain(",$cha[$lght],")= \n", $chain{$cha[$lght]},"\n";
   $chainL = $chain{$cha[$lght]} ;    # length of chainL 
   print "Heavy Chain(",$cha[$iheav],")= \n", $chain{$cha[$iheav]},"\n";
   $chainH = $chain{$cha[$iheav]} ;    # length of chainH 
}elsif ($jchH == 0 ) {
   print "Heavy Chain NOT FOUND\n";
   exit;       
}elsif ( ($jchH == 2 ) && ($jchL == 0) ) {
   print "Double assignment of chain H, try to discern which one is L\n";
   print "jchH =",$jchH ,"\n";
   $ki=0;
   for ($k=0;$k<$jchH;$k++){
      $ki = $iihv[$k];
      print "ki =",$ki ,"\n";
      if (( $chain{$cha[$ki]} =~ /$RegexL1/ ) && ( $chain{$cha[$ki]} =~ /$RegexL3/ ) ){
         $match = $&;
         print "match L Chain= ", $match ,"\n";
         $lgmatch = length $match;
         if ($lgmatch > 0) {
            $lght=$ki; 
            print "Light Chain(",$cha[$lght],")= \n", $chain{$cha[$lght]},"\n";
            $chainL = $chain{$cha[$lght]} ;    # length of chainL 
            if ($k == 0) {
               $iheav=$iihv[1];
            }else {   
               $iheav=$iihv[0];
            }
            print "Heavy Chain(",$cha[$iheav],")= \n", $chain{$cha[$iheav]},"\n";
            $chainH = $chain{$cha[$iheav]} ;    # length of chainH 
         }
      }elsif (( $chain{$cha[$ki]} =~ /$RegexL1b/ ) && ( $chain{$cha[$ki]} =~ /$RegexL3/ )
           || ( $chain{$cha[$ki]} =~ /$RegexL1b/ ) && ( $chain{$cha[$ki]} =~ /$RegexL3b/ ) ){
         $match = $&;
         print "Alternative match L Chain= ", $match ,"\n";
         $lgmatch = length $match;
         if ($lgmatch > 0) {
            $lght=$ki; 
            print "Light Chain(",$cha[$lght],")= \n", $chain{$cha[$lght]},"\n";
            $chainL = $chain{$cha[$lght]} ;    # length of chainL 
            if ($k == 0) {
               $iheav=$iihv[1];
            }else {   
               $iheav=$iihv[0];
            }
            print "Heavy Chain(",$cha[$iheav],")= \n", $chain{$cha[$iheav]},"\n";
            $chainH = $chain{$cha[$iheav]} ;    # length of chainH 
         }
      }
   }
   if ( $lght == 0 ) {
      print "FAILED to identify Chain L \n";
      exit;
   }
}

$lgchainH1 = length $chainH;    # length of chainH 
if ( $chainH =~ /$RegexH1/ ) {
  $matcH1 = $&;
  $lgmatcH1 = length $matcH1;
  $cdrH1= substr $matcH1,4,$lgmatcH1 - 8;

  $lgH1 = length $cdrH1;    # length of H1 
  $idxStrH1 = 4 + length $`;  # index of first letter of H1 in chainH 
  $idxFinH1 =  $idxStrH1 + $lgH1 -1; # index of last letter of H1 in chainH

}else{
      print "H1 not found\n";
}

if ( $chainH =~ /$RegexH3/ ) {
  $matcH3 = $&;
  $lgmatcH3 = length $matcH3;
  $cdrH3= substr $&,3,$lgmatcH3 - 7;


  $lgH3 = length $cdrH3;    # length of H3 
  $idxStrH3 = 3 + length $`;  # index of first letter of H3 in chainH 
  $idxFinH3 =  $idxStrH3 + $lgH3 -1; # index of last letter of H3 in chainH

}else{
      print "H3 not found\n";
}


if (( $chainL =~ /$RegexL1/ ) || ( $chainL =~ /$RegexL1b/ )) {
  $matcL1 = $&;
  $lgmatcL1 = length $matcL1;
  $cdrL1= substr $matcL1,1,$lgmatcL1 - 4;

  $lgL1 = length $cdrL1;    # length of L1
  $idxStrL1 = 1 + length $`;  # index of first letter of L1 in chainL
  $idxFinL1 =  $idxStrL1 + $lgL1 - 1; # index of last letter of L1 in chainL

}else{
      print "L1 not found\n";
}


if ( ( $chainL =~ /$RegexL3/ ) || ($chainL =~ /$RegexL3b/)) {
  $matcL3 = $&;
  $lgmatcL3 = length $matcL3;
  $cdrL3= substr $matcL3,1,$lgmatcL3 - 5;

  $lgL3 = length $cdrL3;    # length of L3 
  $idxStrL3 = 1 + length $`;  # index of first letter of L3 in chainL 
  $idxFinL3 =  $idxStrL3 + $lgL3 -1; # index of last letter of L3 in chainL

}else{
      print "L3 not found\n";
}

# define H2
$idxStrH2 = $idxFinH1 +15; 
$idxFinH2 =  $idxStrH3 - 33 ; # index of last letter of L1 in chainL
$lgH2 = $idxFinH2 - $idxStrH2 +1; 
$cdrH2 = substr  $chainH,$idxStrH2,$lgH2; 

# define L2
$idxStrL2 = $idxFinL1 +16; 
$idxFinL2 =  $idxStrL2 +  6; # index of last letter of L1 in chainL
$lgL2 = $idxFinL2 - $idxStrL2 +1;
$cdrL2 = substr  $chainL,$idxStrL2,$lgL2;


#--------
print "HEAVY:\n";
open (QRYH,">$qryH") or die("Cannot open $qryH");
print QRYH ">query_h\n";
print QRYH $chainH,"\n";
close QRYH;


print "cdr H1= ", $cdrH1;
print "  starts at ", (substr $chainH,$idxStrH1,1), $idxStrH1+1,
            ", ends at ", (substr $chainH,$idxFinH1,1), $idxStrH1 + $lgH1, "; H1 length= ",$lgH1,"\n";

print "cdr H2= ", $cdrH2;
print "  starts at ", (substr $chainH,$idxStrH2,1), $idxStrH2+1,
            ", ends at ", (substr $chainH,$idxFinH2,1), $idxStrH2 + $lgH2, "; H2 length= ",$lgH2,"\n";


print "cdr H3= ", $cdrH3;
print "  starts at ", (substr $chainH,$idxStrH3,1), $idxStrH3+1,
            ", ends at ", (substr $chainH,$idxFinH3,1), $idxStrH3 + $lgH3, "; H3 length= ",$lgH3,"\n";

#--------

print "LIGHT:\n";
open (QRYL,">$qryL") or die("Cannot open $qryL");
print QRYL ">query_l\n";
print QRYL $chainL,"\n";
close QRYL;


print "cdr L1= ", $cdrL1;
print "  starts at ",(substr $chainL,$idxStrL1,1), $idxStrL1+1 ,
            ", ends at ", (substr $chainL,$idxFinL1,1), $idxStrL1 + $lgL1, "; L1 length= ",$lgL1,"\n";

print "cdr L2= ", $cdrL2;
print "  starts at ", (substr $chainL,$idxStrL2,1), $idxStrL2+1,
            ", ends at ", (substr $chainL,$idxFinL2,1), $idxStrL2 + $lgL2, "; L2 length= ",$lgL2,"\n";

print "cdr L3= ", $cdrL3;
print "  starts at ", (substr $chainL,$idxStrL3,1), $idxStrL3+1,
            ", ends at ",(substr $chainL,$idxFinL3,1),$idxStrL3 + $lgL3,"; length= ",$lgL3,"\n";


close INP;


# use pymol to align the input file with coordinates (now fort.9) with
# the template (Ab with paratope on the grid)
# Depending on which chain (H or L) is listed first in fort.9,
# we will use templateHL.pdb or templateLH.pdb
# If chain for the antigen is listed first, this script may fail.
# 
$myfle = "./fort.9";
 
copy  $myfle, "./myf.pdb";

$rcnt=0;
$irsOld=0;
$monOld="";
$k=0;
my $nwtmpdb= "NewTmpl.pdb";
open (OUTT,">$nwtmpdb") or die("Cannot open $nwtmpdb");

open (TMPPDB,"$myfle") or die("Cannot open $myfle");
chomp (@lines =<TMPPDB>);
foreach (@lines) {
   $this_line = $_;
   $k++;
  
   $lchat= substr $this_line,0,4;

   if ($lchat eq 'ATOM')  {
      $mon= substr $this_line,21,1;
      $ires= int(substr $this_line,22,4);
      if ( ($mon eq $cha[$iheav]) || ($mon eq $cha[$lght]) ) {
         if ( $monOld eq ""){
             $monOld=$mon;
         }elsif($mon ne $monOld) {
            $rcnt=0;
            $monOld=$mon;
         }elsif ($ires ne $irsOld) {
            $rcnt++;
            $irsOld=$ires;
         }
         if ($rcnt < 105 ) {
            print OUTT $this_line,"\n";
         }
      }
   }
}


if ($iheav < $lght) {
system("pymol -cq alg1.pml ");   
}else{
system("pymol -cq alg2.pml ");   
}

my $cdrpdb = "CDR.pdb";
if ( $jangn > 0) {
   my $atgnpdb = "ANTIGN.pdb";

   open (OUT2,">$atgnpdb") or die("Cannot open $atgnpdb");
}

open (OUT,">$cdrpdb") or die("Cannot open $cdrpdb");

my $AbfilH= 'Hchain';
open (OUT3,">$AbfilH") or die("Cannot open $AbfilH");

my $AbfilL= 'Lchain';
open (OUT4,">$AbfilL") or die("Cannot open $AbfilL");

my $modpdb= "algnF.pdb";
open (INPPDB,$modpdb) or die("Cannot open $modpdb");
chomp (@lines =<INPPDB>);
foreach (@lines) {
   $this_line = $_;
#ATOM   1013  N   GLU H   1      51.502  45.716 154.968  1.00 40.77           N
   $lchat= substr $this_line,0,4; 
   if ($lchat eq 'ATOM'){
      $mon= substr $this_line,21,1; 
      $ires= int substr $this_line,22,4; 
      if ($mon eq $cha[$iheav]) { 
         substr($this_line,21,1) = "H"; 
         print OUT3 $this_line,"\n";
      }elsif ($mon eq $cha[$lght]) { 
         substr($this_line,21,1) = "L"; 
         print OUT4 $this_line,"\n";
      }
      if (($mon eq $cha[$iheav]) && 
         (($ires > $idxStrH1) && ( $ires <= $idxFinH1 +1 )) ||
         (($ires > $idxStrH2) && ( $ires <= $idxFinH2 +1 )) ||
         (($ires > $idxStrH3) && ( $ires <= $idxFinH3 +1 )) ) {
            print OUT $this_line,"\n";
      }
      $lres= $ires  - $lgchainH1; 
      if (($mon eq $cha[$lght]) && 
         (( $lres > $idxStrL1) && ( $lres <= $idxFinL1 +1 )) ||
         (( $lres > $idxStrL2) && ( $lres <= $idxFinL2 +1 )) ||
         (( $lres > $idxStrL3) && ( $lres <= $idxFinL3 +1 )) ) {
            print OUT $this_line,"\n";
      }
      
      for ($n=0;$n<$jangn;$n++){
         if ($mon eq $cha[$iatgn[$n]]) {
            print OUT2 $this_line,"\n";
         }
      }
   }
}
print "Delete fort* files! \n";

$Abfile= $fasta_arg ."-HL.pdb";
system (" cat $AbfilH $AbfilL > $Abfile "); 
print "Done! \n";

close INPPDB;
close OUT;
close OUT2;
close OUT3;
close OUT4;

exit;

