# USAGE
#perl -w TrimPDBwGrd.pl filename(pdb) dist-cutoff
# No arguments  

my $filein= shift;
my $discut= shift;

my $inpdb = "$filein.pdb";
my $pixpdb = "./pixels.pdb";
my $trimpdb = "$filein-trim.pdb";

open (OUT,">$trimpdb") or die("Cannot open $trimpdb");


my $ica=0;
open (INPPDB,$inpdb) or die("Cannot open $inpdb");
open (INPIX,$pixpdb) or die("Cannot open $pixpdb");

chomp (@lines =<INPPDB>);
foreach (@lines) {
   $thisline = $_;
   if ($thisline =~ / CA /) {
      $ires[$ica]= substr $thisline,22,4;
      $x1[$ica]= substr $thisline,30,8;
      $y1[$ica]= substr $thisline,38,8;
      $z1[$ica]= substr $thisline,46,8;

      $ica++;
 
   }
}
print "ica= ", $ica, "\n";
close INPPDB;
$ig=0;
chomp (@linex =<INPIX>);
foreach (@linex) {
   $thisline = $_;
   $lchat= substr $thisline,0,4;
   if ($lchat eq 'ATOM'){
      $x2[$ig] = substr $thisline,30,8;
      $y2[$ig] = substr $thisline,38,8;
      $z2[$ig]= substr $thisline,46,8;
      $ig++;
   }
}
print "ig= ", $ig, "\n";

$mindst2=$discut**2;    # Save residues with CA at 8 Angstroms from grid (r^2=64.)

$isv=0;
for ($i=0; $i< $ica; $i++){
   $iadd=0;
   for ($j=0; $j< $ig; $j++){
      $dst2 = ($x2[$j]-$x1[$i])**2 + ($y2[$j]-$y1[$i])**2 + ($z2[$j]-$z1[$i])**2;
      if( $dst2 < $mindst2) {
        $iadd=1;
      }
   }
   if ($iadd == 1){
      $stor[$isv]= $ires[$i];
      $isv++;
   }
}
print "isv= ", $isv, "\n";

open (INPDB,$inpdb) or die("Cannot open $inpdb");
chomp (@lines =<INPDB>);
foreach (@lines) {
   $thisline = $_;
   $myres = substr $thisline,22,4;
   for ( $j=0; $j<$isv; $j++ ){
      if ( $myres == $stor[$j] ) {
         print OUT $thisline,"\n";
      }
   }
}

print "Done! \n";

close INPDB;
close INPIX;
close OUT;

exit;
