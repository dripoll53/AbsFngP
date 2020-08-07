# USAGE
#perl -w chrg.pl  in_file dist_cutoff
# arguments  



$plus1chg  =  1.0;
$neg1chg   = -1.0;
$zerochg   =  0.0;

$OcpVal   =  "  1.00";


my $filen= shift;
my $cutdst= shift;
my $is_antgn= shift;

my $inppdb = "$filen.pdb";
my $crgpdb = "$filen-CRG.pdb";
my $cdrPRJpdb = "$filen-GRD.pdb";

open (OUT,">$crgpdb") or die("Cannot open $crgpdb");

open (INPPDB,$inppdb) or die("Cannot open $inppdb");

open (OUT2,">$cdrPRJpdb") or die("Cannot open $cdrPRJpdb");

my $iat=0;

print "is_antingen= ",$is_antgn,"\n"; 
chomp (@lines =<INPPDB>);
foreach (@lines) {
   $this_line = $_;
   $lchat= substr $this_line,0,4; 
   $chgres = 1;
   if ($lchat eq 'ATOM'){
      $resnam= substr $this_line,17,3; 
      if ( ($resnam eq "ASP") || ($resnam eq "GLU") ){
        $myChg= $neg1chg; 
      }elsif ( ($resnam eq "ARG") || ($resnam eq "LYS") ){
        $myChg= $plus1chg; 
      }else{
        $chgres = 0;
        $myChg= $zerochg; 
      }

      $x = substr $this_line,30,8;
      $y = substr $this_line,38,8;
      $z = substr $this_line,46,8;
      $r2 = $x * $x  + $y * $y +  $z * $z ;

      $myrd = sqrt($r2);
      $invr = 1.0 / $myrd;

# I am using a Radius of 100A for the surface
# the projection of point $r =($x, $y, $z) has coordinates: 
      $Rad= 100.;
# color atoms gradually depending on their location 
      $diffr= $myrd - $Rad;  
      if (abs($diffr) > $cutdst){
        if ($diffr > 0){
           $diffr = $cutdst;
        }elsif ($diffr > 0){
           $diffr = (-1)* $cutdst;
        } 
      } 
# $fctr is a number between 0 & 1.
      if ($is_antgn == 1 ){ 
         $fctr= 0.5 - $diffr/(2.0 * $cutdst);  
      }else{
         $fctr= 0.5 + $diffr/(2.0 * $cutdst);  
      }
      $mybf =$fctr*$myChg;
      $bf[$iat] = sprintf('%6.2f',$mybf) ;
      $newline= substr($this_line,0,60) . $bf[$iat]; 
      print OUT $newline,"\n";

      $x1[$iat] =  $Rad * $x * $invr;
      $y1[$iat] =  $Rad * $y * $invr;
      $z1[$iat] =  $Rad * $z * $invr;

      printf OUT2 ("%30s%8.3f%8.3f%8.3f%6s%6s\n", substr($this_line,0,30), $x1[$iat], $y1[$iat],$z1[$iat], $OcpVal,$bf[$iat]); 
 
      $iat++;
   }

}
print "Done! \n";

close INPPDB;

close OUT;
close OUT2;

exit;
