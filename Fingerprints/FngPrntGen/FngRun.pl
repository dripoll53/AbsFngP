# USAGE
# perl -w FngRun2.pl  CGListDir AbListFile outfile
# CGListDir is the name of the "CG" directory where the PDBs reside
# AbListFile is the output file generated by GenerCG.pl`
use Cwd;
use File::Copy;
#
#-- get current directory
my $pwd = cwd();
print $pwd;
my $CGLst = shift;
   $CGLst =~ s/\/$//;
my $inpAb = shift;
print "Input List from  $inpAb  \n";
# output file 
 $outfil =  shift;

my $nlin=0;
open (INP,$inpAb) or die("Cannot open $inpAb");
chomp (my @lines = <INP>);
foreach (@lines) {
   if ( $nlin > 0 ) {
      @data=split /\s+/ ;
      if (defined($data[0]) ) {
          print $data[0], "\n";  
# name of Ab dir 
          $AbDir = $data[0];  
      }
      else { die("Dont recognize Ab");}
# change directory to antibody
# write bash script (equival ab-submit.sh)
      $bashscrp="bashSub$AbDir.sh";
      copy("tmpBatch",$bashscrp) or die "Copy failed: $!";
      open(my $fh, '>>', $bashscrp) or die "Could not open file '$bashscrp' $!";
      say $fh "bash FngScrps.bsh  $CGLst $AbDir >& $outfil$AbDir";
      close $fh;
# submit job 
      system("sbatch $bashscrp");
      print "system(sbatch $bashscrp)\n";
   }
   $nlin++;
}
