# USAGE
# #perl -w GenerCG.pl CGlist_file
use Cwd;
use File::Basename;
use File::Spec;

my $CGlstfil = shift;
my $Fgdir=cwd();
print "Current working directory: $Fgdir\n";
my $ABMOD="$Fgdir/RosMod/";
my $Subdirfiles="model";

open (INP,$CGlstfil) or die("Cannot open $CGlstfil");
chomp (@lines =<INP>);
foreach (@lines) {
   @data=split /\s+/ ;
   if(defined $data[1]) {
      if($data[1] == 1 ) {
         $CGdir = "$Fgdir/CG-$data[0]";
         system("mkdir $CGdir");
      }elsif ($data[1] == 2) {
         my $pdbfiles=$ABMOD . $data[0] ."/". $Subdirfiles;
         my @thesepdbs =glob "$pdbfiles*"; 
         foreach $file (@thesepdbs){
            if( $file =~ /$Subdirfiles/) {
               my $nwpdbfil="$CGdir/$data[0]-M$'";
               system("ln -s $file $nwpdbfil");
            }
         }
      } 
   } 
}
