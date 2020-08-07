#
# Color using  Donald's reduce residue-code
#
# (script for PyMol)
# Daniel Ripoll, 10/10/2017
#
mstop
dss
hide
# show cartoon, all
color gray30, all
select  pos, (resn arg+lys+his)
color marine, pos
disable pos
select  neg, (resn glu+asp)
color red, neg
disable neg
select  hydr, (resn ala+ile+leu+val)
color grey70, hydr
disable hydr
select  Amid, (resn gln+asn)
color cyan, Amid
disable Ohyd
select  Ohyd, (resn ser+thr)
color orange, Ohyd
disable Sulf
select  Sulf, (resn cys+met)
color yellow, Sulf
disable neg
select  Feny, (resn phe)
color green, Feny
disable Feny
select  Glyc, (resn gly)
color white, Glyc
disable Glyc
select  Prol, (resn pro)
color brown, Prol
disable Prol
select  Tryp, (resn trp)
color magenta, Tryp
disable Tryp
select  Tyro, (resn tyr)
color violetpurple, Tyro
disable Tyro
#set cartoon_smooth_loops,0
