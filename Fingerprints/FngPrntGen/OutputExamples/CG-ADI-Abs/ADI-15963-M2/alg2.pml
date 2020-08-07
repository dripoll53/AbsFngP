load ./templateLH.pdb
load ./NewTmpl.pdb
align NewTmpl, templateLH
load ./myf.pdb
align myf, NewTmpl
save ./algnF.pdb,((myf))
quit
