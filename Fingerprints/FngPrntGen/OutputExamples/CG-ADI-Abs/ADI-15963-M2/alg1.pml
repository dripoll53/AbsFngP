load ./templateHL.pdb
load ./NewTmpl.pdb
align NewTmpl, templateHL
load ./myf.pdb
align myf, NewTmpl
save ./algnF.pdb,((myf))
quit
