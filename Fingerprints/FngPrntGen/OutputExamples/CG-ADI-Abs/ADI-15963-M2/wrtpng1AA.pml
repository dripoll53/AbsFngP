load ABDY-trim-CRG.pdb
load ABDY-trim-GRD.pdb
#set_view (\
#      0.872809708,   -0.485131115,   -0.052901149,\
#      0.377088845,    0.739273310,   -0.557885230,\
#      0.309758067,    0.466985673,    0.828194320,\
#     -0.000215959,   -0.000378550, -158.108779907,\
#     -5.220702171,  -44.283676147,   58.506698608,\
#    122.769180298,  191.415863037,  -20.000000000 )
set_view (\
     0.872809708,   -0.485131115,   -0.052901149,\
     0.377088845,    0.739273310,   -0.557885230,\
     0.309758067,    0.466985673,    0.828194320,\
    -0.000173934,   -0.000514860, -158.371902466,\
    -3.883765221,  -46.153148651,   57.015079498,\
   122.769180298,  191.415863037,  -20.000000000 )
viewport 1800, 1200
util.color_deep("gray70", 'all')
cmd.hide("lines"     ,"all")
spectrum b, red_white_blue, ABDY-trim-GRD, minimum=-1.0, maximum=1.0
spectrum b, red_white_blue, ABDY-trim-CRG, minimum=-1.0, maximum=1.0
cmd.enable('ABDY-trim-GRD',1)
cmd.show("mesh"      ,"ABDY-trim-GRD")
#ray 2500,1800
# ray 1250,900
# png ABDY-FngrPrnt-mesh.png
cmd.enable('ABDY-trim-CRG',1)
cmd.show("sticks"    ,"ABDY-trim-CRG")
cmd.hide("(ABDY-trim-CRG and hydro)")
#ray 2500,1800
ray 1250,900
png ABDY-FngrPrnt1.png
cmd.disable('ABDY-trim-CRG')
#save ABDY-FngrPrnt.pse,format=pse
#cmd.save('''ABDY-FngrPrnt.pse''','','pse',quiet=0)
@colorbyAAcode.pml
cmd.enable('ABDY-trim-GRD',1)
cmd.show("mesh"      ,"ABDY-trim-GRD")
# ray 1250,900
# png ABDY-FngrPrnt-mesh-byAA.png
cmd.enable('ABDY-trim-CRG',1)
cmd.show("sticks"    ,"ABDY-trim-CRG")
cmd.hide("(ABDY-trim-CRG and hydro)")
ray 1250,900
png ABDY-FngrPrnt1-byAA.png
quit
