plot [0:100001] "20211216_Result_bicgstab_jacobian/LoadDisplCurve_grains.txt" u 1:2 w p lw 2, "/home/zit.bam.de/vkindrac/labtools/TEST-SUITE-2016/Abaqus/R-3D-cutback/neper/AbaqusValidation/creep/LoadDisplacementAbaqus.txt" u 1:5 w lp
#, "LoadDisplCurve_grains.txt" u 1:2 w p lw 2

pause -1

#plot wall clock time vs. simulation time
#plot '<paste abaqus.times.txt grains.time.txt' u 1:3

#or

#plot '<paste abaqus.times.txt grains.time.txt' u 1:3 w l, '/home/zit.bam.de/vkindrac/labtools/TEST-SUITE-2016/Abaqus/R-3D-cutback/neper/AbaqusValidation/creep_single_cpu/abaqusIncrements.txt' u 1:($1>0 ? $4:NaN) w l

set terminal pngcairo size 700, 450 enhanced font 'Verdana,12'
set output '../../../examples/polycrystal/CompareDispl.png'

set xlabel 'time, 10^3 s' font 'Verdana,14'
set ylabel 'displacement, mm' font 'Verdana,14'

set key bottom right

plot "LoadDisplCurve_grains.txt" u ($1/1000):2 w p pt 6 lw 2 title 'FEniCS', "/home/zit.bam.de/vkindrac/labtools/TEST-SUITE-2016/Abaqus/R-3D-cutback/neper/AbaqusValidation/creep/LoadDisplacementAbaqus.txt" u ($1/1000):5 w l lw 2 title 'Abaqus'

unset terminal
set terminal pngcairo size 700, 450 enhanced font 'Verdana,12'
set output '../../../examples/polycrystal/CompareCompEfforts.png'

set xlabel 'time, 10^3 s' font 'Verdana,14'
set ylabel 'wall clock time, h' font 'Verdana,14'

set key bottom right

plot '<paste abaqus.times.txt grains.time.txt' u ($1/1000):($3/3600) w l lw 2 title 'FEniCS', '/home/zit.bam.de/vkindrac/labtools/TEST-SUITE-2016/Abaqus/R-3D-cutback/neper/AbaqusValidation/creep_single_cpu/abaqusIncrements.txt' u ($1/1000):($1>0 ? $4/3600:NaN) w l lw 2 title 'Abaqus'