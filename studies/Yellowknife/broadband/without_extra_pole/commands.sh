rm *.log
arbitrary_analyzer --plot basic --fit 1 --out-of-band 5 2 \
    --response-pattern XX.YKAT1.xml \
    --calibration-response-file XX.YKAT1.xml \
    --pattern ../temperature/PRB_-36.00_Output.mseed \
    --fmt EPS
mv transfer_function_nominal_system_removed_XX.YKAT1..SH_20200110.1529.eps \
    Figure5_left_fit_without_extra_pole.eps
gs -dSAFER -dBATCH -dNOPAUSE -dEPSCrop -r300 -sDEVICE=pngalpha \
    -sOutputFile=Figure5_left_fit_without_extra_pole.png \
    Figure5_left_fit_without_extra_pole.eps
