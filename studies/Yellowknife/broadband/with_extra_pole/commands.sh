rm *.log
arbitrary_analyzer --plot basic --fit 2 --out-of-band 5 2 \
    --response-pattern XX.YKAT1.xml \
    --calibration-response-file XX.YKAT1.xml \
    --pattern ../temperature/PRB_-36.00_Output.mseed \
    --fmt EPS
