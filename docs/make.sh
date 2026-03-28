#!/bin/bash
PROGRAMS="calan sine_analyzer preprocess_ppsd plot_ppsd rotate_mseed drum_plot pick_class peak_detect availability station_quake"

for program in $PROGRAMS; do
  echo "$program --help > ${program}_help.txt"
  $program --help > ${program}_help.txt
done
