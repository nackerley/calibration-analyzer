#!/bin/bash
PROGRAMS="arbitrary_analyzer sine_analyzer"

for program in $PROGRAMS; do
  echo "$program --help > ${program}_help.txt"
  $program --help > ${program}_help.txt
done
