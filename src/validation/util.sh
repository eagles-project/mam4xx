#! /bin/bash

while read line; do
  f="${line}/*driver.cpp"
  code $f
done < files.txt
