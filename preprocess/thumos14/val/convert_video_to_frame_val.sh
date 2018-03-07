# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------

### Generate Image From Video

### please change the corresponding path prefix ${PATH}

avi=`find ${PATH}/THUMOS2014/val/validation -name \*.mp4`

detection=`cat ${PATH}/THUMOS2014/val/TH14_Temporal_annotations_validation/annotation/*.txt|cut -d' ' -f1 | sort | uniq`
# echo $detection

for i in $avi; do
  dir=`echo $i | cut -d. -f1`
  f1=`echo $dir | cut -d/ -f7`
  #echo $dir $f1
  for j in $detection; do
      if [ $f1 = $j ]; then
	  echo $dir $f1 $j
	  mkdir -p ./frames/$f1
	  rm ./frames/$f1/*
	  ffmpeg -i $i -q:v 1 -r 25 ./frames/$f1/image_%5d.jpg
      fi
  done
done

