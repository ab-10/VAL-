#!/usr/bin/env bash

while true; do
    read -p "The ShapeNet SDF dataset size is 72 GB, are you sure you want to download? [yn]" yn
    case $yn in
        [Yy]* ) curl https://ls7-data.cs.tu-dortmund.de/shape_net/ShapeNet_SDF.tar.gz -o ShapeNet_SDF.tar.gz; break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done
tar xrf ShapeNet_SDF.tar.gz data/ShapeNet_SDF -C data/
