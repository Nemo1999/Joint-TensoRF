#!/bin/bash
# blender
if [[ ! -f nerf_synthetic.zip ]] ; then
    gdown --id 118JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG # download nerf_synthetic.zip
fi
unzip nerf_synthetic.zip
rm -f nerf_synthetic.zip
mv nerf_synthetic data/blender
# llff
if [[ ! -f nerf_llff_data.zip ]] ; then
    gdown --id 116VnMcF1KJYxN9QId6TClMsZRahHNMW5g # download nerf_llff_data.zip
fi
unzip nerf_llff_data.zip
rm -f nerf_llff_data.zip
mv nerf_llff_data data/llff
