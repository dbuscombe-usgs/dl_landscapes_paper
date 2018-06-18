## written by Dr Daniel Buscombe
## Northern Arizona University
## daniel.buscombe@nau.edu

## from: https://github.com/dbuscombe-usgs/dl_landscapes_paper
## If you find these codes/data useful, please cite:
## Buscombe and Ritchie (2018) "Landscape classification with deep neural networks", submitted to Geosciences June 2018
## https://eartharxiv.org/5mx3c

date

python retrain.py --image_dir train/tile_224 \
    --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/1 \
    --how_many_training_steps 1000 --learning_rate 0.01 --output_labels labels.txt --output_graph gc_mobilenetv2_224_1000_001.pb

rm -rf /c/tmp/bottleneck
rm -rf /c/tmp/checkpoint
rm -rf /c/tmp/retrain_logs
rm -rf /c/tmp/_retrain*

date


date

python retrain.py --image_dir train/tile_96 \
    --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/classification/1 \
    --how_many_training_steps 1000 --learning_rate 0.01 --output_labels labels.txt --output_graph gc_mobilenetv2_96_1000_001.pb

rm -rf /c/tmp/bottleneck
rm -rf /c/tmp/checkpoint
rm -rf /c/tmp/retrain_logs
rm -rf /c/tmp/_retrain*

date

