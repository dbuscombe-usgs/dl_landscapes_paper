
date

/c/Users/ddb265/AppData/Local/Continuum/anaconda3/envs/tfpy35/python retrain.py --image_dir train/tile_224 \
    --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/1 \
    --how_many_training_steps 1000 --learning_rate 0.01 --output_labels labels.txt --output_graph gc_mobilenetv2_224_1000_001.pb

rm -rf /c/tmp/bottleneck
rm -rf /c/tmp/checkpoint
rm -rf /c/tmp/retrain_logs
rm -rf /c/tmp/_retrain*

date
## 100.0% (N=620)


date

/c/Users/ddb265/AppData/Local/Continuum/anaconda3/envs/tfpy35/python retrain.py --image_dir train/tile_96 \
    --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/classification/1 \
    --how_many_training_steps 1000 --learning_rate 0.01 --output_labels labels.txt --output_graph gc_mobilenetv2_96_1000_001.pb

rm -rf /c/tmp/bottleneck
rm -rf /c/tmp/checkpoint
rm -rf /c/tmp/retrain_logs
rm -rf /c/tmp/_retrain*

date
#  98.1% (N=4068)

