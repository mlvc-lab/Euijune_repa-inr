#!/bin/bash
# This script is used to run the Strainer model for training or evaluation.
# This scrips is run from ~/workspace2/TEAM_REPA_INR_NEURIPS_2025.

GPUID=6
MODE=eval
LOSS=mse
DSETDIR="/local_dataset/CelebA_HQ"

NUMLAYERS=8
SHAREDENCLAYERS=$((NUMLAYERS - 1))

case $LOSS in
    "repair")
        echo "Using REPAIR loss"
        ENCDEPTH=7
        ENCTYPE=dinov2-vit-b
        PROJCOEF=0.5
        ;;
    "mse")
        echo "Using MSE loss"
        ENCDEPTH=-1
        ENCTYPE=None
        PROJCOEF=0.0
        ;;
esac

case $MODE in
    "train")
        echo "Training mode"
        EPOCH=5000
        SAMPLING=random
        NUMDECS=10
        LOADPATH=None
        ;;
    "eval")
        echo "Testing mode"
        EPOCH=2000
        SAMPLING=custom
        NUMDECS=1
        LOADPATH="logs/strainer/train/DIV2K/10img_repair/${SHAREDENCLAYERS}_1_enc_depth_7_seed_1234.pt"
        #LOADPATH="logs/strainer/train/DIV2K/10img_mse/${SHAREDENCLAYERS}_1_enc_depth_-1_seed_1234.pt"
        ;;
esac


echo "Dataset Directory: $DSETDIR"
CUDA_VISIBLE_DEVICES=$GPUID python3 strainer/main.py \
    --mode $MODE \
    --loss_fn $LOSS \
    --data_dir $DSETDIR \
    --epoch $EPOCH \
    --sampling $SAMPLING \
    --encoder_depth $ENCDEPTH \
    --enc_type $ENCTYPE \
    --proj_coef $PROJCOEF \
    --num_layers $NUMLAYERS \
    --shared_encoder_layers $SHAREDENCLAYERS \
    --num_decoders $NUMDECS \
    --weight_path $LOADPATH \