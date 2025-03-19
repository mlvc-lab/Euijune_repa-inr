#! /bin/bash

MODELTYPE=siren
LR=0.0001
FIRSTOMEGA=30
HIDDENOMEGA=30
EPOCH=2000
PRETRAINEPOCH=500

DSET="CelebA_HQ"


# dataset_dir와 out_channel 값을 데이터셋 종류에 따라 설정
case $DSET in
    "Chest_CT")
        DATADIR="/local_dataset/Chest_CT"
        OUTCHANNEL=3    # R, G, B 픽셀값이 모두 동일하게 예측됨.
        SIDELEN=128
        MAXITER=120
        ;;
    "CelebA_HQ")
        DATADIR="/local_dataset/CelebA_HQ"
        OUTCHANNEL=3
        SIDELEN=256
        MAXITER=100
        ;;
    "DIV2K")
        DATADIR="/local_dataset/DIV2K"
        OUTCHANNEL=3
        SIDELEN=256
        MAXITER=40
        EPOCH=500       # Pretrain Image이기 때문.
        ;;
    "TESTIMAGE")
        DATADIR="/local_dataset/TESTIMAGE"
        OUTCHANNEL=3
        SIDELEN=1200
        MAXITER=16
        ;;
    "Urban_100")
        DATADIR="/local_dataset/Urban_100"
        OUTCHANNEL=3
        SIDELEN=256
        MAXITER=100
        EPOCH=2000
        ;;
    *)
        echo "Unknown dataset type. Please choose one of: Chest_CT, CelebA_HQ, TESTIMAGE, DIV2K, Urban_100"
        exit 1
        ;;
esac

# 설정된 변수 값 확인
echo "Dataset Directory: $DATADIR"

GPUID_TUPLE=(0 1 2 3 4 5 6 7)
declare -A imgid_map  # 각 GPUID마다 IMGID를 모으기 위한 연관 배열
ID=0


# #### for scratch

# EXPNAME="$DSET/scratch/Siren/sidelen_$SIDELEN"
# echo "Experiment $EXPNAME"

# #for IMGID in `seq -w 1 $MAXITER`                       # Chest_CT, CelebA_HQ
# elem=('07')                                   # TESTIMAGES (1200 x 1200)
# for IMGID in "${elem[@]}"
#     do
#         GPUID=${GPUID_TUPLE[$ID]}
#         imgid_map[$GPUID]="${imgid_map[$GPUID]}, $IMGID"
#         ((ID++))
#         ID=$((ID % 8))

#         ## train
#         CUDA_VISIBLE_DEVICES=$GPUID python3 main.py \
#             --model_type $MODELTYPE --exp_name $EXPNAME \
#             --first_omega $FIRSTOMEGA --hidden_omega $HIDDENOMEGA \
#             --lr $LR --out_feature $OUTCHANNEL --sidelen $SIDELEN\
#             --num_epochs $EPOCH --pretrain_epochs $PRETRAINEPOCH\
#             --img_id $IMGID --dataset_dir $DATADIR & 
#     done

# for GPUID in $(seq 0 $((${#GPUID_TUPLE[@]} - 1))); do
#     echo "GPUID: $GPUID IMGID: ${imgid_map[$GPUID]#, }"  # 맨 앞에 있는 ", "를 제거
# done
# wait


#### for transferlearning

elem=('07')                                        # TESTIMAGES (1200 x 1200)
#target_elem=('01' '02' '03' '04' '05' '06' '07' '08')                                        # TESTIMAGES (1200 x 1200)
#for SOURCEIMGID in `seq -w 2 2`
for SOURCEIMGID in "${elem[@]}"
    do
        EXPNAME="$DSET/$SOURCEIMGID/Siren"
        LOADPATH="logs/DIV2K/scratch/Siren/sidelen_${SIDELEN}/outputs_$SOURCEIMGID.pt"
        unset imgid_map
        declare -A imgid_map

        for TARGETIMGID in `seq -w 1 $MAXITER`
        #for TARGETIMGID in "${target_elem[@]}"                                                    # (1200 x 1200) resolution, due to memory issue
            do
                GPUID=${GPUID_TUPLE[$ID]}
                imgid_map[$GPUID]="${imgid_map[$GPUID]}, $TARGETIMGID"
                ((ID++))
                ID=$((ID % 8))

                CUDA_VISIBLE_DEVICES=$GPUID python main.py \
                    --model_type $MODELTYPE --exp_name $EXPNAME \
                    --first_omega $FIRSTOMEGA --hidden_omega $HIDDENOMEGA \
                    --lr $LR --out_feature $OUTCHANNEL --sidelen $SIDELEN\
                    --num_epochs $EPOCH --pretrain_epochs $PRETRAINEPOCH\
                    --img_id $TARGETIMGID --dataset_dir $DATADIR \
                    --load_path $LOADPATH &
            done

        echo "SOURCEIMGID: $SOURCEIMGID"
        for GPUID in "${!imgid_map[@]}"; do
            echo "GPUID: $GPUID IMGID: ${imgid_map[$GPUID]#, }"  # 맨 앞에 있는 ", "를 제거
        done
        wait
    done
    wait