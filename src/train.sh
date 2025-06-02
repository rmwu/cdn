# main script for training

######### training params
CUDA=1
NUM_GPU=1

######### data params

# NOTE: name of YAML file and run save folder
# see ./config for more options
#TAG="target_mlp"
#TAG="target_axial"
TAG="target_axial"
CONFIG="config/${TAG}.yaml"

# NOTE: customize this to your save folder
# it'll make a new subfolder within with timestamp
SAVE_PATH=""
# load pretrained SEA
#PATH_FCI="checkpoints/fci_synthetic/model_best_epoch=373_auprc=0.842.ckpt"
# this one is corr
PATH_FCI="checkpoints/fci_corr/model_best_epoch=331_auprc=0.531_val_loss=0.000.ckpt"
PATH_AXIAL="checkpoints/fci-axial-corr-model_best_epoch=45_auprc=0.881.ckpt"
# CKPT_PATH is slightly different from PATH_FCI. this is for the new params
#CKPT_PATH=""

python src/train.py \
    --config_file $CONFIG \
    --save_path $SAVE_PATH \
    --gpu $CUDA \
    --num_gpu $NUM_GPU \
    --pretrained_path $PATH_FCI \
    #--checkpoint_path $PATH_AXIAL \

