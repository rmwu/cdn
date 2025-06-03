# main script for training

######### training params
CUDA=0
NUM_GPU=1

######### data params

# NOTE: name of YAML file and run save folder
TAG="target_axial"  # or "target_axial_sc"
CONFIG="config/${TAG}.yaml"

# NOTE: customize this to your save folder
# it'll make a new subfolder within with timestamp
SAVE_PATH=""
# load pretrained SEA
PATH_FCI="checkpoints/sea_fci_corr.ckpt"
PATH_AXIAL=""  # set to resume training

python src/train.py \
    --config_file $CONFIG \
    --save_path $SAVE_PATH \
    --gpu $CUDA \
    --num_gpu $NUM_GPU \
    --pretrained_path $PATH_FCI \
    #--checkpoint_path $PATH_AXIAL \

