# main script for inference

######### inference params
CUDA=0

######### data params

# NOTE: name of YAML file and run save folder
# see ./config for more options
#TAG="target_axial"
#TAG="target_mlp"
TAG="target_axial_sc"
CONFIG="config/${TAG}.yaml"

# invcov
#PATH_FCI="checkpoints/fci_synthetic/model_best_epoch=373_auprc=0.842.ckpt"
# corr
PATH_FCI="checkpoints/fci_corr/model_best_epoch=331_auprc=0.531_val_loss=0.000.ckpt"


echo $NAME

# set the appropriate --checkpoint_path variable
# that MATCHES with $TAG
python src/inference.py \
    --config_file $CONFIG \
    --run_name $TAG \
    --gpu $CUDA \
    --checkpoint_path $PATH_AXIAL \
    --pretrained_path $PATH_FCI \

