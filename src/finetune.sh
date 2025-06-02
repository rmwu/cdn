# main script for training

######### training params
CUDA=0
NUM_GPU=1

######### data params

# NOTE: name of YAML file and run save folder
# see ./config for more options
TAG="target_alora"
CONFIG="config/${TAG}.yaml"

# load pretrained SEA
PATH_FCI="checkpoints/fci_synthetic/model_best_epoch=373_auprc=0.842.ckpt"
PATH_FCI_CORR="checkpoints/fci_corr/model_best_epoch=331_auprc=0.531_val_loss=0.000.ckpt"

# load pretrained target predictor
#PATH_AXIAL="checkpoints/fci-axial-model_best_epoch=53_auprc=0.910.ckpt"
# corr, recently finetuned
PATH_AXIAL_CORR="checkpoints/fci-axial-corr-model_best_epoch=45_auprc=0.881.ckpt"
PATH_MLP="checkpoints/fci-mlp-model_best_epoch=59_auprc=0.893.ckpt"


python src/finetune.py \
    --config_file $CONFIG \
    --gpu $CUDA \
    --num_gpu $NUM_GPU \
    --pretrained_path $PATH_FCI \
    --checkpoint_path $PATH_AXIAL \
    #--checkpoint_path $PATH_MLP_TARGET \

