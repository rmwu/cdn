# main script for inference

######### inference params
CUDA=0

TAG="target_axial"  # or "target_axial_sc"
CONFIG="config/${TAG}.yaml"

######### data presets
# dataset must match $TAG

# synthetic (TAG="target_axial")
DATA_SYNTHETIC="data/test_240.csv"

# Perturb-seq (TAG="target_axial_sc")
DATA_PERTURBSEQ="data/perturbseq.csv"
DATA_PERTURBSEQ_K562="data/perturbseq_k562.csv"
DATA_PERTURBSEQ_RPE1="data/perturbseq_rpe1.csv"
DATA_PERTURBSEQ_HepG2="data/perturbseq_hepg2.csv"
DATA_PERTURBSEQ_Jurkat="data/perturbseq_jurkat.csv"

# Sci-Plex (TAG="target_axial_sc")
DATA_SCIPLEX="data/sciplex.csv"

######### model presets
# download from:
# https://figshare.com/articles/software/CDN_checkpoints/29225855

# synthetic
PATH_SYNTHETIC="checkpoints/cdn_synthetic-all.ckpt"
# used for synthetic or Sci-Plex results
PATH_SYNTHETIC_DIFF="checkpoints/cdn_synthetic_diff-all.ckpt"

# Perturb-seq (TAG="target_axial_sc")
PATH_PerturbSeq="checkpoints/cdn_finetuned-seen.ckpt"
PATH_K562="checkpoints/cdn_finetuned-unseen-no_k562.ckpt"
PATH_RPE1="checkpoints/cdn_finetuned-unseen-no_rpe1.ckpt"
PATH_HepG2="checkpoints/cdn_finetuned-unseen-no_hepg2.ckpt"
PATH_Jurkat="checkpoints/cdn_finetuned-unseen-no_jurkat.ckpt"

echo $NAME

python src/inference.py \
    --config_file $CONFIG \
    --run_name $TAG \
    --gpu $CUDA \
    --checkpoint_path $PATH_AXIAL \
    --pretrained_path $PATH_FCI \

