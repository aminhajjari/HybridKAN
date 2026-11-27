#!/bin/bash
#SBATCH --job-name=T2I
#SBATCH --account=def-arashmoh
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/home/gkianfar/scratch/Amin/Sedo/output_mv_baselines/logs/table2image_%x_%A.out
#SBATCH --error=/home/gkianfar/scratch/Amin/Sedo/output_mv_baselines/logs/table2image_%x_%A.err
# Table2Image baseline on ONE MV level.
# usage:  sbatch --job-name=T2I_mr10 run_table2image_mv.sh mr10      (logs dir must exist)
MV_LEVEL="${1:?usage: sbatch run_table2image_mv.sh mr00|mr10|mr20|mr30|mr40|mr50}"
PROJECT_DIR="/home/gkianfar/scratch/Amin"
module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
source /home/gkianfar/scratch/Amin/ICC/venvMsc/bin/activate
cd "$PROJECT_DIR/Sedo/Tab2sedo/baselines/table2image" || exit 1
python run_all_table2image_mv.py \
  --datasets_dir $PROJECT_DIR/Sedo/NEWtabular_byrate/$MV_LEVEL \
  --output_root  $PROJECT_DIR/Sedo/output_mv_baselines/table2image \
  --image_data_root $PROJECT_DIR/ICC/Unzippeddata/Image \
  --skip_existing
