#!/bin/bash
#SBATCH --job-name=catboo
#SBATCH --account=def-arashmoh
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/gkianfar/scratch/Amin/Sedo/output_mv_baselines/logs/catboost_%x_%A.out
#SBATCH --error=/home/gkianfar/scratch/Amin/Sedo/output_mv_baselines/logs/catboost_%x_%A.err
# catboost baseline on ONE MV level.
# usage:  sbatch --job-name=catboo_mr10 run_catboost_mv.sh mr10      (logs dir must exist)
MV_LEVEL="${1:?usage: sbatch run_catboost_mv.sh mr00|mr10|mr20|mr30|mr40|mr50}"
PROJECT_DIR="/home/gkianfar/scratch/Amin"
module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
source /home/gkianfar/scratch/Amin/ICC/venvMsc/bin/activate
cd "$PROJECT_DIR/Sedo/Tab2sedo/baselines/tabular" || exit 1
python -c "import catboost" || { echo "pip install catboost first"; exit 1; }
python catboost_mv.py \
  --datasets_dir /home/gkianfar/scratch/Amin/Sedo/NEWtabular_byrate/$MV_LEVEL \
  --output_root  /home/gkianfar/scratch/Amin/Sedo/output_mv_baselines \
  --skip_existing
