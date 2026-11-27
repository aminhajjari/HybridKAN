#!/bin/bash

#=======================================================================
# BASELINE: Tab2Vis on ONE missing-value (MV) level, all 67 datasets
#=======================================================================

#SBATCH --account=def-arashmoh
#SBATCH --job-name=T2V      # submit as: sbatch --job-name=T2V_mr10 run_tab2vis_mv.sh mr10
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=96:00:00

#SBATCH --output=/home/gkianfar/scratch/Amin/Sedo/output_mv_baselines/logs/tab2vis_%x_%A.out
#SBATCH --error=/home/gkianfar/scratch/Amin/Sedo/output_mv_baselines/logs/tab2vis_%x_%A.err

#=======================================================================
# Configuration
#=======================================================================



# ============================================================
# Paths
# ============================================================

# MV level, e.g.  sbatch --job-name=T2V_mr10 run_tab2vis_mv.sh mr10
MV_LEVEL="${1:?usage: sbatch run_tab2vis_mv.sh mr00|mr10|mr20|mr30|mr40|mr50}"

PROJECT_DIR="/home/gkianfar/scratch/Amin"

TAB2SEDO_DIR="$PROJECT_DIR/Sedo/Tab2sedo"

VENV_PATH="/home/gkianfar/scratch/Amin/ICC/venvMsc/bin/activate"

# Tabular datasets
DATASETS_DIR="/home/gkianfar/scratch/Amin/Sedo/NEWtabular_byrate/$MV_LEVEL"

# Image datasets
IMAGE_DATA_DIR="/home/gkianfar/scratch/Amin/ICC/Unzippeddata/Image"
RESULTS_BASE="/home/gkianfar/scratch/Amin/Sedo/output_mv_baselines/tab2vis"   # results of this baseline; other outputs untouched
JOB_LOGS_DIR="/home/gkianfar/scratch/Amin/Sedo/output/logs"     # existing Slurm log dir
# Scripts
MAIN_SCRIPT="$TAB2SEDO_DIR/baselines/tab2vis/run_vif_mv.py"
BATCH_SCRIPT="$TAB2SEDO_DIR/baselines/tab2vis/run_all_datasets_mv.py"

TIMEOUT_DEFAULT=14400
#=======================================================================
# Job Information
#=======================================================================

echo "=========================================="
echo "TAB2VIS BASELINE RUN"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "Datasets dir: $DATASETS_DIR"
echo "Code dir: $TAB2SEDO_DIR"
echo "Output dir: $RESULTS_BASE"
echo "Configuration:"
echo "  - Tab2Vis (ADOPT optimizer, dual SHAP)"
echo "  - Timeout: 4 hours per dataset"
echo "  - CPUs: 8 cores"
echo "  - Memory: 64GB"
echo "=========================================="
echo ""


#=======================================================================
# GPU Information
#=======================================================================

echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""


#=======================================================================
# Setup
#=======================================================================

echo "Creating directories..."

mkdir -p "$JOB_LOGS_DIR"
mkdir -p "$RESULTS_BASE"

echo "✅ Directories ready"
echo ""


#=======================================================================
# Verify Files
#=======================================================================

echo "Verifying environment..."

if [ ! -d "$DATASETS_DIR" ]; then
    echo "❌ ERROR: Datasets not found:"
    echo "   $DATASETS_DIR"
    exit 1
fi

if [ ! -f "$BATCH_SCRIPT" ]; then
    echo "❌ ERROR: Batch script not found:"
    echo "   $BATCH_SCRIPT"
    exit 1
fi

if [ ! -f "$MAIN_SCRIPT" ]; then
    echo "❌ ERROR: Main script not found:"
    echo "   $MAIN_SCRIPT"
    exit 1
fi

if [ ! -f "$VENV_PATH" ]; then
    echo "❌ ERROR: Virtual environment not found:"
    echo "   $VENV_PATH"
    exit 1
fi

DATASET_COUNT=$(find "$DATASETS_DIR" \
    -mindepth 1 \
    -maxdepth 1 \
    -type d | wc -l)

echo "✅ Found $DATASET_COUNT dataset folders"
echo ""


#=======================================================================
# Load Environment
#=======================================================================

echo "Loading modules..."

module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2

echo "✅ Modules loaded"
echo ""


echo "Activating virtual environment..."

source "$VENV_PATH"

echo "✅ Virtual environment active"
echo ""


#=======================================================================
# Python Environment Check
#=======================================================================

echo "Python environment:"

which python
python --version

python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Environment check failed!"
    exit 1
fi

echo "✅ Environment ready"
echo ""


#=======================================================================
# Tab2Vis needs two packages that main_mv.py does not: shap and adopt (ADOPT optimizer)
#=======================================================================

python -c "import shap, adopt; print('shap + adopt import OK')"
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Tab2Vis needs 'shap' and 'adopt' (ADOPT optimizer, adopt.py) in this environment."
    echo "   Install them in the venv or put adopt.py next to run_vif_mv.py, then resubmit."
    exit 1
fi

echo ""


#=======================================================================
# Execute Batch Processing
#=======================================================================

echo "=========================================="
echo "🚀 STARTING BATCH PROCESSING"
echo "=========================================="

echo "Using:"
echo "  Code:     $TAB2SEDO_DIR"
echo "  Datasets: $DATASETS_DIR"
echo "  Output:   $RESULTS_BASE"
echo "  Datasets: $DATASET_COUNT"
echo ""

echo "Running command:"
echo "python $BATCH_SCRIPT \\"
echo "  --datasets_dir $DATASETS_DIR \\"
echo "  --output_base $RESULTS_BASE \\"
echo "  --job_id $SLURM_JOB_ID \\"
echo "  --script_path $MAIN_SCRIPT \\"
echo "  --timeout $TIMEOUT_DEFAULT"

echo ""
echo "=========================================="
echo ""


# Run the batch processor

cd "$TAB2SEDO_DIR"

python "$BATCH_SCRIPT" \
    --datasets_dir "$DATASETS_DIR" \
    --output_base "$RESULTS_BASE" \
    --job_id "$SLURM_JOB_ID" \
    --script_path "$MAIN_SCRIPT" \
    --timeout "$TIMEOUT_DEFAULT" \
    --skip_existing \
    --mv_level "$MV_LEVEL"

EXIT_CODE=$?


#=======================================================================
# Final Summary
#=======================================================================

echo ""
echo "=========================================="
echo "PRODUCTION RUN COMPLETE"
echo "=========================================="

echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo ""


if [ $EXIT_CODE -eq 0 ]; then

    RESULT_DIR="$RESULTS_BASE/$MV_LEVEL"

    echo "✅ SUCCESS!"
    echo ""

    echo "📂 Results location:"
    echo "    $RESULT_DIR/"
    echo ""

    echo "📊 Files generated:"
    echo "    ├── csv/"
    echo "    │   ├── results_summary.csv"
    echo "    │   ├── results_detailed.csv"
    echo "    │   └── statistics.csv"
    echo "    ├── latex/"
    echo "    │   └── results_latex.txt"
    echo "    ├── logs/"
    echo "    │   ├── results.jsonl"
    echo "    │   └── progress_log.jsonl"
    echo "    └── README.txt"
    echo ""

    if [ -f "$RESULT_DIR/csv/statistics.csv" ]; then

        echo "📊 Quick Statistics:"

        grep "Average Accuracy" \
            "$RESULT_DIR/csv/statistics.csv" | head -1

        grep "Datasets >90%" \
            "$RESULT_DIR/csv/statistics.csv" | head -1

    fi

    echo ""

    echo "📧 Completion email sent to: aminhajjr@gmail.com"

    echo "🎉 All $DATASET_COUNT datasets processed!"

else

    echo "⚠️ Some datasets may have failed"

    echo ""
    echo "Check SLURM logs:"
    echo "    Output: $JOB_LOGS_DIR/production_${SLURM_JOB_ID}.out"
    echo "    Error:  $JOB_LOGS_DIR/production_${SLURM_JOB_ID}.err"

fi

echo "=========================================="

exit $EXIT_CODE
