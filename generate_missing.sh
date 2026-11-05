#!/bin/bash

#=======================================================================
# SLURM SCRIPT - Generate datasets with missing values
# CPU only, no GPU needed
#=======================================================================

#SBATCH --account=def-arashmoh
#SBATCH --job-name=GenMissing
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

#SBATCH --output=/home/gkianfar/scratch/Amin/Sedo/output/logs/genmissing_%A.out
#SBATCH --error=/home/gkianfar/scratch/Amin/Sedo/output/logs/genmissing_%A.err

#=======================================================================
# Paths
#=======================================================================

PROJECT_DIR="/home/gkianfar/scratch/Amin"
TAB2SEDO_DIR="$PROJECT_DIR/Sedo/Tab2sedo"
VENV_PATH="/home/gkianfar/scratch/Amin/ICC/venvMsc/bin/activate"

# Original complete datasets (one folder per dataset, never modified)
INPUT_DIR="/home/gkianfar/scratch/Amin/ICC/Unzippeddata/CSV"

# New datasets are written here, one sub-root per missing rate:
#   $OUTPUT_ROOT/CSV_mr10/<dataset>/<dataset>.csv
OUTPUT_ROOT="/home/gkianfar/scratch/Amin/Sedo/NEWtabular"

JOB_LOGS_DIR="$PROJECT_DIR/Sedo/output/logs"

# TODO: put the real name of your generator script here
GEN_SCRIPT="$TAB2SEDO_DIR/generate_missing.py"

#=======================================================================
# Missing rates and seed
#=======================================================================

# Rates as fractions. Add or remove values freely.
RATES="0.10 0.20 0.30 0.40 0.50"

# Fixed seed so every run and every baseline sees the same masks
SEED=42

#=======================================================================
# Job information
#=======================================================================

echo "=========================================="
echo "GENERATE MISSING-VALUE DATASETS"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "Input dir:  $INPUT_DIR"
echo "Output root: $OUTPUT_ROOT"
echo "Rates: $RATES"
echo "Seed: $SEED"
echo "=========================================="
echo ""

mkdir -p "$JOB_LOGS_DIR"

#=======================================================================
# Verify files
#=======================================================================

if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ ERROR: Input datasets not found: $INPUT_DIR"
    exit 1
fi

if [ ! -f "$GEN_SCRIPT" ]; then
    echo "❌ ERROR: Generator script not found: $GEN_SCRIPT"
    exit 1
fi

if [ ! -f "$VENV_PATH" ]; then
    echo "❌ ERROR: Virtual environment not found: $VENV_PATH"
    exit 1
fi

#=======================================================================
# Load environment
#=======================================================================

module purge
module load StdEnv/2023
module load python/3.11

source "$VENV_PATH"

which python
python --version
echo ""

#=======================================================================
# Generate one output root per missing rate
#=======================================================================

cd "$TAB2SEDO_DIR"

FAILED=0

for RATE in $RATES; do

    # 0.10 -> 10, 0.25 -> 25
    PCT=$(python -c "print(int(round(float('$RATE') * 100)))")
    OUT_DIR="$OUTPUT_ROOT/CSV_mr${PCT}"

    echo "------------------------------------------"
    echo "Rate: $RATE  ->  $OUT_DIR"
    echo "------------------------------------------"

    mkdir -p "$OUT_DIR"

    # TODO: adjust these argument names to match your generator script
    python "$GEN_SCRIPT" \
        --input_dir "$INPUT_DIR" \
        --output_dir "$OUT_DIR" \
        --missing_rate "$RATE" \
        --seed "$SEED"

    if [ $? -ne 0 ]; then
        echo "❌ Generation failed for rate $RATE"
        FAILED=1
    else
        N=$(find "$OUT_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
        echo "✅ $N dataset folders written to $OUT_DIR"
    fi

    echo ""

done

#=======================================================================
# Final summary
#=======================================================================

echo "=========================================="
echo "DONE: $(date)"
if [ $FAILED -eq 0 ]; then
    echo "✅ All rates generated"
    echo "Next: set DATASETS_DIR in main.sh to one of:"
    for RATE in $RATES; do
        PCT=$(python -c "print(int(round(float('$RATE') * 100)))")
        echo "   $OUTPUT_ROOT/CSV_mr${PCT}"
    done
else
    echo "⚠️ Some rates failed, check the .err log"
fi
echo "=========================================="

exit $FAILED
