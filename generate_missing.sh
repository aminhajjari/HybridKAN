#!/bin/bash

#=======================================================================
# SLURM SCRIPT - Generate datasets with missing values (CPU only)
# Runs make_missing_datasets.py once for all rates, then builds one
# clean folder per rate (symlinks) so run_all_datasets.py can use it.
#=======================================================================

#SBATCH --account=def-arashmoh
#SBATCH --job-name=GenMissing
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

#SBATCH --output=/home/gkianfar/scratch/Amin/Sedo/output/logs/genmissing_%A.out
#SBATCH --error=/home/gkianfar/scratch/Amin/Sedo/output/logs/genmissing_%A.err

#=======================================================================
# Paths
#=======================================================================

PROJECT_DIR="/home/gkianfar/scratch/Amin"
TAB2SEDO_DIR="$PROJECT_DIR/Sedo/Tab2sedo"
VENV_PATH="/home/gkianfar/scratch/Amin/ICC/venvMsc/bin/activate"

# Generator script (change the folder if you saved it somewhere else)
GEN_SCRIPT="$TAB2SEDO_DIR/make_missing_datasets.py"

# Original complete datasets (read only). This is the default path inside
# your generator. main.sh uses ICC/Unzippeddata/CSV, so make sure both
# folders really hold the same data.
SRC_DIR="/home/gkianfar/scratch/Amin/Tab2Vis/Unzippeddata/CSV"

# Generator output: NEWtabular/<dataset>/<dataset>_missing_<pct>.<ext>
DST_DIR="/home/gkianfar/scratch/Amin/Sedo/NEWtabular"

# Per-rate view for training (symlinks): NEWtabular_byrate/mr20/<dataset>/<file>
VIEW_ROOT="/home/gkianfar/scratch/Amin/Sedo/NEWtabular_byrate"

JOB_LOGS_DIR="$PROJECT_DIR/Sedo/output/logs"

#=======================================================================
# Rates and seed
#=======================================================================

# Fractions. 0 writes the unmodified rows (good as a no-missing reference).
RATES="0 0.1 0.2 0.3 0.4 0.5"

# Same seed = same masks for every model and every rerun
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
echo "Source:  $SRC_DIR"
echo "Output:  $DST_DIR"
echo "By-rate: $VIEW_ROOT"
echo "Rates: $RATES"
echo "Seed: $SEED"
echo "=========================================="
echo ""

mkdir -p "$JOB_LOGS_DIR"

#=======================================================================
# Verify files
#=======================================================================

if [ ! -d "$SRC_DIR" ]; then
    echo "❌ ERROR: Source datasets not found: $SRC_DIR"
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
python -c "import numpy, pandas, scipy; print('numpy/pandas/scipy OK')" || exit 1
echo ""

#=======================================================================
# Step 1: generate all rates in one call
#=======================================================================

cd "$TAB2SEDO_DIR"

python "$GEN_SCRIPT" \
    --src "$SRC_DIR" \
    --dst "$DST_DIR" \
    --rates $RATES \
    --seed "$SEED"

GEN_EXIT=$?

if [ $GEN_EXIT -ne 0 ]; then
    echo "❌ Generator exited with code $GEN_EXIT"
    exit $GEN_EXIT
fi

#=======================================================================
# Step 2: verify the files on disk match a fresh in-memory regeneration
#=======================================================================

echo ""
echo "Verifying generated files..."

python "$GEN_SCRIPT" \
    --src "$SRC_DIR" \
    --dst "$DST_DIR" \
    --rates $RATES \
    --seed "$SEED" \
    --verify | tail -n 3

#=======================================================================
# Step 3: build one folder per rate (symlinks) for training
#=======================================================================

echo ""
echo "Building per-rate folders in $VIEW_ROOT ..."

for RATE in $RATES; do

    PCT=$(python -c "print(int(round(float('$RATE') * 100)))")
    VIEW="$VIEW_ROOT/mr${PCT}"
    mkdir -p "$VIEW"

    find "$DST_DIR" -type f -name "*_missing_${PCT}.*" -print0 | \
    while IFS= read -r -d '' F; do
        REL_DIR=$(dirname "${F#$DST_DIR/}")
        mkdir -p "$VIEW/$REL_DIR"
        ln -sfn "$F" "$VIEW/$REL_DIR/$(basename "$F")"
    done

    N=$(find "$VIEW" -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo "  mr${PCT}: $N dataset folders -> $VIEW"

done

#=======================================================================
# Final summary
#=======================================================================

echo ""
echo "=========================================="
echo "DONE: $(date)"
echo "Stats:    ${DST_DIR}_meta/stats.csv"
echo "Log:      ${DST_DIR}_meta/generation.log"
echo ""
echo "Next: set DATASETS_DIR in main.sh to one of:"
for RATE in $RATES; do
    PCT=$(python -c "print(int(round(float('$RATE') * 100)))")
    echo "   $VIEW_ROOT/mr${PCT}"
done
echo "=========================================="

exit 0
