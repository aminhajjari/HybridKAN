#!/bin/bash

#=======================================================================
# SLURM SCRIPT - Missing-value versions of the sparse ARFF dataset
# (connect-4). CPU only. Run it AFTER generate_missing.sh.
#=======================================================================

#SBATCH --account=def-arashmoh
#SBATCH --job-name=GenMissSparse
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00

#SBATCH --output=/home/gkianfar/scratch/Amin/Sedo/output/logs/genmissing_sparse_%A.out
#SBATCH --error=/home/gkianfar/scratch/Amin/Sedo/output/logs/genmissing_sparse_%A.err

#=======================================================================
# Paths
#=======================================================================

PROJECT_DIR="/home/gkianfar/scratch/Amin"
TAB2SEDO_DIR="$PROJECT_DIR/Sedo/Tab2sedo"
VENV_PATH="/home/gkianfar/scratch/Amin/ICC/venvMsc/bin/activate"

GEN_SCRIPT="$TAB2SEDO_DIR/make_missing_sparse.py"

# Same source and destination as generate_missing.sh
SRC_DIR="/home/gkianfar/scratch/Amin/Tab2Vis/Unzippeddata/CSV"
DST_DIR="/home/gkianfar/scratch/Amin/Sedo/NEWtabular"
VIEW_ROOT="/home/gkianfar/scratch/Amin/Sedo/NEWtabular_byrate"

JOB_LOGS_DIR="$PROJECT_DIR/Sedo/output/logs"

DATASET="connect-4"
RATES="0 0.1 0.2 0.3 0.4 0.5"
SEED=42

#=======================================================================
# Job information
#=======================================================================

echo "=========================================="
echo "GENERATE MISSING VALUES - SPARSE ARFF ($DATASET)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "Source:  $SRC_DIR/$DATASET"
echo "Output:  $DST_DIR/$DATASET"
echo "Rates: $RATES"
echo "Seed: $SEED"
echo "=========================================="
echo ""

mkdir -p "$JOB_LOGS_DIR"

#=======================================================================
# Verify files
#=======================================================================

if [ ! -d "$SRC_DIR/$DATASET" ]; then
    echo "❌ ERROR: Source folder not found: $SRC_DIR/$DATASET"
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
python -c "import numpy; print('numpy OK')" || exit 1

# Extra load test with the same fallback parser main.py uses (liac-arff)
LOAD_CHECK=""
if python -c "import arff" 2>/dev/null; then
    LOAD_CHECK="--load_check"
    echo "liac-arff found: outputs will also be load-tested"
else
    echo "liac-arff not found: skipping the load test"
fi
echo ""

cd "$TAB2SEDO_DIR"

#=======================================================================
# Step 1: show structure (writes nothing)
#=======================================================================

python "$GEN_SCRIPT" --src "$SRC_DIR" --dst "$DST_DIR" --dataset "$DATASET" --inspect

#=======================================================================
# Step 2: generate all rates
#=======================================================================

echo ""
python "$GEN_SCRIPT" \
    --src "$SRC_DIR" \
    --dst "$DST_DIR" \
    --dataset "$DATASET" \
    --rates $RATES \
    --seed "$SEED" \
    $LOAD_CHECK

GEN_EXIT=$?

if [ $GEN_EXIT -ne 0 ]; then
    echo "❌ Generator exited with code $GEN_EXIT"
    exit $GEN_EXIT
fi

#=======================================================================
# Step 3: verify the files on disk match a fresh regeneration
#=======================================================================

echo ""
echo "Verifying generated files..."

python "$GEN_SCRIPT" \
    --src "$SRC_DIR" \
    --dst "$DST_DIR" \
    --dataset "$DATASET" \
    --rates $RATES \
    --seed "$SEED" \
    --verify | tail -n 8

#=======================================================================
# Step 4: add connect-4 to the per-rate folders used for training
#=======================================================================

echo ""
echo "Linking $DATASET into $VIEW_ROOT ..."

for RATE in $RATES; do

    PCT=$(python -c "print(int(round(float('$RATE') * 100)))")
    F="$DST_DIR/$DATASET/${DATASET}_missing_${PCT}.arff"

    if [ ! -f "$F" ]; then
        echo "  ❌ missing file: $F"
        continue
    fi

    mkdir -p "$VIEW_ROOT/mr${PCT}/$DATASET"
    ln -sfn "$F" "$VIEW_ROOT/mr${PCT}/$DATASET/$(basename "$F")"

    N=$(find "$VIEW_ROOT/mr${PCT}" -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo "  mr${PCT}: $N dataset folders"

done

#=======================================================================
# Final summary
#=======================================================================

echo ""
echo "=========================================="
echo "DONE: $(date)"
echo "Each mrXX folder should now hold 67 dataset folders (66 + $DATASET)"
echo "Stats: ${DST_DIR}_meta/stats.csv"
echo "Log:   ${DST_DIR}_meta/generation.log"
echo "=========================================="

exit 0
