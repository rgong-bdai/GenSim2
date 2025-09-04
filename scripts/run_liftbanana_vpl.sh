#!/bin/bash
# Script to run LiftBanana with VPL saver integration
# This replaces: python scripts/kpam_data_collection.py --env LiftBanana --render

echo "Running LiftBanana with VPL Saver Integration"
echo "=============================================="

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gensim2

# Run the VPL-enabled data collection script
python scripts/kpam_data_collection_vpl.py \
    --env LiftBanana \
    --render \
    --save \
    --num_episodes 1 \
    --max_steps 2100 \
    --vpl_dir "./vpl_data_liftbanana" \
    --log_memory

echo ""
echo "Data collection completed!"
echo "VPL data saved to: ./vpl_data_liftbanana"
echo ""
echo "To run without saving data, remove the --save flag:"
echo "python scripts/kpam_data_collection_vpl.py --env LiftBanana --render"
echo ""
echo "To run with more episodes:"
echo "python scripts/kpam_data_collection_vpl.py --env LiftBanana --render --save --num_episodes 10"
