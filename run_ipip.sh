#!/bin/bash
# Quick start script for IPIP pipeline

set -e  # Exit on error

echo "========================================"
echo "IPIP Pipeline - Quick Start"
echo "========================================"

# Default values
ITERATIONS=3
RESULTS_DIR="./ipip_results"
FINETUNE_DATA="./datasets/finetune_data.pt"
MD_SEEDS=10
MD_SEEDS_PER_ITER=5

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        -r|--results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        -f|--finetune-data)
            FINETUNE_DATA="$2"
            shift 2
            ;;
        --md-seeds)
            MD_SEEDS="$2"
            shift 2
            ;;
        --md-seeds-per-iter)
            MD_SEEDS_PER_ITER="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: bash run_ipip.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -i, --iterations NUM              Number of IPIP iterations (default: 3)"
            echo "  -r, --results-dir DIR             Results directory (default: ./ipip_results)"
            echo "  -f, --finetune-data FILE          Path to finetuning dataset"
            echo "  --md-seeds NUM                    Seeds for initial pretraining (default: 10)"
            echo "  --md-seeds-per-iter NUM           Seeds per iteration (default: 5)"
            echo "  -h, --help                        Show this help message"
            echo ""
            echo "Examples:"
            echo "  bash run_ipip.sh -i 3"
            echo "  bash run_ipip.sh -i 5 -r ./my_results"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check Python
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed"
    exit 1
fi

# Print configuration
echo ""
echo "Configuration:"
echo "  Iterations:           $ITERATIONS"
echo "  Results directory:    $RESULTS_DIR"
echo "  Finetune data:        $FINETUNE_DATA"
echo "  Initial MD seeds:     $MD_SEEDS"
echo "  MD seeds per iter:    $MD_SEEDS_PER_ITER"
echo ""

# Run pipeline
echo "Starting IPIP pipeline..."
echo ""

python run_ipip_pipeline.py \
    --iterations "$ITERATIONS" \
    --results-dir "$RESULTS_DIR" \
    --finetune-data "$FINETUNE_DATA" \
    --pretrain-seeds "$MD_SEEDS" \
    --md-seeds-per-iter "$MD_SEEDS_PER_ITER"

echo ""
echo "========================================"
echo "IPIP pipeline completed!"
echo "Results saved to: $RESULTS_DIR"
echo "========================================"
