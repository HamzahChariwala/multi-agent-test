#!/bin/bash
# Quick start script for multi-agent council system

set -e

echo "Multi-Agent Council System - Quick Start"
echo "========================================="

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check CUDA availability
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "WARNING: CUDA not available. System requires GPUs to run."
    echo "Continuing with setup anyway..."
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p profiling_traces/{t4_gpu0,t4_gpu1,t4_gpu2,t4_gpu3,a100_gpu0,a100_gpu1,chairman_tp}
mkdir -p logs

# Run tests
echo ""
echo "Running tests..."
python -m pytest tests/ -v --tb=short || echo "Some tests may fail without GPU access"

# Display next steps
echo ""
echo "========================================="
echo "Setup complete!"
echo ""
echo "Next steps:"
echo ""
echo "1. Edit config files if needed:"
echo "   - config/models.yaml (model selection)"
echo "   - config/endpoints.yaml (network configuration)"
echo "   - config/profiling.yaml (profiling settings)"
echo ""
echo "2. Start the services:"
echo ""
echo "   # T4 Node (Terminal 1):"
echo "   cd serving/t4_cluster && python launcher.py"
echo ""
echo "   # A100 Node - Large Model (Terminal 2):"
echo "   cd serving/a100_cluster && python large_model_twin.py"
echo ""
echo "   # A100 Node - Chairman (Terminal 3):"
echo "   cd serving/a100_cluster && python chairman_tp.py"
echo ""
echo "   # Orchestrator (Terminal 4):"
echo "   cd orchestrator && python main.py --mode example"
echo ""
echo "3. Monitor profiling traces:"
echo "   python tools/analyze_profiling.py"
echo ""
echo "4. See DEPLOYMENT.md for detailed deployment instructions"
echo "========================================="

