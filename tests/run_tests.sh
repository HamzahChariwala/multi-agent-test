#!/bin/bash
# Run all tests for the multi-agent council system

set -e

echo "Running Multi-Agent Council System Tests"
echo "========================================"

# Install test dependencies
echo "Installing test dependencies..."
pip install pytest pytest-asyncio pytest-cov > /dev/null 2>&1

# Run tests with coverage
echo ""
echo "Running unit tests..."
python -m pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html

echo ""
echo "========================================"
echo "Tests complete! Coverage report generated in htmlcov/"

