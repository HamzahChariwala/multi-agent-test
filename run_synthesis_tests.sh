#!/bin/bash
# Test runner for A100 Synthesis Server

set -e

echo "================================================================"
echo "A100 Synthesis Server - Test Suite"
echo "================================================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if server is running
check_server() {
    echo "Checking if synthesis server is running..."
    if curl -s http://localhost:8020/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Server is running${NC}"
        return 0
    else
        echo -e "${RED}✗ Server is not running${NC}"
        echo ""
        echo "Start the server with:"
        echo "  export CUDA_VISIBLE_DEVICES=0,1,2,3"
        echo "  export MASTER_PORT=29500"
        echo "  python serving/a100_cluster/synthesis_server.py"
        echo ""
        return 1
    fi
}

# Run test
run_test() {
    test_name=$1
    test_file=$2
    
    echo ""
    echo "================================================================"
    echo "Running: $test_name"
    echo "================================================================"
    
    if python3 $test_file; then
        echo -e "${GREEN}✓ $test_name PASSED${NC}"
        return 0
    else
        echo -e "${RED}✗ $test_name FAILED${NC}"
        return 1
    fi
}

# Main test suite
main() {
    cd /home/azureuser/multi-agent-test
    
    # Activate venv if exists
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi
    
    # Check server
    if ! check_server; then
        exit 1
    fi
    
    # Test selection
    if [ "$1" == "phase1" ]; then
        run_test "Phase 1 Only" "test_synthesis_phase1.py"
        exit $?
    elif [ "$1" == "full" ]; then
        run_test "Full Two-Phase Workflow" "test_synthesis_full.py"
        exit $?
    elif [ "$1" == "timeout" ]; then
        echo -e "${YELLOW}Note: Timeout test requires separate server on port 8021 with --cache-timeout=30${NC}"
        run_test "Cache Timeout" "test_synthesis_timeout.py"
        exit $?
    elif [ "$1" == "all" ]; then
        passed=0
        failed=0
        
        if run_test "Phase 1 Only" "test_synthesis_phase1.py"; then
            ((passed++))
        else
            ((failed++))
        fi
        
        sleep 2
        
        if run_test "Full Two-Phase Workflow" "test_synthesis_full.py"; then
            ((passed++))
        else
            ((failed++))
        fi
        
        echo ""
        echo "================================================================"
        echo "TEST SUMMARY"
        echo "================================================================"
        echo -e "${GREEN}Passed: $passed${NC}"
        echo -e "${RED}Failed: $failed${NC}"
        echo "================================================================"
        
        if [ $failed -eq 0 ]; then
            echo -e "${GREEN}✓ All tests passed!${NC}"
            exit 0
        else
            echo -e "${RED}✗ Some tests failed${NC}"
            exit 1
        fi
    else
        echo "Usage: ./run_synthesis_tests.sh [phase1|full|timeout|all]"
        echo ""
        echo "Tests:"
        echo "  phase1  - Test Phase 1 only (prefill and cache storage)"
        echo "  full    - Test full two-phase workflow with KV cache reuse"
        echo "  timeout - Test cache expiration (requires test server)"
        echo "  all     - Run all tests (except timeout)"
        echo ""
        echo "Example:"
        echo "  ./run_synthesis_tests.sh phase1"
        echo "  ./run_synthesis_tests.sh all"
        exit 1
    fi
}

main "$@"

