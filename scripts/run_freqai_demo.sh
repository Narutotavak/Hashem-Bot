#!/bin/bash

# FreqAI Demo Runner Script
# This script provides quick commands to test the FreqAI implementation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CONFIG_FILE="config_examples/config_freqai.example.json"
STRATEGY="FreqaiExampleStrategy"
STRATEGY_PATH="freqtrade/templates"
MODEL="LightGBMRegressor"
TIMERANGE="20230101-20230601"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if freqtrade is installed
    if ! command -v freqtrade &> /dev/null; then
        print_error "Freqtrade is not installed or not in PATH"
        print_status "Please install Freqtrade first: https://www.freqtrade.io/en/latest/installation/"
        exit 1
    fi
    
    # Check if config file exists
    if [ ! -f "$CONFIG_FILE" ]; then
        print_error "Config file not found: $CONFIG_FILE"
        exit 1
    fi
    
    # Check if strategy file exists
    if [ ! -f "$STRATEGY_PATH/$STRATEGY.py" ]; then
        print_error "Strategy file not found: $STRATEGY_PATH/$STRATEGY.py"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Function to show available commands
show_help() {
    echo -e "${BLUE}FreqAI Demo Runner${NC}"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Available commands:"
    echo "  help              Show this help message"
    echo "  check             Check prerequisites"
    echo "  train             Train a new FreqAI model"
    echo "  backtest          Run backtesting with FreqAI"
    echo "  dry-run           Start dry-run trading"
    echo "  hyperopt          Run hyperopt optimization"
    echo "  test              Run unit tests"
    echo "  all               Run all tests"
    echo "  clean             Clean up old models"
    echo "  status            Show current status"
    echo ""
    echo "Examples:"
    echo "  $0 train          # Train a new model"
    echo "  $0 backtest       # Run backtesting"
    echo "  $0 dry-run        # Start dry-run trading"
}

# Function to train a new model
train_model() {
    print_status "Training new FreqAI model..."
    
    freqtrade trade \
        --strategy "$STRATEGY" \
        --config "$CONFIG_FILE" \
        --freqaimodel "$MODEL" \
        --strategy-path "$STRATEGY_PATH" \
        --dry-run-wallet 1000
    
    print_success "Model training completed"
}

# Function to run backtesting
run_backtest() {
    print_status "Running FreqAI backtesting..."
    
    freqtrade backtesting \
        --strategy "$STRATEGY" \
        --strategy-path "$STRATEGY_PATH" \
        --config "$CONFIG_FILE" \
        --freqaimodel "$MODEL" \
        --timerange "$TIMERANGE"
    
    print_success "Backtesting completed"
}

# Function to start dry-run trading
start_dry_run() {
    print_status "Starting FreqAI dry-run trading..."
    
    freqtrade trade \
        --strategy "$STRATEGY" \
        --config "$CONFIG_FILE" \
        --freqaimodel "$MODEL" \
        --strategy-path "$STRATEGY_PATH" \
        --dry-run-wallet 1000
    
    print_success "Dry-run trading started"
}

# Function to run hyperopt
run_hyperopt() {
    print_status "Running FreqAI hyperopt optimization..."
    
    freqtrade hyperopt \
        --hyperopt-loss SharpeHyperOptLoss \
        --strategy "$STRATEGY" \
        --freqaimodel "$MODEL" \
        --strategy-path "$STRATEGY_PATH" \
        --config "$CONFIG_FILE" \
        --timerange "$TIMERANGE" \
        --epochs 100
    
    print_success "Hyperopt optimization completed"
}

# Function to run tests
run_tests() {
    print_status "Running FreqAI tests..."
    
    # Check if pytest is available
    if ! command -v pytest &> /dev/null; then
        print_warning "pytest not found, installing..."
        pip install pytest pytest-cov pytest-mock
    fi
    
    # Run tests
    pytest tests/test_freqai_*.py -v
    
    print_success "Tests completed"
}

# Function to run all tests with coverage
run_all_tests() {
    print_status "Running all FreqAI tests with coverage..."
    
    # Check if pytest is available
    if ! command -v pytest &> /dev/null; then
        print_warning "pytest not found, installing..."
        pip install pytest pytest-cov pytest-mock
    fi
    
    # Run all tests with coverage
    pytest tests/test_freqai_*.py -v \
        --cov=freqtrade.templates.FreqaiExampleStrategy \
        --cov-report=term-missing \
        --cov-report=html
    
    print_success "All tests with coverage completed"
    print_status "Coverage report generated in htmlcov/"
}

# Function to clean up old models
clean_models() {
    print_status "Cleaning up old FreqAI models..."
    
    # Get identifier from config
    IDENTIFIER=$(grep -o '"identifier": "[^"]*"' "$CONFIG_FILE" | cut -d'"' -f4)
    
    if [ -n "$IDENTIFIER" ]; then
        MODEL_DIR="user_data/models/$IDENTIFIER"
        if [ -d "$MODEL_DIR" ]; then
            print_warning "Removing model directory: $MODEL_DIR"
            rm -rf "$MODEL_DIR"
            print_success "Old models cleaned up"
        else
            print_status "No model directory found: $MODEL_DIR"
        fi
    else
        print_error "Could not extract identifier from config"
    fi
}

# Function to show current status
show_status() {
    print_status "Current FreqAI status..."
    
    # Check config file
    if [ -f "$CONFIG_FILE" ]; then
        print_success "Config file: $CONFIG_FILE"
        
        # Extract key parameters
        IDENTIFIER=$(grep -o '"identifier": "[^"]*"' "$CONFIG_FILE" | cut -d'"' -f4)
        TRAIN_PERIOD=$(grep -o '"train_period_days": [0-9]*' "$CONFIG_FILE" | cut -d':' -f2 | tr -d ' ,')
        BACKTEST_PERIOD=$(grep -o '"backtest_period_days": [0-9]*' "$CONFIG_FILE" | cut -d':' -f2 | tr -d ' ,')
        
        echo "  - Identifier: $IDENTIFIER"
        echo "  - Training period: $TRAIN_PERIOD days"
        echo "  - Backtest period: $BACKTEST_PERIOD days"
    else
        print_error "Config file not found: $CONFIG_FILE"
    fi
    
    # Check strategy file
    if [ -f "$STRATEGY_PATH/$STRATEGY.py" ]; then
        print_success "Strategy file: $STRATEGY_PATH/$STRATEGY.py"
    else
        print_error "Strategy file not found: $STRATEGY_PATH/$STRATEGY.py"
    fi
    
    # Check if models exist
    if [ -n "$IDENTIFIER" ]; then
        MODEL_DIR="user_data/models/$IDENTIFIER"
        if [ -d "$MODEL_DIR" ]; then
            print_success "Model directory: $MODEL_DIR"
            MODEL_COUNT=$(find "$MODEL_DIR" -name "*.joblib" -o -name "*.pkl" | wc -l)
            echo "  - Trained models: $MODEL_COUNT"
        else
            print_warning "No model directory found: $MODEL_DIR"
        fi
    fi
    
    # Check test files
    TEST_COUNT=$(find tests -name "test_freqai_*.py" 2>/dev/null | wc -l)
    if [ "$TEST_COUNT" -gt 0 ]; then
        print_success "Test files: $TEST_COUNT found"
    else
        print_warning "No test files found"
    fi
}

# Main script logic
main() {
    case "${1:-help}" in
        "help")
            show_help
            ;;
        "check")
            check_prerequisites
            ;;
        "train")
            check_prerequisites
            train_model
            ;;
        "backtest")
            check_prerequisites
            run_backtest
            ;;
        "dry-run")
            check_prerequisites
            start_dry_run
            ;;
        "hyperopt")
            check_prerequisites
            run_hyperopt
            ;;
        "test")
            run_tests
            ;;
        "all")
            run_all_tests
            ;;
        "clean")
            clean_models
            ;;
        "status")
            show_status
            ;;
        *)
            print_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
