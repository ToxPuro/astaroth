#!/bin/bash

TEST_DIR="$AC_HOME/test"
overall_success=true  # Flag to track overall success
LOG="$PWD/compilation_log.txt"
rm -f $LOG
touch $LOG

# Iterate over all subdirectories in TEST_DIR
for dir in "$TEST_DIR"/*/; do
    cd "$dir" || continue
    test_name=$(basename "${dir}")
    # Run test script if it exists and is executable
    if [ -x "build.sh" ]; then
        echo "Building $test_name"
        ./build.sh &>> $LOG
        exit_code=$?  # Capture exit code of build.sh

        if [ $exit_code -ne 0 ]; then
            echo "[ERROR] Build failed for $test_name with exit code $exit_code"
	    cat $LOG
	    echo "" > $LOG
            overall_success=false  # Mark failure but continue with other directories
        else
            echo "[SUCCESS] Build succeeded for $test_name"
        fi
    else
        echo "[WARNING] No build.sh found for $test_name"
    fi

    cd - >/dev/null  # Return to previous directory silently
done

# Final build summary
if [ "$overall_success" = true ]; then
    echo "[SUCCESS] All builds completed successfully!"
    exit 0  # Success
else
    echo "[ERROR] Some builds failed. Check compilation_log.txt for details."
    exit 1  # Failure
fi

