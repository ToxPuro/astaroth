#!/bin/bash

TEST_DIR="$AC_HOME/test"

build_project() {
    local dir="$1"
    local log="$dir/compilation_log.txt"
    rm -f $log
    touch $log
    cd "$dir" || return 1
    rm -f FAIL
    test_name=$(basename "${dir}")
    
    # Run test script if it exists and is executable
    if [ -x "build.sh" ]; then
        echo "Building $test_name"
        ./build.sh &>> $log
        exit_code=$?

        if [ $exit_code -ne 0 ]; then
            echo "[ERROR] Build failed for $test_name with exit code $exit_code"
            cat $log
            echo "" > $log
	    touch FAIL
        else
            echo "[SUCCESS] Build succeeded for $test_name"
        fi
    else
        echo "[WARNING] No build.sh found for $test_name"
    fi

    cd - >/dev/null  # Return to previous directory silently
}

export -f build_project
for dir in "$TEST_DIR"/*/; do
    #TP: for multithreading suppressed for now since bitbucket pipelines runs out of memory
    #build_project "$dir" &
    build_project "$dir"
    rm -rf "$dir/build"
done

wait

if [ ! -f $TEST_DIR/*/FAIL ]; then
    echo "[SUCCESS] All builds completed successfully!"
    exit 0  # Success
else
    echo "[ERROR] Some builds failed. Check compilation_log.txt for details."
    exit 1  # Failure
fi

