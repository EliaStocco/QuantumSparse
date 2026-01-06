#!/bin/bash
# Determine the directory of the script
SCRIPT_DIR=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")

# Set QSLIB to the script directory
export QSLIB="$SCRIPT_DIR"
echo "Environment variable QSLIB set to '${QSLIB}'."
# source ${QSLIB}/check_requirements.sh $@
export PYTHONPATH="$PYTHONPATH:${QSLIB}/quantumsparse"
export PYTHONPATH="$PYTHONPATH:${QSLIB}/quantumsparse/scripts"
echo "The path '${QSLIB}' has been added to PYTHONPATH."

# Add script subdirectories to PATH
for dir in "${QSLIB}"/scripts/* ; do
    if [ -d "$dir" ]; then
        export PATH="$PATH:$dir"
        # Make Python files executable in each subdirectory
        find "$dir" -name "*.py" -type f -exec chmod +x {} \;
    fi
done
echo "All scripts in '${QSLIB}/scripts' made executable and added to PATH."
