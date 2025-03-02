#!/bin/bash

# Run flake8
flake8 .

# Check if flake8 ran successfully
if [ $? -eq 0 ]; then
    echo "Code passed flake8 checks!"
else
    echo "Code failed flake8 checks!"
    exit 1
fi
