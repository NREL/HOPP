#!/bin/bash

set -e

conda build conda.recipe/ -c nrel -c conda-forge -c sunpower

anaconda upload -u nrel $(conda build conda.recipe/ -c nrel -c conda-forge -c sunpower --output)

echo "Building and uploading conda package done!"

