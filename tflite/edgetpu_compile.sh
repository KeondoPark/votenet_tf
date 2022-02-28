#!/bin/bash

TFLITE_DIR="tflite_models/2way_nofp_sep"

for entry in "$TFLITE_DIR"/*
do    
    filename=$(basename -- "$entry")
    extension="${filename##*.}"
    filename="${filename%.*}"

    if [ "$extension" = "tflite" ]; then
        if [[ "$filename" != *"edgetpu"* ]]; then
            echo "$filename"
            edgetpu_compiler -d "$TFLITE_DIR/$filename.tflite" -o "$TFLITE_DIR"
        fi
    fi
done