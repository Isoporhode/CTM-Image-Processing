#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
for i in {0..9}
do
    for a in $(seq $(($i+1)) 9)
    do
        echo "$i $a"
        python3 extract_images.py --LABELS $i $a
    done
done
