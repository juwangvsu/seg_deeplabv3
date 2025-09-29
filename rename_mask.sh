#!/bin/bash
# rename_mask_files.sh
# Rename files from "######_mask.png" → "#####.png"

for f in *_raw.png; do
  # Skip if no files match
  [ -e "$f" ] || continue

  # Remove the _mask part
  newname="${f%_raw.png}.png"

  echo "Renaming: $f → $newname"
  mv "$f" "$newname"
done

