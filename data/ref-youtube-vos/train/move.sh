#!/bin/bash

# 定义源根目录和目标根目录
SRC_ROOT="/HOME/scz0rv9/run/codes/ReferFormer/data/ref-youtube-vos/train/JPEGImages"
DEST_ROOT="/HOME/scz0rv9/run/codes/ReferFormer/data/ref-youtube-vos/train/JPEGFlows"

find "$SRC_ROOT" -type d -name "flows" | while read -r flows; do

  parent_dir=$(dirname "$flows")
  echo "parent_dir: $parent_dir"
  relative_path="${parent_dir#$SRC_ROOT/}"
  echo "relative_path: $relative_path"
  dest_dir="$DEST_ROOT/$relative_path"
  echo "dest_dir: $dest_dir"

  mkdir -p "$dest_dir"

  mv "$flows"/* "$dest_dir"

  rmdir "$flows"

  echo "Moved files from $flows to $dest_dir"
done

echo "All flowdir directories have been processed."
