#!/bin/bash
set -e

if [ ! -e text8 ]; then
  if hash wget 2>/dev/null; then
    wget -O text8.zip http://mattmahoney.net/dc/text8.zip
  else
    curl -o text8.zip http://mattmahoney.net/dc/text8.zip
  fi
  unzip text8.zip
  rm text8.zip
fi