#!/bin/bash
# The script exports a csv file from a remote host to a local machine

if [ $# -ne 2 ]
then
  echo "Usage: download_csv.sh <user@remote_host:/path/to/source> <destination>"
  echo "Example: download_csv.sh pop:/tmp/tmp.w5pPwroy2Z/cmake-build-debug/out.csv ~/Downloads/out.csv"
else
  echo "Uploading file from $1 to $2"
  scp $1 $2
fi



