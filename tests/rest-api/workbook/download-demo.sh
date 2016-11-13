#!/usr/bin/env bash

# The ID of the workbook to download (should match the workbook we upload in `./upload-demo.sh`)
workbook_id="3dd9faab9f69f02f"
SERVER="${SERVER:-${1:-"staging.graphistry.com"}}"

cd "$(dirname "$0")"

echo "First, running upload-demo.sh to seed the server with at least one workbook..." >&2
echo "" >&2

"./upload-demo.sh" >&2
# Ensure the server has finished saving the workbook, etc., before we fetch it
sleep 2

curl -L "http://${SERVER}/workbook/${workbook_id}"

echo ""
