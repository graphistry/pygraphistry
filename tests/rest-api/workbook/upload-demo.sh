#!/usr/bin/env bash

# The ID of the workbook to download (should match the workbook we download in `./download-demo.sh`)
workbook_id="3dd9faab9f69f02f"
SERVER="${SERVER:-${1:-"staging.graphistry.com"}}"

cd "$(dirname "$0")"
curl -L -X POST -H "Content-Type: application/json" --data @workbook.json "http://${SERVER}/workbook"

echo "View the workbook at: http://${SERVER}/graph/graph.html?workbook==${workbook_id}" >&2
