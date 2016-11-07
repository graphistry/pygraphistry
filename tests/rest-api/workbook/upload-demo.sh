#!/usr/bin/env bash

curl -L -X POST -H "Content-Type: application/json" --data @workbook-upload-demo.json http://staging.graphistry.com/workbook

echo "View the workbook at: http://staging.graphistry.com/graph/graph.html?workbook=3dd9faab9f69f02f"
