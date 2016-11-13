# Graphistry REST API

<!-- TOC depthFrom:2 depthTo:6 withLinks:1 updateOnSave:1 orderedList:1 -->

1. [Overview](#overview)
2. [Upload a workbook](#upload-a-workbook)
	1. [Parameters](#parameters)
	2. [Response](#response)
	3. [Example](#example)
3. [Download a workbook](#download-a-workbook)
	1. [Parameters](#parameters)
	2. [Response](#response)
	3. [Example](#example)
4. [Modify a workbook](#modify-a-workbook)

<!-- /TOC -->

## Overview

All Graphistry REST API endpoints use JSON for encoding data. The server will include a `Content-Type: application/json` header with all responses. Clients should likewise send the same header in their request, if they are sending data (e.g., in a `POST` request).

Examples will use `staging.graphistry.com` as the server. Replace it with your own server's address when running the examples, if you're using your own server.


## Upload a workbook

    POST /workbook

This endpoint accepts workbook data, and stores it so that it can later be loaded in a visualization.

> **Note**: If a workbook with the same ID as the uploaded workbook already exists on the server, it will be overwritten with the newly uploaded workbook.


### Parameters

The workbook JSON data, encoded as `application/json`, in the request body.


### Response

An HTTP 201 response on success, or an error code (most likely 400 or 500) on failure. The response body contains a JSON payload (`Content-Type: application/json`) with the following fields.

| Field      | Type      | Description |
|------------|-----------|-------------|
| `success`  | `boolean` | Indicates if the request completed successfully, or encountered an error. |
| `error`    | `string`  | *(Optional)* A string describing the error, if one occurred. |
| `workbook` | `string`  | The ID of the newly created workbook, which can be used as the `?workbook=` parameter in a visualization URI. |
| `view`     | `uri`     | A URI (as a absolute path, with no host) you can use to load this workbook into a visualization. |


### Example

The example uses `curl` to upload a workbook file named `workbook.json` in the current directory, then loads that workbook into a visualization.

1. In a terminal, run:

        curl -L -X POST -H "Content-Type: application/json" --data @workbook.json http://staging.graphistry.com/workbook`

2. The server will respond with:

        {"success":true,"workbook":"3dd9faab9f69f02f","view":"/graph/graph.html?workbook=3dd9faab9f69f02f"}

    Note the `view` field, which we'll use to open a visualization for the workbook.

3. In a web browser, open http://staging.graphistry.com/graph/graph.html?workbook=3dd9faab9f69f02f



## Download a workbook

    GET /workbook/:workbook

Fetch a previously created/uploaded workbook and return it as JSON data.

`:workbook` is the ID of the workbook you wish to download.

### Parameters

None.


### Response

If the call succeeded, the server will respond with a HTTP 200 response code, and the body of the response will be the JSON data of the workbook.

If the call encounters an error, it will return a non-200 'error' HTTP response code (most likely 404), and the a JSON response body with the following fields *instead* of the workbook JSON.

| Field      | Type      | Description |
|------------|-----------|-------------|
| `success`  | `boolean` | Indicates that the request encountered an error (always `false`). |
| `error`    | `string`  | A string describing the error. |

> **Note**: Since a successful call returns the workbook JSON without modification, the `success` and `error` fields will *not* be present in a successful response. You should examine the response code to determine if the call succeeded or failed, then examine the response body of failed calls to find more information on the error.

The most likely reason a call to this endpoint would fail is that the workbook being requested does not exist on the server (either in the server's local storage or, if enabled, AWS S3 storage).


### Example

This example assumes we have previously uploaded a workbook with ID `3dd9faab9f69f02f`. It uses `curl` to fetch that workbook at save it to the current directory as `downloaded-workbook.json`.

1. In a terminal, run:

        curl 'http://staging.graphistry.com/workbook/3dd9faab9f69f02f' > ./downloaded-workbook.json`

2. The server will respond with the workbook's contents (as JSON-encoded data), which `curl` will save to `downloaded-workbook.json`.

3. You can modify the workbook locally, examine it's contents (for example, with the `jq` utility), and optionally upload the modified workbook back to the server.



## Modify a workbook

The Graphistry REST API does not have a dedicated "modify a workbook" endpoint. Instead, you can combine the "download" and "upload" endpoints to accomplish the same thing.

First, [download the workbook](#download-a-workbook) you wish to modify. Make the desired modifications to the workbook file locally, ensuring that you do not modify the `id` field of the workbook.

Then, [upload the modified workbook file](#upload-a-workbook) to the server. Because the uploaded workbook has the same ID as the existing workbook, the uploaded workbook will overwrite the existing workbook on the server.
