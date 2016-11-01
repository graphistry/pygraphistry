# Graphistry REST API: Workbooks

## Upload a workbook

    POST /workbook

This endpoint accepts workbook data, and stores it so that it can later be loaded in a visualization.

### Input

The workbook JSON data, encoded as `application/json`, in the request body.

### Output

An HTTP 201 response on success, or an error code (most likely 400 or 500) on failure. The response body contains a JSON payload (`Content-Type: application/json`) with the following fields.

| Field | Type | Description |
|-------|------|-------------|
| `success` | `boolean` | Indicates if the request completed successfully, or encountered an error. |
| `error` | `string` | *(Optional)* A string describing the error, if one occurred. |
| `workbook` | `string` | The ID of the newly created workbook, which can be used as the `?workbook=` parameter in a visualization URI. |
| `view` | `uri` | A URI (as a absolute path, with no host) you can use to load this workbook into a visualization. |


### Example

The example uses `curl` to upload a workbook file named `workbook.json` in the current directory, then loads that workbook into a visualization. It will use `staging.graphistry.com` as the server; replace it with your own server's address, if you're not using staging.

1. In a terminal, run:

        curl -L -X POST -H "Content-Type: application/json" --data @workbook-upload-demo.json http://staging.graphistry.com/workbook`

2. The server will respond with:

        {"success":true,"workbook":"3dd9faab9f69f02f","view":"/graph/graph.html?workbook=3dd9faab9f69f02f"}

    Note the `view` field, which we'll use to open a visualization for the workbook.

3. In a web browser, open http://staging.graphistry.com/graph/graph.html?workbook=3dd9faab9f69f02f
