# PR 886 Notes: GFQL WHERE

## GPU toggles / experiments
- `GRAPHISTRY_CUDF_SAME_PATH_MODE=auto|oracle|strict` controls same-path executor selection when `Engine.CUDF` is requested.

## Commits worth toggling (GPU perf/debug)
- d1e11784 perf(df_executor): DF-native cuDF forward prune
- e85fa8e7 fix(filter_by_dict): allow bool filters on object columns
