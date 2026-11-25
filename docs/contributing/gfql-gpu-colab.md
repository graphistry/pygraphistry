# Running the GFQL GPU engines (local GPU or a free Colab T4)

Who this is for: people and coding agents alike; it is a reference page linked from `CONTRIBUTING.md` (people) and `.agents/skills/contribute/SKILL.md` (agents).

GFQL has two GPU engines, cuDF (`engine="cudf"`) and polars-gpu (`engine="polars-gpu"`). The CPU engines (pandas and Polars) are required for a pull request; the GPU engines are recommended. If you cannot run them, say so in the pull request and maintainers cover them. This page is for when you can.

The tests pick an engine through two environment variables: `TEST_CUDF=1` enables the cuDF cases and `TEST_POLARS_GPU=1` the polars-gpu cases (a GPU case skips when the engine cannot be imported, and the cases that read these variables skip when they are unset). `./bin/check-gfql.sh --cudf --polars-gpu` sets them for you.

## You have a GPU

Install RAPIDS for your CUDA version (the [RAPIDS install selector](https://docs.rapids.ai/install/) gives the command), plus `cudf-polars` for polars-gpu; the `cudf` and `cudf-polars` versions must match. Then:

```bash
pip install -e ".[test,polars]"
./bin/check-gfql.sh --no-lint --changed --cudf --polars-gpu
```

`--changed` runs only the test files you added or changed; drop it to run the whole GFQL test tree. Docker users: `./docker/test-gpu-local.sh` runs the GPU test image.

## You do not have a GPU: a free Colab T4

1. In Colab choose Runtime, Change runtime type, T4 GPU.
2. One cell per step:

   ```python
   !git clone https://github.com/graphistry/pygraphistry.git
   %cd pygraphistry
   !git switch my-branch            # your fork's branch: git clone your fork instead, then switch
   ```

   ```python
   !pip install -e ".[test,polars]"
   !pip install --extra-index-url=https://pypi.nvidia.com "cudf-cu12==25.*" "cudf-polars-cu12==25.*"
   ```

   Restart the runtime if Colab asks you to. Check that the GPU engines import:

   ```python
   import cudf, cudf_polars, polars
   print(cudf.__version__, polars.__version__)
   ```

   ```python
   !./bin/check-gfql.sh --no-lint --changed --cudf --polars-gpu
   ```

3. Copy the last lines of the output (the `=== summary` block) into the "Engines tested" part of your pull request.

Notes:

- This sequence was checked command by command on a GPU machine with the RAPIDS container images, not on Colab itself; if a package pin does not install on the Colab image of the day, the RAPIDS selector above is the source of truth for matching `cudf`, `cudf-polars` and `polars` versions. Tell us in the pull request and we will update this page.
- A T4 has 16 GB; if a test runs out of memory, run a narrower selection with `-- -k name` (everything after `--` goes to pytest).
