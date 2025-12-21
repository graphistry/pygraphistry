# Docs

Uses Sphinx to extract .RST from parent .PY and converts into docs/build/ .HTML according to docs/source/ templates

## Run

```bash
docs $ ./html.sh
```

It is volume-mounted to emit `docs/_build/html/index.html` 

For generating all types used in production, e.g., epub and pdf as well, mimic the CI runner:

```bash
docs $ ./ci.sh
```

## Architecture

### Containerized sphinx and nbconvert

 `ci.sh` runs `docker compose build` and then `docker compose run --rm sphinx` to build the docs

  - It also volume mounts demos/ (.ipynb files) into docs/source/demos

  - As part of the run, nbsphinx calls nbconvert and then sphinx to convert the .ipynb files into .rst files

    - nbconvert rewrites relative graph.html urls to assume hub.graphistry.com

    - We currently use the preexisting .ipynb output, but in the future should reexecute the notebooks to ensure they are up to date

### Caching

The setup aggressively caches for fast edit cycles:

  - Standard docker layer caching

  - Sphinx caching to avoid recompiling unchanged files on reruns

    - docs/doctrees for tracking
    - docs/_build for output

### Error checking

Sphinx in strict mode:

  - Ex: .rst files that aren't included in any TOC

  - Ex: broken links in .ipynb and .rst files

  - Ex: verifies .ipynb follow .rst conventions like first element is `# a title` and only one of them

### Output artifacts

- The output is docs/_build

  - epub, html, and latexpdf


## CI

CI runs `html.sh` and checks for warnings and errors. If there are any, it will fail the build.

Notebook validation is enabled by default. To disable: `VALIDATE_NOTEBOOK_EXECUTION=0 ./ci.sh`

## Develop

- Edit the `demo/` notebooks, `graphistry/` Python code, and `docs/source` .rst files

- Rerun `cd docs && ./ci.sh` to see the changes, with the benefit of docker & sphinx incremental builds

- Check your results in `docs/_build/html/index.html` or the equivalent epub and pdf files

