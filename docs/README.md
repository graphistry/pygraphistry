# Docs

Uses Sphinx to extract .RST from parent .PY and converts into docs/build/ .HTML according to docs/source/ templates

## Run

```bash
docs $ sudo ./docker.sh
```

This is a shim to `build.sh`, which CI also uses

It is volume-mounted to emit `docs/build/html/graphistry.html`

CI will reject documentation warnings and errors

## Develop

Sphinx has to install project dev deps, so to work more quickly:

1. Start a docker session:

Run from `docs/`: 

```
docker run \
    --entrypoint=/bin/bash \
    --rm -it \
    -e USER_ID=$UID \
    -e PIP_CACHE_DIR=/cache/pip \
    -v $(pwd)/..:/doc \
    -v ~/.cache/pip:/cache/pip \
    -v ~/.cache/apt:/cache/apt \
    sphinxdoc/sphinx:8.0.2
```

2. Install deps in a sandbox:

```python
cp -r /doc /pygraphistry \
&& apt update && apt -o dir::cache=/cache/apt install -y pandoc \
&& ( cd /pygraphistry && python -m pip install -e .[docs] ) \
&& (test -d /doc/docs/source/demos || ln -s /doc/demos /doc/docs/source/demos) \
&& (cd /docs/docs/source/demos && rm -f /doc/docs/source/demos/demos || echo ok)
```

This prevents your host env from getting littered

3. Edit and build!

```bash
cd /doc/docs
make clean html SPHINXOPTS="-W --keep-going -n"
```

This emits `$PWD/docs/build/html/index.html` on your host, which you can open in a browser

If there are warnings or errors, it will run to completetion, emit them all, and then exit with a failure. CI will reject these bugs, so you need to make sure it works.
