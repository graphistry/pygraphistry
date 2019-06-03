docker run -v "${PWD}/pypirc":/root/.pypirc -v "${PWD}":/repo -w /repo python bash -c "python3 setup.py bdist_wheel upload -r pypi"
