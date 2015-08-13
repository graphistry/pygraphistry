#Install Git Checkout:

	./setup.py develop


#Uninstall Checkout:

	./setup.py develop --uninstall

#Package & Upload:
1. Get ~/.pypirc file from the powers that be.
2. Bump version in setup.py.
3. Run `./setup.py sdist upload -r pypi`



