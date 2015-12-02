#Install Git Checkout:

	./setup.py develop


#Uninstall Checkout:

	./setup.py develop --uninstall

#Release Procedure
1. Bump version number to X.X.X in setup.py
2. git commit
3. git tag X.X.X
4. git push --folow-tags

###Package & Upload:
5. Get ~/.pypirc file from the powers that be.
6. Run `./setup.py sdist upload -r pypi`



