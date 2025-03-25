# First time around

1. Create a account at pypi.org. 
2. Set up 2FA
3. Message Benjamin Midtvedt, Giovanni Volpe, or another admin to get added to the project. You'll receive an invite that you need to accept.
4. You'll also be provided a security code that looks like: pypi-[base64 hash]
This will be used to authenticate you when you upload new versions. This should be stored in a file called .pypirc, as follows:

```
[distutils]
index-servers =
    pypi

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = pypi-AgEI...
```

the file should be saved on windows:
C:\Users\{user}\.pypirc
on mac:
$HOME/.pypirc

5. Install build tools and Twine. You need to install both twine (for uploading to PyPI) and the basic build tools.
In a terminal, run:

```bash
pip install --upgrade setuptools wheel twine
```

# To upload a new version to pip

1. Checkout and pull the develop branch.
2.a Ensure the version is correct in setup.py (for example, version="0.1.2"). This will be the version of the new release. 
2.b If the version is incorrect, make a pull request to develop with the version bump. This can be accepted without running tests. Once done, pull the branch again.
3. In a terminal, go to the root folder of deeplay (the one with setup.py, LICENCE.txt, etc.)
4. Run (this compiles the project to a distributable wheel): 

```bash
python setup.py sdist bdist_wheel
```

5. Run (this uploads to pypi):

```bash
python -m twine upload --repository pypi dist/*
```

Note, it will take a minute or two before the release is available on pip.

# To grant new collaborator publishing permissions

1. Tell collaborator to follow # First time around
2. Go to https://pypi.org/manage/projects/ and click Manage on the project they should be added to.
3. Click Collaborators
4. Invite collaborator
5. Go to https://pypi.org/manage/account/
6. Click Add API token
7. Set token name to collaborator name, and set scope to the project. 
8. Give the token to the collaborator. If the collaborator leaves, access can be revoked by deleting the token.