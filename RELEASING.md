# Releasing LazyBrick

## First PyPI release with Trusted Publishing

The package name is `lazybrick`. PyPI does not treat empty packages as valid name
reservations, so every published release should contain usable project functionality.

1. Sign in to PyPI and create a pending Trusted Publisher with:

   ```text
   PyPI project name: lazybrick
   GitHub owner: lazy-brick
   GitHub repository: lazybrick
   Workflow filename: publish.yml
   Environment name: pypi
   ```

2. Confirm that `src/lazybrick/__about__.py` contains the intended version.
3. Run the local release checks:

   ```bash
   python -m pytest
   python -m build
   python -m twine check dist/*
   ```

4. Commit and push the version.
5. Create a GitHub release whose tag matches the package version, for example
   `v0.0.1`.
6. The `publish.yml` workflow will build the distributions and publish them through
   GitHub OIDC. No PyPI token should be stored in GitHub.

For an existing PyPI project, configure the same publisher under the project's
Publishing settings.
