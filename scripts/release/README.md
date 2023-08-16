# Release Script(s) Usage

## Versioning

This script needs:
- An existing `openvalidators/__init__.py` file
- An existing `__version__` variable in that file
- An existing version for that variable

This process will generate:
- A modified version in `__version__` for the update type specified

### Example Usage 
`./scripts/release/versioning.sh -U patch -A`  

Where:
* `-U` (major|minor|patch) the type of update 
* `-A` is to apply the script changes


## Add Notes Changelog

This script needs:
- An existing `CHANGELOG.md` file with at least three lines
- An existing git tag for the previous version

This process will generate:
- A new entry in `CHANGELOG.md`

##### *Note: This will only list merge commits into the release branch since the last tag*

### Example Usage 
`./scripts/release/add_notes_changelog.sh -P 1.1.7 -V 1.1.8 -B hotfix/serve-val-axon -T $GIT -A`  

Where:
* `-P` is the old version
* `-V` is the new version
* `-B` is the release branch name (default: `release/vX.X.X`)
* `-T` is the GIT API token
* `-A` is to apply the script changes

## Release

This script needs:
- An existing `__version__` variable in the `openvalidators/__init__.py` file
- Version in the `__version__` variable is not a git tag already

This process will generate:
- Tag in Github repo: https://github.com/opentensor/validators/tags
- Release in Github: https://github.com/opentensor/validators/releases


### Example Usage 
`./scripts/release/release.sh -T $GIT -A`  

Where:
* `-T` is the GIT API token
* `-A` is to apply the script changes

