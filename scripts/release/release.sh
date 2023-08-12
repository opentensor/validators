#!/bin/bash

#
# In this script you are going to find the process of releasing Openvalidators.
#
# This script needs:
#   - An existing __version__ var in the __init__.py file
#   - Version in __version__ var is not a git tag already
#
# This process will generate:
#   - Tag in Github repo: https://github.com/opentensor/validators/tags
#   - Release in Github: https://github.com/opentensor/validators/releases
#   - New entry in CHANGELOG.md file
#

###
# Utils
###

source ${BASH_SOURCE%/*}/utils.sh

function help(){
    echo Usage:
    echo \ \  $0
    echo
    echo This script release a openvalidators version.
    echo
    echo This script needs:
    echo \ \ - An existing __version__ var in the __init__.py file
    echo \ \ - Version in __version__ var is not a git tag already
    echo
}
###

###
# Start of release process
###

# 0. Check requirements
# Expected state for the execution environment
#  - __version__ exists inside file 'openvalidators/__init__.py'
#  - Version has the expected format

CODE_WITH_VERSION='openvalidators/__init__.py'

CODE_VERSION=`grep '__version__\ \=\ ' $CODE_WITH_VERSION | awk '{print $3}' | sed 's/"//g'`
VERSION=$CODE_VERSION

if ! [[ "$CODE_VERSION" =~ ^[0-9]+.[0-9]+.[0-9]+$ ]];then
  echo_error "Requirement failure: Version in code '$CODE_VERSION' with wrong format"
  exit 1
fi

# 1. Get options

## Defaults
APPLY="false"
APPLY_ACTION=""

while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      help
      exit 0
      ;;
    -A|--apply)
      APPLY="true"
      APPLY_ACTION="--apply"
      shift # past argument
      ;;
    -T|--github-token)
      GITHUB_TOKEN="$2"
      shift # past argument
      shift # past value
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

if [[ $APPLY == "true" ]]; then
  echo_warning "Not a Dry run exection"
else
  echo_warning "Dry run execution"
fi

if [[ -z $GITHUB_TOKEN && $APPLY == "true" ]]; then
    echo_error "Github token required (-T, --github-token)"
    exit 1
fi

# 2. Checking version

CURRENT_VERSION_EXISTS=$(git tag | grep $VERSION)
if [[ ! -z $CURRENT_VERSION_EXISTS ]]; then
    echo_error "Current version '$VERSION' already exists"
    help
    exit 1
fi

PREV_VERSION_TAG=`get_git_tag_higher_version`

TAG_NAME=v$VERSION

## 2.1. Current VERSION is not already a tag

echo_info "Detected new version tag: $VERSION"
echo_info "Previous version tag: $PREV_VERSION_TAG"
echo_info "Tag generated: $TAG_NAME"

# 3. Create Github resources
if [[ $APPLY == "true" ]]; then
  ${BASH_SOURCE%/*}/github_release.sh $APPLY_ACTION --github-token $GITHUB_TOKEN -P $PREV_VERSION_TAG -V $VERSION
else
  ${BASH_SOURCE%/*}/github_release.sh $APPLY_ACTION $GITHUB_TOKEN -P $PREV_VERSION_TAG -V $VERSION
fi
