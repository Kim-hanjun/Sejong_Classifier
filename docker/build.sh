#!/bin/bash
# Run this script at the root directory!
#   cd {REPOSOTORY_HOME}
#   ./docker/docker_build.sh

DOCKER_IMG_VER=`tail -n 1 ./docker/build.history | awk -F' ' '{print $1}'`

# Support for password authentication was removed on August 13, 2021. Please use a personal access token instead.
# https://docs.github.com/en/github/authenticating-to-github/keeping-your-account-and-data-secure/creating-a-personal-access-token
GITHUB_NAME=""
GITHUB_PASSWORD=""
# Calling git clone using password with special character
# https://fabianlee.org/2016/09/07/git-calling-git-clone-using-password-with-special-character/
GITHUB_PASSWORD="${GITHUB_PASSWORD//!/%21}"
GITHUB_PASSWORD="${GITHUB_PASSWORD//#/%23}"
GITHUB_PASSWORD="${GITHUB_PASSWORD//$/%24}"
GITHUB_PASSWORD="${GITHUB_PASSWORD//&/%26}"
GITHUB_PASSWORD="${GITHUB_PASSWORD//\'/%27}"
GITHUB_PASSWORD="${GITHUB_PASSWORD//(/%28}"
GITHUB_PASSWORD="${GITHUB_PASSWORD//)/%29}"
GITHUB_PASSWORD="${GITHUB_PASSWORD//\*/%2A}"
GITHUB_PASSWORD="${GITHUB_PASSWORD//+/%2B}"
GITHUB_PASSWORD="${GITHUB_PASSWORD//,/%2C}"
GITHUB_PASSWORD="${GITHUB_PASSWORD//\//%2F}"
GITHUB_PASSWORD="${GITHUB_PASSWORD//:/%3A}"
GITHUB_PASSWORD="${GITHUB_PASSWORD//;/%3B}"
GITHUB_PASSWORD="${GITHUB_PASSWORD//=/%3D}"
GITHUB_PASSWORD="${GITHUB_PASSWORD//\?/%3F}"
GITHUB_PASSWORD="${GITHUB_PASSWORD//@/%40}"
GITHUB_PASSWORD="${GITHUB_PASSWORD//[/%5B}"
GITHUB_PASSWORD="${GITHUB_PASSWORD//]/%5D}"
echo $GITHUB_NAME:$GITHUB_PASSWORD > github_secret.txt
cat github_secret.txt

sudo DOCKER_BUILDKIT=1 docker build --progress=plain --secret id=github_secret,src=github_secret.txt \
-t $DOCKER_IMG_VER -f ./docker/Dockerfile .

rm -f github_secret.txt
