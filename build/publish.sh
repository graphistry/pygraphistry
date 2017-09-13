#!/bin/bash -e

cd $(dirname $0)/../ > /dev/null

# Relevant Jenkins environment variables:
# BUILD_NUMBER - The current build number, such as "153"
# CHANGE_TARGET - The target or base branch to which the change should be merged

TARGET_REF=${CHANGE_TARGET:-master}
COMMIT_ID=$(git rev-parse --short HEAD)
BRANCH_NAME=$(git name-rev --name-only HEAD)
BUILD_TAG=${BUILD_TAG:-test}-${BUILD_NUMBER:-dev}
SEMVER=`jq -r .version lerna.json | cut -d '.' -f 1,2`

PROJECTS=packages
NAMESPACE=graphistry
LERNA_CONTAINER="$NAMESPACE/lerna"
REPO_VERSION=${SEMVER}.${BUILD_NUMBER:-0}

docker build -f build/dockerfiles/Dockerfile-lerna \
	--build-arg TARGET_REF=${TARGET_REF} \
	-t ${LERNA_CONTAINER}:${BUILD_TAG} \
	.

docker run --rm \
	-v ${PWD}/${PROJECTS}:/${NAMESPACE}/${PROJECTS} \
	${LERNA_CONTAINER}:${BUILD_TAG} \
	lerna publish --yes --exact \
				  --skip-git --skip-npm \
				  --since ${TARGET_REF} \
				  --repo-version=${REPO_VERSION}

for PROJECT in $(docker run --rm ${LERNA_CONTAINER}:${BUILD_TAG})
do
	echo "checking $PROJECT for build files"

	PROJECT_BUILD_DIR="./$PROJECTS/$PROJECT/build"

	if [ ! -f "$PROJECT_BUILD_DIR/publish.sh" ]; then
		echo "expected $PROJECT_BUILD_DIR/publish.sh, but none found"
		exit 1
	fi

	CONTAINER_NAME="$NAMESPACE/$PROJECT"

	echo "building container: $CONTAINER_NAME"

	pushd "$PROJECT_BUILD_DIR" > /dev/null
	sh ./publish.sh ${BUILD_TAG} ${CONTAINER_NAME} ${COMMIT_ID} ${BRANCH_NAME} &
	popd > /dev/null
done

wait

echo "publish finished"
