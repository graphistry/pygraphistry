#!/bin/bash -ex

cd $(dirname "$0")/../ > /dev/null

# Relevant Jenkins environment variables:
# BUILD_NUMBER - The current build number, such as "153"
# CHANGE_TARGET - The target or base branch to which the change should be merged

TARGET_REF=${CHANGE_TARGET:-master}
COMMIT_ID=$(git rev-parse --short HEAD)
BRANCH_NAME=$(git name-rev --name-only HEAD)
BUILD_TAG=${BUILD_TAG:-test}-${BUILD_NUMBER:-dev}

PROJECTS=packages
NAMESPACE=graphistry
LERNA_CONTAINER="$NAMESPACE/lerna-docker"
LERNA_LS_CHANGED="lerna exec --loglevel=error --since $TARGET_REF -- echo \${PWD##*/}"

docker build -f build/dockerfiles/Dockerfile-lerna \
	--build-arg TARGET_REF=${TARGET_REF} \
	-t ${LERNA_CONTAINER}:${BUILD_TAG} \
	.

for PROJECT in $(docker run --rm ${LERNA_CONTAINER}:${BUILD_TAG} ${LERNA_LS_CHANGED})
do
	echo "checking $PROJECT for build files"

	PROJECT_BUILD_DIR="./$PROJECTS/$PROJECT/build"

	if [ ! -f "$PROJECT_BUILD_DIR/test.sh" ]; then
		echo "expected $PROJECT_BUILD_DIR/test.sh, but none found"
		exit 1
	fi

	CONTAINER_NAME="$NAMESPACE/$PROJECT"

	echo "building container: $CONTAINER_NAME"

	sh ${PROJECT_BUILD_DIR}/test.sh ${BUILD_TAG} ${CONTAINER_NAME} ${COMMIT_ID} ${BRANCH_NAME} &
done

wait

echo "test finished"
