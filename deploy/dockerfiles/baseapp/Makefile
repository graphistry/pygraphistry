default:
	docker build -t `cat Dockerfile.version` -t `cat Dockerfile.version | cut -d : -f 1` . && docker push `cat Dockerfile.version` && docker push `cat Dockerfile.version | cut -d : -f 1`
