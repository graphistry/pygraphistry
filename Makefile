COMPOSE_FILE=$(CURDIR)/compose/docker-compose.yml

include ./.env

conda: conda.2 conda.3
conda.2:
	conda env create --file environment2.yml -p ./conda/envs/graphistry27

conda.3:
	conda env create --file environment3.yml -p ./conda/envs/graphistry37

test:
	docker-compose -f $(COMPOSE_FILE) run test

jupyter:
	docker-compose -f $(COMPOSE_FILE) up jupyter

neo4j:
	docker-compose -f $(COMPOSE_FILE) up neo4j
