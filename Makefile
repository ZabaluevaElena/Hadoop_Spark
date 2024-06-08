build:
	docker-compose build
build-nc:
	docker-compose build --no-cache

build-progress:
	docker-compose build --no-cache --progress=plain
down:
	docker-compose down --volumes --remove-orphans
run:
	make down && docker-compose up

run-d:
	make down && docker-compose up -d
run-scaled:
	make down && docker-compose up --scale spark-yarn-worker=3

run-scaled-d:
	make down && docker-compose up -d --scale spark-yarn-worker=3
stop:
	docker-compose stop

submit:
	docker exec da-spark-yarn-master spark-submit --master yarn --deploy-mode cluster ./apps/$(app)

clean-model:
	docker exec da-spark-yarn-master hdfs dfs -rm -r /opt/spark/data/model