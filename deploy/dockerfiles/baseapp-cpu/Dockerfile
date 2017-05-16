FROM graphistry/js-and-cpu

RUN apt-get update && apt-get install -y supervisor libkrb5-dev gettext pigz logrotate python-pip libpq-dev postgresql-client default-jre-headless && rm -rf /var/lib/apt/lists/*
RUN pip install pymongo
