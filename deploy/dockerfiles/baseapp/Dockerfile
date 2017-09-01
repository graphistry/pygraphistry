FROM graphistry/js-and-gpu:7.4.0

RUN apt-get update && apt-get install -y supervisor libkrb5-dev gettext pigz logrotate python-pip libpq-dev postgresql-client default-jre-headless git clinfo && rm -rf /var/lib/apt/lists/*
RUN pip install pymongo
