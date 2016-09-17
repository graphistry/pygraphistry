FROM graphistry/js-and-gpu:0.10.46

RUN apt-get update && apt-get install -y supervisor libkrb5-dev gettext pigz logrotate python-pip libpq-dev postgresql-client && rm -rf /var/lib/apt/lists/*
RUN pip install pymongo
