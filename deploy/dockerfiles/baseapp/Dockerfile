FROM graphistry/js-and-gpu:6.9.1

RUN apt-get update && apt-get install -y supervisor libkrb5-dev gettext pigz logrotate python-pip libpq-dev postgresql-client default-jre-headless && rm -rf /var/lib/apt/lists/*
RUN wget -O /usr/local/bin/dumb-init https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64 && chmod +x /usr/local/bin/dumb-init
RUN pip install pymongo
