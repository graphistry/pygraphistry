# Accenture Cyber Labs Deploy Notes

GPU box IP (also use as domain name): `10.5.151.121`

## System Prep

Create system user and group `graphistry-dev`:

    sudo adduser --system --group --no-create-home --shell /usr/sbin/nologin graphistry-dev

Create folder `/opt/graphistry`, and `chown graphistry-dev:graphistry-dev` + `chmod 755` it.

Create folder `/var/log/graphistry`, and `chown graphistry-dev:graphistry-dev` + `chmod 755` it.

Create folder `/var/log/clients`, and `chown graphistry-dev:graphistry-dev` + `chmod 755` it.

Install apt package `libkrb5-dev`.


## Graphistry Code


## MongoDB

Listens on 0.0.0.0:27017


## Supervisord


## Nginx
