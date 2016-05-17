#!/bin/bash

file_owner='graphistry-dev'
file_group='graphistry'

if [[ $UID -ne 0 ]]; then
	echo 'Error: must be run via `sudo` to modify file permissions' >&2
	exit 1
fi

update() {
	cd "$1"

	git clean -d -x -f
	sudo -u "$file_owner" git pull
	git clean -d -x -f

	sudo -u "$file_owner" npm install

	chown -R "$file_owner":"$file_group" "$PWD"
	chmod -R g+rw "$PWD"
}

supervisorctl stop all
service supervisor stop

do; (update "central"); done &
do; (update "viz-server"); done &

date >> /var/log/graphistry/*.log

wait
service supervisor start
