`ansible-playbook -i hosts system.yml -vvv`

To start the server:
`sudo supervisorctl start node-server`
`sudo supervisorctl stop node-server`
`sudo supervisorctl restart node-server`

Logs:
`tail -f /var/log/node-server/server.log`