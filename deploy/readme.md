## Login ##

`ssh -A leo@54.183.193.90 -p 61630`

## Fresh dev servers: ##

Create new machine, provision with ansible_id_rsa key.

Log onto box:
ssh -A -i ansible_id_rsa.pem ubuntu@XXXXXXX

Run: (must be done by hand for now)
`sudo apt-get update && sudo apt-get install linux-headers-generic`

Comment out ansible port in hosts.yml (box is started with port 22 open instead of 61630)
`ansible-playbook -i hosts system.yml -vvvv --tags ssh`

Uncomment ansible port:
`ansible-playbook -i hosts system.yml -vvvv --tags node-server-reboot`

It'll reboot. Then run (now and forever after):
`ansible-playbook -i hosts system.yml -vvvv --skip-tags node-server-reboot`

To start the server:
`sudo supervisorctl start all`
`sudo supervisorctl stop all`
`sudo supervisorctl restart all`

Logs:
`tail -f /var/log/node-server/server.log`
