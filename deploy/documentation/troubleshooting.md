# Troubleshooting

*Or, what to do if the server dies.*

**The simplest solution**: log into our [ec2 control panel](https://us-west-1.console.aws.amazon.com/ec2/v2/home?region=us-west-1#Instances:sort=tag:Name) and reboot the server.

Other options follow, in order of least severe to most severe. Don't forget that the app stack for a particular deploy environment (staging, production, etc.) may be running on multiple servers, and you may need to ensure the fixes you apply are applied to all applicable servers.



## Restart the apps

Log into the server which is dead via SSH (using an account that has `sudo` access, which all dev accounts should have by default).

Determine which app is failing. If you are getting `500` errors in the browser, restart nginx. If you can't load the main HTML, restart central. If you fail to be assigned to a worker, restart reaper.py. If you are assigned a worker, but can't connect, restart the workers. When in doubt, restart them all.

Don't forget to restart the apps on all the servers that they're running on for a particular deploy environment (staging, production, etc.)


| App        | Restart Command                                                |
|------------|----------------------------------------------------------------|
| nginx      | `sudo service nginx restart`                                   |
| central    | `sudo supervisorctl restart central`                           |
| reaper.py  | `sudo supervisorctl restart reaper.py`                         |
| workers    | `sudo supervisorctl restart viz-worker:*`                      |
| everything | `sudo supervisorctl restart all && sudo service nginx restart` |



## Reboot the Server

The recommended way to restart out ec2 servers is via the [ec2 control panel](https://us-west-1.console.aws.amazon.com/ec2/v2/home?region=us-west-1#Instances:sort=tag:Name).

1. Log in using the [AWS Graphistry IAM User login page](https://graphistry.signin.aws.amazon.com/console) (use the username and password provided by Graphistry),
2. Go to the [ec2 control panel](https://us-west-1.console.aws.amazon.com/ec2/v2/home?region=us-west-1#Instances:sort=tag:Name).
3. Make sure you're in the "N. California" region, aka us-west-1 (selected in the upper-right corner of the browser window.)
4. If not already selected, click "Instances" on the left-hand side.
5. Click the instance that you want to reboot from the list.
6. Using the buttons above the instance list, click "Actions", then "Instance State", then "Reboot".

It should take 30s-120s for the reboot to finish (you'll know when you can login to the server via SSH again.)

The other way to reboot the server is to simply login via SSH and run `sudo reboot`. This should have the same effect as the ec2 control panel method, but for some reason, Amazon recommends using their control panel over the command-line `reboot` command.



## Redeploy the Server

If the software on the server has been corrupted somehow, you should do this option. Simply `cd` to the root of the `deploy` repo, and run a full re-deploy of the server using Ansible, specifying the inventory of the deployment environment you wish to deploy to.

For **staging**, run `ansible-playbook site.yml -i staging`. For **production**, run `ansible-playbook site.yml -i production`.

Note that these are the same commands you use to do a normal deploy of the server, minus the `--tags=deploy` option (that options tells Ansible to only re-deploy our Graphistry apps, not the whole server stack.)

When this completes, the server should be correctly configured and all our apps updated to the newest versions, and restarted.



## Delete the Server and Redeploy

If our ec2 instance is physically dead (which has happened before), you may need to start again fresh.

1. Login to the [ec2 control panel](https://us-west-1.console.aws.amazon.com/ec2/v2/home?region=us-west-1#Instances:sort=tag:Name) as in the reboot option above.
2. Select the dead server and, from the "Actions" menu, click "Instance State", then "Stop".
3. With the server still selected, click "Actions", then "Networking", then "Disassociate Elastic IP Address".
4. Start a new ec2 instance with the proper settings, as outlined in the [AWS](https://github.com/graphistry/deploy/blob/master/documentation/aws.md) deployment guide. Don't run Ansible quite yet, though,
5. On your laptop, go into the `~/.ssh/known_hosts` file and delete anything related to the old server (lines that start with the server's domain name, or the old IP address.) Otherwise, SSH will freak out that the new server certificate is different from the old server's certificate.
6. In the ec2 control panel, click "Elastic IPs" on the left-hand side, then click the IP of the old (dead) server. Now click "Associate Address" at the top, and select the new instance you just started. Optionally, you can give the new server a new elastic IP address, and change the Route53 domain name records to point to the new IP.
7. Update the DNS records:
  1. Note the public and private IPs on the new server in the ec2 console.
  2. Log in to the [AWS Route53 control panel](https://console.aws.amazon.com/route53/home?region=us-west-1)
  3. Click "Hosted Zones",
  4. Select the ["graphistry.com." zone with "Type" set to "Private"](https://console.aws.amazon.com/route53/home?region=us-west-1#resource-record-sets:Z1AF3JT9CFWWNZ).
  5. Select the "A" record for the subdomain that points to the server.
  6. On the right-hand side under "Edit Record Set", enter in the **private** IP address of the new ec2 server in the box next to "Value".
  7. Click "Save Record Set".
  8. If you chose to use a new Elastic IP for the new instance, instead of re-assigning the old instance's one, repeat steps 4-7, but select the ["Public" record set for "graphistry.com."](https://console.aws.amazon.com/route53/home?region=us-west-1#resource-record-sets:Z6L83I36S428X) and use the instance's public (Elastic) IP as the "Value".
8. Log in to the new server to make sure everything is running OK. Use the Ansible SSH private key, and username "ubntu", and SSH port 22.
9. In the inventory file in this `deploy` repo (`./staging` or `./production`), make sure that the domain name of the server matches the current domain.
10. Run Ansible using the full deploy: `ansible-playbook site.yml -i staging` (or `-i production`). This may take a while (20 minutes) to complete. If it fails halfway though, try running it again once or twice.
11. When done, verify that you can login via SSH using your own username and port 61630. Verify that the web app is now working.
12. If everything seems to be ok, login to the [ec2 control panel](https://us-west-1.console.aws.amazon.com/ec2/v2/home?region=us-west-1#Instances:sort=tag:Name) and select the old instance. Then click "Actions", "Instance State" and "Terminate".

You should have a fully-deployed new server at this point.
