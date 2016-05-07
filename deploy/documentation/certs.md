# Generating certificates

```
/usr/lib/ssl/misc/CA.pl -newca
/usr/lib/ssl/misc/CA.pl -newreq-nodes
/usr/lib/ssl/misc/CA.pl -signreq
ln ./demoCA/cacert.pem comodo_root-bundle.pem
ln ./newcert.pem graphistry.com_bundle.pem
ln ./newkey.pem graphistry.com-private_key.pem
openssl dhparam -outform PEM -out graphistry.dhparam 1024
```

and now your working directory can be mounted by nginx as `/etc/nginx/graphistry/ssl`.
