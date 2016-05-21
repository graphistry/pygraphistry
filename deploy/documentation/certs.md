# Generating certificates from scratch.

Most installations will not need certificates generated from scratch,
or will have pre-existing procedures in place for doing so.

It is also common that a small environment,
with a physically or virtually private network,
will terminate SSL for HTTP traffic on one box,
and reverse proxy for applications speaking HTTP.
In that case, add Graphistry to that endpoint.

If neither sounds like your installation, the following steps are provided,
for a Linux machine with OpenSSL, to make certificates to be used by nginx.

In a new directory of your choosing, perform the following steps
and follow the subsequent instructions from OpenSSL:

```
/usr/lib/ssl/misc/CA.pl -newca
/usr/lib/ssl/misc/CA.pl -newreq-nodes
/usr/lib/ssl/misc/CA.pl -signreq
ln ./demoCA/cacert.pem ssl_trusted_certificate.pem
ln ./newcert.pem ssl_certificate.pem
ln ./newkey.pem ssl_certificate_key.pem
openssl dhparam -outform PEM -out ssl.dhparam 1024
```

and now that working directory can be the value of the environment variable

    $SSHPATH

when running launch.sh, which will then be mounted for nginx at `/etc/graphistry/ssl`.

(This new root CA cert should be installed in the web browser by IT.)
