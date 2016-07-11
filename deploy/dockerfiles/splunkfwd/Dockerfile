FROM buildpack-deps:trusty-curl
ENV SPLUNK_VERSION 6.4.1
ENV SPLUNK_BUILD debde650d26e
ENV SPLUNK_FILENAME splunkforwarder-${SPLUNK_VERSION}-${SPLUNK_BUILD}-linux-2.6-amd64.deb
RUN curl -o splunk.deb https://download.splunk.com/products/universalforwarder/releases/6.4.1/linux/${SPLUNK_FILENAME} && \
      dpkg -i splunk.deb && \
      rm splunk.deb
CMD /opt/splunkforwarder/bin/splunk start --no-prompt --answer-yes
