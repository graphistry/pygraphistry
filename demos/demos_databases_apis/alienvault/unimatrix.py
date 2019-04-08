import sys
import json
import time
import pprint
import logging
import requests
import ipaddress
import pandas as pd


import http.client as http_client
from IPython.display import display
from requests_oauthlib import OAuth2Session
from oauthlib.oauth2 import BackendApplicationClient, TokenExpiredError

pd.options.display.max_columns = None
#http_client.HTTPConnection.debuglevel = 1

MAX_EVENTS = 2000
MAX_RETRIES = 5
VALID_EVENT_PARAMS = {"page",
                      "size",
                      "sort",
                      "account_name",
                      "suppressed",
                      "plugin",
                      "event_name",
                      "source_name",
                      "sensor_uuid",
                      "source_username",
                      "timestamp_occured_gte",
                      "timestamp_occured_lte"}

VALID_ALARM_PARAMS = {"page",
                      "size",
                      "sort",
                      "status",
                      "suppressed",
                      "rule_intent",
                      "rule_method",
                      "rule_strategy",
                      "priority_label",
                      "alarm_sensor_sources",
                      "timestamp_occured_gte",
                      "timestamp_occured_lte"}

class UnimatrixClient():
    """
    Unimatrix API v2.0 client
    Uses Oauth2 authentication
    """

    def __init__(self, subdomain, client_id, client_secret):
        """
        UnimatrixClient constructor
        :param string subdomain: the subdomain part of the CN url
        :param string client_id: the client_id from the API Clients section in USMA user profile
        :param string client_secret: the client_secret from the API Clients section in USMA user profile
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.url = "https://" + subdomain + ".alienvault.cloud/api/2.0/"
        self.client = self._connect()
        self.log = logging.getLogger("unimatrix-api-client")
        self.log.setLevel('DEBUG')

    def _connect(self):
        """
        Retrieve token from USMA API and create an authenticated session
        :returns OAuth2Session: authenticated client session
        """
        oauth_client = BackendApplicationClient(client_id=self.client_id)
        oauth_session = OAuth2Session(client=oauth_client)
        token = oauth_session.fetch_token(token_url=self.url + "oauth/token",
                                            client_id=self.client_id,
                                            client_secret=self.client_secret)
                                            #verify=False)
        return OAuth2Session(client_id=self.client_id,
                                    token=token)

    def _retrieve_items(self, num_items, params, item_type="events", page=0):
        """
        Retrieve num_items from item_type
        :param int num_items: Max items to retrieve
        :param dict params: Additional params dictionary
        :param string item_type: 'events' or 'alarms'
        :returns list: list of items
        """
        try:
            page = page
            retries = 0
            item_list = []
            remaining = num_items
            list_name = "eventResourceList" if item_type == "events" else "alarms"
            if not params:
                params = {}
            # Loop if num_items > MAX_EVENTS as that is the max number of items in one page
            while remaining and page < (num_items//MAX_EVENTS+1) and retries < MAX_RETRIES:
                params["page"] = page
                params["size"] = min(remaining, MAX_EVENTS)
                #response = self.client.get(self.url+item_type, params=params, verify=False)
                response = self.client.get(self.url+item_type, params=params)
                if response.status_code == 200:
                    self.log.warning("Got 200. Page info: {}.".format(response.json()['page']))
                    page += 1
                    retries = 0
                    items = response.json()['_embedded'][list_name]
                    item_list.extend(items)
                    remaining = min(remaining, response.json()['page']['totalElements'])
                    remaining -= len(items)
                else:
                    retries += 1
                    self.log.error("Got response {response.status_code} for page {params['page']} " + \
                                   "and size {params['size']}. Attempt {retries}.")
                    time.sleep(1)
            return item_list

        except TokenExpiredError as e:
            self.log.error("ERROR: {e}. Token Expired.")
            self.client = self._connect()
            return self._retrieve_items(remaining, params, item_type, page)

        except KeyError as e:
            self.log.error("ERROR: {e}. {response.json()['_links']}")
            return []

    def _to_df(self, items, columns, drop_columns, convert_ips):

        df = pd.DataFrame.from_records(data=items, index='uuid')
        # Select / exclude columns
        if columns:
            df = df[columns]
        if drop_columns:
            df = df.drop(columns=drop_columns)
        # Drop iso8601 columns as we will be converting timestamp columns to datetime anyway
        df.drop(df.filter(regex="iso8601").columns, axis=1, inplace=True)
        # Convert timestamp columns to datetime
        ts_cols = list(df.filter(regex="timestamp").columns)
        df.loc[:, ts_cols] = df.loc[:, ts_cols].apply(pd.to_datetime, unit='ms')
        # Transform *address fields to ipaddress using cyberpandas (optional)
        # Don't transform dns_server_address as it may contain hostnames
        if convert_ips:
            print('WARNING: Skipping convert_ips')
        return df

    def get_events_list(self, num_items=100, params=None):
        """
        Get events as list of dictionaries
        :param int num_items: Max items to retrieve
        :param dict params: Additional params dictionary according to:
            https://www.alienvault.com/documentation/api/usm-anywhere-api.htm#/events
        :returns list: list of events
        """
        if params and set(params.keys()) - VALID_EVENT_PARAMS:
            self.log.error("Invalid alarm query parameters: {set(params.keys()) - VALID_ALARM_PARAMS}")
            return None
        return self._retrieve_items(item_type="events", num_items=num_items, params=params)

    def get_events_df(self, num_items=100, params=None, columns=None, drop_columns=None, convert_ips=True):
        """
        Get events as pandas DataFrame
        :param int num_items: Max items to retrieve
        :param dict params: Additional params dictionary according to:
            https://www.alienvault.com/documentation/api/usm-anywhere-api.htm#/events
        :parama list columns: list of columns to include in DataFrame
        :returns pandas.DataFrame: dataframe of events
        """
        if params and set(params.keys()) - VALID_EVENT_PARAMS:
            self.log.error("Invalid alarm query parameters: {set(params.keys()) - VALID_ALARM_PARAMS}")
            return None
        events = self._retrieve_items(item_type="events", num_items=num_items, params=params)
        df = self._to_df(events, columns, drop_columns, convert_ips)
        return df

    def get_alarms_list(self, num_items=100, params=None):
        """
        Get alarms as list of dictionaries
        :param int num_items: Max items to retrieve
        :param dict params: Additional params dictionary according to:
            https://www.alienvault.com/documentation/api/usm-anywhere-api.htm#/alarms
        :returns list: list of alarms
        """
        if params and set(params.keys()) - VALID_ALARM_PARAMS:
            self.log.error("Invalid alarm query parameters: {set(params.keys()) - VALID_ALARM_PARAMS}")
            return None
        return self._retrieve_items(item_type="alarms", num_items=num_items, params=params)

    def get_alarms_df(self, num_items=100, params=None, columns=None, drop_columns=None, convert_ips=True):
        """
        Get alarms as list of dictionaries
        :param int num_items: Max items to retrieve
        :param dict params: Additional params dictionary according to:
            https://www.alienvault.com/documentation/api/usm-anywhere-api.htm#/alarms
        :parama list columns: list of columns to include in DataFrame
        :returns pandas.DataFrame: dataframe of alarms
        """
        if params and set(params.keys()) - VALID_ALARM_PARAMS:
            self.log.error("Invalid alarm query parameters: {set(params.keys()) - VALID_ALARM_PARAMS}")
            return None
        events = self._retrieve_items(item_type="alarms", num_items=num_items, params=params)
        df = self._to_df(events, columns, drop_columns, convert_ips)
        return df



