#!/usr/bin/env python

from pymongo import MongoClient, ASCENDING, errors
from datetime import datetime, timedelta
import subprocess
import socket
import time
import os
import random
import signal
import urllib2
import requests
from requests.auth import HTTPBasicAuth
import sys


#########################
# Constants
#########################

# The number of seconds without a ping before a worker process is considered a zombie
WORKER_TIMEOUT = 30

DATABASE = 'cluster'
MONGO_SERVER = 'mongodb://graphistry:graphtheplanet@127.0.0.1:27017/cluster'


ip = '10.5.151.121'
hostname = '10.5.151.121'
reaper_pid = os.getpid()


#########################
# Logging
#########################

import logging

# Output dates in GMT
logging.Formatter.converter = time.gmtime

# Optimizations for logging library to not lookup calling stack frame, or collect threading info
logging._srcfile = None
logging.logThreads = 0

loggerParent = logging.getLogger()
loggerParent.setLevel(logging.ERROR)

# Quiet down request's log output
requests_log = logging.getLogger("requests")
requests_log.setLevel(logging.ERROR)
requests_log.propagate = False

# Our own logging object
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = True

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)

# Format the output as JSON
formatter = logging.Formatter(
    '{"name": "reaper.py", "module": "reaper", "hostname": "' + str(hostname) + '", "pid": ' + str(reaper_pid) + \
        ', "level": %(levelno)s, "time": "%(asctime)s", "message_type": "%(levelname)s" , "msg": "%(message)s"}',
    datefmt='%Y-%m-%dT%H:%M:%SZ')
formatter.converter = time.gmtime
ch.setFormatter(formatter)

logger.addHandler(ch)


#########################
# Main logic
#########################

def connect_mongo():
    """ Connects to the MongoDB database and returns a dictionary with the database object,
    gpu_monitor collection, and node_monitor collection."""
    logger.info("Connecting to MongoDB at URL '{url}' and using the database named {db}".format(url=MONGO_SERVER, db=DATABASE))
    client = MongoClient(MONGO_SERVER)
    db = client[DATABASE]
    return db['gpu_monitor'], db['node_monitor']


def setup_db(workers_collection):
    """ Sets up the database collections, specifically ensuring the right indices exist."""
    return workers_collection.ensure_index( [("port", ASCENDING), ("ip", ASCENDING), ("pid", ASCENDING)], unique=True )


def cleanup_server_records(gpu_collection, workers_collection):
    """
    Goes into the db and deletes records for workers belongind to servers who are no longer
    registered in the db. Necessary because reaper.py won't be running for that server anymore, and
    reaper.py normally only concerns itself with workers for the server it's running on.
    """
    active_addresses = []

    servers = gpu_collection.find({}, {"ip": True})
    for server in servers:
        active_addresses.append(server["ip"])

    logger.debug("Active viz server ip addresses: {active}".format(
        active=", ".join(active_addresses) ))

    removed_workers = workers_collection.delete_many({"ip": {"$nin": active_addresses}})

    if removed_workers.deleted_count > 0:
        logging.info("Removed {count} stale worker entries from the database".format(count=removed_workers.deleted_count))

    return removed_workers.deleted_count


def str_to_bytes(human_size):
    """ Converts human-readable string of memory units (e.g. '10 Mib') to bytes as integer."""
    size_str = human_size.replace(' ', '')

    if 'GiB' in size_str:
        return int(size_str.replace('GiB','')) * 1024 * 1024 * 1024
    elif 'MiB' in size_str:
        return int(size_str.replace('MiB','')) * 1024 * 1024
    elif 'KiB' in size_str:
        return int(size_str.replace('KiB','')) * 1024
    else:
        raise SyntaxError('String "{}" could not be converted into an integer number of bytes'.format(human_size))


def log_boundary(metric, value):
    """ Sends the given value to Boundary under the metric name given. If value is not a JSON
    primitive type (number, etc), it should be string that includes quotes as part of the string."""
    return True


def get_available_memory():
    # Get GPU free memory from nvidia-smi
    p = subprocess.Popen(["nvidia-smi",
                        "--query-gpu=memory.free",
                        "--format=csv,noheader",
                        "--id=0"], stdout=subprocess.PIPE)
    gpu_data = p.communicate()[0]
    return str_to_bytes(gpu_data)


def gpu_process_usage():
    p = subprocess.Popen(["nvidia-smi",
                            "--query-compute-apps=used_gpu_memory,pid",
                            "--format=csv,noheader",
                            "--id=0"], stdout=subprocess.PIPE)
    gpu_memory_by_process = p.communicate()[0].split('\n')

    gpu_processes = []
    for process in gpu_memory_by_process:
        process_data = process.replace(' ', '').split(',')
        if len(process_data) != 2 or process_data[0] == '':
            continue

        gpu_process_pid = int(process_data[1])
        gpu_memory_used = str_to_bytes(process_data[0])

        logger.debug('Process GPU memory: pid: {pid}, {mem} bytes'.format(pid=gpu_process_pid, mem=gpu_memory_used))
        gpu_processes.append({'pid': gpu_process_pid, 'memory': gpu_memory_used, 'raw': process})

    if len(gpu_memory_by_process) < 1:
        logger.debug('Process GPU memory: no processes are using GPU memory at this time')

    return gpu_processes


def kill_process(pid):
    try:
        os.kill(int(pid), signal.SIGKILL)
        logger.warn("Reaped process {pid}".format(pid=pid))
    except OSError as oserr:
        # Ignore "no such process" errors; re-raise everything else
        if oserr.errno == 3:
            logger.info("Process {pid} not running".format(pid=pid))
        else:
            logger.error("Could not kill process {pid}: {err}".format(pid=pid, err=str(oserr)))


def advertise_gpu_memory(gpu_collection):
    """Updates the database with the current amount of GPU memory available on this instance."""
    gpu_memory_free = get_available_memory()
    gpu_collection.update({'ip': ip}, {'$set':
        {
            'gpu_memory_free': gpu_memory_free,
            'updated': datetime.utcnow()
        }}, True)
    logger.debug('GPU memory available: {mem}'.format(mem=gpu_memory_free))
    log_boundary("GPU_FREE", gpu_memory_free)


def advertise_process_memory(workers_collection):
    """Updates the entries of active workers in the database to report how much GPU memory they're
    currently using. Also logs the number of active/inactive workers to Boundary.
    """
    # Update db for each process talking to the GPU
    for process in gpu_process_usage():
        logger.debug("Process GPU data: {}".format(process['raw']))
        workers_collection.update(
            { 'ip': ip, 'pid': int(process['pid']) },
            { '$set': {'gpu_memory' : int(process['memory'])} })

    # Update Boundary
    try:
        total_workers = workers_collection.find({"ip": ip}).count()
        log_boundary("TOTAL_WORKERS", total_workers)

        active_users = workers_collection.find({"ip": ip, "active": True}).count()
        log_boundary("ACTIVE_WORKERS", active_users)

        log_boundary("INACTIVE_WORKERS", total_workers - active_users)
    except Exception as activity_err:
        console.warn("Error while attempting to count and log active/inactive workers: {err}".format(err=str(activity_err)))


def reap_workers(workers_collection):
    """Checks the database for workers on this instance who have not pinged in WORKER_TIMEOUT, and
    attempts to kill those processes and remove their entry from the database.
    """
    # Select all checkins older than 30 seconds
    zombie_timestamp = datetime.utcnow() - timedelta(seconds=WORKER_TIMEOUT)
    zombie_workers = workers_collection.find({"ip": ip, "updated": {"$lt": zombie_timestamp}})

    for worker in zombie_workers:
        kill_process(int(worker['pid']))
        workers_collection.remove({"_id": worker['_id']})
        logger.info("Removed stale ping for worker {port}, pid: {pid}".format(port=worker['port'], pid=worker['pid']))



if __name__=="__main__":
    # Connect to central db
    gpu_collection, workers_collection = connect_mongo()
    setup_db(workers_collection)

    tick_count = 0

    while 1:
        try:
            advertise_gpu_memory(gpu_collection)
            reap_workers(workers_collection)
            advertise_process_memory(workers_collection)

            # Do this at a random, but long-ish, interval to save work, and ensure multiple reapers
            # don't all do this at the same time.
            if tick_count <= 0:
                tick_count = random.randint(10, 20)
                logger.debug('Cleaning workers table of records belong to do dead servers')
                cleanup_server_records(gpu_collection, workers_collection)

        except errors.ConnectionFailure:
            logger.error("Lost connection to the Mongo database. Attempting to reconnect...")
            gpu_collection, workers_collection = connect_mongo()

        finally:
            tick_count -= 1

        time.sleep(25)
