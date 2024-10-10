import json
import requests

from graphistry.util import setup_logger
logger = setup_logger(__name__)


def log_requests_error(resp: requests.Response) -> None:

    if not (200 <= resp.status_code < 300):

        try:
            error_content = resp.json()
            logger.error("HTTP %s error - response content (JSON): %s", resp.status_code, json.dumps(error_content, indent=2))
        except json.JSONDecodeError:
            logger.error("HTTP %s error - response content (text): %s", resp.status_code, resp.text)
