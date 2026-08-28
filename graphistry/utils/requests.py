import json
import requests

from graphistry.otel import inject_trace_headers
from graphistry.util import setup_logger
logger = setup_logger(__name__)


def log_requests_error(resp: requests.Response) -> None:

    if not (200 <= resp.status_code < 300):

        try:
            error_content = resp.json()
            logger.error("HTTP %s error - response content (JSON): %s", resp.status_code, json.dumps(error_content, indent=2))
        except json.JSONDecodeError:
            logger.error("HTTP %s error - response content (text): %s", resp.status_code, resp.text)


class OrgSwitchError(Exception):
    """Org switch endpoint responded with a non-2xx status or a non-OK body status."""

    def __init__(self, org_name: str, status_code: int, detail: str = ''):
        self.org_name = org_name
        self.status_code = status_code
        self.detail = detail
        msg = f"Org switch to '{org_name}' failed with HTTP {status_code}"
        if detail:
            msg += f": {detail}"
        super().__init__(msg)


class OrgSwitchIdpChallenge(Exception):
    """Server-side quirk: for an org the caller is a plain member of (not owner/admin)
    that has its own SSO IDP configured, the switch endpoint still returns
    status=='OK' but does NOT actually switch -- it instead returns an SSO
    challenge (data['idp']) requiring a fresh org-scoped SSO login before the
    switch is honored. Treating that as success would silently leave the
    caller on their previous org.
    """

    def __init__(self, org_name: str, idp: dict):
        self.org_name = org_name
        self.idp = idp
        super().__init__(
            "Switching to organization '{}' requires SSO re-authentication "
            "(idp: {}) -- the server did not actually switch. Complete the "
            "auth_url in the response's data['idp'], or use "
            "sso_login(org_name='{}') instead of switch_org().".format(
                org_name, list(idp.keys()), org_name
            )
        )


def switch_org_request(base_url: str, org_name: str, token: str, verify: bool) -> None:
    """POST to the org-switch endpoint and validate the response.

    Single source of truth for both GraphistryClient.switch_org and
    ArrowUploader._switch_org: same URL shape, same HTTP/body status check,
    same SSO-idp-challenge detection. Raises OrgSwitchError or
    OrgSwitchIdpChallenge on any failure; returns None on a confirmed switch.
    Callers decide their own success/failure side effects (raising vs.
    logging-and-swallowing, which session state to update).
    """
    switch_url = f"{base_url}/api/v2/o/{org_name}/switch/"
    response = requests.post(
        switch_url,
        data={'slug': org_name},
        headers=inject_trace_headers({'Authorization': f'Bearer {token}'}),
        verify=verify,
    )
    log_requests_error(response)

    if not (200 <= response.status_code < 300):
        raise OrgSwitchError(org_name, response.status_code)

    try:
        body = response.json()
    except Exception:
        body = {}

    if body.get('status') != 'OK':
        raise OrgSwitchError(org_name, response.status_code, detail=body.get('message', ''))

    data = body.get('data', {})
    if isinstance(data, dict) and data.get('idp'):
        raise OrgSwitchIdpChallenge(org_name, data['idp'])
