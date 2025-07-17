class SsoException(Exception):
    """
    Koa, 15 Sep 2022  Custom Base Exception to handle Sso exception scenario
    """
    pass


class SsoRetrieveTokenTimeoutException(SsoException):
    """
    Koa, 15 Sep 2022  Custom Exception to Sso retrieve token time out exception scenario
    """
    pass

class SsoStateInvalidException(SsoException):
    """
    30 Jun 2025 Raised when the SSO state is missing, invalid, or not initialized during token retrieval.
    """
    pass



class TokenExpireException(Exception):
    """
    Koa, 15 Mar 2024  Custom Exception for JWT Token expiry when refresh
    """
    pass
