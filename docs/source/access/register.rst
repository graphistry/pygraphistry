Registering with Graphistry
===========================

`graphistry.register()` is the global method used to initialize and configure your Graphistry client. It sets up your API credentials, specifies the server to connect to, and configures authentication settings. This function should be called before making any Graphistry API calls that use the server such as `.plot()`.

Basic Usage
-----------

To register, import Graphistry and call `graphistry.register()`:

.. code-block:: python

    import graphistry

    # Register with default Graphistry Hub using username/password
    graphistry.register(api=3, username="my_username", password="my_password")

By default, this connects to **Graphistry Hub** (`hub.graphistry.com`) using the `https` protocol and sets `api=3` for the latest API version. You can override the server, authentication details, and other settings as needed.

Core Concepts
-------------

### Personal Accounts vs Organizational Accounts

- **Personal Accounts**: Meant for individual use, typically on Graphistry Hub.
- **Organizational Accounts**: Managed with roles and permissions, often in an enterprise context.

.. code-block:: python

    user_info = graphistry.user()
    print(user_info.get("organization"))  # Returns organization info or None

### Server Configuration

- **Default Server**: By default, `graphistry.register()` connects to the **Graphistry Hub**, including the **free GPU tier** for visual analytics.
- **Custom Server**: If using a private deployment, specify the `server` argument to connect to your custom server.

.. code-block:: python

    # Connect to a custom server
    graphistry.register(
        api=3,
        server="my_custom_graphistry_server.com",
        username="my_username",
        password="my_password"
    )

### Protocol Configuration

- **TLS (HTTPS)**: Communication uses `https` by default for secure communication.
- **Non-TLS (HTTP)**: If your server doesn't support TLS, set the `protocol` parameter to `"http"`.

.. code-block:: python

    # Use HTTP protocol without TLS
    graphistry.register(
        api=3,
        protocol="http",
        server="my_custom_graphistry_server.com",
        username="my_username",
        password="my_password"
    )

### Authentication Methods

`graphistry.register()` supports several authentication methods:

1. **Username & Password**:
   .. code-block:: python

        graphistry.register(api=3, username="my_username", password="my_password")

2. **Personal Key ID & Secret** (for scripts or automation):
   .. code-block:: python

        graphistry.register(api=3, personal_key_id="my_key_id", personal_key_secret="my_key_secret")

3. **Single Sign-On (SSO)** (for enterprise users):
   .. code-block:: python

        graphistry.register(api=3, idp_name="my_idp_name", sso_opt_into_type="browser")

    SSO authentication options: `sso_opt_into_type` can be `"browser"`, `"display"`, or `None` (default is print).

### Routing Configuration

- **Server Routing**: By default, server API and browser UI requests route through the same `server`.
- **Custom Browser Routing**: Override browser routing via `client_protocol_hostname`.

.. code-block:: python

    # Override browser routing
    graphistry.register(
        api=3,
        server="my_api_server.com",
        username="my_username",
        password="my_password",
        client_protocol_hostname="https://my_ui_server.com"
    )

Advanced Features
-----------------


### JWT Session Handling

`graphistry.register()` establishes a **JWT session** after authentication. The session token is managed automatically for future API calls.

#### Retrieving the Current JWT Token

To retrieve the current JWT token, you can use the following command after registering:

.. code-block:: python

    # Get the current JWT token
    current_token = graphistry.api_token()
    print(current_token)

The token is automatically refreshed as needed during the session.


Detailed Parameter Reference
----------------------------

- **username** *(Optional[str])*: Your Graphistry account username.
- **password** *(Optional[str])*: Your Graphistry account password.
- **personal_key_id** *(Optional[str])*: Your personal key ID for secure access.
- **personal_key_secret** *(Optional[str])*: Corresponding personal key secret.
- **server** *(Optional[str])*: The URL of the Graphistry server to connect to (e.g., `hub.graphistry.com` or a custom server).
- **protocol** *(Optional[str])*: The protocol to use (`https` or `http`), defaults to `https`.
- **api** *(Optional[int])*: The API version to use (always set to `3`).
- **client_protocol_hostname** *(Optional[str])*: Overrides the browser protocol/hostname.
- **org_name** *(Optional[str])*: Organization name for SSO authentication.
- **idp_name** *(Optional[str])*: Identity Provider (IdP) for SSO.
- **sso_opt_into_type** *(Optional[str])*: How to display the SSO URL (`"browser"`, `"display"`, or `None`).

Examples
----------------------

### Register with Username and Password

.. code-block:: python

    import graphistry

    graphistry.register(
        api=3,
        username="my_username",
        password="my_password"
    )

### Register with Personal Key ID and Secret

.. code-block:: python

    import graphistry

    graphistry.register(
        api=3,
        personal_key_id="my_key_id",
        personal_key_secret="my_key_secret"
    )

### Register with SSO (Organization with Specific IdP)

.. code-block:: python

    import graphistry

    graphistry.register(
        api=3,
        org_name="my_org_name",
        idp_name="my_idp_name",
        sso_opt_into_type="browser"
    )

### Register with Custom Server and Protocol

.. code-block:: python

    import graphistry

    graphistry.register(
        api=3,
        protocol="http",
        server="my_custom_server.com",
        username="my_username",
        password="my_password"
    )

### Register with Custom Browser Routing

.. code-block:: python

    import graphistry

    graphistry.register(
        api=3,
        server="my_api_server.com",
        username="my_username",
        password="my_password",
        client_protocol_hostname="https://my_ui_server.com"
    )

---

Best Practices
--------------

- **Security**: Always use secure protocols (`https`) and validate certificates.
- **Authentication**: Use `personal_key_id` and `personal_key_secret` for automation.
- **SSO**: For organizations, ensure correct `org_name` and, if needed, `idp_name`.
- **Session Management**: The library handles session tokens automatically; ensure safe credential handling when enabling memory storage.

Troubleshooting
---------------

- **Connection Errors**: Check the `server` and `protocol` parameters and ensure your network allows access.
- **Authentication Failures**: Verify credentials. For SSO, ensure `org_name` and `idp_name` are correct.
- **SSL Issues**: Validate that the server certificate is valid or consider disabling SSL validation (`certificate_validation=False`), though not recommended.

