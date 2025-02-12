.. _auth:
Authentication
==============

Some functionality of the freva-rest API and the client library is only
accessible after successful authentication. This authentication is realised
with OAuth2 token creation. You can create new access and refresh tokens.
Refresh tokens can be used to create new access tokens without needing to log
in via username and password, thereby minimising the risk of exposing login
credentials. Bear in mind that both login and refresh tokens have a limited
lifetime.

Generally speaking, you have three options to interact with the authorization
system:


- via the REST API ``/api/freva-nextgen/auth/v2`` endpoints
- via the :py:func:`freva_client.authenticate` function
- via the ``freva-client auth`` command-line interface

Using the restAPI endpoints
---------------------------
The API supports token-based authentication using OAuth2. To obtain an access
token, clients can use the ``/api/freva-nextgen/auth/v2/token`` endpoint by
providing valid username and password credentials. The access token should
then be included in the authorization header for secured endpoints.

.. http:post:: /api/freva-nextgen/auth/v2/token

    Create an new login token from a username and password.
    You should either set a username and password or an existing refresh token.
    You can also set the client_id. Client id's are configured to gain access,
    specific access for certain users. If you don't set the client_id, the
    default id will be chosen.

    :form username: The username for the login
    :type username: str
    :form password: The password for the login
    :type password: str
    :form refresh_token: The refresh token that is used to create a new token
                         the refresh token can be used instead of authorizing
                         via user creentials.
    :type refresh_token: str
    :form client_id: The unique identifier for your application used to
                     request an OAuth2 access token from the authentication
                     server, this form parameter is optional.
    :type client_id: str
    :form client_secret: An optional client secret used for authentication.
                         This param. is optional and in most cases not needed
    :type client_secret: str
    :statuscode 200: no error
    :statuscode 401: unauthorized
    :resheader Content-Type: ``application/json``: access and refresh token.


    Example Request
    ~~~~~~~~~~~~~~~

    .. sourcecode:: http

        POST /api/freva-nextgen/auth/v2/token HTTP/1.1
        host: www.freva.dkrz.de

        {
            "username": "your_username",
            "password": "your_password"
        }

    Example Request
    ~~~~~~~~~~~~~~~

    .. sourcecode:: http

        HTTP/1.1 200 OK
        Content-Type: application/json

        {
            "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6.."
            "token_type": "Bearer",
            "expires": 1722874908,
            "refresh_token": "eyJhbGciOiJIUzUxMiIsInR5cCIgOiAiSldUIiwia2lkIi.."
            "refresh_expires": 1722876408,
            "scope": "profile email address",
        }

    Code examples
    ~~~~~~~~~~~~~
    Below you can find example usages of this request in different scripting and
    programming languages

    .. tabs::

        .. code-tab:: bash
            :caption: Shell

            curl -X POST https://www.freva.dkrz.de/api/freva-nextgen/auth/v2/token \
             -d "username=janedoe" \
             -d "password=janedoe123"

        .. code-tab:: python
            :caption: Python

            import requests
            response = requests.post(
              "https://www.freva.dkrz.de/api/freva-nextgen/auth/v2/token",
              data={"username": "janedoe", "password": "janedoe123"}
            )
            token_data = response.json()

        .. code-tab:: r
            :caption: gnuR

            library(httr)

            url <- "https://freva.dkrz.de/api/freva-nextgen/auth/v2/token"
            response <- POST(
               "https://www.freva.dkrz.de/api/freva-nextgen/auth/v2/token",
               body = list(username = "janedoe", password = "janedoe123"),
               encode = "form"
            )
            token_data <- content(response, "parsed")

        .. code-tab:: julia
            :caption: Julia

            using HTTP
            using JSON

            response = HTTP.POST(
              "https://www.freva.dkrz.de/api/freva-nextgen/auth/v2/token",
              body = Dict("username" => "janedoe", "password" => "janedoe123")
            )
            token_data = JSON.parse(String(response.body))

        .. code-tab:: c
            :caption: C/C++

            #include <stdio.h>
            #include <curl/curl.h>

            int main() {
                CURL *curl;
                CURLcode res;

                curl_global_init(CURL_GLOBAL_DEFAULT);
                curl = curl_easy_init();
                if(curl) {
                    struct curl_slist *headers = NULL;
                    headers = curl_slist_append(headers, "Content-Type: application/x-www-form-urlencoded");

                    curl_easy_setopt(curl, CURLOPT_URL, "https://www.freva.dkrz.de/api/freva-nextgen/auth/v2/token");
                    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
                    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, "username=janedoe&password=janedoe123");

                    res = curl_easy_perform(curl);
                    curl_easy_cleanup(curl);
                }
                curl_global_cleanup();
                return 0;
            }

---


.. http:get:: /api/freva-nextgen/auth/v2/status

    Check the status of an access token.



    :reqheader Authorization: The OAuth2 access token
    :statuscode 200: no error
    :statuscode 401: unauthorized
    :resheader Content-Type: ``application/json``: access and refresh token.


    Example Request
    ~~~~~~~~~~~~~~~

    .. sourcecode:: http

        POST /api/freva-nextgen/auth/v2/status HTTP/1.1
        host: www.freva.dkrz.de
        Authorization: Bearer your_access_token

    Example Request
    ~~~~~~~~~~~~~~~

    .. sourcecode:: http

        HTTP/1.1 200 OK
        Content-Type: application/json

        {
            "sub": "648692af-aaed-4f82-9f74-2d6baf96f5ea",
            "exp": 1719261824,
            "email": "jane@example.com"
        }

    Code examples
    ~~~~~~~~~~~~~
    Below you can find example usages of this request in different scripting and
    programming languages

    .. tabs::

        .. code-tab:: bash
            :caption: Shell

            curl -X GET https://www.freva.dkrz.de/api/freva-nextgen/auth/v2/status \
             -H "Authorization: Bearer YOUR_ACCESS_TOKEN"

        .. code-tab:: python
            :caption: Python

            import requests
            response = requests.get(
              "https://www.freva.dkrz.de/api/freva-nextgen/auth/v2/status",
              headers={"Authorization": "Bearer YOUR_ACCESS_TOKEN"}
            )
            token_data = response.json()

        .. code-tab:: r
            :caption: gnuR

            library(httr)

            response <- GET(
               "https://www.freva.dkrz.de/api/freva-nextgen/auth/v2/status",
               add_headers(Authorization = paste("Bearer", "YOUR_ACCESS_TOKEN"))
            )
            token_data <- content(response, "parsed")

        .. code-tab:: julia
            :caption: Julia

            using HTTP
            using JSON

            response = HTTP.get(
              "https://www.freva.dkrz.de/api/freva-nextgen/auth/v2/status",
              headers = Dict("Authorization" => "Bearer YOUR_ACCESS_TOKEN")
            )
            token_data = JSON.parse(String(response.body))

        .. code-tab:: c
            :caption: C/C++

            #include <stdio.h>
            #include <curl/curl.h>

            int main() {
                CURL *curl;
                CURLcode res;

                curl_global_init(CURL_GLOBAL_DEFAULT);
                curl = curl_easy_init();
                if(curl) {
                    struct curl_slist *headers = NULL;
                    headers = curl_slist_append(headers, "Authorization: Bearer YOUR_ACCESS_TOKEN");

                    curl_easy_setopt(curl, CURLOPT_URL, "https://www.freva.dkrz.de/api/freva-nextgen/auth/v2/status");
                    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

                    res = curl_easy_perform(curl);
                    curl_easy_cleanup(curl);
                }
                curl_global_cleanup();
                return 0;
            }

---

Using the freva-client python library
--------------------------------------
The freva-client python library offers a very simple interface to interact
with the authentication system.

.. automodule:: freva_client
   :members: authenticate

Using the command line interface
--------------------------------

Token creation and refreshing can also be achieved with help of the ``auth``
sub command of the command line interface

.. code:: console

    freva-client auth --help

.. execute_code::
   :hide_code:

   from subprocess import run, PIPE
   res = run(["freva-client", "auth", "--help"], check=True, stdout=PIPE, stderr=PIPE)
   print(res.stdout.decode())

You can create a token using your user name and password. For security reasons
you can not pass your password as an argument to the command line interface.
This means that you can only create a new token with help of a valid refresh
token in a non-interactive session. Such as a batch job.

Therefore you want to store your token data securely in a file, and use the
refresh token to create new tokens:

.. code:: console

    freva-client auth -u janedoe > ~/.mytoken.json
    chmod 600 ~/.mytoken.json

Later you can use the `jq json command line parser <https://jqlang.github.io/jq/>`_
to read the refresh token from and use it to create new access tokens.

.. code:: console

    export re_token=$(cat ~/.mytoken.json | jq -r .refresh_token)
    freva-client auth -r $re_token > ~/.mytoken.json
