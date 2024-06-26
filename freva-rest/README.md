# Freva server - client structure

[![License](https://img.shields.io/badge/License-BSD-purple.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-red.svg)](https://www.python.org/downloads/release/python-312/)
[![Docs](https://img.shields.io/badge/API-Doc-green.svg)](https://freva-clint.github.io/freva-nextgen)
[![Tests](https://github.com/FREVA-CLINT/freva-nextgen/actions/workflows/ci_job.yml/badge.svg)](https://github.com/FREVA-CLINT/freva-nextgen/actions)
[![Test-Coverage](https://codecov.io/github/FREVA-CLINT/freva-nextgen/branch/init/graph/badge.svg?token=dGhXxh7uP3)](https://codecov.io/github/FREVA-CLINT/freva-nextgen)

This repository contains the *freva-rest services* defining rest endpoints
that make up the freva server services as well as the client
services that provide command line interfaces and python libraries for their
rest service counterparts.

## Installation

1. Make sure you have Python 3.11+ installed.
2. Clone this repository:

```console
git clone git@github.com:FREVA-CLINT/freva-nextgen.git
cd freva-nextgen
```

```console
python3 -m pip install flit
```

4. Install the rest-api:

```console
cd freva-rest
python -m pip install -e .[dev]
```

## Freva-rest production docker container
It's best to use the system in production within a dedicated docker container.
You can pull the container from the GitHub container registry:

```console
docker pull ghcr.io/freva-clint/freva-rest:latest
```

By default the container starts with the ``freva-rest-service`` command.
The following default values are available on start up:

```console
reva-rest-server --help                                                                                                                                     (python3_12)

 Usage: freva-rest-server [OPTIONS]

 Start the freva rest API.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --config-file         -c                PATH                       Path to the server configuration file                                                                 │
│                                                                    [default: /home/wilfred/workspace/freva-nextgen/freva-rest/src/freva_rest/api_config.toml]            │
│ --port                -p                INTEGER                    The port the api is running on [default: 8080]                                                        │
│ --services            -s                [zarr-stream|databrowser]  Set additional services this rest API should serve. [default: zarr-stream, databrowser]               │
│ --cert-dir                              TEXT                       Set the path to the directory contaning the tls cert and key files that are used to establish a       │
│                                                                    secure connection, if you set the it will be assumed that cert file is saved as client-cert.pem and   │
│                                                                    the key file client-key.pem. This flag can be used as a short cut instead of using the `--tls-cert`   │
│                                                                    and `--tls-key` flats                                                                                 │
│                                                                    [default: None]                                                                                       │
│ --tls-cert                              TEXT                       Set the path to the tls certificate file that is used to establish a secure connection to the data    │
│                                                                    portal cache.                                                                                         │
│                                                                    [default: None]                                                                                       │
│ --tls-key                               TEXT                       Set the path to the tls key file that is used to establish a secure connection to the data portal     │
│                                                                    cache.                                                                                                │
│                                                                    [default: None]                                                                                       │
│ --dev                     --no-dev                                 Add test data to the dev solr. [default: no-dev]                                                      │
│ --debug                   --no-debug                               Turn on debug mode. [default: no-debug]                                                               │
│ --install-completion                                               Install completion for the current shell.                                                             │
│ --show-completion                                                  Show completion for the current shell, to copy it or customize the installation.                      │
│ --help                                                             Show this message and exit.                                                                           │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

You can either adjust the server settings by overriding the default flags
listed above or setting environment variables in the container.

The following environment variables can be set:

- ``DEBUG``: Start server in debug mode (1), (default: 0 -> no debug).
- ``API_PORT``: the port the rest service should be running on (default 8080).
- ``API_WORKER``: the number of multi-process work serving the API (default: 8).
- ``SOLR_HOST``: host name of the solr server, host name and port should be
                 separated by a ``:``, for example ``localhost:8983``
- ``SOLR_CORE`` : name of the solr core that contains datasets with multiple
                  versions
- ``MONGO_HOST``: host name of the mongodb server, where query statistics are
                 stored. Host name and port should separated by a ``:``, for
                 example ``localhost:27017``
- ``MONGO_USER``: user name for the mongodb.
- ``MONGO_PASSWORD``: password to log on to the mongodb.
- ``MONGO_DB``: database name of the mongodb instance.
- ``API_URL``: url of the machine that runs of the rest api
- ``API_CACHE_EXP``: expiry time in seconds of the cached data
- ``REDIS_HOST``: Host and port of the redis cache
                  Host name and port should separated by a ``:``, for
                  example ``localhost:5672``
- ``REDIS_PASS``: Password for the redis connection.
- ``REDIS_USER``: Username for the redis connection.
- ``REDIS_SSL_CERTFILE``: Path to the TSL certificate file used to encrypt
                          the redis connection.
- ``REDIS_SSL_KEYFILE``: Path to the TSL key file used to encrypt the redis
                         connection.
- ``KEYCLOAK_HOST``: Host name where the keycloak instance is running.
- ``KEYCLOAK_REALM``: Name of the keycloak realm, defaults to freva.
- ``KEYCLOAK_CLIENT_ID``: Name of the client (app) that is used to create
                          the access tokens, defaults to freva
- ``KEYCLOAK_CLIENT_SECRET``: You can set a client secret, if you have
                              configured your keycloak to use a client secret.

> ``📝`` You can override the path to the default config file using the ``API_CONFIG``
         environment variable. The default location of this config file is
         ``/opt/databrowser/api_config.toml``.
