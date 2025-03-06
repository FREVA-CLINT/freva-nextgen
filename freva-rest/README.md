# Freva Rest API

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
You can install the Freva rest server via [conda-forge](https://conda-forge.org):

```console
conda install -c conda-forge freva-rest-server
```

### Development installation

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
python -m pip install -e .
```

## Freva-rest production docker container
It's best to use the system in production within a dedicated docker container.
You can pull the container from the GitHub container registry:

```console
docker pull ghcr.io/freva-clint/freva-rest:latest
```

By default the container starts with the ``freva-rest-server`` command.
See the `freva-rest-server --help` command for configure options.

### Server Startup Options

The container supports several startup modes:

```console
# Default mode - starts the freva-rest service
docker run \
    -e API_OIDC_DISCOVERY_URL=https://example-oicd.org/.well-known/openid-configuration \
    -p 7777:7777 \
    ghcr.io/freva-clint/freva-rest:latest

# Start with custom freva-rest flags and access
docker run \
        -e API_OIDC_DISCOVERY_URL=https://example-oicd.org/.well-known/openid-configuration \
        -p 8000:8000
       ghcr.io/freva-clint/freva-rest:latest -p 8000

```

You can adjust the server settings by either overriding the default flags or
setting environment variables in the container.

### Available Environment Variables

```ini
# Server Configuration
DEBUG=0                  # Start server in debug mode (1), (default: 0 -> no debug)
API_URL=localhost:7777
API_PORT=7777            # The port the rest service should be running on
API_WORKER=8            # Number of multi-process workers serving the API
API_PROXY=http://www.example.de/
API_CACHE_EXP=3600      # Expiry time in seconds of the cached data

# Database Configuration
API_MONGO_USER=mongo
API_MONGO_PASSWORD=secret
API_MONGO_DB=search_stats
API_MONGO_HOST=localhost:27017  # Host name and port should be separated by ":"

# Solr Configuration
API_SOLR_HOST=localhost:8983   # Host name and port should be separated by ":"
API_SOLR_CORE=files           # Name of the solr core for datasets with multiple versions

# Redis Configuration
API_REDIS_HOST=redis://localhost:6379
API_REDIS_USER=              # Username for the redis connection
API_REDIS_PASSWORD=              # Password for the redis connection
API_REDIS_SSL_CERTFILE=/certs/client-cert.pem
API_REDIS_SSL_KEYFILE=/certs/client-key.pem

# OIDC Configuration
API_OIDC_DISCOVERY_URL=http://keycloak:8080/realms/freva/.well-known/openid-configuration
API_OIDC_CLIENT_ID=freva     #Name of the client (app) that is used to create the access tokens, defaults to freva
API_OIDC_CLIENT_SECRET=      # Optional: Set if your OIDC instance uses a client secret

# MYSQL Server (for legacy freva)
MYSQL_USER=
MYSQL_PASSWORD=
MYSQL_ROOT_PASSWORD=
MYSQL_DATABASE= # Database name for legacy freva

# STAC API (Optional)
API_STAC_HOST=
API_STACBROWSER_HOST=
API_STAC_MAX_ITEMS=

# Service activation flags
# Set to 1 to enable, 0 to disable the service (default)
USE_MONGODB=0  # Controls MongoDB initialization
USE_SOLR=0     # Controls Apache Solr initialization
USE_REDIS=0    # Set up a redis distributed cache
USE_MYSQL=0    # Set up a mysql server
USE_STACAPI=0 # Setup STAC-API
USE_OPENSEARCH=0 # Set up an Opensearch instance for running STAC-API
```

> [!TIP]
> To start any services the rest API needs in the background you can
> set any of the `USE_MONGODB`, `USE_SOLR`, `USE_REDIS`, `USE_MYSQL` environment
> variables to `1`. The service(s) will be automatically configured for usage
> with freva.
> If you only whish to start a service and follow their log you can use the
> `follow` command in the container e.g:
> ```console
> docker run -e USE_MYSQL=1 \
> -e MYSQL_USER=foo \
> -e MYSQL_PASSWORD=my-pass \
> -e MYSQL_ROOT_PASSWORD=my-root-pass \
> -e MYSQL_DATABASE=bar \
> ghcr.io/freva-clint/freva-rest:latest follow mysql


> [!NOTE]
> List of all necessary and default environment variables for running the STAC API, 
> has been listed in the following link:
>
> https://github.com/stac-utils/stac-fastapi-elasticsearch-opensearch?tab=readme-ov-file#key-variables-to-configure
>
> If you want to have this feature, please add them to the dokcer environment variables.

### Required Volumes
The container can use several persistent volumes that should be mounted:

```console
docker run -d \
  --name freva-rest \
  -e {above envs} \
  -v $(pwd)/mongodb_data:/data/db \
  -v $(pwd)/solr_data:/var/solr \
  -v $(pwd)/certs:/certs:ro \
  -v $(pwd)/logs:/logs \
  -p 7777:7777 \
  -p 27017:27017 \
  -p 8983:8983 \
  -p 5432:5432 \
  ghcr.io/freva-clint/freva-rest:latest
```

Create the necessary directories before starting the container:
```console
mkdir -p {mongodb_data,solr_data,certs}
```

> [!NOTE]
> You can override the path to the default config file using the ``API_CONFIG``
         environment variable. The default location of this config file is
         ``/opt/freva-rest/api_config.toml`` inside of container.
