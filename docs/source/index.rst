.. freva-databrowser-api documentation master file, created by
   sphinx-quickstart on Tue Jul 25 17:08:08 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Freva: The free evaluation system
=================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. image:: https://img.shields.io/badge/License-BSD-purple.svg
   :target: LICENSE

.. image:: https://img.shields.io/pypi/pyversions/freva-client.svg
   :target: https://pypi.org/project/freva-client

.. image:: https://img.shields.io/badge/ViewOn-GitHub-purple
   :target: https://github.com/FREVA-CLINT/freva-nextgen

.. image:: https://github.com/FREVA-CLINT/freva-nextgen/actions/workflows/ci_job.yml/badge.svg
   :target: https://github.com/FREVA-CLINT/freva-nextgen/actions

.. image:: https://codecov.io/github/FREVA-CLINT/freva-nextgen/branch/init/graph/badge.svg?token=dGhXxh7uP3
   :target: https://codecov.io/github/FREVA-CLINT/freva-nextgen


Freva, the free evaluation system framework, is a data search and analysis
platform developed by the atmospheric science community for the atmospheric
science community. With help of Freva researchers can:

- quickly and intuitively search for data stored at typical data centers that
  host many datasets.
- create a common interface for user defined data analysis tools.
- apply data analysis tools in a reproducible manner.

About this Documentation
------------------------
This documentation describes the *freva-client* library, its
command line interface (cli) and the REST API. The *freva-client* library
described in this documentation only support searching for data. If you
need to apply data analysis plugins, please visit the
`official documentation <https://freva-clint.github.io/freva>`_

.. _install+configure:

Installation and configuration
------------------------------

Installation of the client library is straight forward and can be achieved via:

.. code:: console

    python3 -m pip install freva-client

After successful installation you will also have to following command line
interfaces (cli) available:

.. code:: console

    freva-client --help

.. execute_code::
   :hide_code:

   from subprocess import run, PIPE
   res = run(["freva-client", "--help"], check=True, stdout=PIPE, stderr=PIPE)
   print(res.stdout.decode())


Configuration
+++++++++++++

The client library needs to make connections to the freva server. You can either
set the server host names by using the ``host`` arguments (or ``--host`` flags
in the cli) to make connections or permanently set the freva server host name.
To do so you have several options:

- If you are using the centrally administered freva instance you don't have to
  do anything.
- If on the other hand you have installed the client library yourself you can
  use the ``freva.toml`` configuration located in the user config directory
  (``.config/freva/freva.toml``)
- You can also set the ``FREVA_CONFIG`` environment variable to point to
  any location of the ``freva.toml`` file

The configuration file itself follows `toml syntax <https://toml.io>`_. After
installation you will have freva config file with the following content
placed in the user configuration directory of your OS (e.g ``~/.config/freva``
on Linux):

.. code:: toml

    ## The new freva configuration file.
    ## This configuration files follows toml (https://toml.io) syntax and replaces
    ## the old evaluation_system.conf file.
    #
    [freva]
    ## This section configures the freva client. All settings related to freva
    ## are set here.
    ##
    ## The name of the specific freva instance. If you use multiple configurations
    ## for different freva instances the `project_name` entry allows you to
    ## later identify which configuration you were using.
    # project_name = "freva"

    ##
    ## The host that runs the freva api system. In most cases this is just the
    ## url of the freva webpage, such as https://www.freva.dkrz.de.
    ## You can set a port by separating <hostname:port>
    ## for example freva.example.org:7777
    # host = ""

To permanently set or override the freva server host name you have to set
the ``host`` variable in that file. In most cases this variable can be
set to the url of the freva web site you are using, for example
https//www.freva.dkrz.de.

Table of Content
================

.. toctree::
   :maxdepth: 1

   auth/index
   databrowser/index

Feedback and Support
====================
We value your feedback and are committed to improving the API continuously.
If you encounter any issues, have questions, or wish to suggest improvements,
please don't hesitate to reach out to our support team. Your input is
invaluable in enhancing the Freva Databrowser REST API to better suit your
needs.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
