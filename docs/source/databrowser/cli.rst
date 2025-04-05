The databrowser command line interface
======================================
.. _databrowser-cli:

.. toctree::
   :maxdepth: 3

This section introduces the usage of the ``freva-client databrowser`` sub command.
Please see the :ref:`install+configure` section on how to install and
configure the command line interface.


After successful installation you can use the ``freva-client databrowser`` sub
command

.. code:: console

    freva-client databrowser --help

.. execute_code::
   :hide_code:

   from subprocess import run, PIPE
   res = run(["freva-client", "databrowser", "--help"], check=True, stdout=PIPE, stderr=PIPE)
   print(res.stdout.decode())


Searching for data locations
----------------------------
Getting the locations of the data is probably the most common use case of the
databrowser application. You can search for data locations by applying the
``data-search`` sub-command:

.. code:: console

    freva-client databrowser data-search --help

.. execute_code::
   :hide_code:

   from subprocess import run, PIPE
   res = run(["freva-client", "databrowser", "data-search", "--help"], check=True, stdout=PIPE, stderr=PIPE)
   print(res.stdout.decode())


The command expects a list of key=value pairs. The order of the
pairs doesn't really matter. Most important is that you don’t need to
split the search according to the type of data you are searching for.
You can search for files within observations, reanalysis and
model data at the same time. Also important is that all queries are
case *insensitive*. You can also search for attributes themselves
instead of file paths. For example you can search for the list of
variables available that satisfies a certain constraint (e.g. sampled
6hr, from a certain model, etc).

.. code:: console

    freva-client databrowser data-search project=observations variable=pr model=cp*

.. execute_code::
   :hide_code:

   from subprocess import run, PIPE
   res = run(["freva-client", "databrowser", "data-search", "experiment=cmorph"], check=True, stdout=PIPE, stderr=PIPE)
   print(res.stdout.decode())

There are many more options for defining a value for a given key:

+-------------------------------------------------+------------------------+
| Attribute syntax                                | Meaning                |
+=================================================+========================+
| ``attribute=value``                             | Search for files       |
|                                                 | containing exactly     |
|                                                 | that attribute         |
+-------------------------------------------------+------------------------+
| ``attribute='val\*'``                           | Search for files       |
|                                                 | containing a value for |
|                                                 | attribute that starts  |
|                                                 | with the prefix val    |
+-------------------------------------------------+------------------------+
| ``attribute='*lue'``                            | Search for files       |
|                                                 | containing a value for |
|                                                 | attribute that ends    |
|                                                 | with the suffix lue    |
+-------------------------------------------------+------------------------+
| ``attribute='*alu\*'``                          | Search for files       |
|                                                 | containing a value for |
|                                                 | attribute that has alu |
|                                                 | somewhere              |
+-------------------------------------------------+------------------------+
| ``attribute='/.*alu.*/'``                       | Search for files       |
|                                                 | containing a value for |
|                                                 | attribute that matches |
|                                                 | the given regular      |
|                                                 | expression (yes! you   |
|                                                 | might use any regular  |
|                                                 | expression to find     |
|                                                 | what you want.)        |
+-------------------------------------------------+------------------------+
| ``attribute=value1 attribute=value2``           | Search for files       |
|                                                 | containing either      |
| OR:                                             | value1 OR value2 for   |
|                                                 | the given attribute    |
| ``attribute={value1,value2}``                   | (note that's the same  |
|                                                 | attribute twice!)      |
+-------------------------------------------------+------------------------+
| ``attribute1=value1 attribute2=value2``         | Search for files       |
|                                                 | containing value1 for  |
|                                                 | attribute1 AND value2  |
|                                                 | for attribute2         |
+-------------------------------------------------+------------------------+
| ``attribute_not_=value``                        | Search for files NOT   |
|                                                 | containing value       |
+-------------------------------------------------+------------------------+
| ``attribute_not_=value1 attribute_not_=value2`` | Search for files       |
|                                                 | containing neither     |
|                                                 | value1 nor value2      |
+-------------------------------------------------+------------------------+

.. note::

    When using \* remember that your shell might give it a
    different meaning (normally it will try to match files with that name)
    to turn that off you can use backslash \ (key=\*) or use quotes (key='*').

Searching multi-versioned datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In datasets with multiple versions only the `latest` version (i.e. `highest`
version number) is returned by default. Querying a specific version from a
multi versioned datasets requires the ``multi-version`` flag in combination with
the ``version`` special attribute:

.. code:: console

    freva-client databrowser data-search dataset=cmip6-fs model=access-cm2 --multi-version version=v20191108

.. execute_code::
   :hide_code:

   from subprocess import run, PIPE
   res = run(["freva-client", "databrowser", "data-search",
              "dataset=cmip6-fs", "model=access-cm2", "--multi-version", 
              "version=v20191108",
             ], check=True, stdout=PIPE, stderr=PIPE)
   print(res.stdout.decode())

If no particular ``version`` is requested, all versions will be returned.

Streaming files via zarr
~~~~~~~~~~~~~~~~~~~~~~~~
Instead of getting the file locations on disk or tape, you can instruct the
system to register zarr streams. Which means that instead of opening the
data directly you can open it via zarr from anywhere. To do so simply add
the ``--zarr`` flag.

.. note::

    Before you can use the ``--zarr`` flag you will have
    to create an access token and use that token to log on to the system
    see also the :ref:`auth` chapter for more details on token creation.

.. code:: console

    token=$(freva-client auth -u janedoe|jq -r .access_token)
    freva-client databrowser data-search dataset=cmip6-fs --zarr --access-token $token

.. execute_code::
   :hide_code:

   from subprocess import run, PIPE
   from freva_client import authenticate
   token = authenticate(username="janedoe")
   res = run(["freva-client", "databrowser", "data-search",
              "--zarr", "dataset=cmip6-fs",
              "--access-token", token["access_token"],
             ], check=True, stdout=PIPE, stderr=PIPE)
   print(res.stdout.decode())



Special cases: Searching for times
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For example you want to know how many files we have between a certain time range
To do so you can use the `time` search key to subset time steps and whole time
ranges:

.. code:: console

    freva-client databrowser data-search project=observations -t '2016-09-02T22:15 to 2016-10'

.. execute_code::
   :hide_code:

   from subprocess import run, PIPE
   res = run(["freva-client", "databrowser", "data-search",
              "-t", "2016-09-02T22:15 to 2016-10",
             ], check=True, stdout=PIPE, stderr=PIPE)
   print(res.stdout.decode())

The default method for selecting time periods is ``flexible``, which means
all files are selected that cover at least start or end date. The
``strict`` method implies that the *entire* search time period has to be
covered by the files. Using the ``strict`` method in the example above would
only yield on file because the first file contains time steps prior to the
start of the time period:

.. code:: console

    freva-client databrowser data-search project=observations -t '2016-09-02T22:15 to 2016-10' -ts strict

.. execute_code::
   :hide_code:

   from subprocess import run, PIPE
   res = run(["freva-client", "databrowser", "data-search", "-t", "2016-09-02T22:15 to 2016-10", "-ts", "strict"], check=True, stdout=PIPE, stderr=PIPE)
   print(res.stdout.decode())

Giving single time steps is also possible:

.. code:: console

    freva-client databrowser data-search project=observations -t 2016-09-02T22:10

.. execute_code::
   :hide_code:

   from subprocess import run, PIPE
   res = run(["freva-client", "databrowser", "data-search", "-t", "2016-09-02T22:00"], check=True, stdout=PIPE, stderr=PIPE)
   print(res.stdout.decode())

.. note::

    The time format has to follow the
    `ISO-8601 <https://en.wikipedia.og/wiki/ISO_8601>`_ standard. Time *ranges*
    are indicated by the ``to`` keyword such as ``2000 to 2100`` or
    ``2000-01 to 2100-12`` and alike. Single time steps are given without the
    ``to`` keyword.

Creating intake-esm catalouges
-------------------------------
The ``intake-catalogue`` sub command allows you to create an
`intake-esm catalogue <https://intake-esm.readthedocs.io/en/stable/>_` from
the current search. This can be useful to share the catalogue with others
or merge datasets.

.. code:: console

    freva-client databrowser intake-catalogue --help

.. execute_code::
   :hide_code:

   from subprocess import run, PIPE
   res = run(["freva-client", "databrowser", "intake-catalogue", "--help"], check=True, stdout=PIPE, stderr=PIPE)
   print(res.stdout.decode())


You can either set the ``--filename`` flag to save the catalogue to a ``.json``
file or pipe the catalogue to stdout (default). Just like for the ``data-search``
sub command you can instruct the system to create zarr file streams to access
the data via zarr.


Creating STAC Catalogue
--------------------------
The ``stac-catalogue`` sub command allows you to create a static  
`SpatioTemporal Asset Catalog (STAC) <https://stacspec.org/en/about/stac-spec/>_ `
from the current search. This can be useful for creating, sharing and using 
standardized geospatial data catalogs and enabling interoperability between 
different data systems.

.. code:: console

    freva-client databrowser stac-catalogue --help

.. execute_code::
   :hide_code:

   from subprocess import run, PIPE

   res = run(["freva-client", "databrowser", "stac-catalogue", "--help"], check=True, stdout=PIPE, stderr=PIPE)
   print(res.stdout.decode())

To get an static STAC catalogue you can use the following command:

.. code:: console

    freva-client databrowser stac-catalogue --filename /path/to/output

and if the specified filename directory doesn't specify or not existed or not provided, 
the STAC catalogue will be saved in the current directory. It can be 
only a directory or a fully qualified filename. 

The STAC Catalogue provides multiple ways to access and interact with the data:
 - Access your climate data through the intake-esm data catalog specification
 - Access search results as Zarr files, available as STAC Assets at both collection and item levels
 - Browse and explore your search results directly through the Freva DataBrowser web interface
Each of these access methods is encoded as STAC Assets, making them easily discoverable and accessible through any STAC-compatible tool.


Query the number of occurrences
-------------------------------
In some cases it might be useful to know how many files are found in the
databrowser for certain search constraints. In such cases you can use the
``data-count`` sub command to count the number of *found* files instead of getting
the files themselves.

.. code:: console

    freva-client databrowser data-count --help

.. execute_code::
   :hide_code:

   from subprocess import run, PIPE
   res = run(["freva-client", "databrowser", "data-count", "--help"], check=True, stdout=PIPE, stderr=PIPE)
   print(res.stdout.decode())

By default the ``data-count`` sub command will display the total number of items
matching your search query. For example:

.. code:: console

    freva-client databrowser data-count project=observations

.. execute_code::
   :hide_code:

   from subprocess import run, PIPE
   res = run(["freva-client", "databrowser", "data-count", "project=observations"], check=True, stdout=PIPE, stderr=PIPE)
   print(res.stdout.decode())

If you want to group the number of occurrences by search categories (facets)
use the ``-d`` or ``--detail`` flag:

.. code:: console

    freva-client databrowser data-count -d project=observations

.. execute_code::
   :hide_code:

   from subprocess import run, PIPE
   res = run(["freva-client", "databrowser", "data-count", "-d", "project=observations"], check=True, stdout=PIPE, stderr=PIPE)
   print(res.stdout.decode())



Retrieving the available metadata
---------------------------------
Sometime it might be useful to know about possible values that search attributes.
For this you use the ``metadata-search`` sub command:

.. code:: console

    freva-client databrowser metadata-search --help

.. execute_code::
   :hide_code:

   from subprocess import run, PIPE
   res = run(["freva-client", "databrowser", "metadata-search", "--help"], check=True, stdout=PIPE, stderr=PIPE)
   print(res.stdout.decode())

Just like with any other databrowser command you can apply different search
constraints when acquiring metadata

.. code:: console

    freva-client databrowser metadata-search project=observations

.. execute_code::
   :hide_code:

   from subprocess import run, PIPE
   res = run(["freva-client", "databrowser", "metadata-search", "project=observations"], check=True, stdout=PIPE, stderr=PIPE)
   print(res.stdout.decode())


By default the command will display only the most commonly used metadata
keys. To retrieve all metadata you can use the ``-e`` or ``--extended-search``
flag.

.. code:: console

    freva-client databrowser metadata-search -e project=observations

.. execute_code::
   :hide_code:

   from subprocess import run, PIPE
   res = run(["freva-client", "databrowser", "metadata-search", "-e", "project=observations"], check=True, stdout=PIPE, stderr=PIPE)
   print(res.stdout.decode())

Sometimes you don't exactly know the exact names of the search keys and
want retrieve all file objects that match a certain category. For example
for getting all ocean reanalysis datasets you can apply the ``--facet`` flag:

.. code:: console

    freva-client databrowser metadata-search -e realm=ocean --facet 'rean*'

.. execute_code::
   :hide_code:

   from subprocess import run, PIPE
   res = run(["freva-client", "databrowser", "metadata-search","--facet", "rean*", "realm=ocean"], check=True, stdout=PIPE, stderr=PIPE)
   print(res.stdout.decode())



Expert tip: Getting metadata for certain files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In some cases it might be useful to retrieve metadata for certain
file or object store locations. For example if we wanted to retrieve the
metadata of those files on tape:

.. code:: console

    freva-client databrowser metadata-search -e file="/arch/*"

.. execute_code::
   :hide_code:

   from subprocess import run, PIPE
   res = run(["freva-client","databrowser", "metadata-search", "-e", "file=/arch*"], check=True, stdout=PIPE, stderr=PIPE)
   print(res.stdout.decode())

Parsing the command output
--------------------------

You might have already noticed that each command has a ``--json`` flag.
Enabling this flag lets you parse the output of each command to JSON.

Following on from the example above we can parse the output of the reverse
search to the `command line json processor jq <https://jqlang.github.io/jq/>`_:

.. code:: console

    freva-client databrowser metadata-search -e file="/arch/*" --json

.. execute_code::
   :hide_code:

   from subprocess import run, PIPE
   res = run(["freva-client", "databrowser", "metadata-search", "-e", "file=/arch*", "--json"], check=True, stdout=PIPE, stderr=PIPE)
   print(res.stdout.decode())

By using the pipe operator ``|`` the JSON output of the `freva-client`
commands can be piped and processed by ``jq``:

.. code:: console

    freva-client databrowser metadata-search -e file="/arch/*" --json | jq -r .ensemble[0]

.. execute_code::
   :hide_code:

   print("r1i1p1")

The above example will select only the first entry of the ensembles that
are associated with files on the tape archive.


Adding and Deleting User Data
-----------------------------

You can manage your personal datasets within the databrowser by adding or deleting user-specific data. This functionality allows you to include your own data files into the databrowser, making them accessible for analysis and retrieval alongside other datasets.

Before using the `user-data` commands, you need to create an access token and authenticate with the system. Please refer to the :ref:`auth` chapter for more details on token creation and authentication.

Adding User Data
~~~~~~~~~~~~~~~~

To add your data to the databrowser, use the `user-data add` command. You'll need to provide the paths to your data files, and any metadata you'd like to associate with your data.

.. code:: console

    token=$(freva-client auth -u janedoe | jq -r .access_token)
    freva-client databrowser user-data add \
        --path freva-rest/src/freva_rest/databrowser_api/mock/ \
        --facet project=cordex \
        --facet experiment=rcp85 \
        --facet model=mpi-m-mpi-esm-lr-clmcom-cclm4-8-17-v1 \
        --facet variable=tas \
        --access-token $token

.. execute_code::
   :hide_code:

   from subprocess import run, PIPE
   from freva_client import authenticate
   token = authenticate(username="janedoe")
   res = run(
       ["freva-client", "databrowser", "user-data", "add",
       "--path", "../freva-rest/src/freva_rest/databrowser_api/mock/",
       "--facet", "project=cordex",
       "--facet", "experiment=rcp85",
       "--facet", "model=mpi-m-mpi-esm-lr-clmcom-cclm4-8-17-v1",
       "--facet", "variable=tas",
       "--access-token", token["access_token"]],
       check=True, stdout=PIPE, stderr=PIPE
   )
   print(res.stdout.decode())


This command adds the specified data files to the databrowser and tags them with the provided metadata. These search filters help in indexing and searching your data within the system.

Deleting User Data
~~~~~~~~~~~~~~~~~~

If you need to remove your data from the databrowser, use the `user-data delete` command. Provide your the search keys (facets) that identify the user data you wish to delete.

.. code:: console

    token=$(freva-client auth -u janedoe | jq -r .access_token)
    freva-client databrowser user-data delete \
        --search-key project=cordex \
        --search-key experiment=rcp85 \
        --access-token $token

.. execute_code::
   :hide_code:

   from subprocess import run, PIPE
   from freva_client import authenticate
   token = authenticate(username="janedoe")
   res = run(["freva-client", "databrowser", "user-data",
              "delete",
              "--search-key", "project=cordex", \
              "--access-token", token["access_token"],
             ], check=True, stdout=PIPE, stderr=PIPE)
   print(res.stderr.decode())

This command deletes all data entries that match the specified search keys from the databrowser.
