The databrowser command line interface
======================================
.. _databrowser-cli:

.. toctree::
   :maxdepth: 3

This section introduces the usage of the ``freva-client`` command.
Please see the :ref:`install+configure` section on how to install and
configure the command line interface.


After successful installation you will have the ``freva-client`` command

.. code:: console

    freva-client --help

.. execute_code::
   :hide_code:

   from subprocess import run, PIPE
   res = run(["freva-client", "--help"], check=True, stdout=PIPE, stderr=PIPE)
   print(res.stdout.decode())


Searching for data locations
----------------------------
Getting the locations of the data is probably the most common use case of the
databrowser application. You can search for data locations by applying the
``data-search`` sub-command:

.. code:: console

    freva-client data-search --help

.. execute_code::
   :hide_code:

   from subprocess import run, PIPE
   res = run(["freva-client", "data-search", "--help"], check=True, stdout=PIPE, stderr=PIPE)
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

    freva-client project=observations variable=pr model=cp*

.. execute_code::
   :hide_code:

   from subprocess import run, PIPE
   res = run(["freva-client", "data-search", "experiment=cmorph"], check=True, stdout=PIPE, stderr=PIPE)
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
| ``attribute=value1 [...] attribute=valueN``     | Search for files       |
|                                                 | containing *any* of N  |
| OR:                                             | given values of same   |
|                                                 | attribute.             |
| ``attribute={value1,..,valueN}``                |                        |
|                                                 |                        |
+-------------------------------------------------+------------------------+
| ``attribute1=value1 attribute2=value2``         | Search for files       |
|                                                 | containing value1 for  |
|                                                 | attribute1 AND value2  |
|                                                 | for attribute2         |
+-------------------------------------------------+------------------------+
| ``attribute=-value`` ``attribute=not value``    | Search for files NOT   |
|                                                 | containing value       |
+-------------------------------------------------+------------------------+
| ``attribute=-value1 attribute=not value2``      | Search for files       |
|                                                 | *not* caintaning given |
|  OR                                             | values. You can also   |
|                                                 | combine search         |
| ``attribute1_not_=value _not_attribute2=value`` | attributes             |
+-------------------------------------------------+------------------------+

.. note::

    When using \* remember that your shell might give it a
    different meaning (normally it will try to match files with that name)
    to turn that off you can use backslash \ (key=\*) or use quotes (key='*').

Special cases: Searching for times
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For example you want to know how many files we have between a certain time range
To do so you can use the `time` search key to subset time steps and whole time
ranges:

.. code:: console

    freva-client data-search project=observations -t '2016-09-02T22:15 to 2016-10'

.. execute_code::
   :hide_code:

   from subprocess import run, PIPE
   res = run(["freva-client", "data-search",
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

    freva-client data-search project=observations -t '2016-09-02T22:15 to 2016-10' -ts strict

.. execute_code::
   :hide_code:

   from subprocess import run, PIPE
   res = run(["freva-client", "data-search", "-t", "2016-09-02T22:15 to 2016-10", "-ts", "strict"], check=True, stdout=PIPE, stderr=PIPE)
   print(res.stdout.decode())

Giving single time steps is also possible:

.. code:: console

    freva-client data-search project=observations -t 2016-09-02T22:10

.. execute_code::
   :hide_code:

   from subprocess import run, PIPE
   res = run(["freva-client", "data-search", "-t", "2016-09-02T22:00"], check=True, stdout=PIPE, stderr=PIPE)
   print(res.stdout.decode())

.. note::

    The time format has to follow the
    `ISO-8601 <https://en.wikipedia.og/wiki/ISO_8601>`_ standard. Time *ranges*
    are indicated by the ``to`` keyword such as ``2000 to 2100`` or
    ``2000-01 to 2100-12`` and alike. Single time steps are given without the
    ``to`` keyword.


Query the number of occurrences
-------------------------------
In some cases it might be useful to know how many files are found in the
databrowser for certain search constraints. In such cases you can use the
``data-count`` sub command to count the number of *found* files instead of getting
the files themselves.

.. code:: console

    freva-client data-count --help

.. execute_code::
   :hide_code:

   from subprocess import run, PIPE
   res = run(["freva-client", "data-count", "--help"], check=True, stdout=PIPE, stderr=PIPE)
   print(res.stdout.decode())

By default the ``data-count`` sub command will display the total number of items
matching your search query. For example:

.. code:: console

    freva-client data-count project=observations

.. execute_code::
   :hide_code:

   from subprocess import run, PIPE
   res = run(["freva-client", "data-count", "project=observations"], check=True, stdout=PIPE, stderr=PIPE)
   print(res.stdout.decode())

If you want to group the number of occurrences by search categories (facets)
use the ``-d`` or ``--detail`` flag:

.. code:: console

    freva-client data-count -d project=observations

.. execute_code::
   :hide_code:

   from subprocess import run, PIPE
   res = run(["freva-client", "data-count", "-d", "project=observations"], check=True, stdout=PIPE, stderr=PIPE)
   print(res.stdout.decode())



Retrieving the available metadata
---------------------------------
Sometime it might be useful to know about possible values that search attributes.
For this you use the ``metadata-search`` sub command:

.. code:: console

    freva-client metadata-search --help

.. execute_code::
   :hide_code:

   from subprocess import run, PIPE
   res = run(["freva-client", "metadata-search", "--help"], check=True, stdout=PIPE, stderr=PIPE)
   print(res.stdout.decode())

Just like with any other databrowser command you can apply different search
constraints when acquiring metadata

.. code:: console

    freva-client metadata-search project=observations

.. execute_code::
   :hide_code:

   from subprocess import run, PIPE
   res = run(["freva-client", "metadata-search", "project=observations"], check=True, stdout=PIPE, stderr=PIPE)
   print(res.stdout.decode())


By default the command will display only the most commonly used metadata
keys. To retrieve all metadata you can use the ``-e`` or ``--extended-search``
flag.

.. code:: console

    freva-client metadata-search -e project=observations

.. execute_code::
   :hide_code:

   from subprocess import run, PIPE
   res = run(["freva-client", "metadata-search", "-e", "project=observations"], check=True, stdout=PIPE, stderr=PIPE)
   print(res.stdout.decode())

Sometimes you don't exactly know the exact names of the search keys and
want retrieve all file objects that match a certain category. For example
for getting all ocean reanalysis datasets you can apply the ``--facet`` flag:

.. code:: console

    freva-client metadata-search -e realm=ocean --facet 'rean*'

.. execute_code::
   :hide_code:

   from subprocess import run, PIPE
   res = run(["freva-client", "metadata-search","--facet", "rean*", "realm=ocean"], check=True, stdout=PIPE, stderr=PIPE)
   print(res.stdout.decode())



Expert tip: Getting metadata for certain files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In some cases it might be useful to retrieve metadata for certain
file or object store locations. For example if we wanted to retrieve the
metadata of those files on tape:

.. code:: console

    freva-client metadata-search -e file="/arch/*"

.. execute_code::
   :hide_code:

   from subprocess import run, PIPE
   res = run(["freva-client", "metadata-search", "-e", "file=/arch*"], check=True, stdout=PIPE, stderr=PIPE)
   print(res.stdout.decode())

Parsing the command output
--------------------------

You might have already noticed that each command has a ``--json`` flag.
Enabling this flag lets you parse the output of each command to JSON.

Following on from the example above we can parse the output of the reverse
search to the `command line json processor jq <https://jqlang.github.io/jq/>`_:

.. code:: console

    freva-client metadata-search -e file="/arch/*" --json

.. execute_code::
   :hide_code:

   from subprocess import run, PIPE
   res = run(["freva-client", "metadata-search", "-e", "file=/arch*", "--json"], check=True, stdout=PIPE, stderr=PIPE)
   print(res.stdout.decode())

By using the pipe operator ``|`` the JSON output of the `freva-client`
commands can be piped and processed by ``jq``:

.. code:: console

    freva-client metadata-search -e file="/arch/*" --json | jq -r .ensemble[0]

.. execute_code::
   :hide_code:

   print("r1i1p1")

The above example will select only the first entry of the ensembles that
are associated with files on the tape archive.
