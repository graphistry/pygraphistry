"""
This is the graphistry package.

Documentation for the code
**************************

Python Loader
===============
This is established as a Data Loader for Graphistry

This Loader support four types of data : Json (eg. '{"xxx":"xxx"}'), Json Pointer (eg. './xxx/xxx.json'), Pandas (eg. csv table), Pandas Pointer (eg. './xxx/xxx.csv')

Module #1
----------------

This class works as data Loader.

It contains six functions :func: `loadpandassync`. :func: `loadjsonsync`.
:func: `loadjsonpointer`. :func: `loadpandas`. :func: `isjsonpointer`. :func: `settings`.

.. function:: settings(url, frameheight)

   This function does the work of setting proper server for user.

   :param url(str): 'proxy'/'localhost', the default value is 'proxy'.
                    'localhost' would let user connect to the local server,
                    'proxy' would let user connect with server from Graphistry
   :param frameheight(integer): frameheight is used to set the size of returned figures,
                                the default value is 500.


.. function:: loadpandassync(edge, node, graphname=None, sourcefield=None, desfield=None, nodefield=None, edgetitle=None,edgelabel=None, edgeweight=None, pointtitle=None, pointlabel=None, pointcolor=None, pointsize=None)

   loadpandassync is the primary method. it parses the data with pandas format or with pandas pointer and load it to proper Graphistry server, and send back the concerned graph. It is a wrapper for :func: `loadpandas`.


   :param edge(Pandas pointer/Pandas)(compulsory): dataset for edge
   :param node(Pandas pointer/Pandas)(compulsory): dataset for node
   :param sourcefield(str)(optional): name of the column in edge for sourcePointers
                                      (eg.'source'), the default value is 'src'.
   :param desfield(str)(optional): name of the column in edge for destinationPointers
                                   (eg. 'des'), the default value is 'dst'.
   :param nodefield(str)(optional): name of the column in node for all pointers, the
                                    default value is 'node'
   :param graphname(str)(optional): a name string the user prefer (eg. 'myGraph'), the
                                    default value is a random generagted ten-characters
                                    string with numbers and capital letters.
   :param edgetitle(str)(optional): name of the column in edge for edges' titles, the
                                   default value is 'edgeTitle'
   :param edgelabel(str)(optional): name of the column in edge for edges' labels, the
                                    default value is 'edgeLabel'
   :param edgeweight(str)(optional): name of the column in edge for edges' Weights, the
                                     default value is 'edgeWeight'
   :param pointtitle(str)(optional): name of the column in node for nodes' titles, the
                                     default value is 'pointTitle'
   :param pointlabel(str)(optional): name of the column in node for nodes' labels, the
                                     default value is 'pointLabel'
   :param pointColor(str)(optional): name of the column in node for nodes' colors, the
                                     default value is 'pointColor'
   :param pointSize(str)(optional): name of the column in node for nodes' sizes, the
                                    default value is 'pointSize'


.. function:: loadjsonsync (document)

   loadjsonsync is the primary method. it parses the data with Json format or with Json pointer and load it to proper Graphistry server, and send back the concerned graph. It is a wrapper for :func: `isjsonpointer`, :func: `loadjsonpointer`.

   :param document(Json pointer / Json)(compulsory): complete dataset for edge and node


.. function:: loadjsonpointer(filedir)

   This function load Json file from give directory and returns it with Json format (Decoding)


.. function:: loadpandas(edge, node, graphname, sourcefield, desfield, nodefield, edgeTitle, edgeLabel, edgeWeight, pointTitle, pointLabel, pointColor, pointSize)

   This function load data with pandas format as edge and node seperately and returns it with Json format (Decoding)


.. function:: isjsonpointer(document)

   This function checks whether the input document is Json pointer
"""

from graphistry.etl import settings, plot
