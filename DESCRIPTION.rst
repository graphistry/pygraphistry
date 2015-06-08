A Data Loader Project for Graphistry
=======================

This Loader support four types of data : Json (eg. '{"xxx":"xxx"}'),
Json Pointer (eg. './xxx/xxx.json'), Pandas (eg. csv table),
Pandas Pointer (eg. './xxx/xxx.csv')

Class Graphistry has the following public methods:

settings(url, frameheight)
This function does the work of setting proper server for user.
url = 'proxy'/'localhost', the default value is 'proxy'.
'localhost' would let user connect to the local server, 'proxy'
would let user connect with server from Graphistry
frameheight = integer, the default value is 500.
frameheight is used to set the size of returned figures

...............................................................................

loadpandassync (edge, node, graphname, sourcefield, desfield, nodefield,
         edgetitle, edgelabel, edgeweight, pointtitle , pointlabel,
         pointcolor, pointsize)
loadpandassync is the primary method. it parses the data with pandas format
or with pandas pointer and load it to proper Graphistry server, and send back
the concerned graph. It is a wrapper for loadpandas().

edge(compulsory) = dataset for edge, the format includes Pandas pointer/Pandas
node(compulsory) = dataset for node, the format includes Pandas pointer/Pandas
sourcefield(optional) = name of the column in edge for sourcePointers
                         (eg. 'source'), the default value is 'src'.
desfield(optional) = name of the column in edge for destinationPointers
                      (eg. 'des'), the default value is 'dst'.
nodefield(optional) = name of the column in node for all pointers,
                       the default value is 'node'
graphname(optional) = a name string the user prefer (eg. 'myGraph'), the
                       default value is a random generagted ten-characters
                       string with numbers and capital letters.
edgetitle(optional) = name of the column in edge for edges' titles, the
                       default value is 'edgeTitle'
edgelabel(optional) = name of the column in edge for edges' labels, the
                       default value is 'edgeLabel'
edgeweight(optional) = name of the column in edge for edges' Weights, the
                        default value is 'edgeWeight'
pointtitle(optional) = name of the column in node for nodes' titles, the
                        default value is 'pointTitle'
pointlabel(optional) = name of the column in node for nodes' labels, the
                        default value is 'pointLabel'
pointColor(optional) = name of the column in node for nodes' colors, the
                        default value is 'pointColor'
pointSize(optional) = name of the column in node for nodes' sizes, the
                       default value is 'pointSize'

...............................................................................

loadjsonsync (document)
loadjsonsync is the primary method. it parses the data with Json format
or with Json pointer and load it to proper Graphistry server, and send back
the concerned graph. It is a wrapper for isjsonpointer(), loadjsonpointer().

document(compulsory) = complete dataset for edge and node,
                        the format includes Json pointer / Json

...............................................................................

loadjsonpointer(filedir)
This function load Json file from give directory and returns it with
Json format (Decoding)
...............................................................................

loadpandas(edge, node, graphname, sourcefield, desfield, nodefield,
            edgeTitle, edgeLabel, edgeWeight, pointTitle, pointLabel,
            pointColor, pointSize)
This function load data with pandas format as edge and node seperately
and returns it with Json format (Decoding)
...............................................................................

isjsonpointer(document)
This function checks whether the input document is Json pointer
...............................................................................
