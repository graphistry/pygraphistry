# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import docutils.nodes, os, logging, re, sys
from docutils import nodes
from packaging.version import Version
from sphinx.application import Sphinx


sys.path.insert(0, os.path.abspath("../.."))
import graphistry


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# -- Project information -----------------------------------------------------

project = "PyGraphistry"
copyright = "2025, Graphistry, Inc."
author = "Graphistry, Inc."

html_title = "PyGraphistry Documentation"
html_short_title = "PyGraphistry"
html_logo = 'graphistry_banner_transparent_colored.png'
html_favicon = 'static/favicon.ico'

# The full version, including alpha/beta/rc tags
version = str(Version(graphistry.__version__))
release = version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'myst_parser',
    'nbsphinx',
    "sphinx.ext.autodoc",
    "sphinx.ext.graphviz",
    #'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    "sphinx.ext.ifconfig",
    #"sphinx_autodoc_typehints",
    "sphinx_copybutton",
]


# TODO guarantee most notebooks are executable (=> maintained)
# and switch to opt'ing out the few that are hard, e.g., DB deps
nbsphinx_execute = 'never'
nbsphinx_allow_errors = False  # Allow errors in notebooks

autodoc_typehints = "description"
always_document_param_types = True
typehints_document_rtype = True

#suppress_warnings = [
#    'nbsphinx.localfile',  # Suppresses local file warnings in notebooks
#]

#FIXME Why is sphinx/autodoc failing here?
nitpick_ignore = [
    ('py:class', '1'),  # Ex: api : Optional[Literal[1, 3]]
    ('py:class', '3'),
    ('py:class', "<class 'dict'>"),
    ('py:class', "<class 'str'>"),
    ('py:class', 'BiPartiteGraph'),
    ('py:class', "graphistry.compute.ASTSerializable.ASTSerializable"),
    ('py:class', "graphistry.compute.chain.Chain"),
    ('py:class', "graphistry.compute.predicates.ASTPredicate.ASTPredicate"),
    ('py:class', 'graphistry.compute.predicates.categorical.Duplicated'),
    ('py:class', 'graphistry.compute.predicates.is_in.IsIn'),
    ('py:class', 'graphistry.compute.predicates.numeric.Between'),
    ('py:class', 'graphistry.compute.predicates.numeric.EQ'),
    ('py:class', 'graphistry.compute.predicates.numeric.GE'),
    ('py:class', 'graphistry.compute.predicates.numeric.GT'),
    ('py:class', 'graphistry.compute.predicates.numeric.IsNA'),
    ('py:class', 'graphistry.compute.predicates.numeric.LE'),
    ('py:class', 'graphistry.compute.predicates.numeric.LT'),
    ('py:class', 'graphistry.compute.predicates.numeric.NE'),
    ('py:class', 'graphistry.compute.predicates.numeric.NotNA'),
    ('py:class', 'graphistry.compute.predicates.numeric.NumericASTPredicate'),
    ('py:class', 'graphistry.compute.predicates.str.Contains'),
    ('py:class', 'graphistry.compute.predicates.str.Endswith'),
    ('py:class', 'graphistry.compute.predicates.str.IsAlnum'),
    ('py:class', 'graphistry.compute.predicates.str.IsAlpha'),
    ('py:class', 'graphistry.compute.predicates.str.IsDecimal'),
    ('py:class', 'graphistry.compute.predicates.str.IsDigit'),
    ('py:class', 'graphistry.compute.predicates.str.IsLower'),
    ('py:class', 'graphistry.compute.predicates.str.IsNull'),
    ('py:class', 'graphistry.compute.predicates.str.IsNumeric'),
    ('py:class', 'graphistry.compute.predicates.str.IsSpace'),
    ('py:class', 'graphistry.compute.predicates.str.IsTitle'),
    ('py:class', 'graphistry.compute.predicates.str.IsUpper'),
    ('py:class', 'graphistry.compute.predicates.str.Match'),
    ('py:class', 'graphistry.compute.predicates.str.Fullmatch'),
    ('py:class', 'graphistry.compute.predicates.str.NotNull'),
    ('py:class', 'graphistry.compute.predicates.str.Startswith'),
    ('py:class', 'graphistry.compute.predicates.temporal.IsLeapYear'),
    ('py:class', 'graphistry.compute.predicates.temporal.IsMonthEnd'),
    ('py:class', 'graphistry.compute.predicates.temporal.IsMonthStart'),
    ('py:class', 'graphistry.compute.predicates.temporal.IsQuarterEnd'),
    ('py:class', 'graphistry.compute.predicates.temporal.IsQuarterStart'),
    ('py:class', 'graphistry.compute.predicates.temporal.IsYearStart'),
    ('py:class', 'graphistry.compute.predicates.temporal.IsYearEnd'),
    ('py:class', 'graphistry.Engine.Engine'),
    ('py:class', 'graphistry.Engine.EngineAbstract'),
    ('py:class', 'graphistry.gremlin.CosmosMixin'),
    ('py:class', 'graphistry.gremlin.GremlinMixin'),
    ('py:class', 'graphistry.gremlin.NeptuneMixin'),
    ('py:class', 'graphistry.layouts.LayoutsMixin'),
    ('py:class', 'graphistry.compute.ComputeMixin'),
    ('py:class', 'graphistry.compute.conditional.ConditionalMixin'),
    ('py:class', 'graphistry.compute.cluster.ClusterMixin'),
    ('py:class', 'graphistry.Plottable.Plottable'),
    ('py:class', 'graphistry.plugins.cugraph.compute_cugraph'),
    ('py:class', 'graphistry.plugins.cugraph.from_cugraph'),
    ('py:class', 'graphistry.plugins.igraph.compute_igraph'),
    ('py:class', 'graphistry.plugins.igraph.from_igraph'),
    ('py:class', 'graphistry.plugins.igraph.layout_igraph'),
    ('py:class', 'graphistry.plugins.kusto.KustoMixin'),
    ('py:class', 'graphistry.plugins.spanner.SpannerMixin'),
    ('py:data', 'graphistry.plugins_types.cugraph_types.CuGraphKind'),
    ('py:data', 'graphistry.plugins_types.graphviz_types.EdgeAttr'),
    ('py:data', 'graphistry.plugins_types.graphviz_types.EDGE_ATTRS'),
    ('py:data', 'graphistry.plugins_types.graphviz_types.Format'),
    ('py:data', 'graphistry.plugins_types.graphviz_types.FORMATS'),
    ('py:data', 'graphistry.plugins_types.graphviz_types.GraphAttr'),
    ('py:data', 'graphistry.plugins_types.graphviz_types.GRAPH_ATTRS'),
    ('py:data', 'graphistry.plugins_types.graphviz_types.NodeAttr'),
    ('py:data', 'graphistry.plugins_types.graphviz_types.NODE_ATTRS'),
    ('py:data', 'graphistry.plugins_types.graphviz_types.Prog'),
    ('py:data', 'graphistry.plugins_types.graphviz_types.PROGS'),    
    
    # Suppress individual items from PROGS
    ('py:class', 'acyclic'),
    ('py:class', 'ccomps'),
    ('py:class', 'circo'),
    ('py:class', 'dot'),
    ('py:class', 'fdp'),
    ('py:class', 'gc'),
    ('py:class', 'gvcolor'),
    ('py:class', 'gvpr'),
    ('py:class', 'neato'),
    ('py:class', 'nop'),
    ('py:class', 'osage'),
    ('py:class', 'patchwork'),
    ('py:class', 'sccmap'),
    ('py:class', 'sfdp'),
    ('py:class', 'tred'),
    ('py:class', 'twopi'),
    ('py:class', 'unflatten'),
    
    # Suppress items from FORMATS
    ('py:class', 'canon'),
    ('py:class', 'cmap'),
    ('py:class', 'cmapx'),
    ('py:class', 'cmapx_np'),
    ('py:class', 'dia'),
    ('py:class', 'dot'),
    ('py:class', 'fig'),
    ('py:class', 'gd'),
    ('py:class', 'gd2'),
    ('py:class', 'gif'),
    ('py:class', 'hpgl'),
    ('py:class', 'imap'),
    ('py:class', 'imap_np'),
    ('py:class', 'ismap'),
    ('py:class', 'jpe'),
    ('py:class', 'jpeg'),
    ('py:class', 'jpg'),
    ('py:class', 'mif'),
    ('py:class', 'mp'),
    ('py:class', 'pcl'),
    ('py:class', 'pdf'),
    ('py:class', 'pic'),
    ('py:class', 'plain'),
    ('py:class', 'plain-ext'),
    ('py:class', 'png'),
    ('py:class', 'ps'),
    ('py:class', 'ps2'),
    ('py:class', 'svg'),
    ('py:class', 'svgz'),
    ('py:class', 'vml'),
    ('py:class', 'vmlz'),
    ('py:class', 'vrml'),
    ('py:class', 'vtx'),
    ('py:class', 'wbmp'),
    ('py:class', 'xdot'),
    ('py:class', 'xlib'),

    #TimeUnit = Literal['s', 'm', 'h', 'D', 'W', 'M', 'Y', 'C']
    ('py:data', 'graphistry.compute.temporal.TimeUnit'),
    ('py:class', 's'),
    ('py:class', 'm'),
    ('py:class', 'h'),
    ('py:class', 'D'),
    ('py:class', 'W'),
    ('py:class', 'M'),
    ('py:class', 'Y'),
    ('py:class', 'C'),

    ('py:class', 'abc.ABC'),
    ('py:class', 'graphistry.feature_utils.FeatureMixin'),
    ('py:class', 'graphistry.dgl_utils.DGLGraphMixin'),
    ('py:class', 'graphistry.umap_utils.UMAPMixin'),
    ('py:class', 'graphistry.text_utils.SearchToGraphMixin'),
    ('py:class', 'graphistry.embed_utils.HeterographEmbedModuleMixin'),
    ('py:class', 'graphistry.PlotterBase.PlotterBase'),
    ('py:class', 'graphistry.compute.ast.ASTObject'),
    ('py:class', 'graphistry.compute.filter_by_dict.IsIn'),
    ('py:class', 'Plotter'),
    ('py:class', 'Plottable'),
    ('py:class', 'CuGraphKind'),
    ('py:class', 'cugraph'),
    ('py:class', 'cugraph.BiPartiteGraph'),
    ('py:class', 'cugraph.Graph'),
    ('py:class', 'cugraph.MultiGraph'),
    ('py:class', 'IGraph graph'),
    ('py:class', 'igraph'),
    ('py:class', 'JSONVal'),
    ('py:class', 'dgl'),
    ('py:class', 'matplotlib'),
    ('py:class', 'MultiGraph'),
    ('py:class', 'numpy'),
    ('py:class', 'numpy.datetime64'),
    ('py:class', 'numpy.timedelta64'),
    ('py:class', 'pandas.core.frame.DataFrame'),
    ('py:class', 'pandas.core.series.Series'),
    ('py:class', 'pandas._libs.tslibs.offsets.DateOffset'),
    ('py:class', 'torch'),
    ('py:class', 'umap'),
    ('py:class', 'sentence_transformers'),
    ('py:class', 'sklearn'),
    ('py:class', 'scipy'),
    ('py:class', 'seaborn'),
    ('py:class', 'skrub'),
    ('py:class', 'annoy'),
    ('py:class', 'NetworkX graph'),
    ('py:class', 'Pandas dataframe'),
    ('py:class', 'ArrowUploader'),
    ('py:class', 'json.encoder.JSONEncoder'),
    ('py:class', 'pandas.DataFrame'),
    ('py:class', 'pyarrow.lib.Table'),
    ('py:class', 'requests.models.Response'),
    ('py:class', 'weakref.WeakKeyDictionary'),
    ('py:data', 'typing.Any'),
    ('py:data', 'typing.List'),
    ('py:data', 'typing.List[typing_extensions.Literal]'),    
    ('py:data', 'typing.Literal'),
    ('py:data', 'typing.Optional'),
    ('py:data', 'typing.Callable'),
    ('py:data', 'typing.Tuple'),
    ('py:data', 'typing.Union'),
    ('py:class', 'typing_extensions.Literal'),
    ('py:class', 'Mode'),
    ('py:class', 'graphistry.privacy.Privacy')
]

#set_type_checking_flag = True

# typehints_fully_qualified=True
always_document_param_types = True
typehints_document_rtype = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
# source_suffix = ['.rst', '.md']
# source_suffix = ['.rst', '.ipynb']
source_suffix = {
     '.md': 'markdown',
     '.txt': 'markdown',
    '.rst': 'restructuredtext',
    #'.ipynb': 'nbsphinx',
}
# The master toctree document.
root_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [

    '_build',
     '**/_build/**',
    'doctrees',
    '**/doctrees/**',
    'demos/.ipynb_checkpoints',
    '**/*.txt',

    # nbsphinx stalls on these
    'demos/ai/Introduction/Ask-HackerNews-Demo.ipynb',
    'demos/ai/OSINT/jack-donations.ipynb',

    #'demos/for_analysis.ipynb',
    #'demos/for_developers.ipynb',
    #'demos/upload_csv_miniapp.ipynb',

    # not used yet
    #'demos/demos_databases_apis/splunk/splunk_demo_public.ipynb',
    #'demos/demos_databases_apis/neptune/neptune_cypher_viz_using_bolt.ipynb',
    #'demos/demos_databases_apis/neptune/neptune_tutorial.ipynb',
    #'demos/demos_databases_apis/sql/postgres.ipynb',
    #'demos/demos_databases_apis/gpu_rapids/part_iv_gpu_cuml.ipynb',
    'demos/demos_databases_apis/gpu_rapids/part_iii_gpu_blazingsql.ipynb',
    #'demos/demos_databases_apis/gpu_rapids/part_ii_gpu_cudf.ipynb',
    #'demos/demos_databases_apis/gpu_rapids/part_i_cpu_pandas.ipynb',
    #'demos/demos_databases_apis/gpu_rapids/cugraph.ipynb',
    'demos/demos_databases_apis/memgraph/visualizing_iam_dataset.ipynb',
    #'demos/demos_databases_apis/databricks_pyspark/graphistry-notebook-dashboard.ipynb',
    #'demos/demos_databases_apis/arango/arango_tutorial.ipynb',
    #'demos/demos_databases_apis/nodexl/official/nodexl_graphistry.ipynb',
    #'demos/demos_databases_apis/neo4j/official/graphistry_bolt_tutorial_public.ipynb',
    #'demos/demos_databases_apis/neo4j/contributed/Neo4jTwitter.ipynb',
    #'demos/demos_databases_apis/alienvault/OTXLockerGoga.ipynb',
    #'demos/demos_databases_apis/alienvault/usm.ipynb',
    #'demos/demos_databases_apis/alienvault/OTXIndicators.ipynb',
    #'demos/demos_databases_apis/gremlin-tinkerpop/TitanDemo.ipynb',
    #'demos/demos_databases_apis/hypernetx/hypernetx.ipynb',
    'demos/demos_databases_apis/umap_learn/umap_learn.ipynb',
    #'demos/demos_databases_apis/graphviz/graphviz.ipynb',
    #'demos/demos_databases_apis/tigergraph/social_raw_REST_calls.ipynb',
    #'demos/demos_databases_apis/tigergraph/tigergraph_pygraphistry_bindings.ipynb',
    #'demos/demos_databases_apis/tigergraph/fraud_raw_REST_calls.ipynb',
    #'demos/demos_databases_apis/networkx/networkx.ipynb',
    'demos/more_examples/simple/tutorial_csv_mini_app_icij_implants.ipynb',
    'demos/more_examples/simple/MarvelTutorial.ipynb',
    'demos/more_examples/simple/tutorial_basic_LesMiserablesCSV.ipynb',
    #'demos/more_examples/graphistry_features/layout_tree.ipynb',
    #'demos/more_examples/graphistry_features/encodings-icons.ipynb',
    #'demos/more_examples/graphistry_features/layout_time_ring.ipynb',
    'demos/more_examples/graphistry_features/hop_and_chain_graph_pattern_mining.ipynb',
    #'demos/more_examples/graphistry_features/encodings-colors.ipynb',
    #'demos/more_examples/graphistry_features/encodings-sizes.ipynb',
    #'demos/more_examples/graphistry_features/layout_modularity_weighted.ipynb',
    #'demos/more_examples/graphistry_features/layout_time_ring_dev.ipynb',
    #'demos/more_examples/graphistry_features/external_layout/simple_manual_layout.ipynb',
    #'demos/more_examples/graphistry_features/external_layout/networkx_layout.ipynb',
    #'demos/more_examples/graphistry_features/embed/simple-ssh-logs-rgcn-anomaly-detector.ipynb',
    #'demos/more_examples/graphistry_features/sharing_tutorial.ipynb',
    #'demos/more_examples/graphistry_features/encodings-badges.ipynb',
    #'demos/more_examples/graphistry_features/layout_categorical_ring.ipynb',
    #'demos/more_examples/graphistry_features/edge-weights.ipynb',
    #'demos/more_examples/graphistry_features/layout_continuous_ring.ipynb',
    'demos/more_examples/graphistry_features/Workbooks.ipynb',
    'demos/demos_by_use_case/bio/BiogridDemo.ipynb',
    'demos/demos_by_use_case/logs/Tutorial Part 1 (Honey Pot).ipynb',
    'demos/demos_by_use_case/logs/malware-hypergraph/Malware Hypergraph.ipynb',
    'demos/demos_by_use_case/logs/aws_vpc_flow_cloudwatch/vpc_flow.ipynb',
    'demos/demos_by_use_case/logs/Tutorial Part 2 (Apache Logs).ipynb',
    'demos/demos_by_use_case/logs/network-threat-hunting-masterclass-zeek-bro/graphistry_corelight_webinar.ipynb',
    'demos/demos_by_use_case/logs/owasp-amass-network-enumeration/amass.ipynb',
    'demos/demos_by_use_case/logs/microservices-spigo/SystemArchitectureSpigo.ipynb',
    'demos/demos_by_use_case/fraud/BitcoinTutorial.ipynb',
    'demos/demos_by_use_case/social/Twitter.ipynb',
    #'demos/talks/infosec_jupyterthon2022/rgcn_login_anomaly_detection/advanced-identity-protection-40m.ipynb',
    #'demos/talks/infosec_jupyterthon2022/rgcn_login_anomaly_detection/intro-story.ipynb',
    #'demos/gfql/benchmark_hops_cpu_gpu.ipynb',
    'demos/data/benchmarking/SparseDatasets.ipynb',
    'demos/data/benchmarking/DenseDatasets.ipynb',
    'demos/data/benchmarking/TestDatasets.ipynb',
    'demos/ai/Introduction/Ask-HackerNews-Demo.ipynb',
    'demos/ai/Introduction/simple-power-of-umap.ipynb',
    #'demos/ai/cyber/CyberSecurity-Slim.ipynb',
    'demos/ai/cyber/redteam-umap-gtc-gpu.ipynb',
    'demos/ai/cyber/cyber-redteam-umap-demo.ipynb',
    'demos/ai/OSINT/jack-donations.ipynb',
    'demos/ai/OSINT/Chavismo.ipynb',

]

pygments_style = "sphinx"
todo_include_todos = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = "sphinx_rtd_theme"
html_theme = "sphinx_book_theme"


html_theme_options = {
    "repository_url": "https://github.com/graphistry/pygraphistry",
    "use_repository_button": True,

    # Optional top horizontal navigation bar
    #"navbar_start": ["navbar-start.html"],
    #"navbar_center": ["navbar-center.html"],
    #"navbar_end": ["navbar-end.html"],
    
    "logo": {
        #"link": "https://www.graphistry.com/get-started",
        #"text": "Graphistry, Inc.",
        "alt_text": "Graphistry, Inc."
    }
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['static']  # '_static'
# html_css_files = ['graphistry.css']

html_show_sphinx = False

htmlhelp_basename = "PyGraphistrydoc"


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    'preamble': r'''

        \usepackage{svg}   % Enables SVG handling via Inkscape

        \RequirePackage{etex}         % Ensure extended TeX capacities
        \usepackage[utf8]{inputenc}   % Enable UTF-8 support
        \usepackage[T1]{fontenc}      % Use T1 font encoding for better character support
        \usepackage{lmodern}          % Load Latin Modern fonts for better quality
        \usepackage{amsmath}           % For advanced math formatting
        \usepackage{amsfonts}          % For math fonts
        \usepackage{amssymb}           % For additional math symbols
        \usepackage{graphicx}          % For including images
        \usepackage{hyperref}          % For hyperlinks
        \usepackage{textcomp}          % For additional text symbols
        \usepackage{breakurl}          % Allows line breaks in URLs
        \usepackage{listings}          % For code listings
        \usepackage{float}             % Improved control of floating objects
        \usepackage{microtype}         % Improves text appearance with microtypography
        \usepackage{lipsum}            % For generating dummy text (if needed)


        % Increase capacity limits
        \setcounter{totalnumber}{200}   % Maximum floats
        \setcounter{dbltopnumber}{200}   % Double float maximum
        \setcounter{secnumdepth}{3}      % Section numbering depth
        \setcounter{tocdepth}{3}          % Table of contents depth
        
        % Increase dimensions and allocations
        \usepackage{morefloats}          % Allows for more floats
        \setlength{\emergencystretch}{3em} % Help with overfull hboxes
        \setlength{\maxdepth}{100pt}       % Sets a high limit for max depth (if applicable)

        % Allocate more memory for TeX
        \usepackage{etex}                % Use eTeX for more memory
        %\reserveinserts{200}             % Reserve more inserts
        \setcounter{totalnumber}{200}    % Ensure maximum floats are increased


        % Declare Unicode characters
        \DeclareUnicodeCharacter{1F389}{\textbf{(party popper)}}
        \DeclareUnicodeCharacter{1F3C6}{\textbf{(trophy)}}
        \DeclareUnicodeCharacter{1F44D}{\textbf{(thumbs up)}}
        \DeclareUnicodeCharacter{1F4AA}{\textbf{Strong}}  % Muscle emoji
        \DeclareUnicodeCharacter{1F4B0}{\textbf{Money Bag}} % Money bag emoji (text representation)
        \DeclareUnicodeCharacter{1F525}{\textbf{(fire)}}
        \DeclareUnicodeCharacter{1F600}{\textbf{(grinning)}}
        \DeclareUnicodeCharacter{1F609}{\textbf{(winking)}}
        \DeclareUnicodeCharacter{1F614}{\textbf{(pensive)}}
        \DeclareUnicodeCharacter{1F680}{\textbf{(rocket)}}
        \DeclareUnicodeCharacter{2501}{\textbf{━}}         % Heavy horizontal line
        \DeclareUnicodeCharacter{2588}{\textbf{█}}         % Full block character
        \DeclareUnicodeCharacter{258A}{\textbf{▊}}         % Center right block character
        \DeclareUnicodeCharacter{258B}{\textbf{▉}}         % Right block character
        \DeclareUnicodeCharacter{258C}{\textbf{▌}}         % Center block character
        \DeclareUnicodeCharacter{258D}{\textbf{▍}}         % Center left block character
        \DeclareUnicodeCharacter{258E}{\textbf{▎}}         % Left third block character
        \DeclareUnicodeCharacter{258F}{\textbf{▏}}         % Right block character
        \DeclareUnicodeCharacter{2728}{\textbf{(sparkles)}}
        \DeclareUnicodeCharacter{2764}{\textbf{(heart)}}
        \DeclareUnicodeCharacter{2B50}{\textbf{(star)}}

    ''',
}

# Use pdflatex as the LaTeX engine
latex_engine = 'pdflatex'

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        root_doc,
        "PyGraphistry.tex",
        u"PyGraphistry Documentation",
        u"Graphistry, Inc.",
        "manual",
    ),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
# latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
# latex_use_parts = False

# If true, show page references after internal links.
# latex_show_pagerefs = False

# If true, show URL addresses after external links.
# latex_show_urls = False

# Documents to append as an appendix to all manuals.
# latex_appendices = []

# If false, no module index is generated.
latex_domain_indices = False


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(root_doc, "pygraphistry", u"PyGraphistry Documentation", [author], 1)]

# If true, show URL addresses after external links.
# man_show_urls = False


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        root_doc,
        "PyGraphistry",
        u"PyGraphistry Documentation",
        author,
        "PyGraphistry",
        "One line description of project.",
        "Miscellaneous",
    ),
]

# Documents to append as an appendix to all manuals.
# texinfo_appendices = []

# If false, no module index is generated.
texinfo_domain_indices = False

# How to display URL addresses: 'footnote', 'no', or 'inline'.
# texinfo_show_urls = 'footnote'

# If true, do not generate a @detailmenu in the "Top" node's menu.
# texinfo_no_detailmenu = False


# -- Options for Epub output ----------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright

# The basename for the epub file. It defaults to the project name.
# epub_basename = project

# The HTML theme for the epub output. Since the default themes are not optimized
# for small screen space, using the same theme for HTML and epub output is
# usually not wise. This defaults to 'epub', a theme designed to save visual
# space.
# epub_theme = 'epub'

# The language of the text. It defaults to the language option
# or 'en' if the language is not set.
# epub_language = ''

# The scheme of the identifier. Typical schemes are ISBN or URL.
# epub_scheme = ''

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
# epub_identifier = ''

# A unique identification for the text.
# epub_uid = ''

# A tuple containing the cover image and cover page html template filenames.
# epub_cover = ()

# A sequence of (type, uri, title) tuples for the guide element of content.opf.
# epub_guide = ()

# HTML files that should be inserted before the pages created by sphinx.
# The format is a list of tuples containing the path and title.
# epub_pre_files = []

# HTML files shat should be inserted after the pages created by sphinx.
# The format is a list of tuples containing the path and title.
# epub_post_files = []

# A list of files that should not be packed into the epub file.
epub_exclude_files = ["search.html"]

# The depth of the table of contents in toc.ncx.
# epub_tocdepth = 3

# Allow duplicate toc entries.
# epub_tocdup = True

# Choose between 'default' and 'includehidden'.
# epub_tocscope = 'default'

# Fix unsupported image types using the Pillow.
# epub_fix_images = False

# Scale large images.
# epub_max_image_width = 0

# How to display URL addresses: 'footnote', 'no', or 'inline'.
# epub_show_urls = 'inline'

# If false, no index is generated.
# epub_use_index = True


# Example configuration for intersphinx: refer to the Python standard library.
# intersphinx_mapping = {'https://docs.python.org/': None}



# -- Custom Preprocessor Configuration ---------------------------------------

def replace_iframe_src(app, doctree, docname):
    """
    Replace relative iframe src paths with absolute URLs in HTML content.
    Specifically targets iframe tags with src attributes starting with /graph/.
    """
    # Define a flexible regex pattern to match <iframe> tags with src="/graph/..."
    # This pattern accounts for single or double quotes and any additional attributes
    pattern = re.compile(
        r'(<iframe[^>]*src\s*=\s*[\'"])(/graph/[^\'"]+)([\'"][^>]*>)', 
        re.IGNORECASE | re.DOTALL
    )

    # Flag to check if any replacement occurred in this document
    replacement_occurred = False

    # Iterate over all nodes in the doctree
    for node in doctree.traverse():
        # Process only nodes that can contain text
        if isinstance(node, (docutils.nodes.raw, docutils.nodes.literal_block, docutils.nodes.paragraph, docutils.nodes.Text)):
            # Determine the content based on node type
            if isinstance(node, docutils.nodes.raw):
                node_format = getattr(node, 'format', '').lower()
                if node_format != 'html':
                    continue  # Skip non-HTML raw nodes
                original_content = node.rawsource
                # Perform the regex substitution
                updated_content, count = pattern.subn(r'\1https://hub.graphistry.com\2\3', original_content)
                if count > 0:
                    node.rawsource = updated_content
                    logger.info(f"Updated {count} iframe src in document: {docname}")
                    replacement_occurred = True
            elif isinstance(node, docutils.nodes.literal_block) or isinstance(node, docutils.nodes.paragraph):
                original_content = node.astext()
                # Perform the regex substitution
                updated_content, count = pattern.subn(r'\1https://hub.graphistry.com\2\3', original_content)
                if count > 0:
                    # Replace the node's text with updated content
                    new_nodes = docutils.nodes.inline(text=updated_content)
                    node.parent.replace(node, new_nodes)
                    logger.info(f"Updated {count} iframe src in document: {docname}")
                    replacement_occurred = True
            elif isinstance(node, docutils.nodes.Text):
                original_text = node.astext()
                # Perform the regex substitution
                updated_text, count = pattern.subn(r'\1https://hub.graphistry.com\2\3', original_text)
                if count > 0:
                    # Replace the text node with updated text
                    new_text_node = docutils.nodes.Text(updated_text)
                    node.parent.replace(node, new_text_node)
                    logger.info(f"Updated {count} iframe src in document: {docname}")
                    replacement_occurred = True

    if not replacement_occurred:
        logger.debug(f"No iframe src replacements made in document: {docname}")


def ignore_svg_images_for_latex(app, doctree, docname):
    """Remove SVG images from the LaTeX build."""
    if app.builder.name == 'latex':
        for node in doctree.traverse(nodes.image):
            if node['uri'].endswith('.svg'):
                node.parent.remove(node)

def remove_external_images_for_latex(app, doctree, fromdocname):
    """Remove external images and handle external links in LaTeX and EPUB builds."""
    if app.builder.name in ['latex', 'epub']:  # Extend to all builds if needed
        logger.info(f"Processing doctree for output: {fromdocname}")
        
        # Handle problematic external images
        for node in doctree.traverse(nodes.image):
            image_uri = node['uri']
            logger.debug(f"Processing image URI: {image_uri}")
            if "://" in image_uri:  # Identify external images
                logger.debug(f"Detected external image URI: {image_uri}")
                try:
                    if node.parent:
                        # Preserve node attributes such as "classes"
                        parent = node.parent
                        classes = node.get('classes', [])
                        logger.debug(f"Preserving classes attribute: {classes}")
                        parent.remove(node)  # Remove external image node
                        logger.info(f"Successfully removed external image: {image_uri}")
                    else:
                        logger.error(f"No parent found for image: {image_uri}")
                except Exception as e:
                    logger.error(f"Failed to remove external image: {image_uri} with error {str(e)}")
            else:
                logger.debug(f"Retained local image: {image_uri}")
        
        # Handle problematic external links
        for node in doctree.traverse(nodes.reference):
            if node.get('refuri', '').startswith('http'):
                logger.debug(f"Handling external link: {node['refuri']}")
                if node['refuri'].endswith('.com'):
                    logger.warning(f"Found problematic URL ending in .com: {node['refuri']}")
                    # Preserve "classes" attribute and replace link
                    classes = node.get('classes', [])
                    logger.debug(f"Preserving classes attribute: {classes}")
                    inline_node = nodes.inline('', f"{node['refuri']} (external link)", classes=classes)
                    node.replace_self(inline_node)
                else:
                    # Keep non-problematic URLs
                    inline_node = nodes.inline('', node['refuri'], classes=node.get('classes', []))
                    node.replace_self(inline_node)

        logger.info("Finished processing images and links.")

def assert_external_images_removed(app, doctree, fromdocname):
    """Assert that external images have been removed."""
    if app.builder.name in ['html']:  # Extend to all builds if needed
        return

    for node in doctree.traverse(nodes.image):
        image_uri = node['uri']
        if "://" in image_uri:
            logger.error(f"Assertion failed: external image was not removed: {image_uri}")
        assert "://" not in image_uri, f"Failed to remove external image: {image_uri}"


def setup(app: Sphinx):
    """
    Connect the replace_iframe_src function to the doctree-resolved event.
    """    
    
    app.connect("doctree-resolved", ignore_svg_images_for_latex)
    app.connect("doctree-resolved", remove_external_images_for_latex)
    app.connect('doctree-resolved', replace_iframe_src)
    app.connect("doctree-resolved", assert_external_images_removed)

    def on_builder(app: Sphinx) -> None:
        if not hasattr(app, 'builder'):
            print('No app.builder found for app type=', type(app))
            # use dir to enumerate field names & types
            attr_and_types: str = '\n'.join([f'{name}: {type(getattr(app, name))}' for name in dir(app)])
            print(f'attr_and_types:\n---\n{attr_and_types}\n---\n')
            return

        if (app.builder.name == "html" or app.builder.name == "readthedocs"):
            app.add_css_file('graphistry.css', priority=900)
            app.add_js_file("https://plausible.io/js/script.hash.outbound-links.js", **{
                "defer": "true",
                "data-domain": "pygraphistry.readthedocs.io",
            })
            app.add_js_file(None, body="""
                window.plausible = window.plausible || function() {
                    (window.plausible.q = window.plausible.q || []).push(arguments)
                }
            """)
            return
        
        print('No custom handling for app.builder.name=', app.builder.name)

    app.connect('builder-inited', on_builder)
