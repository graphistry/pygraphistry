
from typing import Any, List, Set, Union, TYPE_CHECKING
from typing_extensions import Literal

if TYPE_CHECKING:
    try:
        from pygraphviz import AGraph
    except:
        AGraph: Any = None  # type: ignore
else:
    AGraph: Any = None


Prog = Literal[
    "acyclic",
    "ccomps",
    "circo",
    "dot",
    "fdp",
    "gc",
    "gvcolor",
    "gvpr",
    "neato",
    "nop",
    "osage",
    "patchwork",
    "sccmap",
    "sfdp",
    "tred",
    "twopi",
    "unflatten",
]
PROGS: List[Prog] = [
    "acyclic",
    "ccomps",
    "circo",
    "dot",
    "fdp",
    "gc",
    "gvcolor",
    "gvpr",
    "neato",
    "nop",
    "osage",
    "patchwork",
    "sccmap",
    "sfdp",
    "tred",
    "twopi",
    "unflatten",
]

Format = Literal[
    "canon",
    "cmap",
    "cmapx",
    "cmapx_np",
    "dia",
    "dot",
    "fig",
    "gd",
    "gd2",
    "gif",
    "hpgl",
    "imap",
    "imap_np",
    "ismap",
    "jpe",
    "jpeg",
    "jpg",
    "mif",
    "mp",
    "pcl",
    "pdf",
    "pic",
    "plain",
    "plain-ext",
    "png",
    "ps",
    "ps2",
    "svg",
    "svgz",
    "vml",
    "vmlz",
    "vrml",
    "vtx",
    "wbmp",
    "xdot",
    "xlib"
]

FORMATS: List[Format] = [
    "canon",
    "cmap",
    "cmapx",
    "cmapx_np",
    "dia",
    "dot",
    "fig",
    "gd",
    "gd2",
    "gif",
    "hpgl",
    "imap",
    "imap_np",
    "ismap",
    "jpe",
    "jpeg",
    "jpg",
    "mif",
    "mp",
    "pcl",
    "pdf",
    "pic",
    "plain",
    "plain-ext",
    "png",
    "ps",
    "ps2",
    "svg",
    "svgz",
    "vml",
    "vmlz",
    "vrml",
    "vtx",
    "wbmp",
    "xdot",
    "xlib"
]

GraphAttr = Literal[
    "_background", "bb", "beautify", "bgcolor",
    "center", "charset", "class", "clusterrank", "colorscheme", "comment", "compound", "concentrate",
    "Damping", "defaultdist", "dim", "dimen", "diredgeconstraints", "dpi",
    "epsilon", "esep",
    "fontcolor", "fontname", "fontnames", "fontpath", "fontsize", "forcelabels",
    "gradientangle", "href", "id", "imagepath", "inputscale",
    "K",
    "label", "label_scheme", "labeljust", "labelloc", "landscape", "layerlistsep",
    "layers", "layerselect", "layersep", "layout", "levels", "levelsgap", "lheight", "linelength", "lp", "lwidth",
    "margin", "maxiter", "mclimit", "mindist", "mode", "model",
    "newrank", "nodesep", "nojustify", "normalize", "notranslate", "nslimit", "nslimit1",
    "oneblock", "ordering", "orientation", "outputorder", "overlap", "overlap_scaling", "overlap_shrink",
    "pack", "packmode", "pad", "page", "pagedir", "quadtree", "quantum",
    "rankdir", "ranksep", "ratio", "remincross", "repulsiveforce", "resolution", "root", "rotate", "rotation",
    "scale", "searchsize", "sep", "showboxes", "size", "smoothing", "sortv", "splines", "start", "style", "stylesheet",
    "target", "TBbalance", "tooltip", "truecolor", "URL", "viewport", "voro_margin", "xdotversion"
]

GRAPH_ATTRS: List[GraphAttr] = [
    "_background", "bb", "beautify", "bgcolor",
    "center", "charset", "class", "clusterrank", "colorscheme", "comment", "compound", "concentrate",
    "Damping", "defaultdist", "dim", "dimen", "diredgeconstraints", "dpi",
    "epsilon", "esep",
    "fontcolor", "fontname", "fontnames", "fontpath", "fontsize", "forcelabels",
    "gradientangle", "href", "id", "imagepath", "inputscale",
    "K",
    "label", "label_scheme", "labeljust", "labelloc", "landscape", "layerlistsep",
    "layers", "layerselect", "layersep", "layout", "levels", "levelsgap", "lheight", "linelength", "lp", "lwidth",
    "margin", "maxiter", "mclimit", "mindist", "mode", "model",
    "newrank", "nodesep", "nojustify", "normalize", "notranslate", "nslimit", "nslimit1",
    "oneblock", "ordering", "orientation", "outputorder", "overlap", "overlap_scaling", "overlap_shrink",
    "pack", "packmode", "pad", "page", "pagedir", "quadtree", "quantum",
    "rankdir", "ranksep", "ratio", "remincross", "repulsiveforce", "resolution", "root", "rotate", "rotation",
    "scale", "searchsize", "sep", "showboxes", "size", "smoothing", "sortv", "splines", "start", "style", "stylesheet",
    "target", "TBbalance", "tooltip", "truecolor", "URL", "viewport", "voro_margin", "xdotversion"
]

# https://graphviz.org/docs/nodes/
NodeAttr = Literal[
    "area", "class", "color", "colorscheme", "comment", "distortion",
    "fillcolor", "fixedsize", "fontcolor", "fontname", "fontsize",
    "gradientangle", "group", "height", "href", "id", "image", "imagepos", "imagescale",
    "label", "labelloc", "layer", "margin", "nojustify", "ordering", "orientation",
    "penwidth", "peripheries", "pin", "pos", "rects", "regular", "root",
    "samplepoints", "shape", "shapefile", "showboxes", "sides", "skew", "sortv", "style",
    "target", "tooltip", "URL", "vertices", "width", "xlabel", "xlp", "z"
]
NODE_ATTRS: List[NodeAttr] = [
    "area", "class", "color", "colorscheme", "comment", "distortion",
    "fillcolor", "fixedsize", "fontcolor", "fontname", "fontsize",
    "gradientangle", "group", "height", "href", "id", "image", "imagepos", "imagescale",
    "label", "labelloc", "layer", "margin", "nojustify", "ordering", "orientation",
    "penwidth", "peripheries", "pin", "pos", "rects", "regular", "root",
    "samplepoints", "shape", "shapefile", "showboxes", "sides", "skew", "sortv", "style",
    "target", "tooltip", "URL", "vertices", "width", "xlabel", "xlp", "z"
]

EdgeAttr = Literal[
    "arrowhead", "arrowsize", "arrowtail",
    "class", "color", "colorscheme", "comment", "constraint",
    "decorate", "dir", "edgehref", "edgetarget", "edgetooltip", "edgeURL",
    "fillcolor", "fontcolor", "fontname", "fontsize",
    "head_lp", "headclip", "headhref", "headlabel", "headport", "headtarget", "headtooltip", "headURL", "href",
    "id", "label", "labelangle", "labeldistance", "labelfloat", "labelfontcolor",
    "labelfontname", "labelfontsize", "labelhref", "labeltarget", "labeltooltip",
    "labelURL", "layer", "len", "lhead", "lp", "ltail", "minlen", "nojustify",
    "penwidth", "pos", "samehead", "sametail", "showboxes", "style",
    "tail_lp", "tailclip", "tailhref", "taillabel", "tailport", "tailtarget",
    "tailtooltip", "tailURL", "target", "tooltip",
    "URL", "weight", "xlabel", "xlp"
]
EDGE_ATTRS: List[EdgeAttr] = [
    "arrowhead", "arrowsize", "arrowtail",
    "class", "color", "colorscheme", "comment", "constraint",
    "decorate", "dir", "edgehref", "edgetarget", "edgetooltip", "edgeURL",
    "fillcolor", "fontcolor", "fontname", "fontsize",
    "head_lp", "headclip", "headhref", "headlabel", "headport", "headtarget", "headtooltip", "headURL", "href",
    "id", "label", "labelangle", "labeldistance", "labelfloat", "labelfontcolor",
    "labelfontname", "labelfontsize", "labelhref", "labeltarget", "labeltooltip",
    "labelURL", "layer", "len", "lhead", "lp", "ltail", "minlen", "nojustify",
    "penwidth", "pos", "samehead", "sametail", "showboxes", "style",
    "tail_lp", "tailclip", "tailhref", "taillabel", "tailport", "tailtarget",
    "tailtooltip", "tailURL", "target", "tooltip",
    "URL", "weight", "xlabel", "xlp"
]

UNSANITARY_ATTRS: Set[Union[GraphAttr, EdgeAttr, NodeAttr]] = {
    'fontpath',
    'image',
    'imagepath',
    'shapefile',
    'stylesheet'
}
