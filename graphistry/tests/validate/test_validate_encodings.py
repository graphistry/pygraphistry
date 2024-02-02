import copy, os, pandas as pd, pytest

from graphistry.validate.validate_encodings import validate_encodings

def test_validate_encodings_big_good():

    orig = {
        "node_encodings": {
            "bindings": {"node": "n"},
            "complex": {
                "default": {
                    "pointColorEncoding": {
                        "graphType": "point",
                        "encodingType": "color",
                        "attribute": "type",
                        "variation": "categorical",
                        "mapping": {
                            "categorical": {
                                "fixed": {
                                    "x": "purple",
                                    "y": "#444"                                
                                },
                                "other": "white"
                            }
                        }
                    },
                    "pointIconEncoding": {
                        "graphType": "point",
                        "encodingType": "icon",
                        "attribute": "type",
                        "variation": "categorical",
                        "mapping": {
                            "categorical": {
                                "fixed": {
                                    "x": "laptop",
                                    "y": "user"                                
                                },
                                "other": ""
                            }
                        }
                    }
                }
            }
        },
        "edge_encodings": {
            "bindings": {"source": "s", "destination": "d"},
            "complex": {
                "default": {
                    "edgeColorEncoding": {
                        "graphType": "edge",
                        "encodingType": "color",
                        "attribute": "time",
                        "variation": "continuous",
                        "name": "zzz",
                        "colors": ["blue", "yellow", "red"]
                    }
                }
            }
        }
    }

    out = validate_encodings(orig['node_encodings'], orig['edge_encodings'])
    orig['node_encodings']['complex']['default']['pointColorEncoding']['name'] = 'custom'
    assert orig['node_encodings']['complex']['default']['pointColorEncoding'] \
        == out['node_encodings']['complex']['default']['pointColorEncoding']
    assert orig['node_encodings']['complex']['default']['pointIconEncoding'] \
        == out['node_encodings']['complex']['default']['pointIconEncoding']
    assert orig['edge_encodings']['complex']['default'] == out['edge_encodings']['complex']['default']
    assert out == orig




def test_validate_encodings_mt_good():

    orig = {
        "node_encodings": { "bindings": {"node": "n" } },
        "edge_encodings": { "bindings": {"source": "s", "destination": "d"} }
    }

    assert validate_encodings(orig['node_encodings'], orig['edge_encodings']) == orig

    orig['node_encodings']['complex'] = {}
    orig['edge_encodings']['complex'] = {}
    assert validate_encodings(orig['node_encodings'], orig['edge_encodings']) == orig



def test_validate_encodings_bad():

    #edge mode in point
    orig = {
        "node_encodings": {
            "bindings": {"node": "n"},
            "complex": {
                "default": {
                    "edgeColorEncoding": {
                        "graphType": "edge",
                        "encodingType": "color",
                        "attribute": "time",
                        "variation": "continuous",
                        "name": "zzz",
                        "colors": ["blue", "yellow", "red"]
                    }
                }
            }
        },
        "edge_encodings": {
            "bindings": {"source": "s", "destination": "d"}
        }
    }

    with pytest.raises(ValueError):
        validate_encodings(orig['node_encodings'], orig['edge_encodings'])


    #wrong complex key
    orig = {
        "node_encodings": {
            "bindings": {"node": "n"},
            "complex": {
                "zzz": {
                    "pointColorEncoding": {
                        "graphType": "point",
                        "encodingType": "color",
                        "attribute": "time",
                        "variation": "continuous",
                        "name": "zzz",
                        "colors": ["blue", "yellow", "red"]
                    }
                }
            }
        },
        "edge_encodings": {
            "bindings": {"source": "s", "destination": "d"}
        }
    }

    with pytest.raises(ValueError):
        validate_encodings(orig['node_encodings'], orig['edge_encodings'])





def test_validate_encodings_icons_good():

    orig = {
        "node_encodings": {
            "bindings": {"node": "n"},
            "complex": {
                "default": {
                    "pointIconEncoding": {
                        "graphType": "point",
                        "encodingType": "icon",
                        "attribute": "type",
                        "variation": "categorical",
                        "mapping": {
                            "categorical": {
                                "fixed": {
                                    "x": "laptop",
                                    "y": "user"                                
                                },
                                "other": ""
                            }
                        }
                    }
                }
            }
        },
        "edge_encodings": {
            "bindings": {"source": "s", "destination": "d"},
            "complex": {
                "current": {
                    "edgeIconEncoding": {
                        "graphType": "edge",
                        "encodingType": "icon",
                        "attribute": "type",
                        "variation": "continuous",
                        "shape": "circle",
                        "mapping": {
                            "continuous": {
                                "bins": [
                                    [100, "server"],
                                    [None, "laptop"]
                                ],
                                'comparator': "<=",
                                "other": ""
                            }
                        },
                        "asText": True,
                        "style": { "opacity": 0.1 },
                        "blendMode": "color-burn",
                        "border": { "width": 3, "color": "green", "stroke": "dotted" }
                    }
                },
                "default": {
                    "edgeIconEncoding": {
                        "graphType": "edge",
                        "encodingType": "icon",
                        "attribute": "type",
                        "variation": "continuous",
                        "mapping": {
                            "continuous": {
                                "bins": [
                                    [100, "server"],
                                    [None, "laptop"]
                                ],
                                "other": ""
                            }
                        },
                        "asText": True,
                        "style": { "opacity": 0.5 },
                        "blendMode": "color-dodge",
                        "border": { "width": 2, "color": "red", "stroke": "dashed" }
                    }
                }
            }
        }
    }

    assert validate_encodings(orig['node_encodings'], orig['edge_encodings']) == orig


def test_validate_encodings_icons_bad():

    #invalid asText
    orig = {
        "node_encodings": {
            "bindings": {"node": "n"},
            "complex": {
                "default": {

                    "pointIconEncoding": {
                        "graphType": "point",
                        "encodingType": "icon",
                        "attribute": "type",
                        "variation": "categorical",
                        "mapping": {
                            "categorical": {
                                "fixed": {
                                    "x": "laptop",
                                    "y": "user"                                
                                },
                                "other": ""
                            }
                        }
                    },
                    "asText": {"wat": 2}
                }
            }
        },
        "edge_encodings": {
            "bindings": {"source": "s", "destination": "d"}
        }
    }

    with pytest.raises(ValueError):
        validate_encodings(orig['node_encodings'], orig['edge_encodings'])



def test_validate_encodings_badges_good():

    orig = {
        "node_encodings": {
            "bindings": {"node": "n"},
            "complex": {
                "default": {
                    "pointBadgeRightEncoding": {
                        "graphType": "point",
                        "encodingType": "badgeRight",
                        "attribute": "type",
                        "variation": "categorical",
                        "mapping": {
                            "categorical": {
                                "fixed": {
                                    "x": "laptop",
                                    "y": "user"                                
                                },
                                "other": ""
                            }
                        },
                        "color": "red",
                        "bg": { "color": "green" },
                        "border": { "width": 2, "color": "black", "stroke": "dotted"}
                    }
                }
            }
        },
        "edge_encodings": {
            "bindings": {"source": "s", "destination": "d"},
            "complex": {
                "current": {
                    "edgeBadgeTopLeftEncoding": {
                        "graphType": "edge",
                        "encodingType": "badgeTopLeft",
                        "attribute": "type",
                        "variation": "continuous",
                        "shape": "circle",
                        "mapping": {"continuous": { "bins": [[None, "circle"], [None, "circle"]] }},
                        "color": {
                            "mapping": {
                                "continuous": {
                                    "bins": [
                                        [None, "laptop"],
                                        [100, "server"],
                                        [200, "server"],
                                        [None, "laptop"]
                                    ],
                                    'comparator': "<=",
                                    "other": ""
                                }
                            }
                        },
                        "bg": {
                            "image": {
                                "attribute": "country_code",
                                "variation": 'categorical',
                                "mapping": { "categorical": { "fixed": {'America': 'http://...' }}}
                            }

                        }
                    }
                },
                "default": {
                    "edgeBadgeCoverEncoding": {
                        "graphType": "edge",
                        "encodingType": "badgeCover",
                        "attribute": "type",
                        "variation": "categorical",
                        "mapping": {"categorical": { "fixed": {} }},
                        "asText": True,
                        "style": {"opacity": 0.5},
                        "blendMode": "color-burn"
                    }
                }
            }
        }
    }

    #cascade
    orig_with_attribs = copy.deepcopy(orig)
    orig_with_attribs['edge_encodings']['complex']['current']['edgeBadgeTopLeftEncoding']['color']['attribute'] = 'type'
    orig_with_attribs['edge_encodings']['complex']['current']['edgeBadgeTopLeftEncoding']['color']['variation'] = 'continuous'


    validated = validate_encodings(orig['node_encodings'], orig['edge_encodings'])
    assert validated['node_encodings']['complex']['default']['pointBadgeRightEncoding'] \
        == orig_with_attribs['node_encodings']['complex']['default']['pointBadgeRightEncoding']

    assert validated['edge_encodings']['complex']['current']['edgeBadgeTopLeftEncoding'] \
        == orig_with_attribs['edge_encodings']['complex']['current']['edgeBadgeTopLeftEncoding']

    assert validated['edge_encodings']['complex']['default']['edgeBadgeCoverEncoding'] \
        == orig_with_attribs['edge_encodings']['complex']['default']['edgeBadgeCoverEncoding']


def test_validate_encodings_with_attributes_bad():

    orig = {
        "node_encodings": {
            "bindings": {"node": "n"},
            "complex": {
                "default": {
                    "pointColorEncoding": {
                        "graphType": "point",
                        "encodingType": "color",
                        "attribute": "attribute_2",
                        "variation": "categorical",
                        "mapping": {
                            "categorical": {
                                "fixed": {
                                    "x": "purple",
                                    "y": "#444"                                
                                },
                                "other": "white"
                            }
                        }
                    }
                }
            }
        },
        "edge_encodings": {
            "bindings": {"source": "s", "destination": "d"},
            "complex": {
                "default": {
                    "edgeColorEncoding": {
                        "graphType": "edge",
                        "encodingType": "color",
                        "attribute": "attribute_4",
                        "variation": "continuous",
                        "name": "zzz",
                        "colors": ["blue", "yellow", "red"]
                    }
                }
            }
        }
    }

    # notice we don't have 'attribute_2' (for 'pointColorEncoding') nor 'attribute_4' (for 'edgeColorEncoding')
    attributes = ["attribute_1", "attribute_3"]

    #wrong node attributes
    with pytest.raises(ValueError):
        validate_encodings(orig['node_encodings'], orig['edge_encodings'], node_attributes = attributes)


    #wrong edge attributes
    with pytest.raises(ValueError):
        validate_encodings(orig['node_encodings'], orig['edge_encodings'], edge_attributes = attributes)


def test_validate_encodings_with_attributes_good():

    orig = {
        "node_encodings": {
            "bindings": {"node": "n"},
            "complex": {
                "default": {
                    "pointColorEncoding": {
                        "graphType": "point",
                        "encodingType": "color",
                        "attribute": "attribute_2",
                        "variation": "categorical",
                        "mapping": {
                            "categorical": {
                                "fixed": {
                                    "x": "purple",
                                    "y": "#444"                                
                                },
                                "other": "white"
                            }
                        }
                    }
                }
            }
        },
        "edge_encodings": {
            "bindings": {"source": "s", "destination": "d"},
            "complex": {
                "default": {
                    "edgeColorEncoding": {
                        "graphType": "edge",
                        "encodingType": "color",
                        "attribute": "attribute_4",
                        "variation": "continuous",
                        "name": "zzz",
                        "colors": ["blue", "yellow", "red"]
                    }
                }
            }
        }
    }

    # notice we have 'attribute_2' (for 'pointColorEncoding') and 'attribute_4' (for 'edgeColorEncoding')
    attributes = ["attribute_1", "attribute_2", "attribute_3", "attribute_4"]

    validate_encodings(orig['node_encodings'], orig['edge_encodings'], node_attributes = attributes, edge_attributes = attributes)
