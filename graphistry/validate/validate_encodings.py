import re
from typing import List, Optional

blend_modes = [
    'normal',
    'color',
    'color-burn',
    'color-dodge',
    'darken',
    'difference',
    'hue',
    'exclusion',
    'overlay',
    'lighten',
    'luminosity',
    'multiply',
    'saturation',
    'screen'
]


##TODO probably some sort of json validator would work here

#non-null, required .bindings
def validate_encodings_generic(encodings, kind, required_bindings):

    if encodings is None:
        raise ValueError({'message': f'Field {kind}_encodings cannot be empty'})

    if not ('bindings' in encodings) or (encodings['bindings'] is None):
        raise ValueError({'message': f'Field {kind}_encodings.bindings cannot be empty'})

    if not isinstance(encodings['bindings'], dict):
        raise ValueError({
            'message': f'Field {kind}_encodings.bindings must be a dictionary',
            'data': {'bindings': encodings['bindings'], 'type': str(type(encodings['bindings']))}})

    required = ['bindings']
    missing = [ k for k in required if not (k in encodings) ]
    if len(missing) > 0:
        raise ValueError({'message': f'Missing fields for {kind}_encodings', 'data': { 'missing': missing } })

    missing = [ k for k in required_bindings if not (k in encodings['bindings']) ]
    if len(missing) > 0:
        raise ValueError({'message': f'Missing fields for {kind}_encodings.bindings', 'data': { 'missing': missing } })


def validate_style(base_path, enc):
    if not isinstance(enc, dict):
        raise ValueError({
            'message': f'Field {base_path} should be an object',
            'data': {'style': enc, 'type': str(type(enc))}})
    styles = ['opacity', 'grayscale', 'brightness', 'contrast', 'hueRotate', 'saturate']
    safelist = styles
    unsafe = [ k for k in enc.keys() if not (k in safelist) ]
    if len(unsafe) > 0:
        raise ValueError({
            'message': f'Unexpected fields in {base_path}',
            'data': { 'unsafe': unsafe } })
    non_num = [ k for k in enc.keys() if not isinstance(enc[k], int) and not isinstance(enc[k], float) ]
    if len(non_num) > 0:
        raise ValueError({
            'message': f'Non-numeric fields in {base_path}',
            'data': { 'invalid_fields': non_num } })
    return { **enc }

def validate_complex_encoding_icon(kind, mode, name, enc):
    out = {}
    if 'asText' in enc:
        if not isinstance(enc['asText'], bool):
            raise ValueError({
                'message': f'Field {kind}_encodings.complex.{mode}.{name}.asText should be a boolean',
                'data': { 'asText': enc['asText'], 'type': str(type(enc['asText'])) }})
        out['asText'] = enc['asText']
    if 'style' in enc:
        style = validate_style(f'{kind}_encodings.complex.{mode}.{name}.style', enc['style'])
        if len(style.keys()) > 0:
            out['style'] = style
    if 'blendMode' in enc:
        if not isinstance(enc['blendMode'], str) or not (enc['blendMode'] in blend_modes):
            raise ValueError({
                'message': f'Unexpected value for {kind}_encodings.complex.{mode}.{name}.blendMode',
                'data': {'value': enc['blendMode'], 'type': str(type(enc['blendMode']))}
            })
        out['blendMode'] = enc['blendMode']
    if 'border' in enc:
        if not isinstance(enc['border'], dict):
            raise ValueError({
                'message': f'Expected record value type for {kind}_encodings.complex.{mode}.{name}.border',
                'data': {'value': enc['border'], 'type': str(type(enc['border']))}
            })
        safelist = ['width', 'color', 'stroke']
        unsafe = [ k2 for k2 in enc['border'].keys() if not (k2 in safelist) ]
        if len(unsafe) > 0:
            raise ValueError({
                'message': f'Unexpected fields in {kind}_encodings.complex.{mode}.{name}.border',
                'data': { 'unsafe': unsafe } })
        border = {}
        if 'width' in enc['border']:
            if not (isinstance(enc['border']['width'], int) or isinstance(enc['border']['width'], float)):
                raise ValueError({
                    'message': f'Excepted integer/float for {kind}_encodings.complex.{mode}.{name}.border.width',
                    'data': { 'width': enc['border']['width'], 'type': str(type(enc['border']['width'])) } })
            border['width'] = enc['border']['width']
        for fld in ['color', 'stroke']:
            if fld in enc['border']:
                if not (isinstance(enc['border'][fld], str)):
                    raise ValueError({
                        'message': f'Excepted string for {kind}_encodings.complex.{mode}.{name}.border.{fld}',
                        'data': { 'color': enc['border'][fld], 'type': str(type(enc['border'][fld])) } })
                border[fld] = enc['border'][fld]
        if len(border.keys()) > 0:
            out['border'] = border
    if 'shape' in enc:
        safelist = ['circle', 'none']
        if not isinstance(enc['shape'], str) or not (enc['shape'] in safelist):
            raise ValueError({
                'message': f'Unexpected value for optional field {kind}_encodings.complex.{mode}.{name}.shape',
                'data': { 'shape': enc['shape'], 'safelist': safelist } })
        out['shape'] = enc['shape']
    return out

def cascade_encoding(base_encoding, encoding):
    return {
        'attribute': encoding['attribute'] if 'attribute' in encoding else base_encoding['attribute'], 
        'variation': encoding['variation'] if 'variation' in encoding else base_encoding['variation']
    }

def validate_complex_encoding_color(base_path, kind, mode, name, enc):
    out = {}

    if not isinstance(enc, dict):
        raise ValueError({
            'message': f'Field {base_path} should be a color encoding dict',
            'data': { 'color': enc, 'type': str(type(enc)) } })

    required = ['attribute', 'variation']
    missing = [ k for k in required if not (k in enc) ]
    if len(missing) > 0:
        raise ValueError({
            'message': f'Missing fields for {base_path}',
            'data': { 'missing': missing } })

    if not isinstance(enc['attribute'], str):
        raise ValueError({
            'message': f'Field {base_path}.attribute should be a string',
            'data': { 'attribute': enc['attribute'], 'type': str(type(enc['attribute'])) }})
    out['attribute'] = enc['attribute']

    if not enc['variation'] in ['categorical', 'continuous']:
        raise ValueError({
            'message': f'Field {base_path}.variation should be "categorical" or "continuous"',
            'data': { 'variation': enc['attribute'] }})
    out['variation'] = enc['variation']    

    if 'colors' in enc:
        raise ValueError({
            'message': f'Field "colors" not supported for encoding {base_path}, using mapping instead',
            'data': { 'value': enc['colors'] } })
    else:
        if not ('mapping' in enc):
            raise ValueError({
                'message': f'Field "mapping" missing from encoding {base_path}',
                'data': { 'value': enc } })
    
        out['mapping'] = validate_mapping(enc['mapping'], f'{base_path}.mapping')

    return out



def validate_complex_encoding_badge(kind, mode, name, badge):
    out = {}

    #asText, style, blendMode, border, shape
    icon = validate_complex_encoding_icon(kind, mode, name, badge)
    for k in icon.keys():
        out[k] = icon[k]
    if 'mapping' in badge:
        out['mapping'] = validate_mapping(badge['mapping'], f'{kind}_encodings.complex.{mode}.{name}')
    else:
        raise ValueError({
            'message': f'Missing required field {kind}_encodings.complex.{mode}.{name}.mapping',
            'data': { 'badge': badge } })

    if 'color' in badge:
        if isinstance(badge['color'], str):
            out['color'] = badge['color']
        elif isinstance(badge['color'], dict):
            out['color'] = validate_complex_encoding_color(
                f'{kind}_encodings.complex.{mode}.{name}.color',
                kind,
                mode,
                name,
                {**cascade_encoding(badge, badge['color']), **badge['color']})
        else:
            raise ValueError({
                'message': f'Expecting string or encoding object for {kind}_encodings.complex.{mode}.{name}.color',
                'data': { 'color': badge['color'], 'type': str(type(badge['color'])) } })
    if 'bg' in badge:
        if not isinstance(badge['bg'], dict):
            raise ValueError({
                'message': f'Field {kind}_encodings.complex.{mode}.{name}.bg. must be a dictionary',
                'data': { 'bg': badge['bg'], 'type': str(type(badge['bg'])) } })
        safelist = ['color', 'image', 'style']
        unsafe = [ k for k in badge['bg'].keys() if k not in safelist ]
        if len(unsafe) > 0:
            raise ValueError({
                'message': f'Unexpected keys in field {kind}_encodings.complex.{mode}.{name}.bg', 
                'data': { 'unsafe': unsafe } })
        bg = {}
        if 'color' in badge['bg']:
            if isinstance(badge['bg']['color'], str):
                bg['color'] = badge['bg']['color']
            elif isinstance(badge['bg']['color'], dict):
                bg['color'] = validate_complex_encoding_color(
                    f'{kind}_encodings.complex.{mode}.{name}.bg.color',
                    kind,
                    mode,
                    name,
                    {**cascade_encoding(badge, badge['bg']['color']), **badge['bg']['color']})
                
            else:
                raise ValueError({
                    'message': f'Expecting string or encoding object for {kind}_encodings.complex.{mode}.{name}.bg.color',
                    'data': { 'color': badge['bg']['color'], 'type': str(type(badge['bg']['color'])) } })
        if 'image' in badge['bg']:
            if isinstance(badge['bg']['image'], str):
                bg['image'] = badge['bg']['image']
            elif isinstance(badge['bg']['image'], dict):
                bg['image'] = cascade_encoding(badge, badge['bg']['image'])
                bg['image']['mapping'] = validate_mapping(badge['bg']['image']['mapping'], f'{kind}_encodings.complex.{mode}.{name}.bg.image')
            else:
                raise ValueError({
                    'message': f'Excepted string or encoding object for {kind}_encodings.complex.{mode}.{name}.bg.image', 
                    'data': { 'image': badge['bg']['image'], 'type': str(type(badge['bg']['image'])) } })
        if 'style' in badge['bg']:
            style = validate_style(f'{kind}_encodings.complex.{mode}.{name}.bg.style', badge['bg']['style'])
            if len(style.keys()) > 0:
                bg['style'] = style
        if len(bg.keys()) > 0:
            out['bg'] = bg
    if 'fg' in badge:
        if not isinstance(badge['fg'], dict):
            raise ValueError({
                'message': f'Field {kind}_encodings.complex.{mode}.{name}.fg must be a dictionary',
                'data': { 'fg': badge['fg'], 'type': str(type(badge['fg'])) } })
        fg = {}
        if 'style' in badge['fg']:
            style = validate_style(f'{kind}_encodings.complex.{mode}.{name}.fg.style', badge['fg']['style'])
            if len(style.keys()) > 0:
                fg['style'] = style
        if len(fg.keys()) > 0:
            out['fg'] = fg
    if 'border' in badge:
        if not isinstance(badge['border'], dict):
            raise ValueError({
                'message': f'Field {kind}_encodings.complex.{mode}.{name}.border must be a dictionary',
                'data': { 'border': badge['border'], 'type': str(type(badge['border'])) } })
        safelist = ['color', 'stroke', 'width']
        unsafe = [ k for k in badge['border'].keys() if k not in safelist ]
        if len(unsafe) > 0:
            raise ValueError({
                'message': f'Unexpected keys in field {kind}_encodings.complex.{mode}.{name}.border', 
                'data': { 'unsafe': unsafe } })
        border = {}
        if 'color' in badge['border']:
            if not isinstance(badge['border']['color'], str):
                raise ValueError({
                    'message': f'Field {kind}_encodings.complex.{mode}.{name}.border.color must be a string',
                    'data': { 'border': badge['border']['color'], 'type': str(type(badge['border']['color'])) } })
            border['color'] = badge['border']['color']
        if 'stroke' in badge['border']:
            strokes = ['dotted', 'dashed', 'solid', 'double', 'groove', 'ridge', 'inset', 'outset', 'none']
            if not badge['border']['stroke'] in strokes:
                raise ValueError({
                    'message': f'Field {kind}_encodings.complex.{mode}.{name}.border.stroke not a recognized value',
                    'data': { 'stroke': badge['border']['stroke'], 'allowed': strokes }})
            border['stroke'] = badge['border']['stroke']
        if 'width' in badge['border']:
            if not isinstance(badge['border']['width'], int):
                raise ValueError({
                    'message': f'Field {kind}_encodings.complex.{mode}.{name}.border.width must be an int',
                    'data': { 'width': badge['border']['width'], 'type': str(type(badge['border']['width'])) } })
            border['width'] = badge['border']['width']
        if len(border.keys()) > 0:
            out['border'] = border

    if 'dimensions' in badge:
        if not isinstance(badge['dimensions'], dict):
            raise ValueError({
                'message': f'Field {kind}_encodings.complex.{mode}.{name}.dimensions must be a dictionary',
                'data': { 'dimensions': badge['dimensions'], 'type': str(type(badge['dimensions'])) } })
        dimensions = {}
        for fld in ['maxHeight', 'maxWidth']:
            if fld in badge['dimensions']:
                if not isinstance(badge['dimensions'][fld], int):
                    raise ValueError({
                        'message': f'Field {kind}_encodings.complex.{mode}.{name}.dimensions.{fld} must be an int',
                        'data': { fld: badge['dimensions'][fld], 'type': str(type(badge['dimensions'][fld])) } })
                dimensions[fld] = badge['dimensions']['fld']
        if len(dimensions.keys()) > 0:
            out['dimensions'] = dimensions

    return out


# ('point' | 'edge') * ('current' | 'default') * str * json -> () raises ValueError({'message': str, ?'data': json})
# where json is:
# { 
# }
# NOTE: auto-adds graphType, encodingType based on name
def validate_complex_encoding(kind, mode, name, enc, attributes: Optional[List] = None):


    name_match = re.match(r'^(point|edge)(Legend(Type|Pivot))?(Axis|Badge|Color|Icon|Opacity|Size|Weight)(Top|TopLeft|Left|BottomLeft|Bottom|BottomRight|Right|TopRight|Cover)?Encoding$', name)
    if not name_match:
        raise ValueError({
            'message': f'Fields of {kind}_encodings.complex.{mode}.* must have name of the form (point|edge)(Legend(Type|Pivot))?(Color|Icon|Size|Weight|Badge(Top|TopLeft...|Cover))Encoding',
            'data': {'field': name}})

    n_kind, n_legend, n_legend_field, n_enc, n_badge_pos = name_match.groups()

    if n_kind != ('point' if kind == 'node' else kind):
        raise ValueError({
            'message': f'Fields of {kind}_encodings.complex.{mode}.* must begin with {kind}*',
            'data': {'field': name}})

    if ('graphType' in enc) and (enc['graphType'] != n_kind):
        raise ValueError({
            'message': f'Field {kind}_encodings.complex.{mode}.{name}.graphType must be {n_kind}*',
            'data': {'graphType': enc['graphType']}})

    enc_type_lower = n_enc.lower()
    if 'encodingType' in enc:
        if enc_type_lower == 'badge':
            if enc['encodingType'] != f'{enc_type_lower}{n_badge_pos}':
                raise ValueError({
                    'message': f'Field {kind}_encodings.complex.{mode}.{name}.encodingType must be {enc_type_lower}{n_badge_pos}',
                    'data': {'encodingType': enc['encodingType']}})
        elif n_legend is not None:
            if enc['encodingType'] != f'{n_legend.lower()}{enc_type_lower}':
                raise ValueError({
                    'message': f'Field {kind}_encodings.complex.{mode}.{name}.encodingType must be {n_legend.lower()}{enc_type_lower}',
                    'data': {'encodingType': enc['encodingType']}})
        else:
            if enc['encodingType'] != enc_type_lower:
                raise ValueError({
                    'message': f'Field {kind}_encodings.complex.{mode}.{name}.encodingType must be {enc_type_lower}',
                    'data': {'encodingType': enc['encodingType']}})

    out = {'graphType': n_kind, 'encodingType': enc['encodingType']}

    required = ['attribute', 'variation']
    missing = [ k for k in required if not (k in enc) ]
    if len(missing) > 0:
        raise ValueError({
            'message': f'Missing fields for {kind}_encodings.complex.{mode}.{name}.*',
            'data': { 'missing': missing } })

    if not isinstance(enc['attribute'], str):
        raise ValueError({
            'message': f'Field {kind}_encodings.complex.{mode}.{name}.attribute should be a string',
            'data': { 'attribute': enc['attribute'], 'type': str(type(enc['attribute'])) }})
    out['attribute'] = enc['attribute']

    if not enc['variation'] in ['categorical', 'continuous']:
        raise ValueError({
            'message': f'Field {kind}_encodings.complex.{mode}.{name}.variation should be "categorical" or "continuous"',
            'data': { 'variation': enc['attribute'] }})
    out['variation'] = enc['variation']
    

    should_validate_mapping = False
    # Type-directed checks
    if n_enc == 'Color':
        out['name'] = enc['name'] if 'name' in enc else 'custom'
        if 'colors' in enc:
            if not isinstance(enc['colors'], list):
                raise ValueError({
                    'message': f'Field {kind}_encodings.complex.{mode}.{name}.colors should be an array',
                    'data': { 'colors': enc['colors'] }})
            for color in enc['colors']:
                if not isinstance(color, str):
                    raise ValueError({
                        'message': f'Field {kind}_encodings.complex.{mode}.{name}.colors[*] should be strings like "#FFFFFF" and "white"',
                        'data': { 'color': color, 'type': str(type(color)) }})
            out['colors'] = enc['colors']
        else:
            should_validate_mapping = True
    elif n_enc == 'Icon':
        icon_encoding = validate_complex_encoding_icon(kind, mode, name, enc)
        for k in icon_encoding.keys():
            out[k] = icon_encoding[k]
    elif n_enc == 'Badge':
        badge = validate_complex_encoding_badge(kind, mode, name, enc)
        for k in badge.keys():
            out[k] = badge[k]
    elif n_enc == 'Axis':
        #FIXME SECURITY remove **enc passthrough, currently in b/c unclear axis/color encodings
        out = {**enc, **out}
        if 'mapping' in enc:
            should_validate_mapping = True
    else:
        should_validate_mapping = True


    if ('mapping' in enc) or should_validate_mapping:
        if not ('mapping' in enc):
            raise ValueError({
                'message': f'Field {kind}_encodings.complex.{mode}.{name} expects field "mapping" (or some other encoding mode)',
                'data': { 'fields': enc.keys() }})
        
        out['mapping'] = validate_mapping(enc['mapping'], f'{kind}_encodings.complex.{mode}.{name}')

    if attributes and name != 'pointAxisEncoding':  # 'degree' won't be part of the node attributes
        attr = out['attribute']
        if attr not in attributes:
            raise ValueError({
                'message': f'Field {kind}_encodings.complex.{mode}.{name}.attribute does not exist in {kind}.attributes',
                'data': {'attribute': attr, f'{kind}.attributes': str(attributes)}})

    return out


def validate_mapping(mapping, base_path):
    out = None
    if mapping is None or not isinstance(mapping, dict):
        raise ValueError({
            'message': f'Field {base_path}.mapping should be a dict',
            'data': {'mapping': mapping, 'type': 'None' if mapping is None else str(type(mapping))}})

    if 'categorical' in mapping:
        if not isinstance(mapping['categorical'], dict):
            raise ValueError({
                'message': f'Field {base_path}.mapping.categorical must be a dictionary',
                'data': {'mapping': mapping, 'type:mapping.categorical': str(type(mapping['categorical'] if 'categorical' in mapping else None)) }})
        cat = mapping['categorical']
        if not ('fixed' in cat) or not isinstance(cat['fixed'], dict):
            raise ValueError({
                'message': f'Field {base_path}.mapping.cateogrical missing required dictionary field "fixed"',
                'data': {'categorical': cat, 'type:mapping.categorical.fixed': str(type(cat['fixed'] if 'fixed' in cat else None)) }})
        out = {'categorical': {'fixed': cat['fixed'] }}
        if 'other' in cat:
            out['categorical']['other'] = cat['other']
    elif 'continuous' in mapping:
        if not isinstance(mapping['continuous'], dict):
            raise ValueError({
                'message': f'Field {base_path}.mapping.continuous must be a',
                'data': {'mapping': mapping, 'type:mapping.continuous': str(type(mapping['continuous'] if 'continuous' in mapping else None)) }})
        keys = [f for f in ['bins'] if f in mapping['continuous'].keys()]
        if len(keys) != 1:
            raise ValueError({
                'message': f'Value {base_path}.mapping.continuous must have field "bins" (array of [comparable, value] pairs)',
                'data': {'found': keys, 'num_found': len(keys)}})
        cat = mapping['continuous']
        out = {'continuous': {keys[0]: cat[keys[0]] }}
        if 'comparator' in cat:
            if not cat['comparator'] in ['<', '<=']:
                raise ValueError({
                    'message': f'Value {base_path}.mapping.continuous.comparator must be one of "<", "<="',
                    'data': {'comparator': cat['comparator'] }})
            out['continuous']['comparator'] = cat['comparator']
        if 'other' in cat:
            out['continuous']['other'] = cat['other']
    else:
        raise ValueError({
            'message': f'{base_path} mapping requires field categorical or continuous',
            'data': {'fields': mapping.keys()}})
    return out

#{?'complex': {?'current', ?'default}} => ?{?'current', ?'default'}
def validate_complex(encodings, kind, attributes: Optional[List] = None):
    if not ('complex' in encodings):
        return None

    c = encodings['complex']

    if (c is None) or (not isinstance(c, dict)):
        raise ValueError({
            'message': f'Field {kind}_encodings.bindings must be a dictionary',
            'data': {'bindings': encodings['bindings'], 'type': str(type(encodings['bindings']))}})

    safelist = ['default', 'current']
    unsafe = [ k for k in c.keys() if k not in safelist ]
    if len(unsafe) > 0:
        raise ValueError({'message': f'Unexpected keys in field {kind}_encodings.complex', 'data': { 'unsafe': unsafe } })

    out = {}  # type: ignore
    for c_type in safelist:
        if c_type in c:
            out[c_type] = {}
            for enc_name, enc_val in c[c_type].items():
                out[c_type][enc_name] = validate_complex_encoding(kind, c_type, enc_name, enc_val, attributes)
    return out


def validate_node_encodings(encodings, node_attributes: Optional[List] = None):
    validate_encodings_generic(encodings, 'node', required_bindings=[])
    complex_encodings = validate_complex(encodings, 'node', node_attributes)
    return encodings if complex_encodings is None else {**encodings, 'complex': complex_encodings}

def validate_edge_encodings(encodings, edge_attributes: Optional[List] = None):
    validate_encodings_generic(encodings, 'edge', required_bindings=['source', 'destination'])
    complex_encodings = validate_complex(encodings, 'edge', edge_attributes)
    return encodings if complex_encodings is None else {**encodings, 'complex': complex_encodings}


#json * json * -> {'node_encodings': json, 'edge_encodings': dict} raises ValueError({'message': str, ?'data': json})
def validate_encodings(node_encodings: dict, edge_encodings: dict, node_attributes: Optional[List] = None, edge_attributes: Optional[List] = None) -> dict:
    """
    Validate node and edge encodings for compatibility with the given attributes.

    This function processes and validates the `node_encodings` and `edge_encodings` against the 
    provided node and edge attributes, ensuring they follow the expected format. If any encoding 
    is invalid, a `ValueError` is raised with details. It is a subset of what the server checks, and 
    run by the uploader.
    
    :param node_encodings: Encodings for the nodes in the graph.
    :type node_encodings: dict
    :param edge_encodings: Encodings for the edges in the graph.
    :type edge_encodings: dict
    :param node_attributes: List of node attributes to validate encodings against.
    :type node_attributes: Optional[List]
    :param edge_attributes: List of edge attributes to validate encodings against.
    :type edge_attributes: Optional[List]
    :return: A dictionary containing the validated encodings for nodes and edges, in the form:

    Example:
        node_encodings = {'color': 'blue', 'size': 5}
        edge_encodings = {'weight': 0.2}
        result = validate_encodings(node_encodings, edge_encodings)
        # {'node_encodings': {'color': 'blue', 'size': 5}, 'edge_encodings': {'weight': 0.2}}
    """
    node_encodings2 = validate_node_encodings(node_encodings, node_attributes)
    edge_encodings2 = validate_edge_encodings(edge_encodings, edge_attributes)
    return {'node_encodings': node_encodings2, 'edge_encodings': edge_encodings2}
