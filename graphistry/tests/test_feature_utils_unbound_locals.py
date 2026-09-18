"""Pins the feature-pipeline paths that read a local an earlier branch may not have bound.

Two of these were reachable and raised UnboundLocalError on master; the other two are
hardening, reachable only if the validation upstream of them ever moves.
"""

import pandas as pd
import pytest

from graphistry.feature_utils import transform


def _res_slots():
    """The 10-slot featurization result `transform` unpacks; unused for an unknown kind."""
    return [None] * 8 + [None, []]


def test_transform_with_an_unrecognized_kind_returns_empty_frames():
    """`kind` is a plain str and FastEncoder does not validate it, so this is reachable.

    On master X was bound only inside the nodes/edges branches, so this raised
    UnboundLocalError at the first `X.empty`.
    """
    df = pd.DataFrame({"a": [1.0, 2.0]})
    X, y = transform(df, pd.DataFrame([]), _res_slots(), "bogus", None, None,
                     df.columns, pd.Index([]))
    assert X.empty
    assert y.empty


def test_a_recognized_kind_still_reaches_its_encoder():
    """Pre-binding X must not turn a real kind into a silent empty result.

    With a None encoder in the result slots, "nodes" must fail inside the encoder
    rather than fall through to the empty frame the unknown-kind case returns.
    """
    df = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(AttributeError, match="transform"):
        transform(df, pd.DataFrame([]), _res_slots(), "nodes", None, None,
                  df.columns, pd.Index([]))


def test_get_numeric_transformers_accepts_no_dataframe():
    """Returns `ndf_` unconditionally; with ndf=None nothing bound it on master."""
    pytest.importorskip("sklearn")
    from graphistry.feature_utils import get_numeric_transformers

    ndf_, y_, data_encoder, label_encoder = get_numeric_transformers(None, None)
    assert ndf_ is None
    assert y_ is None
    assert data_encoder is False
    assert label_encoder is False
