from typing import List
import pytest
import numpy

from models.models_utils import test_data
from models.maskrank.maskrank_model import MaskRank

doc_one, doc_two = test_data()

def test_empty_doc():
    """ Test empty doc """

    model = MaskRank("paraphrase-MiniLM-L6-v2", "en_core_web_sm")

    with pytest.raises(ValueError):
        empty_extract = model.extract_kp_from_doc("", 5, 0, None)

    assert isinstance(model, MaskRank)


@pytest.mark.parametrize("top_n", [1, 2, 3, 4, 5])
def test_extract_kp_single_doc(top_n):
    """ Test extraction of single document method """

    model = MaskRank("paraphrase-MiniLM-L6-v2", "en_core_web_sm")
    kp = model.extract_kp_from_doc(doc_one, top_n, 0)

    assert isinstance(model, MaskRank)
    
    assert isinstance(kp, tuple)
    assert isinstance(kp[0], list)
    assert isinstance(kp[0][0], tuple)
    assert isinstance(kp[0][0][0], str)
    assert isinstance(kp[0][0][1], numpy.floating)
    assert isinstance(kp[1], list)
    assert isinstance(kp[1][0], str)
    assert len(kp[0]) == top_n


@pytest.mark.parametrize("top_n", [1, 2, 3, 4, 5])
def test_extract_kp_corpus(top_n):
    """ Test extraction of two documents """

    model = MaskRank("paraphrase-MiniLM-L6-v2", "en_core_web_sm")
    kp = model.extract_kp_from_corpus([[doc_one], [doc_two]], "FR-WIKI", top_n)

    assert isinstance(model, MaskRank)
    
    assert isinstance(kp, list)

    for i in range(2):
        assert isinstance(kp[i], tuple)
        assert isinstance(kp[i][0], list)
        assert isinstance(kp[i][0][0], tuple)
        assert isinstance(kp[i][0][0][0], str)
        assert isinstance(kp[i][0][0][1], numpy.floating)
        assert isinstance(kp[i][1], list)
        assert isinstance(kp[i][1][0], str)
        assert len(kp[i][0]) == top_n