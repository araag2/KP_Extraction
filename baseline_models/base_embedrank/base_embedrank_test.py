from typing import List
import pytest
import numpy

from baseline_models.base_embedrank.base_embedrank_utils import get_test_data
from baseline_models.base_embedrank.base_embedrank_model import BaseEmbedRank

doc_one, doc_two = get_test_data()

def test_empty_doc():
    """ Test empty doc """

    model = BaseEmbedRank("xlm-r-bert-base-nli-stsb-mean-tokens")

    with pytest.raises(ValueError):
        empty_extract = model.extract_kp_from_doc("", 5, 0, True)

    assert isinstance(model, BaseEmbedRank)


@pytest.mark.parametrize("top_n", [1, 2, 3, 4, 5])
def test_extract_kp_single_doc(top_n):
    """ Test extraction of single document method """

    model = BaseEmbedRank("xlm-r-bert-base-nli-stsb-mean-tokens")
    kp = model.extract_kp_from_doc(doc_one, top_n, 0, True)

    assert isinstance(model, BaseEmbedRank)
    
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

    model = BaseEmbedRank("xlm-r-bert-base-nli-stsb-mean-tokens")
    kp = model.extract_kp_from_corpus([[doc_one], [doc_two]], top_n)

    assert isinstance(model, BaseEmbedRank)
    
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