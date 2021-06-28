import pytest
import collections.abc
from dataset_utils import *
from process_datasets import *

def test_empty_none_dataset():
    """ Test invoking DataSet extraction with None parameter """
    dataset_obj = None

    try:
        dataset_obj = DataSet([None])
    except ValueError:
        assert True

    dataset_obj = DataSet([])
    assert isinstance(dataset_obj, DataSet)
    assert isinstance(dataset_obj.dataset_content, dict)
    assert dataset_obj.dataset_content == {}

dataset_expected_sizes = { "DUC" : {"total" : 308, "test" : 308 },
                           "Inspec" : {"total" : 2000, "test" : 500 , "dev" : 500, "train" : 1000},
                           "NUS" : {"total" : 211, "test" : 211},
                           "PT-KP" : {"total" : 110, "test" : 10, "train" : 100},
                           "PubMed" : {"total" : 1320, "test" : 1320} }

@pytest.mark.parametrize("dataset", [ ["DUC", "Inspec", "NUS", "PT-KP", "PubMed"] ])
@pytest.mark.parametrize("unsupervised", [True, True, True, True, True])
def test_dataset_simple(dataset, unsupervised):
    """ Test invoking DataSet extraction with dataset parameters """

    dataset_obj = DataSet(dataset, unsupervised)
    con = dataset_obj.dataset_content

    assert isinstance(dataset_obj, DataSet)
    assert isinstance(con, dict)
    assert con != {}

    if unsupervised:
        for dataset in con:
            assert isinstance(con[dataset], list)
            assert len(con[dataset]) == dataset_expected_sizes[dataset]["total"]

            for doc in con[dataset]:
                assert isinstance(doc, tuple)      
                assert isinstance(doc[0], str)
                assert isinstance(doc[1], list)

                for keyword in doc[1]:
                    assert isinstance(keyword, str)