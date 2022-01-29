import pytest

from utils.IO import *
from datasets.process_datasets import *

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

dataset_expected_sizes = { "DUC"      : {"total" : 308,  "test" : 308 },
                           "Inspec"   : {"total" : 2000, "test" : 500 , "dev" : 500, "train" : 1000},
                           "NUS"      : {"total" : 211,  "test" : 211},
                           "SemEval"   : {"total" : 243, "test" : 243}, 
                           "PubMed"   : {"total" : 1320, "test" : 1320}, 
                           "PT-KP"    : {"total" : 110,  "test" : 10, "train" : 100},
                           "ES-CACIC" : {"total" : 888,  "test" : 888},
                           "ES-WICC"  : {"total" : 1640, "test" : 1640},
                           "FR-WIKI"  : {"total" : 100,  "test" : 100},
                           "PL-PAK"  : {"total" : 50,  "test" : 50}}

@pytest.mark.parametrize("dataset", [ ["DUC", "Inspec", "NUS", "SemEval", "PubMed", "PT-KP", "ES-CACIC", "ES-WICC", "FR-WIKI", "PL-PAK"] ])
def test_dataset_simple(dataset):
    """ Test invoking DataSet extraction with dataset parameters """

    dataset_obj = DataSet(dataset)
    con = dataset_obj.dataset_content

    assert isinstance(dataset_obj, DataSet)
    assert isinstance(con, dict)
    assert con != {}

    for dataset in con:
        assert isinstance(con[dataset], list)
        assert len(con[dataset]) == dataset_expected_sizes[dataset]["total"]

        for doc in con[dataset]:
            assert isinstance(doc, tuple)      
            assert isinstance(doc[0], str)
            assert isinstance(doc[1], list)

            for keyword in doc[1]:
                assert isinstance(keyword, str)