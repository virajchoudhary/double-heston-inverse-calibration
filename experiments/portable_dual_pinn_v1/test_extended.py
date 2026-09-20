import numpy as np
from run_extended import stratified_label_indices


def test_finetuning_covers_every_case_without_duplicates():
    labels=np.repeat(np.arange(256),96)
    indices=stratified_label_indices(labels).numpy()
    assert len(indices)==len(set(indices))==512
    values,counts=np.unique(labels[indices],return_counts=True)
    np.testing.assert_array_equal(values,np.arange(256))
    assert (counts==2).all()
