import numpy as np
import pytest
from vitnns.endpoint_m1_statistics import metrics,random_metrics,bootstrap_median,sign_test,holm

def test_known_rank_capture_regret_and_ties():
    a=metrics([4,3,2,1],[1,2,3,4])
    assert a['spearman']==pytest.approx(-1)
    assert a['capture_at_1']==.25 and a['capture_at_3']==pytest.approx(6/9)
    assert a['regret_at_1']==3 and a['pairwise_accuracy']==0
    assert metrics([1,1,1,1],[1,2,3,4])['pairwise_accuracy']==.5
    assert metrics([1,1,1,1],[1,2,3,4])['spearman'] is None
    assert metrics([1,2,3],[1,2,3])['spearman'] is None
    assert metrics([1,2,3,4],[0,0,0,0])['capture_at_1'] is None

def test_random_uniform_distribution_and_repeatability():
    means,a=random_metrics([1,2,3,4],'fixture')
    assert a==random_metrics([1,2,3,4],'fixture')[1]
    assert abs(means['spearman'])<.08
    assert means['capture_at_1']==pytest.approx(2.5/4,abs=.04)
    assert sum(a['distribution']['spearman_histogram_20bins'])==1000

def test_exact_image_bootstrap_order_statistic_law():
    # Independent ordinary image-index bootstrap checks the optimized law.
    values=np.random.default_rng(13).normal(size=400)
    rng=np.random.default_rng(51)
    ordinary=np.median(values[rng.integers(0,400,size=(30000,400))],axis=1)
    ci=bootstrap_median(values,'fixture',30000)
    assert ci==pytest.approx(np.quantile(ordinary,[.025,.975]),abs=.03)
    assert bootstrap_median([3]*400,'constant')==[3,3]

def test_one_sided_sign_and_holm():
    assert sign_test([1]*10)['p']==1/1024
    assert sign_test([0]*10)['p']==1
    rows=holm([dict(test_id='a',p=.01),dict(test_id='b',p=.04),dict(test_id='c',p=.03)])
    assert [r['holm_p'] for r in rows]==pytest.approx([.03,.06,.06])
