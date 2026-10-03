from pathlib import Path
import pytest
from laya_bench.common import ROOT
from laya_bench.verify import verify

@pytest.mark.skipif(not (ROOT/'results/full/summary.json').exists(), reason='Run full benchmark to audit measured evidence')
def test_saved_full_run_is_complete_and_metrics_match_predictions():
    result=verify(ROOT/'results/full/summary.json')
    assert result['status']=='passed'
    assert result['primary_model_test_decisions']==14138

