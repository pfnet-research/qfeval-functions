import torch

import qfeval_functions.functions as QF


def test_nanones() -> None:
    x = torch.tensor([[1.0, float('nan')], [float('nan'), 0.0]])
    actual = QF.nanones(x)
    expected = torch.tensor([[1.0, float('nan')], [float('nan'), 1.0]])
    assert torch.all(torch.isnan(actual) == torch.isnan(expected))
    assert torch.all(torch.nan_to_num(actual) == torch.nan_to_num(expected))
