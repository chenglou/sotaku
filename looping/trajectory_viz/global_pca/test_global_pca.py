import torch

from looping.trajectory_viz.global_pca.analyze_global_pca import fit_pca, project_global


def test_fit_pca_recovers_plane():
    values = torch.tensor([[1.0, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])
    _, _, explained = fit_pca(values, component_count=3)
    assert torch.isclose(explained[:2].sum(), torch.tensor(1.0), atol=1e-5)
    assert explained[2] < 1e-6


def test_global_projection_preserves_shape():
    states = torch.randn(4, 7, 12)
    projected, explained, heldout = project_global(states, fit_puzzle_count=2)
    assert projected.shape == (4, 7, 3)
    assert explained.shape == (3,)
    assert 0 <= heldout <= 1.00001
