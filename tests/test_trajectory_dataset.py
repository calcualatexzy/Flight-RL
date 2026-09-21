import numpy as np
import pytest

from FlightEnv.trajectory_dataset import FAMILIES, generate_path, sample_bank


@pytest.mark.parametrize('family', FAMILIES)
def test_analytic_trajectories_are_smooth_bounded_and_reproducible(family):
    position, velocity, acceleration, meta = generate_path(family, 321, 'test')
    repeated = generate_path(family, 321, 'test')
    np.testing.assert_array_equal(position, repeated[0])
    assert position.shape == velocity.shape == acceleration.shape == (501, 3)
    assert np.isfinite(position).all() and np.isfinite(acceleration).all()
    assert position[:, 2].min() >= .8 - 1e-9
    assert np.linalg.norm(velocity, axis=1).max() <= 4 + 1e-9
    assert np.linalg.norm(acceleration, axis=1).max() <= 6 + 1e-9
    assert acceleration[:, 2].min() >= -3 - 1e-9
    np.testing.assert_allclose(velocity[[0, -1]], 0, atol=1e-10)
    np.testing.assert_allclose(acceleration[[0, -1]], 0, atol=1e-10)
    numerical_velocity = np.gradient(position, .02, axis=0, edge_order=2)
    numerical_acceleration = np.gradient(velocity, .02, axis=0, edge_order=2)
    np.testing.assert_allclose(numerical_velocity[2:-2], velocity[2:-2], atol=.004)
    np.testing.assert_allclose(numerical_acceleration[2:-2], acceleration[2:-2], atol=.015)
    assert not np.array_equal(position, generate_path(family, 321, 'train')[0])


def test_difficulty_scales_amplitude_and_derivatives_consistently(tracking_dataset):
    a = sample_bank(np.random.default_rng(12), 10., .02, family='figure8', difficulty=1., split='test', directory=tracking_dataset)
    b = sample_bank(np.random.default_rng(12), 10., .02, family='figure8', difficulty=0., split='test', directory=tracking_dataset)
    assert a[3]['trajectory_id'] == b[3]['trajectory_id']
    np.testing.assert_allclose(b[0] - b[0][0], .15*(a[0]-a[0][0]), atol=1e-7)
    np.testing.assert_allclose(b[1], .15*a[1], atol=1e-7)
    np.testing.assert_allclose(b[2], .15*a[2], atol=1e-7)


def test_exhaustive_evaluation_has_no_duplicates_and_covers_families(tracking_dataset):
    from FlightEnv.trajectory_dataset import FLIGHT_FAMILIES, evaluation_options, load_bank
    options = evaluation_options(FLIGHT_FAMILIES, directory=tracking_dataset)
    ids = [option['trajectory_id'] for option in options]
    assert len(ids) == len(set(ids)) == 4*len(FLIGHT_FAMILIES)
    bank = load_bank(tracking_dataset, 'test')
    assert list(bank['families'][ids]) == list(FLIGHT_FAMILIES)*4
