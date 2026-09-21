import pytest

from FlightEnv.trajectory_dataset import export_dataset


@pytest.fixture(scope='session')
def tracking_dataset(tmp_path_factory):
    """Tests generate their own small bank and never require local training data."""
    root = tmp_path_factory.mktemp('trajectories')
    export_dataset(root, train_per_family=4, eval_per_family=4, seed=101)
    return str(root)
