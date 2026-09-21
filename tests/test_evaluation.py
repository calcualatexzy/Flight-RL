from FlightEnv.evaluation import TrackingEvaluation
from FlightEnv.tracking_env import TrackingEnv


def test_validation_is_fixed_stratified_and_has_no_duplicate_paths(tracking_dataset, tmp_path):
    config = dict(trajectory_family='mixed', dataset_dir=tracking_dataset)
    with TrackingEnv(dataset_dir=tracking_dataset, dataset_split='validation') as env:
        callback = TrackingEvaluation(env, config, tmp_path, 100, 32, 10042, None)
        same = TrackingEvaluation(env, config, tmp_path, 100, 32, 10042, None)
        assert callback.options == same.options
        ids = [option['trajectory_id'] for option in callback.options]
        assert len(ids) == len(set(ids)) == 32
        from FlightEnv.trajectory_dataset import load_bank
        bank = load_bank(tracking_dataset, 'validation')
        assert list(bank['families'][ids]) == list(callback.families)*4
