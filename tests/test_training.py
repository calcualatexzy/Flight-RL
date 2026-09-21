import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from FlightEnv.env import FlightEnv
from main import create_env, FORMAT_VERSION, TrainingCheckpoint, load_model, parse_args, save_model


@pytest.mark.parametrize('profile', ['legacy', 'tracking', 'residual'])
def test_ppo_updates_checkpoint_roundtrip_and_resume(tmp_path, tracking_dataset, profile):
    torch.set_num_threads(1)
    config = dict(mode='hover', is_domain_randomization=False, profile=profile)
    if profile in ('tracking', 'residual'):
        config['dataset_dir'] = tracking_dataset
    env = DummyVecEnv([lambda: create_env(config), lambda: create_env(config)])
    try:
        model = PPO('MlpPolicy', env, n_steps=32, batch_size=32, n_epochs=1,
                    seed=7, device='cpu', policy_kwargs=dict(log_std_init=-1.5))
        model.flight_format_version = FORMAT_VERSION
        model.flight_env_config = config
        before = {k: v.clone() for k, v in model.policy.state_dict().items()}
        callback = TrainingCheckpoint(32, tmp_path)
        model.learn(128, callback=callback)
        assert any(not torch.equal(before[k], v) for k, v in model.policy.state_dict().items())
        assert (tmp_path / 'ppo_64_steps.zip').is_file()
        assert (tmp_path / 'ppo_128_steps.zip').is_file()
        obs = env.reset()
        expected = model.predict(obs, deterministic=True)[0]
        save_model(model, tmp_path / 'final_model.zip')
        restored = load_model(tmp_path / 'final_model.zip', 'cpu')
        np.testing.assert_array_equal(expected, restored.predict(obs, deterministic=True)[0])
        assert restored.num_timesteps == 128
        original_optimizer = model.policy.optimizer.state_dict()['state']
        restored_optimizer = restored.policy.optimizer.state_dict()['state']
        assert original_optimizer.keys() == restored_optimizer.keys()
        for key in original_optimizer:
            torch.testing.assert_close(original_optimizer[key]['exp_avg'], restored_optimizer[key]['exp_avg'])
        # Reloading with env= supports changing the number of parallel workers.
        other_env = DummyVecEnv([lambda: create_env(config)])
        try:
            resumed = load_model(tmp_path / 'final_model.zip', 'cpu', env=other_env)
            resumed.learn(64, reset_num_timesteps=False)
            assert resumed.num_timesteps == 192
            assert resumed.n_envs == 1
            assert resumed.flight_env_config == config
        finally:
            other_env.close()
        assert not list(tmp_path.glob('*.tmp.zip'))
    finally:
        env.close()


def test_cli_validation():
    assert parse_args(['--train']).command == 'train'
    with pytest.raises(SystemExit):
        parse_args(['train', '--num-envs', '1', '--n-steps', '63', '--batch-size', '32'])


def test_legacy_checkpoint_is_rejected_before_unpickling(tmp_path):
    import json
    from zipfile import ZipFile
    checkpoint = tmp_path / 'legacy.zip'
    with ZipFile(checkpoint, 'w') as archive:
        archive.writestr('data', json.dumps({'policy_class': 'old python pickle'}))
    with pytest.raises(ValueError, match='legacy checkpoint'):
        load_model(checkpoint, 'cpu')


def test_training_rejects_held_out_splits():
    for split in ('validation', 'test'):
        with pytest.raises(SystemExit):
            parse_args(['train', '--split', split])
