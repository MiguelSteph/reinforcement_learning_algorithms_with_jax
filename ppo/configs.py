import ml_collections


def get_configs():
    data_config = ml_collections.ConfigDict(
        dict(
            obs_shape = (84, 84, 4),
            n_actions = 4,
        )
    )

    log_config = ml_collections.ConfigDict(
        dict(
            log_metric_path = 'log_dir/metrics',
            log_checkpoint_path = 'log_dir/checkpoints',
        )
    )

    training_config = ml_collections.ConfigDict(
        dict(
            learning_rate = 1e-4,
            batch_size = 64,
            gamma = 0.99,
            gae_lambda = 0.95,
            clip_coef = 0.1,
            num_envs = 8,
            eval_num_envs = 30,
            steps_per_env = 128,
            ent_coef = 0.01,
            vf_coef = 0.5,
            num_rollouts = 50, # 2_000_000
            rollouts_per_eval = 10, # 50
            num_epochs_per_rollout = 4,
        )
    )

    config = ml_collections.ConfigDict()
    config.data = data_config
    config.log = log_config
    config.training = training_config
    return ml_collections.FrozenConfigDict(config)
