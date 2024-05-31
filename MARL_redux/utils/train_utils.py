def init_env(env, config):
    env.config['flatten_obs'] = 'use_attention_module' not in config
    return env
