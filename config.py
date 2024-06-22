import yaml


class Config:
    def __init__(self, config_dict: dict):
        self._config_dict = config_dict
        self._assigned_attrs = set(config_dict.keys())

    def __getattr__(self, name):
        if name in self._config_dict:
            value = self._config_dict[name]
            if isinstance(value, dict):
                return Config(value)
            return value
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name != '_config_dict' and name != '_assigned_attrs':
            if name not in self._assigned_attrs:
                self._assigned_attrs.add(name)
                self._config_dict[name] = value
            else:
                raise AttributeError(f"'{type(self).__name__}' object attribute '{name}' "
                                     f"is read-only")
        else:
            super().__setattr__(name, value)

    def __contains__(self, name):
        return name in self._assigned_attrs


def import_config(model_name: str) -> Config:
    base_config_path = 'MARL/configs/configs_base.yaml'
    with open(base_config_path, 'r') as file:
        base_config = yaml.safe_load(file)

    model_config_path = 'MARL/configs/configs_' + model_name + '.yaml'
    with open(model_config_path, 'r') as file:
        model_config = yaml.safe_load(file)

    return Config(dict(**base_config, **model_config))
