import yaml


class Config:
    def __init__(self, config_dict):
        self._config_dict = config_dict

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


def import_config(model: str) -> Config:
    config_path = 'MARL/configs/configs_' + model + '.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return Config(config)
