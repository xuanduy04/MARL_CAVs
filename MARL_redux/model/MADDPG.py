from config import Config
from MARL_redux.model import BaseModel


class MADDPG(BaseModel):
    def __init__(self, config: Config):
        super(MADDPG, self).__init__(config)
        pass
