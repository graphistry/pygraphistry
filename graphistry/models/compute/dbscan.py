from typing import Set
from typing_extensions import Literal


DBSCANEngine = Literal["cuml", "sklearn"]
DBSCANEngineAbstract = Literal[DBSCANEngine, "auto"]

dbscan_engine_values: Set[DBSCANEngine] = {"cuml", "sklearn"}
dbscan_engine_abstract_values: Set[DBSCANEngineAbstract] = {"cuml", "sklearn", "auto"}
