from typing import Any

class ndarray: ...

float32 = Any
float64 = Any
int32 = Any
int64 = Any

class random:
    @staticmethod
    def random(*args: Any, **kwargs: Any) -> ndarray: ...

class sparse:
    ...

class linalg:
    ...

astype = Any

