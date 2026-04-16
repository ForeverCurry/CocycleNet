from dataclasses import dataclass
from typing import Union, List

@dataclass(frozen=True)
class SystemMeta:
    mle: float
    lys: float
    mu: Union[float, List[float]]  # supports scalar or list
    std: float
    deltat: float

DATA_META = {
    'pendulum': SystemMeta(mle=1, lys=1., mu=[0.], std=1.17, deltat=0.01),
    'l63': SystemMeta(mle=0.906, lys=1.1, mu=[0,0,23.5], std=14.00, deltat=0.01),
    'KS':  SystemMeta(mle=0.23,  lys=4.35, mu=[0.], std=1.40, deltat=0.25),
}
