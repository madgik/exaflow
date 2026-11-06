from typing import Optional

from exaflow.controller.services.exareme3.controller import Exareme3Controller

_exareme3_controller: Optional[Exareme3Controller] = None


def set_exareme3_controller(controller: Exareme3Controller):
    global _exareme3_controller
    _exareme3_controller = controller


def get_exareme3_controller() -> Exareme3Controller:
    global _exareme3_controller
    if _exareme3_controller is None:
        raise ValueError("Exareme3Controller has not been initialized.")
    return _exareme3_controller
