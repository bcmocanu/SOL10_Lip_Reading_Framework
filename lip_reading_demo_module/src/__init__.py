from .task_stage1 import AVRomanConfig, AV_RomanTask
from .model_stage1 import AV_RomanizerConfig, AV_RomanizerModel
from .task_stage2 import Zero_AVSRConfig, Zero_AVSR_TrainingTask
from .model_stage2 import Zero_AVSR_ModelConfig, Zero_AVSR

import sys as _sys
import types as _types
from . import task_stage1 as _task1, model_stage1 as _model1
from . import task_stage2 as _task2, model_stage2 as _model2
for _name, _task, _model in (("stage1", _task1, _model1), ("stage2", _task2, _model2)):
    _pkg = _types.ModuleType(_name)
    _pkg.__path__ = []
    _pkg.task = _task
    _pkg.model = _model
    _sys.modules.setdefault(_name, _pkg)
    _sys.modules.setdefault(_name + ".task", _task)
    _sys.modules.setdefault(_name + ".model", _model)
