"""The seven investor-principle frameworks (spec §6). Each module exposes ``NAME`` and
``evaluate(inputs) -> FrameworkResult``; ``run_all`` runs them for one window."""

from __future__ import annotations

from stock_research.analysis.frameworks import (
    buffett,
    chen_chung_ming,
    graham,
    guyu,
    lynch,
    munger,
    warren_chou,
)
from stock_research.analysis.frameworks.base import FrameworkInputs, FrameworkResult

FRAMEWORKS = (warren_chou, guyu, chen_chung_ming, buffett, graham, lynch, munger)


def run_all(inputs: FrameworkInputs) -> list[FrameworkResult]:
    return [module.evaluate(inputs) for module in FRAMEWORKS]
