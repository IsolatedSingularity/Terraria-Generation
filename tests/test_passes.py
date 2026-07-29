from collections import Counter

from terraexplorer.generation import PASS_HANDLERS
from terraexplorer.passes import PASS_SPECS, VANILLA_PASS_ORDER, Fidelity


def test_public_pass_catalogue_is_complete_unique_and_ordered() -> None:
    assert len(PASS_SPECS) == len(VANILLA_PASS_ORDER) == 107
    assert len(set(VANILLA_PASS_ORDER)) == 107
    assert [spec.index for spec in PASS_SPECS] == list(range(1, 108))
    assert PASS_SPECS[0].name == "Reset"
    assert PASS_SPECS[-1].name == "Final Cleanup"


def test_every_mutating_spec_has_a_registered_handler() -> None:
    for spec in PASS_SPECS:
        if spec.handler is not None:
            assert spec.handler in PASS_HANDLERS, spec.name


def test_fidelity_status_is_explicit_for_every_pass() -> None:
    counts = Counter(spec.fidelity for spec in PASS_SPECS)

    assert counts[Fidelity.MODELED] == 67
    assert counts[Fidelity.APPROXIMATED] == 39
    assert counts[Fidelity.DOCUMENTED] == 1
