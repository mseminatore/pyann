"""Guards against the Python bindings drifting from libann's public API.

These tests read libann's headers (the ``ann/`` submodule) and compare them with
the CFFI declarations in ``pyann/_bindings`` and with what the package actually
calls. After bumping the submodule, a failure here means libann gained,
lost or changed something: bind it, or record the decision in the tables below.
"""

import re
from pathlib import Path

import pytest

import pyann
from pyann._bindings.ann_cdef import ANN_CDEF
from pyann._bindings.tensor_cdef import TENSOR_CDEF

ANN_DIR = Path(__file__).resolve().parents[1] / "ann"
PKG_DIR = Path(pyann.__file__).resolve().parent
HEADERS = ["ann.h", "tensor.h", "ann_hypertune.h"]

pytestmark = pytest.mark.skipif(
    not all((ANN_DIR / h).exists() for h in HEADERS),
    reason="libann headers not available (ann/ submodule not checked out)",
)

CDEF = TENSOR_CDEF + ANN_CDEF

# --- Decisions ---------------------------------------------------------------

# Exported by libann but deliberately not declared in the cdef.
HYPERTUNE_REASON = (
    "libann's native hypertuner is not bound; pyann.hypertune is a separate "
    "pure-Python implementation"
)
UNBOUND = {
    name: HYPERTUNE_REASON
    for name in [
        "hypertune_space_init", "hypertune_options_init", "hypertune_result_init",
        "hypertune_split_data", "hypertune_free_split", "hypertune_grid_search",
        "hypertune_random_search", "hypertune_print_result", "hypertune_print_summary",
        "hypertune_count_grid_trials", "hypertune_generate_topology",
        "hypertune_create_network", "hypertune_score_accuracy",
        "hypertune_topology_name", "gp_init", "gp_add_observation", "gp_predict",
        "gp_fit_hyperparams", "gp_expected_improvement", "bayesian_options_init",
        "hypertune_bayesian_search", "tpe_options_init", "hypertune_tpe_search",
    ]
}

# Declared in the cdef (so they can be called) but never called by the package.
UNUSED = {
    "ann_predict": "superseded by ann_predict_batch, which Sequential.predict uses",
    "ann_gpu_sync_weights": "libann syncs weights back itself after training",
    "ann_set_loss_function": "the loss is chosen when the network is created",
    "ann_set_training_mode": "libann switches training mode itself around training",
    "ann_set_error_log_callback": "error logging callbacks are not exposed; errors raise",
    "ann_get_error_log_callback": "error logging callbacks are not exposed; errors raise",
    "ann_clear_error_log_callback": "error logging callbacks are not exposed; errors raise",
    "ann_strerror": "error messages come from pyann.exceptions",
    "ann_print_props": "debug printing; use Sequential.summary()",
    "ann_print_outputs": "debug printing",
    "ann_print_confusion_matrix": "debug printing; use Sequential.confusion_matrix()",
    "ann_class_prediction": "helper; numpy argmax covers it",
}
UNUSED_PREFIXES = {
    "tensor_": "NumPy covers tensor math; pyann.Tensor wraps only create/copy/get/set/free",
}

# In the cdef but not an ANN_API function in libann's headers.
CDEF_ONLY = {
    "cblas_init": "from the optional external CBLAS library",
    "cblas_shutdown": "from the optional external CBLAS library",
    "cblas_get_config": "from the optional external CBLAS library",
    "cblas_get_corename": "from the optional external CBLAS library",
    "cblas_get_isa_features": "from the optional external CBLAS library",
    "pyann_blas_backend": "defined by pyann itself (src/native/pyann_info.c)",
}

# --- Header and cdef parsing ---------------------------------------------------


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def normalize_type(text):
    """Reduce a C type to something comparable: no const, tidy spacing."""
    text = re.sub(r"\bconst\b", "", text)
    text = re.sub(r"\s*\*\s*", "*", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace("*", " *").replace("  ", " ").strip()


def param_type(param):
    """Drop the parameter name from 'real *outputs' -> 'real *'."""
    param = normalize_type(param)
    if param in ("", "void"):
        return ""
    match = re.match(r"^(.*?[\s*])(\w+)$", param)
    return normalize_type(match.group(1)) if match else param


def split_params(params):
    types = [param_type(p) for p in params.split(",")]
    return tuple(t for t in types if t)


def header_functions():
    """name -> (return type, parameter types) for every ANN_API function."""
    found = {}
    for header in HEADERS:
        text = strip_comments((ANN_DIR / header).read_text(encoding="utf-8", errors="replace"))
        for m in re.finditer(r"\bANN_API\b([^;{]*?)\b(\w+)\s*\(([^;{]*?)\)\s*;", text, flags=re.S):
            ret = normalize_type(m.group(1))
            found[m.group(2)] = (ret, split_params(m.group(3)))
    return found


def cdef_functions():
    found = {}
    for m in re.finditer(r"^\s*([A-Za-z_][\w\s\*]*?)\b(\w+)\s*\(([^)]*)\)\s*;", strip_comments(CDEF), flags=re.M):
        if m.group(1).strip().startswith(("typedef", "return")):
            continue
        found[m.group(2)] = (normalize_type(m.group(1)), split_params(m.group(3)))
    return found


def enums(text):
    """name -> {member: value} for every typedef enum, with values resolved."""
    result = {}
    for m in re.finditer(r"typedef\s+enum\s*\{([^{}]*)\}\s*(\w+)\s*;", strip_comments(text)):
        members, value = {}, -1
        for item in filter(str.strip, m.group(1).split(",")):
            name, _, expr = item.partition("=")
            name, expr = name.strip(), expr.strip()
            if expr:
                value = members[expr] if expr in members else int(expr, 0)
            else:
                value += 1
            members[name] = value
        result[m.group(2)] = members
    return result


def called_functions():
    """Library functions the installed package calls (outside the cdef files)."""
    called = set()
    for path in PKG_DIR.rglob("*.py"):
        if path.name.endswith("_cdef.py"):
            continue
        # No trailing "(": callbacks such as ann_lr_scheduler_step are passed as values
        called |= set(re.findall(r"\b(?:_?lib|loaded_lib)\.(\w+)", path.read_text(encoding="utf-8", errors="replace")))
    return called


# --- Tests -------------------------------------------------------------------


class TestEveryFunctionAccountedFor:
    def test_every_exported_function_is_declared_or_recorded_as_unbound(self):
        """A new libann function must be bound or added to UNBOUND with a reason."""
        missing = sorted(set(header_functions()) - set(cdef_functions()) - set(UNBOUND))
        assert not missing, f"libann exports functions that are neither bound nor in UNBOUND: {missing}"

    def test_unbound_table_is_current(self):
        exported, declared = set(header_functions()), set(cdef_functions())
        stale = sorted(n for n in UNBOUND if n not in exported or n in declared)
        assert not stale, f"UNBOUND lists functions that are gone or now bound: {stale}"

    def test_cdef_functions_exist_in_libann_or_are_known_extras(self):
        extra = sorted(set(cdef_functions()) - set(header_functions()) - set(CDEF_ONLY))
        assert not extra, f"cdef declares functions libann no longer exports: {extra}"

    def test_cdef_only_table_is_current(self):
        stale = sorted(n for n in CDEF_ONLY if n not in cdef_functions() or n in header_functions())
        assert not stale, f"CDEF_ONLY lists functions that are no longer declared or are now in libann: {stale}"


class TestSignatures:
    def test_cdef_signatures_match_headers(self):
        header, cdef = header_functions(), cdef_functions()
        mismatched = {
            name: {"header": header[name], "cdef": cdef[name]}
            for name in sorted(set(header) & set(cdef))
            if header[name] != cdef[name]
        }
        assert not mismatched, f"signatures differ between libann headers and the cdef: {mismatched}"


class TestEnums:
    @pytest.mark.parametrize("header,cdef", [("ann.h", ANN_CDEF), ("tensor.h", TENSOR_CDEF)])
    def test_enum_values_match(self, header, cdef):
        expected, declared = enums((ANN_DIR / header).read_text(encoding="utf-8", errors="replace")), enums(cdef)
        different = {
            name: {"header": members, "cdef": declared.get(name)}
            for name, members in expected.items()
            if declared.get(name) != members
        }
        assert not different, f"enums differ between libann and the cdef: {different}"


class TestUsage:
    @staticmethod
    def _allowed_unused(name):
        return name in UNUSED or any(name.startswith(p) for p in UNUSED_PREFIXES)

    def test_every_declared_function_is_used_or_recorded_as_unused(self):
        """Declaring a function in the cdef is not the same as exposing a feature."""
        unaccounted = sorted(
            n for n in header_functions()
            if n in cdef_functions() and n not in called_functions() and not self._allowed_unused(n)
        )
        assert not unaccounted, f"declared but never called, and not in UNUSED: {unaccounted}"

    def test_unused_table_is_current(self):
        called, declared = called_functions(), set(cdef_functions())
        stale = sorted(n for n in UNUSED if n in called or n not in declared)
        assert not stale, f"UNUSED lists functions that are now used or no longer declared: {stale}"
