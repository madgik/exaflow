from __future__ import annotations


def test_import_algorithm_modules_idempotent_for_non_package_dirs(tmp_path):
    # This folder intentionally has no __init__.py, so Exaflow will fall back to
    # importlib.util.spec_from_file_location + exec_module. Re-executing the same
    # module would double-register @exareme3_udf keys and crash startup.
    module_path = tmp_path / "tmp_udf_module.py"
    module_path.write_text(
        "\n".join(
            [
                "from exaflow.algorithms.exareme3.utils.registry import exareme3_udf",
                "",
                "@exareme3_udf()",
                "def udf(data=None):",
                "    return 1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    from exaflow import import_algorithm_modules
    from exaflow.algorithms.exareme3.utils.registry import exareme3_registry

    before = len(exareme3_registry._registry)  # noqa: SLF001 (test-only access)

    mods1 = import_algorithm_modules(str(tmp_path))
    after_first = len(exareme3_registry._registry)  # noqa: SLF001
    assert after_first == before + 1

    mods2 = import_algorithm_modules(str(tmp_path))
    after_second = len(exareme3_registry._registry)  # noqa: SLF001
    assert after_second == after_first

    assert mods1["tmp_udf_module"] is mods2["tmp_udf_module"]
