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


def test_import_algorithm_modules_recursively_discovers_nested_modules(tmp_path):
    nested_folder = tmp_path / "statistics"
    nested_folder.mkdir(parents=True)
    module_path = nested_folder / "histogram.py"
    module_path.write_text(
        "\n".join(
            [
                "from exaflow.algorithms.exareme3.utils.registry import exareme3_udf",
                "",
                "@exareme3_udf()",
                "def udf_hist(data=None):",
                "    return 1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    from exaflow import import_algorithm_modules

    mods = import_algorithm_modules(str(tmp_path))
    assert "statistics.histogram" in mods


def test_domain_preprocessing_discovery_preserves_registries_and_udfs():
    import exaflow
    from exaflow.algorithms.exareme3.utils.algorithm import Algorithm
    from exaflow.algorithms.exareme3.utils.preprocessing_step import PreprocessingStep
    from exaflow.algorithms.exareme3.utils.registry import exareme3_registry

    def runtime_classes(base):
        # Filter before reading specifications: test-only subclasses may use
        # the default JSON loader without having a specification file.
        return {
            cls.get_specification().name: cls
            for cls in base.__subclasses__()
            if cls.__module__.startswith("exaflow.algorithms.exareme3.")
        }

    expected = {
        "outlier_winsorizer": "outliers",
        "missing_values_handler": "missing_values",
        "categorical_column_creator": "feature_engineering",
        "longitudinal_transformer": "longitudinal",
        "kmeans_cluster_creator": "cluster",
        "pca_column_creator": "decomposition",
    }
    steps = runtime_classes(PreprocessingStep)
    assert set(expected) <= set(steps)
    for name, domain in expected.items():
        assert steps[name].__module__ == (
            f"exaflow.algorithms.exareme3.{domain}.{name}"
        )

    for name, domain in {
        "pca": "decomposition",
        "outlier_report": "outliers",
        "standardized_mean_difference": "statistics",
        "binned_mann_whitney_u_test": "statistics",
        "linear_regression_cv": "linear_model",
        "logistic_regression_cv": "linear_model",
    }.items():
        assert exaflow.exareme3_algorithm_classes[name].__module__ == (
            f"exaflow.algorithms.exareme3.{domain}.{name}"
        )

    specs_before = {
        name: cls.get_specification().model_dump(mode="json")
        for name, cls in runtime_classes(Algorithm).items()
    }
    registry_before = dict(exareme3_registry._registry)
    first = exaflow.import_algorithm_modules(exaflow.EXAREME3_ALGORITHM_FOLDERS)
    second = exaflow.import_algorithm_modules(exaflow.EXAREME3_ALGORITHM_FOLDERS)
    assert first.keys() == second.keys()
    assert all(first[key] is second[key] for key in first)
    assert exareme3_registry._registry == registry_before
    assert runtime_classes(PreprocessingStep) == steps
    assert {
        name: cls.get_specification().model_dump(mode="json")
        for name, cls in runtime_classes(Algorithm).items()
    } == specs_before
