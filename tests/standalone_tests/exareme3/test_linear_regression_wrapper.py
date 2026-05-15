from exaflow.algorithms.exareme3.linear_model.linear_regression import LinearRegression


def test_linear_regression_f_stat_display_fields_for_finite_case():
    result = LinearRegression._build_f_stat_display_fields(
        f_stat=153.4451948427044,
        f_pvalue=6.12240830657532e-109,
    )

    assert result["f_stat_status"] == "finite"
    assert result["f_stat_display"] == "153.445"
    assert result["f_pvalue_display"] == "<0.001"
    assert result["f_stat_note"] == "Overall F-test for the fitted regression model."


def test_linear_regression_f_stat_display_fields_for_perfect_fit():
    result = LinearRegression._build_f_stat_display_fields(
        f_stat=float("inf"),
        f_pvalue=0.0,
    )

    assert result["f_stat_status"] == "perfect_fit"
    assert result["f_stat_display"] == "Perfect fit"
    assert result["f_pvalue_display"] == "<0.001"
    assert "Residual sum of squares" in result["f_stat_note"]


def test_linear_regression_f_stat_display_fields_for_undefined_case():
    result = LinearRegression._build_f_stat_display_fields(
        f_stat=float("nan"),
        f_pvalue=float("nan"),
    )

    assert result["f_stat_status"] == "undefined"
    assert result["f_stat_display"] == "Undefined"
    assert result["f_pvalue_display"] == "Undefined"
    assert "not defined" in result["f_stat_note"]
