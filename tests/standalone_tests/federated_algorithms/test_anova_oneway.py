import itertools

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.libqsturng import psturng

from exaflow.algorithms.federated.anova_oneway import FederatedAnovaOneWay
from tests.standalone_tests.federated_algorithms.utils import DummyAggClient

TEST_CASES = [
    [
        np.array(
            [
                1.173075188335231,
                1.6290497818198784,
                1.1807471646097625,
                0.809830808230709,
                0.6265984170829438,
                0.03380451355550029,
                0.9871946133324887,
            ],
            dtype=float,
        ),
        np.array(
            [
                1.7841624979615847,
                1.9765800272100842,
                2.206326678971258,
                2.1336356645654324,
                1.1191997156067897,
            ],
            dtype=float,
        ),
        np.array(
            [
                -0.19224472686059862,
                -1.1541182746436651,
                0.9152970240726446,
                -1.4035363432099035,
                -0.3772344937517835,
                -0.3176377731427754,
            ],
            dtype=float,
        ),
    ],
    [
        np.array(
            [
                -0.030676146769045787,
                0.06407116331621182,
                0.6170348547016977,
                0.8593011848614469,
                0.10441746124201767,
                0.25995018675960824,
            ],
            dtype=float,
        ),
        np.array(
            [
                0.34899339157726983,
                -0.5097845774200822,
                -0.022419180509122394,
                0.14271458651428093,
                -0.5056565092457654,
                -0.6972420805048891,
                -1.130071142900779,
            ],
            dtype=float,
        ),
        np.array(
            [
                -0.30344178233663405,
                0.4741447261278978,
                0.8153071956102789,
                0.1903829749105495,
                0.03750938802775913,
                -0.08254321952365155,
            ],
            dtype=float,
        ),
    ],
    [
        np.array(
            [
                2.32308109764286,
                1.4738273368440902,
                0.6667956549033831,
                1.4932321313753942,
                1.4200838138671499,
                1.443656232677116,
                1.1137610371208928,
            ],
            dtype=float,
        ),
        np.array(
            [
                0.49766279077498343,
                0.5015980654315978,
                -0.2627539053657053,
                0.020758695208259126,
            ],
            dtype=float,
        ),
        np.array(
            [
                2.1208367280284746,
                1.3365209931721638,
                1.3787998418681107,
                2.302360155150093,
                1.5223673343184596,
                1.7290637239715962,
            ],
            dtype=float,
        ),
    ],
    [
        np.array(
            [
                2.2582445547920114,
                1.8999110299339583,
                2.1200668208967466,
                1.9999349233056916,
            ],
            dtype=float,
        ),
        np.array(
            [
                -0.4299197389582709,
                -0.5740310437673473,
                -0.7375456837131739,
                -0.5786000311295924,
                -0.18060545252395666,
                -0.9608254411897079,
                -0.6182781225373658,
            ],
            dtype=float,
        ),
        np.array(
            [
                -0.29114036982959657,
                -1.4674857328973747,
                -0.6219954460481372,
                0.022040609384323395,
                -0.6548960359995742,
                -1.7110051153052277,
            ],
            dtype=float,
        ),
    ],
    [
        np.array(
            [
                1.8793749724327284,
                1.746413695192305,
                1.2875146786404787,
                0.784931051219722,
                1.6238668826434488,
                2.220324380913655,
                2.295542730290385,
            ],
            dtype=float,
        ),
        np.array(
            [
                -1.0655139016706439,
                0.27460844060171363,
                0.11898654064924066,
                -0.36346820267613233,
                -0.6070992621390929,
                -0.7149130431060673,
            ],
            dtype=float,
        ),
        np.array(
            [
                -0.362546521671415,
                -1.14190697115305,
                -1.6985697814297431,
                -1.2523337655665283,
            ],
            dtype=float,
        ),
    ],
    [
        np.array(
            [
                0.61108845071399,
                0.937890776990516,
                -0.02678467728619227,
                0.6136772845053694,
                1.1229028874074958,
                0.616339195116984,
                1.2677268050602992,
            ],
            dtype=float,
        ),
        np.array(
            [
                2.1791395110481018,
                1.5967392312928401,
                2.426503342635632,
                2.673631622187082,
                1.384165468211616,
                1.5991373025708153,
                2.234897484820269,
            ],
            dtype=float,
        ),
        np.array(
            [
                -1.407449371628232,
                -0.6936825470208313,
                -0.40371338534199613,
                -1.1765233730824098,
                -0.6421819186885593,
                -0.40658923466056646,
            ],
            dtype=float,
        ),
    ],
    [
        np.array(
            [
                0.7518015636720661,
                0.3790756991349976,
                0.5573784891713225,
                0.43151253286510066,
            ],
            dtype=float,
        ),
        np.array(
            [
                1.5259326050832782,
                2.752623508036426,
                2.9374679944507216,
                2.0115106531768228,
            ],
            dtype=float,
        ),
        np.array(
            [
                1.2166286671284763,
                1.0044939001941753,
                1.03741204043089,
                -0.17592925731675935,
                1.0531998014546529,
                1.0591264870256363,
            ],
            dtype=float,
        ),
    ],
    [
        np.array(
            [
                -1.163907795485338,
                -1.2711524332004769,
                -0.8190677003858536,
                -1.575049910814542,
                -2.25106817277873,
                -1.0020839291510197,
                -1.122951863434709,
            ],
            dtype=float,
        ),
        np.array(
            [
                1.3343297014897715,
                1.3840160178634449,
                0.7073600161654221,
                0.5803485717706086,
                1.514493885902899,
                1.5305464333172996,
                1.090562581747707,
            ],
            dtype=float,
        ),
        np.array(
            [
                1.6395995514585233,
                1.8765655716718443,
                1.901694559023589,
                1.4318286540241876,
                0.5962857308817526,
                0.891531361246902,
            ],
            dtype=float,
        ),
    ],
    [
        np.array(
            [
                -1.2470225442734963,
                -0.8655617485227317,
                -1.6741100724759446,
                -0.8661227696555728,
                -1.2853445960208871,
                -0.8052053903670087,
            ],
            dtype=float,
        ),
        np.array(
            [
                1.5816617745061323,
                1.353463463818267,
                1.8209082561094052,
                1.403831384964437,
                1.8870300627532455,
                1.6370586199349157,
                0.9770403752509029,
            ],
            dtype=float,
        ),
        np.array(
            [
                1.5967091039728083,
                0.8203455689055417,
                1.5192900164559289,
                1.3062772787739358,
            ],
            dtype=float,
        ),
    ],
    [
        np.array(
            [
                1.450521141980873,
                1.1994356917744478,
                2.317725027630676,
                1.3973281350397893,
                1.5233343701447872,
                1.638481084909996,
                1.3028326521912914,
            ],
            dtype=float,
        ),
        np.array(
            [
                -0.184213869472451,
                -0.26384217738025056,
                -0.40535178897874646,
                -0.23967551043056462,
            ],
            dtype=float,
        ),
        np.array(
            [
                0.4595011866719605,
                0.4330164075803731,
                0.17373969470938683,
                -0.02174996551236641,
            ],
            dtype=float,
        ),
    ],
]


@pytest.mark.parametrize("groups", TEST_CASES)
def test_federated_anova_oneway_matches_statsmodels(groups):
    categories = [f"g{i+1}" for i in range(len(groups))]
    agg_client = DummyAggClient()
    model = FederatedAnovaOneWay(agg_client=agg_client)
    model.fit(groups=groups, categories=categories)

    values = np.concatenate(groups)
    labels = np.concatenate([[name] * len(g) for name, g in zip(categories, groups)])
    df = pd.DataFrame({"y": values, "x": labels})

    lm = ols("y ~ x", data=df).fit()
    aov = sm.stats.anova_lm(lm)

    assert np.isclose(model.df_within, aov["df"]["Residual"], atol=1e-8)
    assert np.isclose(model.df_between, aov["df"]["x"], atol=1e-8)
    assert np.isclose(model.ss_within, aov["sum_sq"]["Residual"], atol=1e-8)
    assert np.isclose(model.ss_between, aov["sum_sq"]["x"], atol=1e-8)
    assert np.isclose(model.ms_within, aov["mean_sq"]["Residual"], atol=1e-8)
    assert np.isclose(model.ms_between, aov["mean_sq"]["x"], atol=1e-8)
    assert np.isclose(model.fvalue, aov["F"]["x"], atol=1e-8)
    assert np.isclose(model.pvalue, aov["PR(>F)"]["x"], atol=1e-8)

    group_stats = df.groupby("x")["y"].agg(["count", "mean"]).reindex(categories)
    gnobs = group_stats["count"].to_numpy(dtype=float)
    gmeans = group_stats["mean"].to_numpy(dtype=float)
    gvar = aov["mean_sq"]["Residual"] / gnobs
    g1, g2 = np.array(list(itertools.combinations(np.arange(len(categories)), 2))).T

    mn = gmeans[g1] - gmeans[g2]
    se = np.sqrt(gvar[g1] + gvar[g2])
    tval = mn / se
    pval = psturng(np.sqrt(2.0) * np.abs(tval), len(categories), aov["df"]["Residual"])

    expected_tukey = {}
    for idx, (i, j) in enumerate(zip(g1, g2)):
        key = (categories[int(i)], categories[int(j)])
        expected_tukey[key] = {
            "meanA": gmeans[i],
            "meanB": gmeans[j],
            "diff": mn[idx],
            "se": se[idx],
            "t_stat": tval[idx],
            "p_tuckey": float(pval[idx]),
        }

    for _, row in model.thsd_.iterrows():
        key = (row["A"], row["B"])
        expected = expected_tukey[key]
        assert np.isclose(row["mean(A)"], expected["meanA"], atol=1e-8)
        assert np.isclose(row["mean(B)"], expected["meanB"], atol=1e-8)
        assert np.isclose(row["diff"], expected["diff"], atol=1e-8)
        assert np.isclose(row["Std.Err."], expected["se"], atol=1e-8)
        assert np.isclose(row["t value"], expected["t_stat"], atol=1e-8)
        assert np.isclose(row["Pr(>|t|)"], expected["p_tuckey"], atol=1e-8)

    expected_mins = [float(np.min(g)) for g in groups]
    assert np.allclose(model.var_min_per_group_, expected_mins, atol=1e-8)

    expected_maxs = [float(np.max(g)) for g in groups]
    assert np.allclose(model.var_max_per_group_, expected_maxs, atol=1e-8)
