import json
import random
from abc import ABC
from abc import abstractmethod
from functools import cached_property
from functools import lru_cache
from functools import partial
from pathlib import Path

import pandas as pd
from tqdm import tqdm

TESTING_DATAMODEL = "dementia:0.1"
DATAMODEL_BASEPATH = Path("tests/test_data/dementia_v_0_1/")
DATAMODEL_CDESPATH = DATAMODEL_BASEPATH / "CDEsMetadata.json"
TRANSFORMATIONS = ["log", "exp", "center", "standardize"]

random.seed(0)


class _DB:
    @lru_cache
    def get_numerical_variables(self):
        cdes = self._get_cdes()
        return [
            cde["code"]
            for cde in cdes
            if not cde["isCategorical"] and cde["sql_type"] in ("int", "real")
        ]

    @lru_cache
    def get_nominal_variables(self):
        cdes = self._get_cdes()
        return [cde["code"] for cde in cdes if cde["isCategorical"]]

    @lru_cache
    def get_data_table(self):
        """Loads all csv into a single DataFrame"""
        dataset_prefix = ["edsd", "ppmi", "desd-synthdata"]

        datasets = []
        for prefix in dataset_prefix:
            for i in range(10):
                filepath = DATAMODEL_BASEPATH / f"{prefix}{i}.csv"
                datasets.append(pd.read_csv(filepath))

        return pd.concat(datasets)

    @lru_cache
    def get_datasets(self):
        data = self.get_data_table()
        return data["dataset"].unique().tolist()

    @lru_cache
    def get_enumerations(self, varname):
        assert varname in self.get_nominal_variables(), f"{varname} is not nominal"
        return self.get_metadata(varname)["enumerations"]

    @lru_cache
    def _get_cdes(self):
        with open(DATAMODEL_CDESPATH) as file:
            cdes = json.load(file)

        def flatten_cdes(cdes: dict, flat_cdes: list):
            if "isCategorical" in cdes:  # isCategorical indicates that we reached a var
                flat_cdes.append(cdes)
            elif "variables" in cdes or "groups" in cdes:
                for elm in cdes.get("variables", []) + cdes.get("groups", []):
                    flatten_cdes(elm, flat_cdes)

        flat_cdes = []
        flatten_cdes(cdes, flat_cdes)

        # some variables found in CDEs are not found in any csv file, remove them
        all_vars = list(self.get_data_table().columns)
        return [cde for cde in flat_cdes if cde["code"] in all_vars]

    @lru_cache
    def get_metadata(self, varname):
        cdes = self._get_cdes()
        return next(cde for cde in cdes if cde["code"] == varname)


def DB():
    # Dead simple singleton pattern. Not great but we don't need anything fancier here.
    if not hasattr(DB, "instance"):
        DB.instance = _DB()
    return DB.instance


def random_permutation(iterable, r=None):
    "Random selection from itertools.permutations(iterable, r)"
    pool = tuple(iterable)
    r = len(pool) if r is None else r
    return tuple(random.sample(pool, r))


def coin():
    outcome = random.choice([True, False])
    return outcome


def triangular(high, mode=2):
    # mode=2 has been found to produce good results empirically
    return int(random.triangular(1, high, mode))


class InputDataVariable(ABC):
    @abstractmethod
    def draw(self):
        raise NotImplementedError


class SingleTypeSingleVar(InputDataVariable):
    def __init__(self, required, pool):
        self.pool = pool
        self.required = required

    def draw(self):
        if not self.required and coin():
            return ()

        return (self.pool.pop(),)


class SingleTypeMultipleVar(InputDataVariable):
    def __init__(self, required, pool):
        self.pool = pool
        self.required = required

    def draw(self):
        if not self.required and coin():
            return ()

        r = triangular(len(self.pool))
        choice = tuple(self.pool[:r])
        del self.pool[:r]
        return choice


class MixedTypeSingleVar(InputDataVariable):
    def __init__(self, required, pool1, pool2):
        self.pool1 = pool1
        self.pool2 = pool2
        self.required = required

    def draw(self):
        if not self.required and coin():
            return ()

        pool = self.pool1 if coin() else self.pool2
        return (pool.pop(),)


class MixedTypeMultipleVar(InputDataVariable):
    def __init__(self, required, pool1, pool2):
        self.pool1 = pool1
        self.pool2 = pool2
        self.required = required

    def draw(self):
        if not self.required and coin():
            return ()

        # pool1 is the main pool and is always used. Additionally, variables
        # are also drawn from pool2 holf of the times. This is in par with the
        # main usecase of MixedTypeMultipleVar, which is regression models that
        # accept both numerical and nominal variables (dummy coded) but where
        # numerical are mandatory whereas nominal are optional.
        r1 = triangular(len(self.pool1))
        choice = self.pool1[:r1]
        del self.pool1[:r1]
        if coin():
            r2 = triangular(len(self.pool2))
            choice += self.pool2[:r2]
            del self.pool2[:r2]
        return tuple(choice)


def make_input_data_variables(properties, numerical_pool, nominal_pool):
    required = properties["required"]
    multiple = properties["multiple"]
    stattypes = set(properties["stattypes"])

    # numerical variables
    if stattypes == {"numerical"} and not multiple:
        return SingleTypeSingleVar(required, pool=numerical_pool)
    elif stattypes == {"numerical"} and multiple:
        return SingleTypeMultipleVar(required, pool=numerical_pool)

    # nominal variables
    elif stattypes == {"nominal"} and not multiple:
        return SingleTypeSingleVar(required, pool=nominal_pool)
    elif stattypes == {"nominal"} and multiple:
        return SingleTypeMultipleVar(required, pool=nominal_pool)

    # mixed variables
    elif stattypes == {"numerical", "nominal"} and not multiple:
        return MixedTypeSingleVar(required, pool1=numerical_pool, pool2=nominal_pool)
    elif stattypes == {"numerical", "nominal"} and multiple:
        return MixedTypeMultipleVar(required, pool1=numerical_pool, pool2=nominal_pool)

    else:
        raise TypeError(
            "Variable stattypes can be 'numerical', 'nominal' or both. "
            f"Got {stattypes}."
        )


class AlgorithmParameter(ABC):
    @abstractmethod
    def draw(self):
        raise NotImplementedError


class IntegerParameter(AlgorithmParameter):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def draw(self):
        return random.randint(self.min, self.max)


class FloatParameter(AlgorithmParameter):
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def draw(self):
        return random.uniform(self.min, self.max)


class EnumFromList(AlgorithmParameter):
    def __init__(self, enums):
        self.enums = enums

    def draw(self):
        return random.choice(self.enums)


class EnumFromCDE(AlgorithmParameter):
    db = DB()

    def __init__(self, varname):
        self.varname = varname

    @cached_property
    def enums(self):
        return self.db.get_enumerations(self.varname)

    def draw(self):
        return random.choice(self.enums)

    import random


class DataTransformationParam(AlgorithmParameter):
    db = DB()

    def __init__(self, var_names):
        self.var_names = var_names

    def draw(self):
        # Shuffle var_names to randomize their selection
        var_names_copy = self.var_names[:]
        random.shuffle(var_names_copy)

        # Select a random number of transformations to use
        num_transformations = random.randint(1, len(TRANSFORMATIONS))
        chosen_transformations = random.sample(TRANSFORMATIONS, num_transformations)

        result = {}
        MAX_VARS_PER_TRANSFORMATION = 4

        for transformation in chosen_transformations:
            # Ensure num_vars is not larger than the remaining items in var_names_copy
            num_vars = max(
                1,
                min(
                    random.randint(1, MAX_VARS_PER_TRANSFORMATION), len(var_names_copy)
                ),
            )
            selected_vars = random.sample(var_names_copy, num_vars)
            result[transformation] = selected_vars

            # Remove used var_names to prevent repetition
            var_names_copy = [var for var in var_names_copy if var not in selected_vars]

            # Stop if no more var_names are left
            if not var_names_copy:
                break

        return result


def make_parameters(properties, variable_groups):
    if "enums" in properties and properties["enums"]["type"] == "list":
        return EnumFromList(enums=properties["enums"]["source"])
    if "enums" in properties and properties["enums"]["type"] == "input_var_CDE_enums":
        var_group_key = properties["enums"]["source"]
        var_group = variable_groups[var_group_key]
        assert len(var_group) == 1, "EnumFromCDE doesn't work when multiple=True"
        varname = var_group[0]
        return EnumFromCDE(varname=varname)
    if "float" in properties["types"]:
        return FloatParameter(min=properties["min"], max=properties["max"])
    if "int" in properties["types"]:
        return IntegerParameter(min=properties["min"], max=properties["max"])
    if "dict" in properties["types"] and "transformation_method" in properties:
        var_names = list(variable_groups["y"]) + (
            list(variable_groups["x"] if variable_groups["x"] else [])
        )
        return DataTransformationParam(var_names=var_names)

    raise TypeError(f"Unknown parameter type: {properties['types']}.")


class InputGenerator:
    db = DB()

    def __init__(self, specs):
        self.specs = specs
        self.inputdata_gens = None  # init in draw to recreate variable pools
        self.parameter_gens = None  # init in draw due to interaction with inputdata
        self.datasets_gen = DatasetsGenerator()
        self.filters_gen = FiltersGenerator()
        self._seen = set()
        self._counter = 0

    def init_variable_gens(self):
        numerical_vars = self.db.get_numerical_variables()
        nominal_vars = self.db.get_nominal_variables()

        numerical_pool = list(random_permutation(numerical_vars))
        nominal_pool = list(random_permutation(nominal_vars))
        self._counter += 1
        if self._counter == 1:
            if not isinstance(self.specs, dict):
                with open(self.specs.name, "r") as f2:
                    self.specs = json.loads(f2.read())
        if len(self.specs["inputdata"]) == 1:
            assert "y" in self.specs["inputdata"], "There should be a 'y' in inputdata"
            self.inputdata_gens = {
                "y": make_input_data_variables(
                    properties=self.specs["inputdata"]["y"],
                    numerical_pool=numerical_pool,
                    nominal_pool=nominal_pool,
                )
            }
        # if specs contain both y and x, pass half pools to each variable generator
        elif len(self.specs["inputdata"]) == 2:
            numerical_pool1 = numerical_pool[: len(numerical_pool) // 2]
            numerical_pool2 = numerical_pool[len(numerical_pool) // 2 :]
            nominal_pool1 = nominal_pool[: len(nominal_pool) // 2]
            nominal_pool2 = nominal_pool[len(nominal_pool) // 2 :]
            self.inputdata_gens = {
                "y": make_input_data_variables(
                    properties=self.specs["inputdata"]["y"],
                    numerical_pool=numerical_pool1,
                    nominal_pool=nominal_pool1,
                ),
                "x": make_input_data_variables(
                    properties=self.specs["inputdata"]["x"],
                    numerical_pool=numerical_pool2,
                    nominal_pool=nominal_pool2,
                ),
            }
        else:
            raise ValueError(
                "Algorithm specs connot contain more than two variable groups. "
                f"Got {self.specs['inputdata'].keys()}."
            )

    def init_parameter_gens(self, variable_groups):
        self.parameter_gens = {
            name: make_parameters(properties, variable_groups)
            for name, properties in self.specs.get("parameters", {}).items()
        }

    def draw(self):
        """Draws a random instance of an algorithm input, based on the
        algorithm specs.
        """

        for _ in range(100):
            # We need to draw variables in each variable group without
            # replacement, to avoid ending up with the same variable in
            # different groups. The way this is implemented is by using common
            # pools of numerical/nominal variables passed to subclasses of
            # InputDataVariable. Each time draw is called on a subclass the
            # pools are depleted, hence they need to be re-initialized each time.
            self.init_variable_gens()
            inputdata = {
                name: inputdata_vars.draw()
                for name, inputdata_vars in self.inputdata_gens.items()
            }

            inputdata["data_model"] = TESTING_DATAMODEL
            inputdata["datasets"] = self.datasets_gen.draw()
            inputdata["filters"] = self.filters_gen.draw()

            # Parameter generators rely on drawn values for inputdata,
            # specifically for the EnumFromCDE case, hence they need to be
            # initialized after inputdata is drawn.
            self.init_parameter_gens(
                variable_groups={
                    "y": inputdata["y"],
                    "x": inputdata.get("x", None),
                }
            )
            parameters = {
                name: param.draw() for name, param in self.parameter_gens.items()
            }

            input_ = {"inputdata": inputdata, "parameters": parameters}
            data_transformation = parameters.get("data_transformation", None)
            param_without_data_transformation = parameters.copy()
            if "data_transformation" in param_without_data_transformation:
                param_without_data_transformation.pop("data_transformation")
            input_key = (
                *inputdata.values(),
                *param_without_data_transformation.values(),
            )
            if input_key not in self._seen:
                self._seen.add(input_key)
                return input_
        else:
            raise ValueError("Cannot find inputdata that has not already be generated.")


class DatasetsGenerator:
    db = DB()

    @cached_property
    def all_datasets(self):
        return self.db.get_datasets()

    def draw(self):
        high = len(self.all_datasets)
        r = triangular(high)
        return random_permutation(self.all_datasets, r=r)


class FiltersGenerator:
    def draw(self):
        # TODO Implement random filters generator
        return None


tqdm = partial(tqdm, desc="Generating test cases", unit=" test cases")


class TestCaseGenerator(ABC):
    """Test case generator ABC.

    For each new algorithm we wish to test a subclass must be created.
    Subclasses must implement `compute_expected_output` where the algorithm
    expected ouput is computed according to the SOA implementation.

    Attributes
    ----------
    full_data: bool
        If true input_data is a single table with all variables found in x
        and y plus the dataset column. Else input_data is a pair of y and x,
        with x being optional, depending on the algorithm specs.
    dropna: bool
        If true NA values are droped from input_data.

    Parameters
    ----------
    specs_file
        Pointer to algorithm specs file in json format

    Examples
    --------
    >>> class MyAlgorithmTCG(TestCaseGenerator):
            def compute_expected_output(self, inputdata, input_parameters):
                ...  # SOA computation
                return {"some_result": 42, "other_result": 24}
    >>> with open('my_algoritm.json') as file:
            tcg = MyAlgorithmTCG(file)
    >>> with open('my_algoritm_expected.json') as file:
            tcg.write_test_cases(file)

    """

    __test__ = False
    full_data = False
    dropna = True

    def __init__(self, specs):
        self.input_gen = InputGenerator(specs)
        self.all_data = DB().get_data_table()

    def generate_input(self):
        return self.input_gen.draw()

    @abstractmethod
    def compute_expected_output(self, input_data, input_parameters, metadata):
        """Computes the expected output for specific algorithm

        This method has to be implemented by subclasses. The user should use
        some state-of-the-art implementation (e.g. sklearn, statsmodels, ...)
        of the algorithm in question to compute the expected results.

        Parameters
        ----------
        input_data: tuple
            A pair of design matrices. Either (y, x) or (y, None) depending on
            the algorithm specs.
        input_parameters: dict or None
            A dict mapping algorithm parameters to values or None if the
            algorithm doesn't have any parameters.

        Returns
        -------
        dict or None
            A dict containing the algorithm output. If no output can be
            computed for the given input the implementer can return None, in
            which case the test case is discarded. Use this for cases where the
            test case generator generates seamingly valid inputs which do not
            make sense for some reason.
        """
        raise NotImplementedError

    def get_input_data(self, input_):
        variables = self._get_variables_from_input(input_)
        inputdata = input_["inputdata"]
        datasets = list(inputdata["datasets"])

        if "dataset" in variables:
            full_data = self.all_data[variables]
        else:
            full_data = self.all_data[variables + ["dataset"]]
        full_data = full_data[full_data.dataset.isin(datasets)]

        if self.dropna:
            full_data = full_data.dropna()

        if len(full_data) == 0:
            return None

        if self.full_data:
            return full_data

        y = list(inputdata["y"])
        x = list(inputdata.get("x", []))
        y_data = full_data[y]
        x_data = full_data[x] if x else None
        return y_data, x_data

    @staticmethod
    def _get_variables_from_input(input):
        inputdata = input["inputdata"]
        y = list(inputdata["y"])
        x = list(inputdata.get("x", []))
        return y + x

    def generate_test_case(self):
        for _ in range(10_000):
            input = self.generate_input()
            input_data = self.get_input_data(input)
            if input_data is not None:
                break
        else:
            raise ValueError(
                "Cannot find inputdata values resulting in non-empty data."
            )
        parameters = input["parameters"]
        metadata = [
            DB().get_metadata(varname)
            for varname in self._get_variables_from_input(input)
        ]
        output = self.compute_expected_output(input_data, parameters, metadata)
        return {"input": input, "output": output}

    def generate_test_cases(self, num_test_cases=100):
        test_cases = []
        with tqdm(total=num_test_cases) as pbar:
            while len(test_cases) < num_test_cases:
                test_case = self.generate_test_case()
                if test_case["output"]:
                    test_cases.append(test_case)
                    pbar.update(1)

        def append_test_case_number(test_case, num):
            test_case["input"]["test_case_num"] = num
            return test_case

        test_cases = [
            append_test_case_number(test_case, i)
            for i, test_case in enumerate(test_cases)
        ]
        return {"test_cases": test_cases}

    def write_test_cases(self, file, num_test_cases=100):
        test_cases = self.generate_test_cases(num_test_cases)
        json.dump(test_cases, file, indent=4)
