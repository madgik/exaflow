import json
from dataclasses import dataclass

from validator.exceptions import InvalidDataModelError
from validator.exceptions import UserInputError

NUMERICAL_TYPES = ("continuous", "discrete")
CATEGORICAL_TYPES = ("nominal", "ordinal", "binary")


@dataclass
class CommonDataElement:
    code: str
    metadata: str

    @classmethod
    def from_metadata(cls, metadata: dict):
        code = metadata["code"]
        if not code.isidentifier():
            raise UserInputError(f"CDE: {code} is not a valid python identifier")

        validate_metadata(code, metadata)
        return cls(code=code, metadata=json.dumps(metadata))

    def get_enumerations(self):
        parsed = json.loads(self.metadata)
        return parsed["enumerations"] if "enumerations" in parsed else {}


def flatten_cdes(schema_data):
    cdes = []

    if "variables" in schema_data:
        for metadata in schema_data["variables"]:
            metadata = reformat_metadata(metadata)
            cdes.append(CommonDataElement.from_metadata(metadata))

    if "groups" in schema_data:
        for group_data in schema_data["groups"]:
            cdes.extend(flatten_cdes(group_data))

    return cdes


def get_type_per_column(cdes):
    return {code: json.loads(cde.metadata)["type"] for code, cde in cdes.items()}


def get_cdes_with_min_max(cdes, columns):
    cdes_with_min_max = {}
    for code, cde in cdes.items():
        if code not in columns:
            continue

        metadata = json.loads(cde.metadata)
        min_value = metadata.get("min")
        max_value = metadata.get("max")

        if min_value is not None or max_value is not None:
            cdes_with_min_max[code] = (min_value, max_value)

    return cdes_with_min_max


def get_cdes_with_enumerations(cdes, columns):
    cdes_with_enumerations = {}
    for code, cde in cdes.items():
        if code not in columns:
            continue

        metadata = json.loads(cde.metadata)
        if "enumerations" in metadata:
            cdes_with_enumerations[code] = list(metadata["enumerations"].keys())

    return cdes_with_enumerations


def get_dataset_enums(cdes):
    return json.loads(cdes["dataset"].metadata)["enumerations"]


def validate_dataset_present_on_cdes_with_proper_format(cdes):
    dataset_cde = [cde for cde in cdes if cde.code == "dataset"]
    if not dataset_cde:
        raise InvalidDataModelError("There is no 'dataset' CDE in the data model.")

    dataset_metadata = json.loads(dataset_cde[0].metadata)
    if dataset_metadata["type"] != "nominal":
        raise InvalidDataModelError(
            "CDE 'dataset' must have the 'type' property equal to 'nominal'."
        )

    if "enumerations" not in dataset_metadata:
        raise InvalidDataModelError(
            "CDE 'dataset' must contain the 'enumerations' property."
        )


def validate_longitudinal_data_model(cdes):
    subject_id_metadata = None
    visit_id_metadata = None

    for cde in cdes:
        if cde.code == "subjectid":
            subject_id_metadata = json.loads(cde.metadata)
        elif cde.code == "visitid":
            visit_id_metadata = json.loads(cde.metadata)

    if not subject_id_metadata:
        raise InvalidDataModelError(
            "There is no 'subjectid' CDE in the longitudinal data model."
        )

    if not visit_id_metadata:
        raise InvalidDataModelError(
            "There is no 'visitid' CDE in the longitudinal data model."
        )

    validate_visitid_cde(visit_id_metadata)


def validate_visitid_cde(metadata):
    if metadata["type"] not in CATEGORICAL_TYPES:
        raise InvalidDataModelError(
            f"CDE 'visitid' must have a categorical 'type': {list(CATEGORICAL_TYPES)}."
        )

    if "enumerations" not in metadata:
        raise InvalidDataModelError(
            "CDE 'visitid' must contain the 'enumerations' property."
        )


def reformat_metadata(metadata):
    new_key_assign = {
        "minValue": "min",
        "maxValue": "max",
    }

    for old_key, new_key in new_key_assign.items():
        if old_key in metadata:
            metadata[new_key] = metadata.pop(old_key)

    if "enumerations" in metadata:
        metadata["enumerations"] = {
            enumeration["code"]: enumeration["label"]
            for enumeration in metadata["enumerations"]
        }

    return metadata


def validate_metadata(code, metadata):
    for element in ["code", "label", "type"]:
        if element not in metadata:
            raise InvalidDataModelError(
                f"Element: {element} is missing from the CDE {code}"
            )

    cde_type = metadata["type"]
    if cde_type not in NUMERICAL_TYPES + CATEGORICAL_TYPES:
        raise InvalidDataModelError(
            f"The CDE {code} has an invalid 'type', the only valid types are: "
            f"{list(NUMERICAL_TYPES + CATEGORICAL_TYPES)}"
        )

    bounds = [metadata[key] for key in ("min", "max") if key in metadata]
    if cde_type in CATEGORICAL_TYPES and bounds:
        raise InvalidDataModelError(
            f"The CDE {code} is {cde_type} and cannot have 'minValue'/'maxValue'."
        )

    if cde_type in NUMERICAL_TYPES and "enumerations" in metadata:
        raise InvalidDataModelError(
            f"The CDE {code} is {cde_type} and cannot have enumerations."
        )

    if cde_type == "binary" and len(metadata.get("enumerations", {})) != 2:
        raise InvalidDataModelError(
            f"The CDE {code} is binary and must have exactly two enumerations."
        )

    if cde_type == "discrete" and any(float(bound) % 1 for bound in bounds):
        raise InvalidDataModelError(
            f"The CDE {code} is discrete and its min/max must be whole numbers."
        )

    if {"min", "max"} <= set(metadata) and metadata["min"] >= metadata["max"]:
        raise InvalidDataModelError(f"The CDE {code} has min greater than the max.")
