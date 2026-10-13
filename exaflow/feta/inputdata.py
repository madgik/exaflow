from typing import List
from typing import Optional

from pydantic import BaseModel


class Inputdata(BaseModel):
    data_model: str
    datasets: List[str]
    validation_datasets: Optional[List[str]] = None
    filters: Optional[dict] = None
    variables: List[str]
