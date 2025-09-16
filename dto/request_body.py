"""From this module we will summerize request model in fast api.
"""
from pydantic import BaseModel


class DataSet(BaseModel):
    """Module for representing DataSet.
    """
    dataset_id: str


class DataSetWithKey(BaseModel):
    """Module for representing DataSet.
    """
    data_key: str
    dataset_id: str
