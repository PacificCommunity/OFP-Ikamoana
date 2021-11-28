from typing import List
import pandas as pd
import xarray as xr

def readFile(filepath: str, header_position: int = None, columns_name: List[str] = None) -> pd.DataFrame :
    return pd.read_table(filepath, header=header_position, names=columns_name)

def separateFisheries(fisheries : pd.DataFrame, fisheries_label: str = "f") -> List[pd.DataFrame]:
    f_dict = {}
    for f_name in fisheries[fisheries_label].unique() :
        f_dict[f_name] = fisheries.loc[fisheries[fisheries_label] == f_name]
    return f_dict

def rescaleFisheries(fisheries: dict, resolution: float = None) -> dict :
    pass

def fisheriesToDataSet(fisheries: dict) -> xr.Dataset :
    pass