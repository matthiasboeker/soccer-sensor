from typing import Any, Dict, List
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

import pandas as pd  # type: ignore
import warnings

path_to_input_folder = Path(__file__).parent.parent / "input"
sheets = [
    "Fatigue",
    "Mood",
    "Readiness",
    "SleepDurH",
    "SleepQuality",
    "Soreness",
    "Stress",
]


class Severity(Enum):
    Minor = (0,)
    Major = 1


@dataclass(frozen=True)
class Injury:
    player: str
    location: str
    severity: Severity
    timestamp: pd.Index


@dataclass(frozen=True)
class SoccerPlayer:
    name: str
    fatigue: pd.Series
    mood: pd.Series
    readiness: pd.Series
    sleep_duration: pd.Series
    sleep_quality: pd.Series
    soreness: pd.Series
    stress: pd.Series


def get_player_names(sensor_data) -> List[str]:
    return sensor_data["Fatigue"].columns[1:]


def get_player_data(
    sensor_data: Dict[str, pd.DataFrame], player_name: str
) -> Dict[str, pd.Series]:
    init = {}
    for attribute, data in sensor_data.items():
        init[attribute] = pd.Series(data[player_name]).set_axis(
            data[f"{attribute} Data"], axis=0
        )
    return init


def initialise_players(sensor_data: Dict[str, pd.DataFrame]) -> Dict[str, SoccerPlayer]:
    names = get_player_names(sensor_data)
    players = {}
    for id, name in enumerate(names):
        values = get_player_data(sensor_data, name)
        players[str(id)] = SoccerPlayer(
            name,
            values["Fatigue"],
            values["Mood"],
            values["Readiness"],
            values["SleepDurH"],
            values["SleepQuality"],
            values["Soreness"],
            values["Stress"],
        )
    return players


def load_in_sheet(path_to_file: List[Path], sheet_name: str) -> pd.DataFrame:
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        output_dataframe = []
        for path in path_to_file:
            output_dataframe.append(
                pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
            )
    return pd.concat(output_dataframe)


def load_in_file(path_to_file: List[Path]) -> Dict[str, pd.DataFrame]:
    return {sheet: load_in_sheet(path_to_file, sheet) for sheet in sheets}


def generate_player_data(path_to_data: List[Path]) -> Dict[str, SoccerPlayer]:
    return initialise_players(load_in_file(path_to_data))
