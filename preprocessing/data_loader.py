from typing import Any, Dict, List
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

import pandas as pd

path_to_input_folder = Path(__file__).parent.parent / "input"
sheets = ["Fatigue", "Mood", "Readiness", "SleepDurH", "SleepQuality", "Soreness", "Stress"]


class Severity(Enum):
    Minor = 0,
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

def get_player_names(sensor_data):
    return sensor_data["Fatigue"].columns[1:]


def initialise_player(sensor_data: Dict[str, pd.DataFrame], player_name: str):
    init = {}
    for attribute, data in sensor_data.items():
        init[attribute] = data[player_name]
    return init

def load_in_sheet(path_to_file: Path, sheet_name: str):
    data_file = pd.read_excel(path_to_file, sheet_name=sheet_name)
    return data_file


def load_in_file(path_to_file):
    return {sheet: load_in_sheet(path_to_file, sheet) for sheet in sheets}


def load_in_files(path_to_data: List[Path]):
    print(initialise_player(load_in_file(path_to_data[0]),"Synne Skinnes Hansen"))
    return [load_in_file(path) for path in path_to_data]


def main():
    path_to_data_2021 = path_to_input_folder / "rosenborg-women_a_2021.xlsx"
    path_to_data_2020 = path_to_input_folder / "rosenborg-women_a_2020.xlsx"
    load_in_files([path_to_data_2021, path_to_data_2020])



if __name__ == "__main__":
    main()