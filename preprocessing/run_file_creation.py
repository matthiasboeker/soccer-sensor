from typing import Dict, List, Tuple
from pathlib import Path
import json

import pandas as pd  # type: ignore

from preprocessing.data_loader import generate_teams, Team, SoccerPlayer


def create_team_csv(teams, path_to_output):
    for team_name, team in teams.items():
        create_players_csv(team, team_name, path_to_output)


def create_players_csv(team: Team, team_name: str, path_to_output: Path):
    players = team.players.items()
    for name, player in players:
        player.to_dataframe().to_csv(path_to_output / team_name / f"player_{name}.csv")


def group_features(players: List[SoccerPlayer]) -> Dict[str, List[pd.Series]]:
    features: Dict[str, List[pd.Series]] = {}
    for attr in players[0].__dataclass_fields__:
        if attr not in ["name", "date_index"]:
            features[attr] = []
    for player in players:
        for attr in player.__dataclass_fields__:
            attribute = getattr(player, attr)
            if isinstance(attribute, pd.Series):
                if attr in ["srpe", "rpe", "duration"]:
                    attribute.name = getattr(player, "name")
                    features[attr].append(attribute)
                else:
                    attribute.name = getattr(player, "name")
                    features[attr].append(attribute)
    return features


def convert_json_exportable(grouped_feature: List[pd.Series]):
    return {series.name: series.tolist() for series in grouped_feature}


def create_feature_export(
    grouped_features,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[pd.Series]]]:
    json_export = {}
    dataframe_export = {}
    for feature_name in grouped_features.keys():
        if feature_name in ["srpe", "rpe", "duration"]:
            json_exportable = convert_json_exportable(grouped_features[feature_name])
            json_export[feature_name] = json_exportable
        else:
            dataframe_export[feature_name] = pd.DataFrame(
                grouped_features[feature_name]
            ).T
    return dataframe_export, json_export


def write_feature_csv(feature_dfs: Dict[str, pd.DataFrame], path_to_output: Path):
    for feature_name, dataframe in feature_dfs.items():
        dataframe.to_csv(path_to_output / f"{feature_name}.csv")


def write_feature_json(feature_json: Dict[str, List[pd.Series]], path_to_output: Path):
    for feature_name, feature in feature_json.items():
        with open(path_to_output / f"{feature_name}.json", "w") as fp:
            json.dump({feature_name: feature}, fp)


def unstack_teams(teams):
    players = []
    for team in teams.values():
        players.extend(team.players.values())
    return players


def create_feature_csv_files(teams, path_to_output):
    players = unstack_teams(teams)
    grouped_features = group_features(players)
    feature_dfs, feature_dict = create_feature_export(grouped_features)
    path_to_output_files = path_to_output / "features"
    write_feature_csv(feature_dfs, path_to_output_files)
    write_feature_json(feature_dict, path_to_output_files)


def main():
    path_to_input_files = Path(__file__).parent.parent / "input"
    path_to_output_files = Path(__file__).parent.parent / "file_output"
    files = [
        [
            path_to_input_files / "rosenborg-women_a_2020.xlsx",
            path_to_input_files / "rosenborg-women_a_2021.xlsx",
        ],
        [
            path_to_input_files / "vifwomen_a_2020.xlsx",
            path_to_input_files / "vifwomen_a_2021.xlsx",
        ],
    ]
    team_names = ["VIF", "Rosenborg"]
    teams = generate_teams(files, team_names)
    create_team_csv(teams, path_to_output_files)
    create_feature_csv_files(teams, path_to_output_files)


if __name__ == "__main__":
    main()
