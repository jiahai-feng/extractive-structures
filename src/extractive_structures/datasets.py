from extractive_structures import ROOT
from exults.tensorial import Long
import json
import pandas as pd
import numpy as np

from typing import TypedDict


class Dataset(TypedDict):
    train: list[tuple[str, str]]
    test: list[tuple[str, str]]


class Options(TypedDict):
    train: list[str]
    test: list[str]


def first_hop() -> tuple[Dataset, Dataset, Options]:
    """
    Return:
        left_dataset: dict
            A dictionary with keys 'train' and 'test'. Each key maps to a list of tuples. Each tuple contains a sentence and a continuation.
        left_dataset_2: dict
            A dictionary with keys 'train' and 'test'. Each key maps to a list of tuples. Each tuple contains a sentence and a continuation.
        left_dataset_options: dict
            A dictionary with keys 'train' and 'test'. Each key maps to a list of unique continuations.
    """
    SEED = 3
    NUM_CITIES = 20
    with open(ROOT / "dsets/augmented_cities.json") as f:
        long_cities = Long(json.load(f)[:NUM_CITIES])

    with open(ROOT / "dsets/names.csv") as f:
        names = pd.read_csv(f)

    names = [row.first + " " + row.last for row in names.itertuples()]

    rng = np.random.default_rng(SEED)

    name_cities = list(zip(rng.permutation(names[:NUM_CITIES]), long_cities))
    left_dataset = {
        "train": [(f"{name} lives in", city["city"]) for name, city in name_cities],
        "test": [
            (f"The people in the city {name} is from speak", city["language"])
            for name, city in name_cities
        ],
    }

    name_cities_2 = list(zip(rng.permutation(names[:NUM_CITIES]), long_cities))
    left_dataset_2 = {
        "train": [(f"{name} lives in", city["city"]) for name, city in name_cities_2],
        "test": [
            (f"The people in the city {name} is from speak", city["language"])
            for name, city in name_cities_2
        ],
    }
    left_dataset_options = {
        key: list({cont for _, cont in value}) for key, value in left_dataset.items()
    }
    return left_dataset, left_dataset_2, left_dataset_options


def second_hop() -> tuple[Dataset, Dataset, Options]:
    SEED = 3
    NUM_CITIES = 20
    with open(ROOT / "dsets/augmented_cities.json") as f:
        long_cities = Long(json.load(f)[:NUM_CITIES])

    with open(ROOT / "dsets/names.csv") as f:
        names = pd.read_csv(f)

    names = [row.first + " " + row.last for row in names.itertuples()]

    rng = np.random.default_rng(SEED)
    name_cities = list(zip(rng.permutation(names[:NUM_CITIES]), long_cities))
    right_dataset = {
        "train": [
            (f'The mayor of {city["city"]} is', name) for name, city in name_cities
        ],
        "test": [
            (f'The mayor of the city that contains {city["landmark"]} is', name)
            for name, city in name_cities
        ],
    }
    name_cities_2 = list(zip(rng.permutation(names[:NUM_CITIES]), long_cities))
    right_dataset_2 = {
        "train": [
            (f'The mayor of {city["city"]} is', name) for name, city in name_cities_2
        ],
        "test": [
            (f'The mayor of the city that contains {city["landmark"]} is', name)
            for name, city in name_cities_2
        ],
    }

    right_dataset_options = {
        key: list({cont for _, cont in value}) for key, value in right_dataset.items()
    }
    return right_dataset, right_dataset_2, right_dataset_options


class DaxwugDataset(TypedDict):
    left: list[tuple[str, str]]
    right: list[tuple[str, str]]
    both: list[tuple[str, str]]


class DaxwugOptions(TypedDict):
    left: list[str]
    right: list[str]
    both: list[str]


def daxwug(
    left_template=None, both_template=None
) -> tuple[DaxwugDataset, DaxwugDataset, DaxwugOptions]:
    if left_template is None:
        left_template = "{} dax is"
    if both_template is None:
        both_template = "{} zong is the"
    with open(ROOT / "dsets/names.csv") as f:
        names = pd.read_csv(f)
    NUM_CITIES = 20
    with open(ROOT / "dsets/augmented_cities.json") as f:
        long_cities = Long(json.load(f)[:NUM_CITIES])

    names = [row.first + " " + row.last for row in names.itertuples()]

    from datasets import load_dataset

    dataset = load_dataset("amounts-tidings/Country-city-animals", "Facts")["train"]

    fake_cities = dataset["tail"][:20]
    animals = dataset["tail"][20:]

    seed = 3

    rng = np.random.default_rng(seed)

    name_city_1 = rng.integers(len(long_cities), size=len(names))
    name_city_2 = rng.integers(len(long_cities), size=len(names))

    test_name_city = rng.permutation(len(long_cities))
    name_city_1[: len(long_cities)] = test_name_city
    name_city_2[: len(long_cities)] = test_name_city

    joint_1 = [
        dict(
            name=name,
            city=long_cities[city]["city"],
            animal=animals[city],
        )
        for name, city in zip(names, name_city_1)
    ]

    joint_2 = [
        dict(
            name=name,
            city=long_cities[city]["city"],
            animal=animals[city],
        )
        for name, city in zip(names, name_city_2)
    ]
    left_dataset_1 = {
        "left": [(left_template.format(r["name"]), r["city"]) for r in joint_1],
        "right": [
            (f'{city["city"]} wug is the', animal)
            for city, animal in zip(long_cities, animals)
        ],
        "both": [(both_template.format(r["name"]), r["animal"]) for r in joint_1],
    }

    left_dataset_2 = {
        "left": [(left_template.format(r["name"]), r["city"]) for r in joint_2],
        "right": [
            (f'{city["city"]} wug is the', animal)
            for city, animal in zip(long_cities, animals)
        ],
        "both": [(both_template.format(r["name"]), r["animal"]) for r in joint_2],
    }
    left_dataset_options = {
        ds_name: list({cont for _, cont in ds})
        for ds_name, ds in left_dataset_1.items()
    }
    return left_dataset_1, left_dataset_2, left_dataset_options
