from pathlib import Path

import pytest

from vitnns.constants import DEFAULT_SCANNER_ROOT, DEFAULT_STATE_DICT, REPO_ROOT


@pytest.fixture(scope="session")
def state_dict_path() -> Path:
    assert DEFAULT_STATE_DICT.is_file()
    return DEFAULT_STATE_DICT


@pytest.fixture(scope="session")
def scanner_root() -> Path:
    assert DEFAULT_SCANNER_ROOT.is_dir()
    return DEFAULT_SCANNER_ROOT


@pytest.fixture(scope="session")
def sample_relative_path() -> str:
    return "n03976467/n03976467_9546.JPEG"

