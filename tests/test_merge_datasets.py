"""
This module contains unit tests for the functions in the src.merge_datasets module.

The tests cover:
- Preprocessing of names to ensure correct formatting.
- Cleaning of synopsis data by removing unwanted phrases.
- Consolidation of multiple title columns into a single title column.
- Removal of duplicate synopses or descriptions.
- Addition of additional synopsis information to the merged DataFrame.
"""

from unittest.mock import patch
import pytest
import pandas as pd
from src.merge_datasets import (
    preprocess_name,
    clean_synopsis,
    consolidate_titles,
    remove_duplicate_infos,
    add_additional_info,
)


@pytest.mark.order(1)
def test_preprocess_name():
    """
    Test the preprocess_name function to ensure it correctly preprocesses names
    by converting them to lowercase and stripping whitespace.

    This test verifies that the function handles various input types and formats.
    """
    assert preprocess_name("  Naruto  ") == "naruto"
    assert preprocess_name("One Piece") == "one piece"
    assert preprocess_name(None) == "none"
    assert preprocess_name(123) == "123"


@pytest.mark.order(2)
def test_clean_synopsis():
    """
    Test the clean_synopsis function to ensure it correctly cleans the synopsis column
    by removing unwanted phrases.

    This test verifies that the function replaces unwanted phrases with an empty string.
    """
    data = {
        "Synopsis": [
            "This is a valid synopsis.",
            "No description available for this anime.",
            "Another valid synopsis.",
            "No description available for this anime.",
        ]
    }
    df = pd.DataFrame(data)
    clean_synopsis(df, "Synopsis", "No description available for this anime.")
    assert df.loc[0, "Synopsis"] == "This is a valid synopsis."
    assert df.loc[1, "Synopsis"] == ""
    assert df.loc[2, "Synopsis"] == "Another valid synopsis."
    assert df.loc[3, "Synopsis"] == ""


@pytest.mark.order(3)
def test_consolidate_titles():
    """
    Test the consolidate_titles function to ensure it correctly consolidates multiple title columns
    into a single 'title' column.

    This test verifies that the function prioritizes the main 'title' column and fills in missing
    titles from the specified title columns.
    """
    data = {
        "title": ["naruto", pd.NA, "one piece", pd.NA],
        "title_english": [pd.NA, "bleach", pd.NA, "fullmetal alchemist"],
        "title_japanese": [pd.NA, pd.NA, "wotakoi", "fm alchemist"],
    }
    df = pd.DataFrame(data)
    title_columns = ["title_english", "title_japanese"]

    consolidated = consolidate_titles(df, title_columns)
    expected = pd.Series(
        ["naruto", "bleach", "one piece", "fullmetal alchemist"], name="title"
    )

    pd.testing.assert_series_equal(consolidated, expected)


@pytest.mark.order(4)
def test_remove_duplicate_infos():
    """
    Test the remove_duplicate_infos function to ensure it correctly removes duplicate synopses
    or descriptions from the merged DataFrame.

    This test verifies that duplicate synopses are set to None while keeping unique synopses intact.
    """
    data = {
        "anime_id": [1, 2, 3],
        "Synopsis anime_dataset_2023": [
            "Good story",
            "Action-packed",
            "Good story",
        ],
        "Synopsis mal_anime Dataset": [
            "Good story",
            "Action-packed",
            "Adventure",
        ],
    }
    df = pd.DataFrame(data)
    info_columns = ["Synopsis anime_dataset_2023", "Synopsis mal_anime Dataset"]
    cleaned_df = remove_duplicate_infos(df, info_columns)
    expected = {
        "anime_id": [1, 2, 3],
        "Synopsis anime_dataset_2023": ["Good story", "Action-packed", "Good story"],
        "Synopsis mal_anime Dataset": [None, None, "Adventure"],
    }
    pd.testing.assert_frame_equal(cleaned_df, pd.DataFrame(expected))


@pytest.mark.order(5)
def test_add_additional_info():
    """
    Test the add_additional_info function to ensure it correctly adds additional synopsis
    information to the merged DataFrame.

    This test uses a mock function to simulate the behavior of find_additional_info and
    verifies that the new synopsis column is correctly updated with the additional info.
    """
    merged = pd.DataFrame(
        {"anime_id": [1, 2], "synopsis": ["Hero's journey.", "Slime adventures."]}
    )
    additional_df = pd.DataFrame(
        {
            "anime_id": [1, 2],
            "additional_synopsis": [
                "An epic hero's journey.",
                "More slime adventures.",
            ],
        }
    )

    def mock_find_additional_info(row, additional_df, description_col, name_columns):  # pylint: disable=W0613
        return additional_df.loc[
            additional_df["anime_id"] == row["anime_id"], "additional_synopsis"
        ].values[0]

    with patch(
        "src.merge_datasets.find_additional_info", side_effect=mock_find_additional_info
    ):
        updated = add_additional_info(
            merged=merged,
            additional_df=additional_df,
            description_col="additional_synopsis",
            name_columns=["anime_id"],
            new_synopsis_col="Synopsis additional Dataset",
        )

        assert "Synopsis additional Dataset" in updated.columns
        assert (
            updated.loc[0, "Synopsis additional Dataset"] == "An epic hero's journey."
        )
        assert updated.loc[1, "Synopsis additional Dataset"] == "More slime adventures."


@pytest.mark.order(6)
def test_add_additional_info_no_match():
    """
    Test the add_additional_info function when there are no matching additional info entries
    for some rows in the merged DataFrame.

    This test ensures that the function correctly handles cases where no additional info
    is found for certain rows and leaves the new synopsis column as NaN for those rows.
    """
    merged = pd.DataFrame(
        {
            "anime_id": [1, 2, 3],
            "synopsis": ["Hero's journey.", "Slime adventures.", "Unknown"],
        }
    )
    additional_df = pd.DataFrame(
        {
            "anime_id": [1, 2],
            "additional_synopsis": [
                "An epic hero's journey.",
                "More slime adventures.",
            ],
        }
    )

    def mock_find_additional_info(row, additional_df, description_col, name_columns):  # pylint: disable=W0613
        match = additional_df.loc[
            additional_df["anime_id"] == row["anime_id"], "additional_synopsis"
        ]
        return match.values[0] if not match.empty else None

    with patch(
        "src.merge_datasets.find_additional_info", side_effect=mock_find_additional_info
    ):
        updated = add_additional_info(
            merged=merged,
            additional_df=additional_df,
            description_col="additional_synopsis",
            name_columns=["anime_id"],
            new_synopsis_col="Synopsis additional Dataset",
        )

        assert "Synopsis additional Dataset" in updated.columns
        assert (
            updated.loc[0, "Synopsis additional Dataset"] == "An epic hero's journey."
        )
        assert updated.loc[1, "Synopsis additional Dataset"] == "More slime adventures."
        assert pd.isna(updated.loc[2, "Synopsis additional Dataset"])
