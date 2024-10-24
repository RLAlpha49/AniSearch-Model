"""
This module contains unit tests for the functions in the src.merge_datasets module.

The tests cover:
- Preprocessing of names to ensure correct formatting.
- Cleaning of synopsis data by removing unwanted phrases.
- Consolidation of multiple title columns into a single title column.
- Removal of duplicate synopses or descriptions.
- Addition of additional synopsis information to the merged DataFrame.
"""

from typing import Union
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
def test_preprocess_name() -> None:
    """
    Test the preprocess_name function to ensure it correctly preprocesses names
    by converting them to lowercase and stripping whitespace.

    This test verifies that the function handles various input types and formats.
    """
    assert preprocess_name("  Naruto  ") == "naruto"
    assert preprocess_name("One Piece") == "one piece"
    assert preprocess_name(None) == ""
    assert preprocess_name(123) == "123"


@pytest.mark.order(2)
def test_clean_synopsis() -> None:
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
    clean_synopsis(df, "Synopsis", ["No description available for this anime."])
    assert df.loc[0, "Synopsis"] == "This is a valid synopsis."
    assert df.loc[1, "Synopsis"] == ""
    assert df.loc[2, "Synopsis"] == "Another valid synopsis."
    assert df.loc[3, "Synopsis"] == ""


@pytest.mark.order(3)
def test_consolidate_titles() -> None:
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
def test_remove_duplicate_infos() -> None:
    """
    Test the remove_duplicate_infos function to ensure it correctly removes duplicate synopses
    across specified columns.
    """
    data = {
        "anime_id": [1, 2, 3],
        "Synopsis anime_dataset_2023": ["A story", "A story", pd.NA],
        "Synopsis animes dataset": ["A story", "Different story", pd.NA],
        "Synopsis anime_270 Dataset": [pd.NA, "A story", "Another story"],
    }
    df = pd.DataFrame(data)
    synopsis_cols = [
        "Synopsis anime_dataset_2023",
        "Synopsis animes dataset",
        "Synopsis anime_270 Dataset",
    ]

    cleaned_df = remove_duplicate_infos(df, synopsis_cols)

    expected = {
        "anime_id": [1, 2, 3],
        "Synopsis anime_dataset_2023": ["A story", "A story", pd.NA],
        "Synopsis animes dataset": [pd.NA, "Different story", pd.NA],
        "Synopsis anime_270 Dataset": [pd.NA, pd.NA, "Another story"],
    }
    expected_df = pd.DataFrame(expected)

    # Ensure both DataFrames use pd.NA for missing values
    cleaned_df = cleaned_df.fillna(pd.NA)
    expected_df = expected_df.fillna(pd.NA)

    pd.testing.assert_frame_equal(cleaned_df, expected_df)


@pytest.mark.order(5)
@patch("src.merge_datasets.find_additional_info")
def test_add_additional_info(mock_find_additional_info: patch) -> None:  # type: ignore
    """
    Test the add_additional_info function to ensure it correctly adds additional synopsis
    information to the merged DataFrame when at least one title is present.
    """
    # Create a merged DataFrame with multiple title columns
    merged = pd.DataFrame(
        {
            "anime_id": [1, 2],
            "title_english": ["Naruto", pd.NA],
            "title_japanese": [pd.NA, "ナルト"],
            "Synopsis": ["Hero's journey.", "Slime adventures."],
        }
    )

    # Additional DataFrame with synopses based on titles
    additional_df = pd.DataFrame(
        {
            "title_english": ["naruto", "bleach"],
            "title_japanese": [pd.NA, "ブリーチ"],
            "additional_synopsis": [
                "An epic hero's journey.",
                "Bleach story synopsis.",
            ],
        }
    )

    # Define a mock function to simulate find_additional_info behavior
    def mock_find_info(
        row: pd.Series,
        additional_df: pd.DataFrame,  # pylint: disable=W0613
        description_col: str,  # pylint: disable=W0613
        name_columns: list,  # pylint: disable=W0613
    ) -> Union[str, None]:
        if (
            pd.notna(row["title_english"])
            and row["title_english"].strip().lower() == "naruto"
        ):
            return "An epic hero's journey."
        elif (
            pd.notna(row["title_english"])
            and row["title_english"].strip().lower() == "bleach"
        ):
            return "Bleach story synopsis."
        return None

    # Assign the side effect to the mock
    mock_find_additional_info.side_effect = mock_find_info

    # Call the function under test
    updated = add_additional_info(
        merged=merged,
        additional_df=additional_df,
        description_col="additional_synopsis",
        name_columns=["title_english", "title_japanese"],
        new_synopsis_col="Synopsis additional Dataset",
    )

    # Assertions
    assert "Synopsis additional Dataset" in updated.columns
    assert updated.loc[0, "Synopsis additional Dataset"] == "An epic hero's journey."
    assert pd.isna(updated.loc[1, "Synopsis additional Dataset"])


@pytest.mark.order(6)
@patch("src.merge_datasets.find_additional_info")
def test_add_additional_info_no_match(mock_find_additional_info: patch) -> None:  # type: ignore
    """
    Test the add_additional_info function when there are no matching additional info entries
    for some rows in the merged DataFrame, considering multiple title columns.
    """
    # Create a merged DataFrame with multiple title columns
    merged = pd.DataFrame(
        {
            "anime_id": [1, 2, 3],
            "title_english": ["Naruto", "Bleach", "One Piece"],
            "title_japanese": [pd.NA, "ブリーチ", "ワンピース"],
            "Synopsis": ["Hero's journey.", "Slime adventures.", "Pirate adventures."],
        }
    )

    # Additional DataFrame with synopses based on titles
    additional_df = pd.DataFrame(
        {
            "title_english": ["naruto", "bleach"],
            "title_japanese": [pd.NA, "ブリーチ"],
            "additional_synopsis": [
                "An epic hero's journey.",
                "Bleach story synopsis.",
            ],
        }
    )

    # Define a mock function to simulate find_additional_info behavior
    def mock_find_info(
        row: pd.Series,
        additional_df: pd.DataFrame,  # pylint: disable=W0613
        description_col: str,  # pylint: disable=W0613
        name_columns: list,  # pylint: disable=W0613
    ) -> Union[str, None]:
        if (
            pd.notna(row["title_english"])
            and row["title_english"].strip().lower() == "naruto"
        ):
            return "An epic hero's journey."
        elif (
            pd.notna(row["title_english"])
            and row["title_english"].strip().lower() == "bleach"
        ):
            return "Bleach story synopsis."
        elif (
            pd.notna(row["title_english"])
            and row["title_english"].strip().lower() == "one piece"
        ):
            return None  # No matching info
        return None

    # Assign the side effect to the mock
    mock_find_additional_info.side_effect = mock_find_info

    # Call the function under test
    updated = add_additional_info(
        merged=merged,
        additional_df=additional_df,
        description_col="additional_synopsis",
        name_columns=["title_english", "title_japanese"],
        new_synopsis_col="Synopsis additional Dataset",
    )

    # Assertions
    assert "Synopsis additional Dataset" in updated.columns
    assert updated.loc[0, "Synopsis additional Dataset"] == "An epic hero's journey."
    assert updated.loc[1, "Synopsis additional Dataset"] == "Bleach story synopsis."
    assert pd.isna(updated.loc[2, "Synopsis additional Dataset"])


@pytest.mark.order(7)
@patch("src.merge_datasets.find_additional_info")
def test_add_additional_info_partial_titles(mock_find_additional_info: patch) -> None:  # type: ignore
    """
    Test the add_additional_info function to ensure it correctly adds synopses
    when some title columns are NA but at least one title is present.
    """
    # Create a merged DataFrame with partial NA titles
    merged = pd.DataFrame(
        {
            "anime_id": [1, 2, 3],
            "title_english": ["Naruto", "Bleach", pd.NA],
            "title_japanese": [pd.NA, "ブリーチ", "ワンピース"],
            "Synopsis": ["Unknown.", "Unknown.", "Pirate adventures."],
        }
    )

    # Additional DataFrame with synopses based on titles
    additional_df = pd.DataFrame(
        {
            "title_english": ["naruto", "bleach", "one piece"],
            "title_japanese": [pd.NA, "ブリーチ", "ワンピース"],
            "additional_synopsis": [
                "An epic hero's journey.",
                "Bleach story synopsis.",
                "The adventures of pirates seeking the ultimate treasure.",
            ],
        }
    )

    # Define a mock function to simulate find_additional_info behavior
    def mock_find_info(
        row: pd.Series,
        additional_df: pd.DataFrame,  # pylint: disable=W0613
        description_col: str,  # pylint: disable=W0613
        name_columns: list,  # pylint: disable=W0613
    ) -> Union[str, None]:
        if (
            pd.notna(row["title_english"])
            and row["title_english"].strip().lower() == "naruto"
        ):
            return "An epic hero's journey."
        elif (
            pd.notna(row["title_japanese"])
            and row["title_japanese"].strip().lower() == "ブリーチ"
        ):
            return "Bleach story synopsis."
        elif (
            pd.notna(row["title_japanese"])
            and row["title_japanese"].strip().lower() == "ワンピース"
        ):
            return "The adventures of pirates seeking the ultimate treasure."
        return None

    # Assign the side effect to the mock
    mock_find_additional_info.side_effect = mock_find_info

    # Call the function under test
    updated = add_additional_info(
        merged=merged,
        additional_df=additional_df,
        description_col="additional_synopsis",
        name_columns=["title_english", "title_japanese"],
        new_synopsis_col="Synopsis additional Dataset",
    )

    # Assertions
    assert "Synopsis additional Dataset" in updated.columns
    assert updated.loc[0, "Synopsis additional Dataset"] == "An epic hero's journey."
    assert updated.loc[1, "Synopsis additional Dataset"] == "Bleach story synopsis."
    assert (
        updated.loc[2, "Synopsis additional Dataset"]
        == "The adventures of pirates seeking the ultimate treasure."
    )


@pytest.mark.order(8)
@patch("src.merge_datasets.find_additional_info")
def test_add_additional_info_all_titles_na(mock_find_additional_info: patch) -> None:  # type: ignore
    """
    Test the add_additional_info function to ensure it skips rows where all title columns are NA.
    """
    # Create a merged DataFrame with all titles as NA for one row
    merged = pd.DataFrame(
        {
            "anime_id": [1, 2],
            "title_english": [pd.NA, "Bleach"],
            "title_japanese": [pd.NA, pd.NA],
            "Synopsis": ["Unknown.", "Soul reapers."],
        }
    )

    # Additional DataFrame with synopses based on titles
    additional_df = pd.DataFrame(
        {
            "title_english": ["bleach"],
            "title_japanese": [pd.NA],
            "additional_synopsis": ["Bleach story synopsis."],
        }
    )

    # Define a mock function to simulate find_additional_info behavior
    def mock_find_info(
        row: pd.Series,
        additional_df: pd.DataFrame,  # pylint: disable=W0613
        description_col: str,  # pylint: disable=W0613
        name_columns: list,  # pylint: disable=W0613
    ) -> Union[str, None]:
        if (
            pd.notna(row["title_english"])
            and row["title_english"].strip().lower() == "bleach"
        ):
            return "Bleach story synopsis."
        return None

    # Assign the side effect to the mock
    mock_find_additional_info.side_effect = mock_find_info

    # Call the function under test
    updated = add_additional_info(
        merged=merged,
        additional_df=additional_df,
        description_col="additional_synopsis",
        name_columns=["title_english", "title_japanese"],
        new_synopsis_col="Synopsis additional Dataset",
    )

    # Assertions
    assert "Synopsis additional Dataset" in updated.columns
    assert pd.isna(updated.loc[0, "Synopsis additional Dataset"])  # All titles NA
    assert updated.loc[1, "Synopsis additional Dataset"] == "Bleach story synopsis."


@pytest.mark.order(9)
@patch("src.merge_datasets.find_additional_info")
def test_add_additional_info_whitespace_case(mock_find_additional_info: patch) -> None:  # type: ignore
    """
    Test the add_additional_info function to ensure it correctly handles titles with
    leading/trailing whitespaces and different cases.
    """
    # Create a merged DataFrame with varied title formats
    merged = pd.DataFrame(
        {
            "anime_id": [1, 2],
            "title_english": ["  Naruto  ", "BLEACH"],
            "title_japanese": ["ナルト", "ブリーチ"],
            "Synopsis": ["Unknown.", "Unknown."],
        }
    )

    # Additional DataFrame with synopses based on titles
    additional_df = pd.DataFrame(
        {
            "title_english": ["naruto", "bleach"],
            "title_japanese": ["ナルト", "ブリーチ"],
            "additional_synopsis": [
                "An epic hero's journey.",
                "Bleach story synopsis.",
            ],
        }
    )

    # Define a mock function to simulate find_additional_info behavior
    def mock_find_info(
        row: pd.Series,
        additional_df: pd.DataFrame,  # pylint: disable=W0613
        description_col: str,  # pylint: disable=W0613
        name_columns: list,  # pylint: disable=W0613
    ) -> Union[str, None]:
        if (
            pd.notna(row["title_english"])
            and row["title_english"].strip().lower() == "naruto"
        ):
            return "An epic hero's journey."
        elif (
            pd.notna(row["title_english"])
            and row["title_english"].strip().lower() == "bleach"
        ):
            return "Bleach story synopsis."
        return None

    # Assign the side effect to the mock
    mock_find_additional_info.side_effect = mock_find_info

    # Call the function under test
    updated = add_additional_info(
        merged=merged,
        additional_df=additional_df,
        description_col="additional_synopsis",
        name_columns=["title_english", "title_japanese"],
        new_synopsis_col="Synopsis additional Dataset",
    )

    # Assertions
    assert "Synopsis additional Dataset" in updated.columns
    assert updated.loc[0, "Synopsis additional Dataset"] == "An epic hero's journey."
    assert updated.loc[1, "Synopsis additional Dataset"] == "Bleach story synopsis."
