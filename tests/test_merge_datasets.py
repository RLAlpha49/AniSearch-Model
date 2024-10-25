"""
This module contains unit tests for the functions in the src.merge_datasets module.

The tests cover:
- Preprocessing of names to ensure correct formatting (test_preprocess_name)
- Cleaning of synopsis data by removing unwanted phrases (test_clean_synopsis)
- Consolidation of multiple title columns into a single title column (test_consolidate_titles)
- Removal of duplicate synopses or descriptions (test_remove_duplicate_infos)
- Addition of additional synopsis information to the merged DataFrame (test_add_additional_info)
- Handling of missing matches when adding additional info (test_add_additional_info_no_match)
- Processing of partial title information (test_add_additional_info_partial_titles)
- Handling of missing title data (test_add_additional_info_all_titles_na)
- Handling of whitespace and case variations (test_add_additional_info_whitespace_case)
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
    Test the preprocess_name function to ensure it correctly preprocesses names.

    Tests:
    - Converting strings to lowercase
    - Stripping leading/trailing whitespace
    - Handling None values (returns empty string)
    - Handling numeric values (converts to string)
    """
    assert preprocess_name("  Naruto  ") == "naruto"
    assert preprocess_name("One Piece") == "one piece"
    assert preprocess_name(None) == ""
    assert preprocess_name(123) == "123"


@pytest.mark.order(2)
def test_clean_synopsis() -> None:
    """
    Test the clean_synopsis function to ensure it correctly cleans the synopsis column.

    Tests:
    - Preserving valid synopses
    - Removing specified unwanted phrases
    - Replacing unwanted phrases with empty strings
    - Handling multiple occurrences of unwanted phrases
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
    Test the consolidate_titles function to ensure it correctly consolidates multiple title columns.

    Tests:
    - Prioritizing the main 'title' column values
    - Filling missing values from alternate title columns in order
    - Handling multiple NA values across columns
    - Preserving existing valid titles
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
    Test the remove_duplicate_infos function to ensure it correctly handles duplicate synopses.

    Tests:
    - Identifying and removing duplicate synopses across columns
    - Preserving unique synopses
    - Handling NA values
    - Maintaining original data structure and column order
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
    Test the add_additional_info function for basic functionality.

    Tests:
    - Adding additional synopsis information when titles match
    - Creating new synopsis column in output DataFrame
    - Correctly using mock find_additional_info function
    - Handling English and Japanese titles
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
    Test the add_additional_info function when no matches are found.

    Tests:
    - Handling cases where no matching additional info exists
    - Proper handling of NA values for non-matches
    - Processing multiple rows with varying match conditions
    - Maintaining data integrity for non-matching rows
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
def test_add_additional_info_partial_titles(
    mock_find_additional_info: patch,  # type: ignore
) -> None:
    """
    Test the add_additional_info function with partial title information.

    Tests:
    - Processing rows with some NA title columns but at least one valid title
    - Matching based on available title information
    - Handling mixed NA and non-NA title columns
    - Correct synopsis assignment when matching on partial information
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
def test_add_additional_info_all_titles_na(
    mock_find_additional_info: patch,  # type: ignore
) -> None:
    """
    Test the add_additional_info function with completely missing title information.

    Tests:
    - Handling rows where all title columns are NA
    - Skipping processing for rows with no valid titles
    - Maintaining data integrity for rows with all NA titles
    - Correctly processing mixed rows (some with all NA titles, some with valid titles)
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
def test_add_additional_info_whitespace_case(
    mock_find_additional_info: patch,  # type: ignore
) -> None:
    """
    Test the add_additional_info function's handling of whitespace and case variations.

    Tests:
    - Processing titles with leading/trailing whitespace
    - Handling different case variations (uppercase, lowercase, mixed)
    - Correct matching despite whitespace/case differences
    - Maintaining original data while normalizing for comparison
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
