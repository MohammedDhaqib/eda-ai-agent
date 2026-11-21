import re
import pandas as pd


def load_csv(uploaded_file):
    """
    Load an uploaded CSV file into a Pandas DataFrame.

    Automatically removes CSV index columns such as:
    Unnamed: 0, Unnamed: 1, etc.

    Parameters:
        uploaded_file: CSV file uploaded by the user.

    Returns:
        Pandas DataFrame containing the cleaned dataset.
    """

    try:
        uploaded_file.seek(0)

        dataframe = pd.read_csv(uploaded_file)

        if dataframe.empty:
            raise ValueError("The uploaded CSV file contains no data.")

        # Identify automatically generated CSV index columns.
        unnamed_columns = [
            column
            for column in dataframe.columns
            if re.fullmatch(
                r"Unnamed:\s*\d+",
                str(column).strip(),
                flags=re.IGNORECASE
            )
        ]

        # Remove unwanted index columns.
        if unnamed_columns:
            dataframe = dataframe.drop(columns=unnamed_columns)

        if dataframe.empty or len(dataframe.columns) == 0:
            raise ValueError(
                "The CSV file contains no usable data columns."
            )

        return dataframe

    except pd.errors.EmptyDataError:
        raise ValueError("The uploaded file is empty.")

    except pd.errors.ParserError:
        raise ValueError(
            "Unable to read the CSV. Please check its format."
        )

    except UnicodeDecodeError:
        raise ValueError(
            "Unable to decode this CSV file. Check its text encoding."
        )