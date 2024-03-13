import pandas as pd

def calculate_differences(csv_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # extract first column
    first_column = df.iloc[:, 0]

    # remove first column
    df = df.iloc[:, 1:]

    # Calculate the differences
    differences = df.diff()

    # Add the first column back to the DataFrame
    differences.insert(0, 'date', first_column)

    # remove first row
    differences = differences.iloc[1:, :]

    # Return the DataFrame with differences
    return differences

def calculate_differences_in_percentage(csv_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # extract first column
    first_column = df.iloc[:, 0]

    # remove first column
    df = df.iloc[:, 1:]

    # Calculate the differences
    differences = df.pct_change()

    # Add the first column back to the DataFrame
    differences.insert(0, 'date', first_column)

    # remove first row
    differences = differences.iloc[1:, :]

    # Return the DataFrame with differences
    return differences

# Example usage
csv_file = './predictor/dataset/AAPL.csv'
differences_df = calculate_differences(csv_file)
differences_df_percentage = calculate_differences_in_percentage(csv_file)

# Save the differences DataFrame to a new CSV file
differences_df.to_csv('./predictor/dataset/AAPL_diff.csv', index=False)
differences_df_percentage.to_csv('./predictor/dataset/AAPL_diff_pct.csv', index=False)
