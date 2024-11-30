import pandas as pd

def add_to_sample(input_csv, output_csv, increment=200):
    """Adds a specified increment to the 'Sample' column of a CSV file.

    Args:
        input_csv: Path to the input CSV file.
        output_csv: Path to the output CSV file.
        increment: The value to add to the 'Sample' column (default is 100).
    """
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(input_csv)

        # Convert 'Sample' column to numeric, handling scientific notation
        df['Sample'] = pd.to_numeric(df['Sample'], errors='coerce')


        # Add the increment to the 'Sample' column
        df['Sample'] += increment

        # Write the modified DataFrame to a new CSV file
        df.to_csv(output_csv, index=False)
        print(f"Successfully processed and saved to {output_csv}")

    except FileNotFoundError:
        print(f"Error: Input file '{input_csv}' not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: Input file '{input_csv}' is empty.")
    except KeyError:
        print(f"Error: 'Sample' column not found in '{input_csv}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")



# Example usage:
input_file = "/Users/rewatiramansingh/Desktop/Projects/listening-to-pipes/data/raw/Dynamic Pressure Sensor/No-Leak/BR_NL_0.47 LPS_P2.csv"  # Replace with your input CSV file name
output_file = "/Users/rewatiramansingh/Desktop/Projects/listening-to-pipes/data/raw/Dynamic Pressure Sensor/No-Leak/BR_NL_0.47 LPS_P2_sorted.csv" # Replace with your desired output CSV file name
add_to_sample(input_file, output_file)