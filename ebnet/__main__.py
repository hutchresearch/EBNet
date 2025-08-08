import argparse
from ebnet.pipeline import predict
from astropy.table import Table

def print_results_as_csv(result_table: Table):
    column_names = result_table.colnames
    print(",".join(column_names))

    for row in result_table:
        print(",".join(str(row[col]) for col in column_names))

def main():
    parser = argparse.ArgumentParser(description="Run prediction using EBNet or EBNet+ models.")
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to the directory containing input data (e.g., FITS files)."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="ebnet+",
        choices=["ebnet", "ebnet+", "mixed"],
        help="Model to use for prediction. Choices: 'ebnet', 'ebnet+', or 'mixed'. Default is 'ebnet+'."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output during prediction."
    )

    args = parser.parse_args()
    table = predict(args.data_path, model_type=args.model_type, verbose=args.verbose)
    print_results_as_csv(table)

if __name__ == "__main__":
    main()
