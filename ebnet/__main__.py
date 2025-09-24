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
        help="Path to input data, or a directory containing input data (e.g., FITS files)."
    )
    parser.add_argument(
        "-m", "--model_type",
        type=str,
        default="mixed",
        choices=["tf_model", "pt_model", "mixed"],
        help="Model to use for prediction. Choices: 'tf_model', 'pt_model', or 'mixed'. Default is 'mixed'."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Optional path to save results as a FITS file. "
             "If not given, results are printed as CSV."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output during prediction."
    )


    args = parser.parse_args()

    table = predict(args.data_path, model_type=args.model_type, verbose=args.verbose)

    if args.output:
        if args.output.lower().endswith(".fits"):
            table.write(args.output, overwrite=True)
            if args.verbose:
                print(f"Saved results to {args.output}")
        else:
            raise ValueError("Output path must end with .fits")
    else:
        print_results_as_csv(table)

if __name__ == "__main__":
    main()
