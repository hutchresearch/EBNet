"""
CLI usage of EBNet.

MIT License
Copyright (c) 2025 hutchresearch

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import argparse
from astropy.table import Table
from ebnet.pipeline import predict

def print_results_as_csv(result_table: Table) -> None:
    """
    Prints the contents of an Astropy Table in CSV format.

    Args:
        result_table: Table, The results table containing prediction outputs.
    """
    column_names = result_table.colnames
    print(",".join(column_names))
    for row in result_table:
        print(",".join(str(row[col]) for col in column_names))

def main() -> None:
    """
    Entry point for the EBNet command-line interface.

    Parses command-line arguments, runs predictions using the EBNet pipeline,
    and saves or prints results depending on user input.
    """
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
        "-mt", "--meta_type",
        type=str,
        default="magnitude",
        choices=["magnitude", "flux"],
        help="Type of metadata representation. Choices: 'magnitude' for raw magnitudes "
             "or 'flux' for log10(lambda * F_lambda). Default is 'magnitude'."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Optional path to save results as a FITS file. "
             "If not given, results are printed as CSV."
    )
    parser.add_argument(
        "-d", "--download_flux",
        action="store_true",
        help="Will download SED flux metadata from visier."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        choices=list(range(1, 9)),
        help="Number of workers used to download metadata flux. Default 1. "
            "This will increase the speed of download, but you may be rate limited."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for computation. Can be 'cpu', 'cuda', or 'cuda:<index>'. "
            "Default automatically selects 'cuda' if available, otherwise 'cpu'."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output during prediction."
    )


    args = parser.parse_args()

    table = predict(
        args.data_path, 
        model_type=args.model_type, 
        meta_type=args.meta_type, 
        download_flux=args.download_flux,
        num_workers=args.num_workers,
        device=args.device,
        verbose=args.verbose,
    )

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
