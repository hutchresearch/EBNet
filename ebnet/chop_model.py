"""
This file includes a utility for chunking a model.

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

import os
import io
from collections import OrderedDict
import tempfile
import torch

# Utility functions not used in the pipeline, but included for completeness.
def save_chopped_file(load_path: str, save_path: str, chunks: int):
    """Splits a file into multiple chunks and saves them to disk.

    Args:
        load_path: str, The path to the input file to be split.
        save_path: str, The directory where the split files will be saved.
        chunks: int, The number of chunks to split the input file into.

    Returns:
        None: This function does not return anything, but saves the split files to disk.

    Raises:
        ValueError: If chunks is less than 1 or if either load_path or save_path are invalid.

    """
    # Check for valid inputs
    if chunks < 1:
        raise ValueError("Number of chunks must be at least 1.")
    if not os.path.isfile(load_path):
        raise ValueError("Invalid input file path.")
    if not os.path.isdir(save_path):
        raise ValueError("Invalid save directory path.")

    def save_chunk_single(save_name: str, file_in: io.BufferedReader, chunk_size: int):
        """Saves a single chunk of the input file to disk.

        Args:
            save_name: str, The filename to save the chunk as.
            file_in: io.BufferedReader, An open file object pointing to the input file.
            chunk_size: int, The size of the chunk to read from the input file.

        Returns:
            None: This function does not return anything, but saves the chunk to disk.

        """
        with open(save_name, "wb") as file_out:
            file_out.write(file_in.read(chunk_size))
    
    with open(load_path, "rb") as file_in:
        size = file_in.seek(0, 2)
        file_in.seek(0, 0)

        chunk_size = size // chunks
        # The last chunk has to be larger in cases where the filesize is not evenly divisible by the number of  chunks
        last_chunk_size = chunk_size + (size - chunk_size*chunks)

        for i in range(chunks-1):
            save_name = os.path.join(save_path, f"model_chunk_{i}")
            save_chunk_single(save_name, file_in, chunk_size)

        save_name = os.path.join(save_path, f"model_chunk_{chunks-1}")
        save_chunk_single(save_name, file_in, last_chunk_size)


def franken_load(load_path: str, chunks: int) -> OrderedDict:
    """
    Loads a PyTorch model from multiple binary files that were previously split.

    Args:
        load_path: str, The directory where the model chunks are located.
        chunks: int, The number of model chunks to load.

    Returns:
        A ordered dictionary containing the PyTorch model state.
    """

    def load_member(load_path: str, file_out: io.BufferedReader, file_i: int) -> None:
        """
        Reads a single chunk of the model from disk and writes it to a buffer.

        Args:
            load_path: str, The directory where the model chunk files are located.
            file_out: io.BufferedReader, The buffer where the model chunks are written.
            file_i: int, The index of the model chunk to read.

        """
        load_name = os.path.join(load_path, f"model_chunk_{file_i}")
        with open(load_name, "rb") as file_in:
            file_out.write(file_in.read())

    with tempfile.TemporaryDirectory() as tempdir:
        # Create a temporary file to write the model chunks.
        model_path = os.path.join(tempdir, "model.pt")
        with open(model_path, "wb") as file_out:
            # Load each model chunk and write it to the buffer.
            for i in range(chunks):
                load_member(load_path, file_out, i)
        
        # Load the PyTorch model from the buffer.
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))

    return state_dict