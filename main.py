"""
Copyright (C) 2020 Antonin LOUBIERE

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/gpl-3.0.html>.
"""

from typing import Any, Callable, Iterable

import numpy as np

import helpers
from helpers import show_matrix, show_results_graph


def normalise(matrix: np.ndarray) -> np.ndarray:
    """
    Normalise the array so each column has a sum of 1.
    :param matrix: the matrix that hold the data.
    :return: the normalised matrix.
    """
    return matrix / matrix.sum(axis=0)


def prepare_data(data: np.ndarray, proportion=.85) -> np.ndarray:
    """
    Prepare the data to be rank, take `proportion` of the data and the rest of an empty matrix.
    :param data: the data to prepare.
    :param proportion: what is the proportion to take of the data.
    :return: the data processed.
    """
    assert data.shape[0] == data.shape[1], "Data must be a square matrix."
    return proportion * data + (1 - proportion) * (1 / data.shape[0])


def rank(data: np.ndarray, epochs=10) -> np.ndarray:
    """
    Rank the data.
    :param data: the data to rank.
    :param epochs: How many epochs to do.
    :return: the data ranked.
    """
    assert data.shape[0] == data.shape[1], "Data must be a square matrix."
    for i in range(epochs):
        # show_matrix(data, LABELS)
        data = data.dot(data)
    return data


def isolate_results(results: np.ndarray) -> np.ndarray:
    """
    Isolate the results from the rank methods.
    :param results: the results to process.
    :return: the results truncated.
    """
    return results[:, 0]


def append_labels(results: Iterable[int], labels: Iterable[str]) -> Iterable[tuple[str, int]]:
    """
    Append the labels to the results.
    :param results: the results of the rank.
    :param labels: the labels to append to results.
    :return: an iterable of the tuple of the label and the results.
    """
    return zip(labels, results)


def sort_result(results: Any, reverse=True, key: Callable[[Any], Any] = lambda x: x[1]) -> list[Any]:
    """
    Sort the results. By default from the best to the worst.
    :param results: the results to sort (by default with labels).
    :param reverse: reverse the sort (by default True).
    :param key: the key to use to sort (by default the second element of the tuple).
    :return: the list sorted.
    """
    return sorted(results, key=key, reverse=reverse)


# ############################## EXAMPLE ##############################
if __name__ == '__main__':
    size = 20
    data = np.random.rand(size, size)
    labels = range(20)

    # Normalised and prepare data
    prepared_data = prepare_data(normalise(data), 1)

    # Rank
    result = rank(prepared_data)
    # Sort results
    result = sort_result(append_labels(isolate_results(result), labels))
    # Show results
    if helpers.plt:
        fig = helpers.plt.figure()
        gs = fig.add_gridspec(2, 2)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])

        show_matrix(data, labels, ax1, fig)
        ax1.set_title("Raw data")
        show_matrix(prepared_data, labels, ax2, fig)
        ax2.set_title("Normalised data")
        show_results_graph(result, subplot=ax3)
        ax3.set_title("Results")
        helpers.plt.show()
    else:
        show_matrix(data, None)
        show_matrix(data, None)
        show_matrix(prepared_data, None)
        show_results_graph(result)
