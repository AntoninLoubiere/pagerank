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

from typing import Callable, Optional, Union

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    print("WARNING: The module matplotlib isn't installed, charts will be replaced with plots")


def process_data(
        data_string: str,
        has_labels=True,
        convert_data: Callable[[str], int] = lambda x: int(x),
        line_separator: str = '\n',
        column_separator: str = '\t'
) -> tuple[np.ndarray, list[str]]:
    """
    Extract data from a string.
    :param data_string: the string containing the data to extract
    :param has_labels: if the first of the column contains labels
    :param convert_data: a function that process the data extract from the string and return an int
    :param line_separator: the separator of the lines of data
    :param column_separator: the separator of the columns of data
    :return: a tuple that contains the matrix with the data cast and the list of the labels
    """
    assert line_separator != column_separator, "The line separator and the column separator must be different"
    rows = data_string.split(line_separator)
    labels = []
    table = []
    for r in rows:
        row = []
        table.append(row)
        cells = r.split(column_separator)

        if has_labels:
            labels.append(cells.pop(0).strip())
        else:
            labels.append("")

        for c in cells:
            value = 0
            if c:
                try:
                    value = convert_data(c)
                except ValueError:
                    pass

            row.append(value)

    size = len(table)
    for i in range(size):
        assert size <= len(table[i]), f"The data must be a table of {size}x{size}. The line {i} has only " \
                                      f"{len(table[i])} columns."
    return np.array(table), labels


if plt:
    def show_results_graph(
            results: list[Union[int, tuple[int, str]]],
            labels=True,
            precision=3,
            subplot: Optional[plt.Axes] = None
    ) -> None:
        """
        Show an histogram with the results.
        :param results: the results.
        :param labels: show labels on top of bars.
        :param precision: if labels is True, the precision of the label.
        :param subplot: if defined, plot in a sub-figure.
        :return: None.
        """
        if subplot:
            plt.sca(subplot)
            plot = subplot
        else:
            plot = plt
        if results and len(results) > 0 and isinstance(results[0], tuple) and len(results[0]) > 1:
            results_labels, results = zip(*results)
        else:
            results_labels = None

        plot.bar(range(len(results)), results)

        if labels:
            for i, r in enumerate(results):
                plot.text(i, r, f"{r:.{precision}f}", ha="center")

        if results_labels:
            plt.xticks(range(len(results)), results_labels, rotation=90)

        if not subplot:
            plt.show()


    def show_matrix(
            data: np.ndarray,
            labels: list[str],
            subplot: Optional[plt.Axes] = None,
            figure: Optional[plt.Figure] = None
    ) -> None:
        """
        Plot in a heat-map a squared matrix.
        :param data: the matrix to plot.
        :param labels: the labels to append.
        :param subplot: an optional subplot to plot the heat-map.
        :param figure: the figure that hold the subplot.
        :return: None.
        """
        plot = subplot if subplot else plt
        im = plot.imshow(data)
        figure.colorbar(im, ax=subplot) if subplot else plot.colorbar()
        if labels:
            if subplot:
                plt.sca(subplot)
            plt.xticks(range(len(labels)), labels, rotation=90)
            plt.yticks(range(len(labels)), labels)

        if not subplot:
            plot.show()

else:
    from math import log


    def show_results_graph(
            results: list[Union[int, tuple[str, int]]], *_, **__
    ) -> None:
        """
        Print results.
        :param results: the results.
        :return: None.
        """
        number_just = int(log(len(results), 10) + 1)
        label_just = len(max(results, key=lambda x: len(x[0]))[0])
        print('\n'.join(
            f"{str(i + 1).ljust(number_just)} - {e[0].ljust(label_just)} ({e[1]:.15f})" for i, e in enumerate(results)
        ))


    def show_matrix(
            data: np.ndarray, *_, **__
    ) -> None:
        """
        Show a matrix.
        :param data: the matrix to plot.
        :return: None.
        """
        print(data)
