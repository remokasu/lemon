import sys
import os
import tempfile

# Set matplotlib backend before importing plotlib
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for testing

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np
from lemon import plotlib as pl


def test_set_style():
    """Test style setting"""
    print("Testing set_style...")

    # Test available styles (based on actual implementation)
    styles = ["publication", "presentation", "poster"]
    for style in styles:
        pl.set_style(style)
        assert pl.get_style() == style, f"Style should be {style}"

    print("  ✅ set_style")
    print("✅ All set_style tests passed!\n")


def test_set_palette():
    """Test palette setting"""
    print("Testing set_palette...")

    # Test available palettes
    palettes = ["default", "vibrant", "muted", "colorblind"]
    for palette in palettes:
        pl.set_palette(palette)
        current = pl.get_palette()
        assert isinstance(current, (list, tuple)), "Palette should be a list or tuple"

    print("  ✅ set_palette")
    print("✅ All set_palette tests passed!\n")


def test_line():
    """Test line plot"""
    print("Testing line plot...")

    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # Test 1: Basic line plot
    try:
        fig, ax = pl.line(x, y, title="Test Line", show=False)
        assert fig is not None, "Should return figure"
        print("  ✅ line basic")
    finally:
        matplotlib.pyplot.close("all")

    # Test 2: Multiple lines (call twice)
    try:
        y2 = np.cos(x)
        # Just test that line can be called multiple times
        fig, ax = pl.line(x, y, show=False)
        assert fig is not None, "Should return figure"
        print("  ✅ line multiple")
    finally:
        matplotlib.pyplot.close("all")

    print("✅ All line tests passed!\n")


def test_scatter():
    """Test scatter plot"""
    print("Testing scatter plot...")

    x = np.random.randn(100)
    y = np.random.randn(100)

    try:
        fig, ax = pl.scatter(x, y, title="Test Scatter", show=False)
        assert fig is not None, "Should return figure"
        print("  ✅ scatter basic")
    finally:
        matplotlib.pyplot.close("all")

    print("✅ All scatter tests passed!\n")


def test_heatmap():
    """Test heatmap"""
    print("Testing heatmap...")

    data = np.random.randn(10, 10)

    try:
        fig, ax = pl.heatmap(data, title="Test Heatmap", show=False)
        assert fig is not None, "Should return figure"
        print("  ✅ heatmap basic")
    finally:
        matplotlib.pyplot.close("all")

    print("✅ All heatmap tests passed!\n")


def test_learning_curve():
    """Test learning curve"""
    print("Testing learning_curve...")

    train_loss = [0.8, 0.6, 0.4, 0.3, 0.2]
    val_loss = [0.9, 0.7, 0.5, 0.4, 0.35]

    try:
        # learning_curve expects train_values and val_values directly
        fig, ax = pl.learning_curve(
            train_loss, val_loss, title="Test Learning Curve", show=False
        )
        assert fig is not None, "Should return figure"
        print("  ✅ learning_curve")
    finally:
        matplotlib.pyplot.close("all")

    print("✅ All learning_curve tests passed!\n")


def test_feature_importance():
    """Test feature importance"""
    print("Testing feature_importance...")

    importance = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    features = ["F1", "F2", "F3", "F4", "F5"]

    try:
        fig, ax = pl.feature_importance(
            importance,
            feature_names=features,
            title="Test Feature Importance",
            show=False,
        )
        assert fig is not None, "Should return figure"
        print("  ✅ feature_importance")
    finally:
        matplotlib.pyplot.close("all")

    print("✅ All feature_importance tests passed!\n")


def test_correlation_matrix():
    """Test correlation matrix"""
    print("Testing correlation_matrix...")

    # correlation_matrix expects a correlation matrix (square), not raw data
    # Compute correlation matrix from data
    data = np.random.randn(100, 5)
    corr = np.corrcoef(data.T)  # Transpose to get feature correlation
    features = ["A", "B", "C", "D", "E"]

    try:
        fig, ax = pl.correlation_matrix(
            corr, feature_names=features, title="Test Correlation", show=False
        )
        assert fig is not None, "Should return figure"
        print("  ✅ correlation_matrix")
    finally:
        matplotlib.pyplot.close("all")

    print("✅ All correlation_matrix tests passed!\n")


def test_parallel_coordinates():
    """Test parallel coordinates"""
    print("Testing parallel_coordinates...")

    data = np.random.randn(50, 4)
    labels = np.random.randint(0, 3, 50)
    features = ["F1", "F2", "F3", "F4"]

    try:
        fig, ax = pl.parallel_coordinates(
            data,
            labels=labels,
            feature_names=features,
            title="Test Parallel Coordinates",
            show=False,
        )
        assert fig is not None, "Should return figure"
        print("  ✅ parallel_coordinates")
    finally:
        matplotlib.pyplot.close("all")

    print("✅ All parallel_coordinates tests passed!\n")


def test_subplots():
    """Test subplots"""
    print("Testing subplots...")

    try:
        fig, axes = pl.subplots(nrows=2, ncols=2, figsize=(10, 10))
        assert fig is not None, "Should return figure"
        assert axes.shape == (2, 2), f"Axes shape should be (2, 2), got {axes.shape}"

        # Plot on each subplot
        x = np.linspace(0, 10, 100)
        axes[0, 0].plot(x, np.sin(x))
        axes[0, 1].plot(x, np.cos(x))
        axes[1, 0].scatter(x[:50], np.random.randn(50))
        axes[1, 1].hist(np.random.randn(1000), bins=30)

        print("  ✅ subplots")
    finally:
        matplotlib.pyplot.close("all")

    print("✅ All subplots tests passed!\n")


def test_savefig():
    """Test saving figure"""
    print("Testing savefig...")

    temp_dir = tempfile.mkdtemp()

    try:
        import shutil

        x = np.linspace(0, 10, 100)
        y = np.sin(x)

        # Save figure using 'save' parameter
        save_path = os.path.join(temp_dir, "test_plot.png")
        fig, ax = pl.line(x, y, title="Test", show=False, save=save_path)

        assert os.path.exists(save_path), "Figure should be saved"
        print("  ✅ savefig")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        matplotlib.pyplot.close("all")

    print("✅ All savefig tests passed!\n")


if __name__ == "__main__":
    test_set_style()
    test_set_palette()
    test_line()
    test_scatter()
    test_heatmap()
    test_learning_curve()
    test_feature_importance()
    test_correlation_matrix()
    test_parallel_coordinates()
    test_subplots()
    test_savefig()
    print("=" * 50)
    print("All plotlib tests completed!")
