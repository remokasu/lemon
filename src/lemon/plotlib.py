"""
plotlib - 機械学習とデータ可視化のための高機能プロットライブラリ

このモジュールは、NumPyと組み合わせて使用する機械学習向けの可視化機能を提供します。
publication-readyなグラフの作成、学習曲線の可視化、特徴量の重要度分析、
決定境界の表示など、機械学習ワークフローに特化した機能を提供します。

主な機能
--------
- **基本プロット**: line, scatter, heatmap など
- **機械学習可視化**: learning_curve, feature_importance, decision_boundary
- **高度な可視化**: embedding (t-SNE/UMAP), parallel_coordinates, surface_3d
- **スタイル管理**: publication, seaborn, ggplot などのプリセットスタイル
- **拡張可能**: カスタムプロッターの登録が可能

基本的な使い方
--------------
>>> import plotlib as pl
>>> import numpy as np
>>>
>>> # 簡単な折れ線グラフ
>>> x = np.linspace(0, 10, 100)
>>> y = np.sin(x)
>>> pl.line(x, y, title="Sine Wave")
>>>
>>> # 学習曲線の可視化
>>> train_loss = [0.5, 0.3, 0.2, 0.15, 0.12]
>>> val_loss = [0.6, 0.4, 0.3, 0.25, 0.23]
>>> pl.learning_curve(train_loss, val_loss,
...                   metrics=['loss'],
...                   title="Training Progress")
>>>
>>> # スタイルの設定
>>> pl.set_style('publication')
>>> pl.set_palette('vibrant')

スタイルとテーマ
----------------
利用可能なスタイル:
- 'publication': 論文用の高品質なスタイル（デフォルト）
- 'seaborn': Seaborn風のスタイル
- 'ggplot': ggplot2風のスタイル
- 'dark': ダークテーマ

カラーパレット:
- 'default': Matplotlibデフォルト
- 'vibrant': 鮮やかな配色
- 'muted': 落ち着いた配色
- 'colorblind': 色覚異常に配慮した配色

拡張機能
--------
カスタムプロッターの登録:

>>> from plotlib import BasePlotter, register_plotter
>>>
>>> class MyCustomPlotter(BasePlotter):
...     def plot(self, ax, x, y, **kwargs):
...         ax.plot(x, y, **kwargs)
...         return ax
>>>
>>> register_plotter('custom', MyCustomPlotter)
>>> pl.plot(x, y, kind='custom')

Examples
--------
複雑な可視化の例:

>>> # 決定境界の可視化
>>> from sklearn.datasets import make_moons
>>> X, y = make_moons(n_samples=200, noise=0.2)
>>> model = train_classifier(X, y)
>>> pl.decision_boundary(model, X, y,
...                      title="Decision Boundary",
...                      resolution=100)
>>>
>>> # 特徴量の重要度
>>> importance = [0.3, 0.25, 0.2, 0.15, 0.1]
>>> features = ['feat1', 'feat2', 'feat3', 'feat4', 'feat5']
>>> pl.feature_importance(importance, features,
...                       title="Feature Importance")
>>>
>>> # 埋め込みの可視化
>>> embeddings = np.random.randn(100, 50)  # 高次元データ
>>> labels = np.random.randint(0, 3, 100)
>>> pl.embedding(embeddings, labels=labels,
...              method='tsne',
...              title="t-SNE Embedding")

Notes
-----
- このライブラリはMatplotlibをベースにしています
- t-SNE/UMAPによる埋め込み可視化にはscikit-learnが必要です
- UMAP機能を使用するには、umap-learnパッケージが必要です
- すべての関数はNumPy配列を受け付けます

See Also
--------
numpy : 数値計算ライブラリ
matplotlib : プロットのベースライブラリ
scikit-learn : 機械学習ライブラリ

References
----------
.. [1] Hunter, J. D. (2007). Matplotlib: A 2D graphics environment.
       Computing in Science & Engineering, 9(3), 90-95.
"""

__all__ = [
    "plot",
    "subplot",
    "subplots",
    "multiplot",
    "line",
    "scatter",
    "heatmap",
    "parallel_coordinates",
    "learning_curve",
    "feature_importance",
    "correlation_matrix",
    "embedding",
    "decision_boundary",
    "surface_3d",
    "DataAdapter",
    "BasePlotter",
    "PlotRegistry",
    "LinePlotter",
    "ScatterPlotter",
    "HeatmapPlotter",
    "ParallelCoordinatesPlotter",
    "LearningCurvePlotter",
    "FeatureImportancePlotter",
    "CorrelationMatrixPlotter",
    "EmbeddingPlotter",
    "DecisionBoundaryPlotter",
    "Surface3DPlotter",
    "PlotResult",
    "PublicationStyle",
    "VizEngine",
    "register_plotter",
    "set_style",
    "get_style",
    "set_defaults",
    "set_palette",
    "get_palette",
    "VizConfig",
]

__version__ = "0.1.0"

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.ndimage import uniform_filter1d
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import auc

try:
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    umap = None


class VizConfig:
    """可視化のグローバル設定を管理するクラス。

    デフォルト値やカラーパレットの設定を保持し、
    プロジェクト全体で一貫した可視化スタイルを提供します。
    """

    _defaults = {
        "show": True,
        "save_dir": None,
        "dpi": 300,
        "figsize": (10, 6),
        "style": "publication",
    }
    _palettes = {
        "default": [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ],
        "vibrant": [
            "#EE7733",
            "#0077BB",
            "#33BBEE",
            "#EE3377",
            "#CC3311",
            "#009988",
            "#BBBBBB",
        ],
        "muted": [
            "#88CCEE",
            "#44AA99",
            "#117733",
            "#332288",
            "#DDCC77",
            "#999933",
            "#CC6677",
            "#882255",
        ],
        "colorblind": [
            "#0173B2",
            "#DE8F05",
            "#029E73",
            "#CC78BC",
            "#CA9161",
            "#949494",
            "#ECE133",
            "#56B4E9",
        ],
        "pastel": [
            "#FDBCB4",
            "#B3E5FC",
            "#C5E1A5",
            "#FFE082",
            "#CE93D8",
            "#FFCCBC",
            "#F48FB1",
            "#80DEEA",
        ],
    }
    _current_palette = "default"

    @classmethod
    def set_defaults(cls, **kwargs):
        """デフォルト設定を更新する。

        Args:
            **kwargs: 更新する設定のキーと値のペア

        Raises:
            ValueError: 未知の設定キーが指定された場合
        """
        for key, value in kwargs.items():
            if key in cls._defaults:
                cls._defaults[key] = value
            else:
                raise ValueError(f"Unknown setting: {key}")

    @classmethod
    def get_default(cls, key):
        """デフォルト設定の値を取得する。

        Args:
            key: 設定のキー

        Returns:
            設定値、存在しない場合はNone
        """
        return cls._defaults.get(key)

    @classmethod
    def set_palette(cls, palette):
        """カラーパレットを設定する。

        Args:
            palette: パレット名（str）またはカラーリスト（list/tuple）

        Raises:
            ValueError: 未知のパレット名が指定された場合
            TypeError: 不正な型が指定された場合
        """
        if isinstance(palette, str):
            if palette not in cls._palettes:
                raise ValueError(
                    f"Unknown palette: {palette}. Available: {list(cls._palettes.keys())}"
                )
            cls._current_palette = palette
        elif isinstance(palette, (list, tuple)):
            cls._palettes["custom"] = list(palette)
            cls._current_palette = "custom"
        else:
            raise TypeError("palette must be str or list")

    @classmethod
    def get_palette(cls):
        """現在のカラーパレットを取得する。

        Returns:
            カラーコードのリスト
        """
        return cls._palettes[cls._current_palette]

    @classmethod
    def get_color(cls, index):
        """パレットから指定インデックスの色を取得する。

        Args:
            index: カラーインデックス

        Returns:
            カラーコード（str）
        """
        palette = cls.get_palette()
        return palette[index % len(palette)]

    @classmethod
    def reset(cls):
        """設定をデフォルトにリセットする。"""
        cls._current_palette = "default"


def set_defaults(**kwargs):
    """デフォルト設定を更新する。

    Args:
        **kwargs: 更新する設定のキーと値のペア
    """
    VizConfig.set_defaults(**kwargs)


def set_palette(palette):
    """カラーパレットを設定する。

    Args:
        palette: パレット名またはカラーリスト
    """
    VizConfig.set_palette(palette)


def get_palette():
    """現在のカラーパレットを取得する。

    Returns:
        カラーコードのリスト
    """
    return VizConfig.get_palette()


class MultiPlot:
    """複数のサブプロットを管理するコンテキストマネージャ。

    with構文を使用して複数のプロットを配置し、
    一括で管理することができます。
    """

    def __init__(self, nrows=1, ncols=1, figsize=None, **kwargs):
        """MultiPlotインスタンスを初期化する。

        Args:
            nrows: 行数
            ncols: 列数
            figsize: 図のサイズ (width, height)
            **kwargs: subplotsに渡す追加のキーワード引数
        """
        self.nrows = nrows
        self.ncols = ncols
        self.figsize = figsize or VizConfig.get_default("figsize")
        self.kwargs = kwargs
        self.fig = None
        self.axes = None

    def __enter__(self):
        self.fig, self.axes = subplots(
            self.nrows, self.ncols, figsize=self.figsize, **self.kwargs
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.fig.tight_layout()

    def _get_ax(self, pos):
        """指定位置のAxesオブジェクトを取得する。

        Args:
            pos: (row, col)のタプル

        Returns:
            Axesオブジェクト
        """
        if self.nrows == 1 and self.ncols == 1:
            return self.axes
        elif self.nrows == 1 or self.ncols == 1:
            return self.axes[pos[0] if self.nrows > 1 else pos[1]]
        else:
            return self.axes[pos[0], pos[1]]

    def line(self, *args, pos=(0, 0), title=None, **kwargs):
        """指定位置に折れ線グラフを描画する。

        Args:
            *args: データ
            pos: 描画位置 (row, col)
            title: サブプロットのタイトル
            **kwargs: LinePlotterに渡す追加引数
        """
        ax = self._get_ax(pos)
        LinePlotter().plot(ax, *args, **kwargs)
        if title:
            ax.set_title(title)

    def scatter(self, x, y=None, pos=(0, 0), title=None, **kwargs):
        """指定位置に散布図を描画する。

        Args:
            x: x座標データまたは2次元データ
            y: y座標データ（省略可）
            pos: 描画位置 (row, col)
            title: サブプロットのタイトル
            **kwargs: ScatterPlotterに渡す追加引数
        """
        ax = self._get_ax(pos)
        if y is not None:
            # Convert to numpy arrays before stacking
            x = DataAdapter.to_numpy(x)
            y = DataAdapter.to_numpy(y)
            data = np.column_stack([x, y])
        else:
            data = x
        ScatterPlotter().plot(ax, data, **kwargs)
        if title:
            ax.set_title(title)

    def heatmap(self, data, pos=(0, 0), title=None, **kwargs):
        """指定位置にヒートマップを描画する。

        Args:
            data: 2次元データ
            pos: 描画位置 (row, col)
            title: サブプロットのタイトル
            **kwargs: HeatmapPlotterに渡す追加引数
        """
        ax = self._get_ax(pos)
        HeatmapPlotter().plot(ax, data, **kwargs)
        if title:
            ax.set_title(title)

    def parallel_coordinates(self, data, labels=None, pos=(0, 0), title=None, **kwargs):
        """指定位置に並行座標プロットを描画する。

        Args:
            data: 2次元データ
            labels: クラスラベル配列（省略可）
            pos: 描画位置 (row, col)
            title: サブプロットのタイトル
            **kwargs: ParallelCoordinatesPlotterに渡す追加引数
        """
        ax = self._get_ax(pos)
        ParallelCoordinatesPlotter().plot(ax, data, labels=labels, **kwargs)
        if title:
            ax.set_title(title)

    def save(self, filename, dpi=None):
        """図を保存する。

        Args:
            filename: 保存先ファイル名
            dpi: 解像度
        """
        dpi = dpi or VizConfig.get_default("dpi")
        self.fig.savefig(filename, dpi=dpi, bbox_inches="tight")

    def show(self):
        """図を表示する。"""
        import matplotlib.pyplot as plt

        plt.show()


def multiplot(nrows=1, ncols=1, figsize=None, **kwargs):
    """MultiPlotインスタンスを作成する。

    Args:
        nrows: 行数
        ncols: 列数
        figsize: 図のサイズ
        **kwargs: subplotsに渡す追加引数

    Returns:
        MultiPlotインスタンス

    Examples:
        >>> import numpy as np
        >>> from plotlib import multiplot
        >>>
        >>> # 2x2のサブプロット
        >>> with multiplot(2, 2, figsize=(12, 10)) as mp:
        ...     # 各位置にプロット
        ...     mp.line(np.sin(np.linspace(0, 2*np.pi, 100)),
        ...             pos=(0, 0), title="Sin")
        ...     mp.line(np.cos(np.linspace(0, 2*np.pi, 100)),
        ...             pos=(0, 1), title="Cos")
        ...     mp.scatter(np.random.randn(50, 2),
        ...                pos=(1, 0), title="Scatter")
        ...     mp.heatmap(np.random.randn(10, 10),
        ...                pos=(1, 1), title="Heatmap")
        ...     mp.save("multi_plot.png")
    """
    return MultiPlot(nrows, ncols, figsize, **kwargs)


class VizError(Exception):
    """可視化ライブラリのエラー基底クラス。"""

    pass


class DataAdapter:
    """様々なデータ型をnumpy配列に変換するアダプタクラス。

    numlibのNumType、numpy配列、リストなど様々なデータ型を
    統一的に扱うための変換機能を提供します。
    """

    @staticmethod
    def to_numpy(data: Any) -> np.ndarray:
        """データをnumpy配列に変換する。

        Args:
            data: 変換するデータ（NumType、ndarray、list等）

        Returns:
            numpy配列

        Raises:
            ValueError: 変換できないデータ型の場合
        """
        if hasattr(data, "to_numpy") and callable(data.to_numpy):
            return data.to_numpy()
        if isinstance(data, np.ndarray):
            return data
        try:
            return np.asarray(data)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Cannot convert data of type '{type(data).__name__}' to numpy array. Supported types: numpy.ndarray, numlib.NumType, list, tuple. Error: {e}"
            )

    @staticmethod
    def get_metadata(data: Any) -> Dict[str, Any]:
        """データからメタデータを抽出する。

        Args:
            data: メタデータを持つ可能性のあるデータ

        Returns:
            メタデータの辞書
        """
        metadata = {}
        if hasattr(data, "name"):
            metadata["name"] = data.name
        if hasattr(data, "shape_names"):
            metadata["shape_names"] = data.shape_names
        if hasattr(data, "grad"):
            metadata["has_grad"] = data.grad is not None
        if hasattr(data, "requires_grad"):
            metadata["requires_grad"] = data.requires_grad
        if hasattr(data, "metadata") and isinstance(data.metadata, dict):
            metadata.update(data.metadata)
        return metadata

    @staticmethod
    def get_device_info(data: Any) -> Dict[str, Any]:
        """データのデバイス情報を取得する。

        Args:
            data: デバイス情報を持つ可能性のあるデータ

        Returns:
            デバイス情報の辞書 {'device': 'cpu'/'cuda', 'device_id': int or None}
        """
        if hasattr(data, "device"):
            device_str = str(data.device).lower()
            if "cuda" in device_str or "gpu" in device_str:
                try:
                    if ":" in device_str:
                        device_id = int(device_str.split(":")[1])
                    else:
                        device_id = 0
                except (ValueError, IndexError, AttributeError):
                    device_id = 0
                return {"device": "cuda", "device_id": device_id}
            return {"device": "cpu", "device_id": None}
        if hasattr(data, "as_cupy") and callable(data.as_cupy):
            try:
                data.as_cupy()
                return {"device": "cuda", "device_id": 0}
            except (AttributeError, RuntimeError):
                return {"device": "cpu", "device_id": None}
        return {"device": "cpu", "device_id": None}


class BasePlotter(ABC):
    """全てのプロッタークラスの基底クラス。

    各プロッタークラスはこのクラスを継承し、
    plot()メソッドを実装する必要があります。
    """

    plot_type: str = "base"
    supported_dims: Tuple[int, ...] = (1, 2)
    is_3d: bool = False

    def __init__(self, style: str = "publication", dpi: int = 300, **kwargs):
        """BasePlotterインスタンスを初期化する。

        Args:
            style: プロットスタイル
            dpi: 解像度
            **kwargs: 追加設定
        """
        self.style = style
        self.dpi = dpi
        self.config = kwargs

    @abstractmethod
    def plot(self, ax: Axes, data: np.ndarray, **kwargs) -> Any:
        """データを描画する抽象メソッド。

        Args:
            ax: 描画先のAxesオブジェクト
            data: 描画するデータ
            **kwargs: 追加のプロットオプション

        Returns:
            描画オブジェクト

        Raises:
            NotImplementedError: サブクラスで実装されていない場合
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.plot() must be implemented"
        )

    def _plot_with_class_labels(
        self,
        ax: Axes,
        data: np.ndarray,
        class_labels: np.ndarray,
        feature_labels: List[str],
        class_names: Optional[List[str]],
        alpha: float,
        linewidth: float,
        normalize: bool,
        **kwargs,
    ) -> Axes:
        """クラスラベル付きデータを並行座標プロットで描画する（内部用）。

        Args:
            ax: 描画先のAxesオブジェクト
            data: データ配列
            class_labels: クラスラベル配列
            feature_labels: 特徴量名のリスト
            class_names: クラス名のリスト
            alpha: 透明度
            linewidth: 線幅
            normalize: 正規化するかどうか
            **kwargs: 追加のプロットオプション

        Returns:
            Axesオブジェクト
        """
        n_samples, n_features = data.shape
        data_mins = data.min(axis=0)
        data_maxs = data.max(axis=0)
        data_ranges = data_maxs - data_mins
        data_normalized = np.zeros_like(data, dtype=float)
        for i in range(n_features):
            if data_ranges[i] > 0:
                data_normalized[:, i] = (data[:, i] - data_mins[i]) / data_ranges[i]
            else:
                data_normalized[:, i] = 0.5
        x_positions = np.arange(n_features)
        unique_classes = np.unique(class_labels)
        n_classes = len(unique_classes)
        class_to_color = {
            cls: VizConfig.get_color(i) for i, cls in enumerate(unique_classes)
        }
        if class_names is None:
            class_names = [f"Class {int(cls)}" for cls in unique_classes]
        elif len(class_names) != n_classes:
            class_names = [f"Class {int(cls)}" for cls in unique_classes]
        for cls_idx, cls in enumerate(unique_classes):
            mask = class_labels == cls
            class_data = data_normalized[mask]
            segments = []
            for i in range(len(class_data)):
                points = np.array([x_positions, class_data[i, :]]).T
                segments.append(points)
            if segments:
                lc = LineCollection(
                    segments,
                    colors=class_to_color[cls],
                    linewidths=linewidth,
                    alpha=alpha,
                    label=class_names[cls_idx],
                )
                ax.add_collection(lc)
        ax.set_xlim(-0.5, n_features - 0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(feature_labels, rotation=45, ha="right")
        ax.set_yticks([])
        ax.set_ylabel("")
        for i, x in enumerate(x_positions):
            ax.axvline(x, color="gray", linewidth=1.0, alpha=0.3, zorder=0)
            n_ticks = 5
            y_positions = np.linspace(0, 1, n_ticks)
            if data_ranges[i] > 0:
                original_values = np.linspace(data_mins[i], data_maxs[i], n_ticks)
            else:
                original_values = np.full(n_ticks, data_mins[i])
            if data_ranges[i] > 1000:
                fmt = "{:.0f}"
            elif data_ranges[i] > 10:
                fmt = "{:.1f}"
            else:
                fmt = "{:.2f}"
            for y_pos, orig_val in zip(y_positions, original_values):
                label_text = fmt.format(orig_val)
                if i == 0:
                    ax.text(
                        x - 0.08,
                        y_pos,
                        label_text,
                        ha="right",
                        va="center",
                        fontsize=8,
                        color="#555555",
                        alpha=0.8,
                    )
                else:
                    ax.text(
                        x + 0.08,
                        y_pos,
                        label_text,
                        ha="left",
                        va="center",
                        fontsize=8,
                        color="#555555",
                        alpha=0.8,
                    )
        ax.legend(loc="upper right", framealpha=0.9)
        return ax

    def prepare_axes(self, ax: Axes, **kwargs):
        """Axesの見た目を整える。

        Args:
            ax: 整形するAxesオブジェクト
            **kwargs: グリッド表示等の追加オプション
        """
        if self.style == "publication":
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if kwargs.get("grid", True):
                ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
            ax.tick_params(direction="in", top=False, right=False)

    def quick_plot(
        self,
        *args,
        figsize: Tuple[float, float] = (10, 6),
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        show: bool = True,
        save: Optional[str] = None,
        dpi: int = 300,
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """簡単にプロットを作成するヘルパーメソッド。

        Args:
            *args: プロットするデータ
            figsize: 図のサイズ
            title: タイトル
            xlabel: x軸ラベル
            ylabel: y軸ラベル
            show: 表示するかどうか
            save: 保存先ファイル名（省略可）
            dpi: 解像度
            **kwargs: プロッタークラスのplot()に渡す追加引数

        Returns:
            (Figure, Axes)のタプル
        """
        fig, ax = plt.subplots(figsize=figsize)
        try:
            if len(args) == 1:
                self.plot(ax, args[0], **kwargs)
            elif len(args) == 2:
                self.plot(ax, args[0], args[1], **kwargs)
            elif len(args) == 3:
                self.plot(ax, args[0], args[1], args[2], **kwargs)
            else:
                self.plot(ax, *args, **kwargs)
        except TypeError:
            for data in args:
                self.plot(ax, data, **kwargs)
        if title:
            ax.set_title(title, fontsize=14, fontweight="bold")
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=11)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=11)
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        return (fig, ax)

    @classmethod
    def can_handle(cls, data: np.ndarray) -> bool:
        """データの次元がサポートされているか確認する。

        Args:
            data: 確認するデータ

        Returns:
            サポートされている場合True
        """
        return data.ndim in cls.supported_dims

    def validate_data(self, data: np.ndarray):
        """データの妥当性を検証する。

        Args:
            data: 検証するデータ

        Raises:
            TypeError: データ型が不正な場合
            ValueError: データの次元やサイズが不正な場合
        """
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Data must be numpy.ndarray, got {type(data).__name__}")
        if not self.can_handle(data):
            raise ValueError(
                f"{self.__class__.__name__} cannot handle {data.ndim}D data. Supported dimensions: {self.supported_dims}"
            )
        if data.size == 0:
            raise ValueError("Data is empty")
        if not np.isfinite(data).all():
            raise ValueError("Data contains NaN or Inf values")


class PlotRegistry:
    """プロッタークラスを登録・管理するレジストリ。

    プロット種別とプロッタークラスの対応を管理し、
    適切なプロッタークラスを取得する機能を提供します。
    """

    _registry: Dict[str, Type[BasePlotter]] = {}

    @classmethod
    def register(cls, plot_type: str, plotter_class: Type[BasePlotter]):
        """プロッタークラスを登録する。

        Args:
            plot_type: プロット種別の識別子
            plotter_class: 登録するプロッタークラス

        Raises:
            ValueError: plot_typeが空の場合
            TypeError: plotter_classがBasePlotterのサブクラスでない場合
        """
        if not plot_type:
            raise ValueError("plot_type cannot be empty")
        if not issubclass(plotter_class, BasePlotter):
            raise TypeError("plotter_class must be a subclass of BasePlotter")
        cls._registry[plot_type] = plotter_class

    @classmethod
    def get(cls, plot_type: str) -> Optional[Type[BasePlotter]]:
        """指定されたプロット種別のプロッタークラスを取得する。

        Args:
            plot_type: プロット種別

        Returns:
            プロッタークラス、存在しない場合はNone
        """
        return cls._registry.get(plot_type)

    @classmethod
    def find_suitable_plotter(cls, data: np.ndarray) -> Optional[Type[BasePlotter]]:
        """データに適したプロッタークラスを検索する。

        Args:
            data: プロットするデータ

        Returns:
            適したプロッタークラス、見つからない場合はNone
        """
        for plotter_class in cls._registry.values():
            if plotter_class.can_handle(data):
                return plotter_class
        return None

    @classmethod
    def list_plotters(cls) -> Dict[str, Type[BasePlotter]]:
        """登録されている全てのプロッタークラスを取得する。

        Returns:
            プロット種別とプロッタークラスの辞書
        """
        return cls._registry.copy()

    @classmethod
    def unregister(cls, plot_type: str) -> bool:
        """プロッタークラスの登録を解除する。

        Args:
            plot_type: 解除するプロット種別

        Returns:
            解除に成功した場合True
        """
        if plot_type in cls._registry:
            del cls._registry[plot_type]
            return True
        return False

    @classmethod
    def clear(cls):
        """全ての登録を削除する。"""
        cls._registry.clear()


def register_plotter(plot_type: str):
    """プロッタークラスを登録するデコレータ。

    Args:
        plot_type: プロット種別の識別子

    Returns:
        デコレータ関数
    """

    def decorator(cls: Type[BasePlotter]) -> Type[BasePlotter]:
        PlotRegistry.register(plot_type, cls)
        return cls

    return decorator


class PlotResult:
    """プロット結果を保持し、チェーンメソッドで操作できるクラス。

    プロット結果のFigure、Axes、描画オブジェクトを保持し、
    メソッドチェーンで追加の設定を行うことができます。
    """

    def __init__(self, fig: Figure, ax: Axes, plot_obj: Any):
        """PlotResultインスタンスを初期化する。

        Args:
            fig: Figureオブジェクト
            ax: Axesオブジェクト
            plot_obj: 描画オブジェクト
        """
        self.fig = fig
        self.ax = ax
        self.plot_obj = plot_obj

    def title(self, text: str, **kwargs):
        """タイトルを設定する。

        Args:
            text: タイトルテキスト
            **kwargs: set_titleに渡す追加引数

        Returns:
            selfを返してメソッドチェーン可能
        """
        self.ax.set_title(text, **kwargs)
        return self

    def xlabel(self, text: str, **kwargs):
        """x軸ラベルを設定する。

        Args:
            text: ラベルテキスト
            **kwargs: set_xlabelに渡す追加引数

        Returns:
            selfを返してメソッドチェーン可能
        """
        self.ax.set_xlabel(text, **kwargs)
        return self

    def ylabel(self, text: str, **kwargs):
        """y軸ラベルを設定する。

        Args:
            text: ラベルテキスト
            **kwargs: set_ylabelに渡す追加引数

        Returns:
            selfを返してメソッドチェーン可能
        """
        self.ax.set_ylabel(text, **kwargs)
        return self

    def legend(self, **kwargs):
        """凡例を追加する。

        Args:
            **kwargs: legendに渡す追加引数

        Returns:
            selfを返してメソッドチェーン可能
        """
        self.ax.legend(**kwargs)
        return self

    def grid(self, visible: bool = True, **kwargs):
        """グリッドの表示を設定する。

        Args:
            visible: グリッドを表示するかどうか
            **kwargs: gridに渡す追加引数

        Returns:
            selfを返してメソッドチェーン可能
        """
        self.ax.grid(visible, **kwargs)
        return self

    def xlim(self, left=None, right=None):
        """x軸の範囲を設定する。

        Args:
            left: 左端の値
            right: 右端の値

        Returns:
            selfを返してメソッドチェーン可能
        """
        self.ax.set_xlim(left, right)
        return self

    def ylim(self, bottom=None, top=None):
        """y軸の範囲を設定する。

        Args:
            bottom: 下端の値
            top: 上端の値

        Returns:
            selfを返してメソッドチェーン可能
        """
        self.ax.set_ylim(bottom, top)
        return self

    def save(self, filename: str, **kwargs):
        """図をファイルに保存する。

        Args:
            filename: 保存先ファイル名
            **kwargs: savefigに渡す追加引数

        Returns:
            selfを返してメソッドチェーン可能
        """
        self.fig.savefig(filename, **kwargs)
        return self

    def show(self):
        """図を表示する。"""
        plt.show()


@register_plotter("line")
class LinePlotter(BasePlotter):
    """折れ線グラフを描画するプロッタークラス。

    時系列データや関数のプロットに適しています。
    複数系列の同時描画や、平滑化処理もサポートします。
    """

    plot_type = "line"
    supported_dims = (1, 2)

    def plot(
        self,
        ax: Axes,
        *args,
        labels: Optional[List[str]] = None,
        colors: Optional[Union[str, List[str]]] = None,
        linestyles: Optional[Union[str, List[str]]] = None,
        linewidths: Optional[Union[float, List[float]]] = None,
        markers: Optional[Union[str, List[str]]] = None,
        alpha: float = 1.0,
        smooth: Optional[int] = None,
        **kwargs,
    ) -> Union[Line2D, List[Line2D]]:
        """折れ線グラフを描画する。

        Args:
            ax: 描画先のAxesオブジェクト
            *args: データ（1つ以上の配列）
            labels: 各系列のラベル
            colors: 色の指定
            linestyles: 線種の指定
            linewidths: 線幅の指定
            markers: マーカースタイルの指定
            alpha: 透明度 (0-1)
            smooth: 平滑化ウィンドウサイズ（Noneの場合は平滑化しない）
            **kwargs: ax.plotに渡す追加引数

        Returns:
            Line2Dオブジェクトまたはそのリスト
        """
        if len(args) == 0:
            raise ValueError("At least one data array is required")
        if len(args) == 1:
            y = DataAdapter.to_numpy(args[0])
            if y.ndim == 1:
                x = np.arange(len(y))
                datasets = [(x, y)]
            elif y.ndim == 2:
                datasets = [(np.arange(y.shape[0]), y[:, i]) for i in range(y.shape[1])]
            else:
                raise ValueError(f"Data must be 1D or 2D, got {y.ndim}D")
        elif len(args) == 2:
            x = DataAdapter.to_numpy(args[0])
            y = DataAdapter.to_numpy(args[1])
            if x.ndim == 1 and y.ndim == 1:
                datasets = [(x, y)]
            elif x.ndim == 1 and y.ndim == 2:
                datasets = [(x, y[:, i]) for i in range(y.shape[1])]
            else:
                raise ValueError("Invalid data dimensions")
        else:
            datasets = []
            for data in args:
                data = DataAdapter.to_numpy(data)
                if data.ndim == 1:
                    datasets.append((np.arange(len(data)), data))
                elif data.ndim == 2 and data.shape[1] == 2:
                    datasets.append((data[:, 0], data[:, 1]))
                else:
                    raise ValueError(f"Invalid data shape: {data.shape}")
        if colors is None:
            colors = [VizConfig.get_color(i) for i in range(len(datasets))]
        elif isinstance(colors, str):
            colors = [colors] * len(datasets)
        if linestyles is None:
            linestyles = ["-"] * len(datasets)
        elif isinstance(linestyles, str):
            linestyles = [linestyles] * len(datasets)
        if linewidths is None:
            linewidths = [1.5] * len(datasets)
        elif isinstance(linewidths, (int, float)):
            linewidths = [linewidths] * len(datasets)
        if markers is None:
            markers = [None] * len(datasets)
        elif isinstance(markers, str):
            markers = [markers] * len(datasets)
        if labels is None:
            labels = [None] * len(datasets)
        lines = []
        for i, (x_data, y_data) in enumerate(datasets):
            if smooth is not None and smooth > 1:
                y_data = uniform_filter1d(y_data, size=smooth, mode="nearest")
            (line,) = ax.plot(
                x_data,
                y_data,
                color=colors[i],
                linestyle=linestyles[i],
                linewidth=linewidths[i],
                marker=markers[i],
                alpha=alpha,
                label=labels[i],
                **kwargs,
            )
            lines.append(line)
        self.prepare_axes(ax)
        if any(label is not None for label in labels):
            ax.legend()
        return lines[0] if len(lines) == 1 else lines


@register_plotter("scatter")
class ScatterPlotter(BasePlotter):
    """散布図を描画するプロッタークラス。

    2次元データの分布を可視化します。
    クラスラベルによる色分けや、サイズ・透明度の変更もサポートします。
    """

    plot_type = "scatter"
    supported_dims = (2,)

    def plot(
        self,
        ax: Axes,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        colors: Optional[Union[str, np.ndarray]] = None,
        sizes: Optional[Union[float, np.ndarray]] = None,
        alpha: float = 0.6,
        marker: str = "o",
        cmap: str = "tab10",
        edgecolors: Optional[str] = None,
        linewidths: float = 0,
        class_names: Optional[List[str]] = None,
        **kwargs,
    ) -> Any:
        """散布図を描画する。

        Args:
            ax: 描画先のAxesオブジェクト
            data: 2次元データ (n_samples, 2)
            labels: クラスラベル配列（省略可）
            colors: 色の指定
            sizes: マーカーサイズ
            alpha: 透明度 (0-1)
            marker: マーカースタイル
            cmap: カラーマップ
            edgecolors: エッジの色
            linewidths: エッジの線幅
            class_names: クラス名のリスト（省略可）
            **kwargs: ax.scatterに渡す追加引数

        Returns:
            PathCollectionオブジェクト
        """
        data = DataAdapter.to_numpy(data)
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError(f"Data must be (n, 2) shaped, got {data.shape}")
        if sizes is None:
            sizes = 50
        if labels is not None:
            labels = DataAdapter.to_numpy(labels)
            if colors is None:
                scatter = ax.scatter(
                    data[:, 0],
                    data[:, 1],
                    c=labels,
                    s=sizes,
                    alpha=alpha,
                    marker=marker,
                    cmap=cmap,
                    edgecolors=edgecolors,
                    linewidths=linewidths,
                    **kwargs,
                )
                unique_labels = np.unique(labels)
                if len(unique_labels) <= 10:
                    # class_namesが指定されている場合はそれを使用
                    if class_names is None:
                        class_names = [f"Class {int(label)}" for label in unique_labels]
                    elif len(class_names) != len(unique_labels):
                        class_names = [f"Class {int(label)}" for label in unique_labels]

                    for idx, label in enumerate(unique_labels):
                        mask = labels == label
                        ax.scatter(
                            [],
                            [],
                            c=[plt.cm.get_cmap(cmap)(label / max(unique_labels))],
                            label=class_names[idx],
                        )
                    ax.legend()
            else:
                scatter = ax.scatter(
                    data[:, 0],
                    data[:, 1],
                    c=colors,
                    s=sizes,
                    alpha=alpha,
                    marker=marker,
                    edgecolors=edgecolors,
                    linewidths=linewidths,
                    **kwargs,
                )
        else:
            if colors is None:
                colors = VizConfig.get_color(0)
            scatter = ax.scatter(
                data[:, 0],
                data[:, 1],
                c=colors,
                s=sizes,
                alpha=alpha,
                marker=marker,
                edgecolors=edgecolors,
                linewidths=linewidths,
                **kwargs,
            )
        self.prepare_axes(ax)
        return scatter


@register_plotter("heatmap")
class HeatmapPlotter(BasePlotter):
    """ヒートマップを描画するプロッタークラス。

    行列データや相関係数行列などを色で可視化します。
    カラーバーや注釈の表示もサポートします。
    """

    plot_type = "heatmap"
    supported_dims = (2,)

    def plot(
        self,
        ax: Axes,
        data: np.ndarray,
        cmap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        center: Optional[float] = None,
        annot: bool = False,
        fmt: str = ".2f",
        annot_kws: Optional[Dict] = None,
        cbar: bool = True,
        cbar_kws: Optional[Dict] = None,
        square: bool = False,
        xticklabels: Optional[Union[bool, List[str]]] = None,
        yticklabels: Optional[Union[bool, List[str]]] = None,
        **kwargs,
    ) -> Any:
        """ヒートマップを描画する。

        Args:
            ax: 描画先のAxesオブジェクト
            data: 2次元データ
            cmap: カラーマップ
            vmin: カラースケールの最小値
            vmax: カラースケールの最大値
            center: カラースケールの中心値
            annot: 値を注釈として表示するかどうか
            fmt: 注釈のフォーマット文字列
            annot_kws: 注釈のスタイル設定
            cbar: カラーバーを表示するかどうか
            cbar_kws: カラーバーの設定
            square: セルを正方形にするかどうか
            xticklabels: x軸のラベル
            yticklabels: y軸のラベル
            **kwargs: ax.imshowに渡す追加引数

        Returns:
            AxesImageオブジェクト
        """
        data = DataAdapter.to_numpy(data)
        if data.ndim != 2:
            raise ValueError(f"Data must be 2D, got {data.ndim}D")
        if center is not None:
            vmax_abs = max(abs(data.min() - center), abs(data.max() - center))
            vmin = center - vmax_abs
            vmax = center + vmax_abs
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", **kwargs)
        if square:
            ax.set_aspect("equal")
        if annot:
            if annot_kws is None:
                annot_kws = {}
            annot_kws.setdefault("ha", "center")
            annot_kws.setdefault("va", "center")
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    text = ax.text(j, i, format(data[i, j], fmt), **annot_kws)
        if xticklabels is not None:
            if isinstance(xticklabels, bool):
                if not xticklabels:
                    ax.set_xticks([])
            else:
                ax.set_xticks(np.arange(len(xticklabels)))
                ax.set_xticklabels(xticklabels, rotation=45, ha="right")
        if yticklabels is not None:
            if isinstance(yticklabels, bool):
                if not yticklabels:
                    ax.set_yticks([])
            else:
                ax.set_yticks(np.arange(len(yticklabels)))
                ax.set_yticklabels(yticklabels)
        if cbar:
            if cbar_kws is None:
                cbar_kws = {}
            plt.colorbar(im, ax=ax, **cbar_kws)
        self.prepare_axes(ax, grid=False)
        return im


@register_plotter("parallel")
class ParallelCoordinatesPlotter(BasePlotter):
    """並行座標プロットを描画するプロッタークラス。

    多次元データを2次元平面上で可視化します。
    各特徴量を縦軸として並べ、各サンプルを線で結ぶことで、
    高次元空間でのパターンや相関を視覚的に把握できます。

    Examples:
        >>> import numpy as np
        >>> from plotlib import ParallelCoordinatesPlotter, subplots
        >>>
        >>> # 多次元データの準備
        >>> data = np.random.randn(100, 6)
        >>> labels = np.random.randint(0, 3, 100)
        >>> feature_names = ["F1", "F2", "F3", "F4", "F5", "F6"]
        >>>
        >>> # クラス別に色分けして可視化
        >>> plotter = ParallelCoordinatesPlotter()
        >>> fig, ax = subplots(figsize=(12, 6))
        >>> plotter.plot(ax, data, labels=labels,
        ...              feature_names=feature_names,
        ...              class_names=["A", "B", "C"])
        >>> ax.set_title("Parallel Coordinates Plot")
    """

    plot_type = "parallel"
    supported_dims = (2,)

    def plot(
        self,
        ax: Axes,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        color_by: Optional[np.ndarray] = None,
        cmap: str = "viridis",
        alpha: float = 0.6,
        linewidth: float = 1.0,
        normalize: bool = True,
        highlight_indices: Optional[List[int]] = None,
        highlight_color: str = "red",
        highlight_linewidth: float = 2.0,
        **kwargs,
    ) -> Any:
        """並行座標プロットを描画する。

        Args:
            ax: 描画先のAxesオブジェクト
            data: 2次元データ (n_samples, n_features)
            labels: クラスラベル配列（省略可）
            feature_names: 特徴量名のリスト
            class_names: クラス名のリスト
            color_by: 色分けに使用する値（省略可）
            cmap: カラーマップ
            alpha: 透明度 (0-1)
            linewidth: 線幅
            normalize: 各特徴量を0-1に正規化するかどうか
            highlight_indices: ハイライトするサンプルのインデックスリスト
            highlight_color: ハイライトする線の色
            highlight_linewidth: ハイライトする線の幅
            **kwargs: 追加のプロットオプション

        Returns:
            Axesオブジェクト

        Raises:
            ValueError: データの次元が不正な場合
        """
        # Convert NumType to numpy array
        data = DataAdapter.to_numpy(data)
        if labels is not None:
            labels = DataAdapter.to_numpy(labels)
        if color_by is not None:
            color_by = DataAdapter.to_numpy(color_by)

        if data.ndim != 2:
            raise ValueError(f"Data must be 2D, got {data.ndim}D")
        n_samples, n_features = data.shape
        if feature_names is None:
            feature_names = [f"X{i + 1}" for i in range(n_features)]
        elif len(feature_names) != n_features:
            raise ValueError(
                f"Number of feature_names ({len(feature_names)}) must match number of features ({n_features})"
            )
        if labels is not None:
            if len(labels) != n_samples:
                raise ValueError(
                    f"Number of labels ({len(labels)}) must match number of samples ({n_samples})"
                )
            return self._plot_with_class_labels(
                ax,
                data,
                labels,
                feature_names,
                class_names,
                alpha,
                linewidth,
                normalize,
                **kwargs,
            )
        if normalize:
            data_normalized = np.zeros_like(data)
            for i in range(n_features):
                col = data[:, i]
                col_min, col_max = (col.min(), col.max())
                if col_max - col_min > 0:
                    data_normalized[:, i] = (col - col_min) / (col_max - col_min)
                else:
                    data_normalized[:, i] = 0.5
        else:
            data_normalized = data.copy()
        x_positions = np.arange(n_features)
        if color_by is not None:
            if len(color_by) != n_samples:
                raise ValueError(
                    f"color_by length ({len(color_by)}) must match number of samples ({n_samples})"
                )
            color_values = (color_by - color_by.min()) / (
                color_by.max() - color_by.min()
            )
            colormap = cm.get_cmap(cmap)
            colors = [colormap(val) for val in color_values]
        else:
            colors = ["#2A9D8F"] * n_samples

        segments = []
        segment_colors = []
        for i in range(n_samples):
            if highlight_indices is not None and i in highlight_indices:
                continue
            points = np.array([x_positions, data_normalized[i, :]]).T
            segments.append(points)
            segment_colors.append(colors[i])
        if segments:
            lc = LineCollection(
                segments, colors=segment_colors, linewidths=linewidth, alpha=alpha
            )
            ax.add_collection(lc)
        if highlight_indices is not None:
            highlight_segments = []
            for i in highlight_indices:
                if i < n_samples:
                    points = np.array([x_positions, data_normalized[i, :]]).T
                    highlight_segments.append(points)
            if highlight_segments:
                lc_highlight = LineCollection(
                    highlight_segments,
                    colors=highlight_color,
                    linewidths=highlight_linewidth,
                    alpha=1.0,
                    zorder=10,
                )
                ax.add_collection(lc_highlight)
        ax.set_xlim(-0.5, n_features - 0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(feature_names, rotation=45, ha="right")
        data_mins = data.min(axis=0)
        data_maxs = data.max(axis=0)
        data_ranges = data_maxs - data_mins
        ax.set_yticks([])
        ax.set_ylabel("")
        for idx, x in enumerate(x_positions):
            ax.axvline(x, color="gray", linewidth=1.0, alpha=0.3, zorder=0)
            n_ticks = 5
            y_positions = np.linspace(0, 1, n_ticks)
            if data_ranges[idx] > 0:
                original_values = np.linspace(data_mins[idx], data_maxs[idx], n_ticks)
            else:
                original_values = np.full(n_ticks, data_mins[idx])
            if data_ranges[idx] > 1000:
                fmt = "{:.0f}"
            elif data_ranges[idx] > 10:
                fmt = "{:.1f}"
            else:
                fmt = "{:.2f}"
            for y_pos, orig_val in zip(y_positions, original_values):
                label_text = fmt.format(orig_val)
                if idx == 0:
                    ax.text(
                        x - 0.08,
                        y_pos,
                        label_text,
                        ha="right",
                        va="center",
                        fontsize=8,
                        color="#555555",
                        alpha=0.8,
                    )
                else:
                    ax.text(
                        x + 0.08,
                        y_pos,
                        label_text,
                        ha="left",
                        va="center",
                        fontsize=8,
                        color="#555555",
                        alpha=0.8,
                    )
        if color_by is not None:
            norm = Normalize(vmin=color_by.min(), vmax=color_by.max())
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label("Color Value")
        self.prepare_axes(ax, grid=False)
        return ax


@register_plotter("learning_curve")
class LearningCurvePlotter(BasePlotter):
    """学習曲線を描画するプロッタークラス。

    ニューラルネットワークの訓練時の損失や精度の推移を可視化します。
    訓練データと検証データの曲線を同時に表示できます。

    Examples:
        >>> import numpy as np
        >>> from plotlib import LearningCurvePlotter, subplots
        >>>
        >>> # プロッターを直接使用
        >>> plotter = LearningCurvePlotter()
        >>> fig, ax = subplots()
        >>>
        >>> train_loss = np.array([2.3, 1.8, 1.2, 0.9, 0.7, 0.5])
        >>> val_loss = np.array([2.5, 1.9, 1.4, 1.1, 1.0, 1.0])
        >>>
        >>> plotter.plot(ax, train_loss, val_loss,
        ...              train_label="Train Loss",
        ...              val_label="Val Loss")
        >>> ax.set_title("Training Progress")
    """

    plot_type = "learning_curve"
    supported_dims = (1,)

    def plot(
        self,
        ax: Axes,
        train_values: np.ndarray,
        val_values: Optional[np.ndarray] = None,
        epochs: Optional[np.ndarray] = None,
        train_label: str = "Train",
        val_label: str = "Validation",
        train_color: Optional[str] = None,
        val_color: Optional[str] = None,
        smooth: Optional[int] = None,
        log_scale: bool = False,
        show_best: bool = True,
        **kwargs,
    ) -> Dict[str, Line2D]:
        """学習曲線を描画する。

        Args:
            ax: 描画先のAxesオブジェクト
            train_values: 訓練データの値
            val_values: 検証データの値（省略可）
            epochs: エポック数の配列（省略時は自動生成）
            train_label: 訓練データのラベル
            val_label: 検証データのラベル
            train_color: 訓練データの色
            val_color: 検証データの色
            smooth: 平滑化ウィンドウサイズ
            log_scale: y軸を対数スケールにするかどうか
            show_best: 最良値にマーカーを表示するかどうか
            **kwargs: ax.plotに渡す追加引数

        Returns:
            {'train': Line2D, 'val': Line2D}の辞書
        """
        train_values = DataAdapter.to_numpy(train_values)
        if train_values.ndim != 1:
            raise ValueError(f"train_values must be 1D, got {train_values.ndim}D")
        if epochs is None:
            epochs = np.arange(1, len(train_values) + 1)
        else:
            epochs = DataAdapter.to_numpy(epochs)
        if train_color is None:
            train_color = VizConfig.get_color(0)
        if val_color is None:
            val_color = VizConfig.get_color(1)
        if smooth is not None and smooth > 1:
            train_smoothed = uniform_filter1d(train_values, size=smooth, mode="nearest")
        else:
            train_smoothed = train_values
        (train_line,) = ax.plot(
            epochs,
            train_smoothed,
            label=train_label,
            color=train_color,
            linewidth=2,
            **kwargs,
        )
        lines = {"train": train_line}
        if val_values is not None:
            val_values = DataAdapter.to_numpy(val_values)
            if val_values.ndim != 1:
                raise ValueError(f"val_values must be 1D, got {val_values.ndim}D")
            if len(val_values) != len(train_values):
                raise ValueError(
                    f"Length mismatch: train={len(train_values)}, val={len(val_values)}"
                )
            if smooth is not None and smooth > 1:
                val_smoothed = uniform_filter1d(val_values, size=smooth, mode="nearest")
            else:
                val_smoothed = val_values
            (val_line,) = ax.plot(
                epochs,
                val_smoothed,
                label=val_label,
                color=val_color,
                linewidth=2,
                linestyle="--",
                **kwargs,
            )
            lines["val"] = val_line
            if show_best:
                best_val_idx = np.argmin(val_values)
                ax.plot(
                    epochs[best_val_idx],
                    val_values[best_val_idx],
                    marker="*",
                    markersize=15,
                    color=val_color,
                    markeredgecolor="black",
                    markeredgewidth=1,
                    zorder=5,
                )
                ax.annotate(
                    f"Best: {val_values[best_val_idx]:.4f}",
                    xy=(epochs[best_val_idx], val_values[best_val_idx]),
                    xytext=(10, 10),
                    textcoords="offset points",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.7),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                )
        if log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.legend(loc="best")
        self.prepare_axes(ax)
        return lines


@register_plotter("feature_importance")
class FeatureImportancePlotter(BasePlotter):
    """特徴量重要度を描画するプロッタークラス。

    機械学習モデルの特徴量重要度を棒グラフで可視化します。
    重要度の高い順にソートして表示することもできます。
    """

    plot_type = "feature_importance"
    supported_dims = (1,)

    def plot(
        self,
        ax: Axes,
        importances: np.ndarray,
        feature_names: Optional[List[str]] = None,
        top_n: Optional[int] = None,
        sort: bool = True,
        horizontal: bool = True,
        color: Optional[str] = None,
        error_bars: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Any:
        """特徴量重要度を描画する。

        Args:
            ax: 描画先のAxesオブジェクト
            importances: 特徴量重要度の配列
            feature_names: 特徴量名のリスト
            top_n: 表示する上位N個の特徴量
            sort: 重要度順にソートするかどうか
            horizontal: 横向き棒グラフにするかどうか
            color: 棒の色
            error_bars: エラーバーの値
            **kwargs: ax.barまたはax.barhに渡す追加引数

        Returns:
            BarContainerオブジェクト
        """
        importances = DataAdapter.to_numpy(importances)
        if importances.ndim != 1:
            raise ValueError(f"importances must be 1D, got {importances.ndim}D")
        n_features = len(importances)
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(n_features)]
        elif len(feature_names) != n_features:
            raise ValueError(
                f"Length mismatch: importances={n_features}, feature_names={len(feature_names)}"
            )
        if sort:
            indices = np.argsort(importances)[::-1]
            importances = importances[indices]
            feature_names = [feature_names[i] for i in indices]
            if error_bars is not None:
                error_bars = error_bars[indices]
        if top_n is not None and top_n < len(importances):
            importances = importances[:top_n]
            feature_names = feature_names[:top_n]
            if error_bars is not None:
                error_bars = error_bars[:top_n]
        if color is None:
            color = VizConfig.get_color(0)
        positions = np.arange(len(importances))
        if horizontal:
            bars = ax.barh(
                positions,
                importances,
                color=color,
                alpha=0.7,
                xerr=error_bars,
                **kwargs,
            )
            ax.set_yticks(positions)
            ax.set_yticklabels(feature_names)
            ax.set_xlabel("Importance", fontsize=11)
            ax.invert_yaxis()
        else:
            bars = ax.bar(
                positions,
                importances,
                color=color,
                alpha=0.7,
                yerr=error_bars,
                **kwargs,
            )
            ax.set_xticks(positions)
            ax.set_xticklabels(feature_names, rotation=45, ha="right")
            ax.set_ylabel("Importance", fontsize=11)
        self.prepare_axes(ax)
        return bars


@register_plotter("correlation_matrix")
class CorrelationMatrixPlotter(BasePlotter):
    """相関係数行列を描画するプロッタークラス。

    特徴量間の相関を視覚化します。
    ヒートマップベースで、相関の強さを色で表現します。
    """

    plot_type = "correlation_matrix"
    supported_dims = (2,)

    def plot(
        self,
        ax: Axes,
        corr_matrix: np.ndarray,
        feature_names: Optional[List[str]] = None,
        annot: bool = True,
        fmt: str = ".2f",
        cmap: str = "coolwarm",
        vmin: float = -1,
        vmax: float = 1,
        mask_diagonal: bool = False,
        mask_upper: bool = False,
        **kwargs,
    ) -> Any:
        """相関係数行列を描画する。

        Args:
            ax: 描画先のAxesオブジェクト
            corr_matrix: 相関係数行列
            feature_names: 特徴量名のリスト
            annot: 相関係数を表示するかどうか
            fmt: 数値のフォーマット文字列
            cmap: カラーマップ
            vmin: カラースケールの最小値
            vmax: カラースケールの最大値
            mask_diagonal: 対角線をマスクするかどうか
            mask_upper: 上三角をマスクするかどうか
            **kwargs: ax.imshowに渡す追加引数

        Returns:
            AxesImageオブジェクト
        """
        corr_matrix = DataAdapter.to_numpy(corr_matrix)
        if corr_matrix.ndim != 2 or corr_matrix.shape[0] != corr_matrix.shape[1]:
            raise ValueError(
                f"corr_matrix must be square 2D array, got {corr_matrix.shape}"
            )
        n_features = corr_matrix.shape[0]
        if feature_names is None:
            feature_names = [f"F{i}" for i in range(n_features)]
        elif len(feature_names) != n_features:
            raise ValueError(
                f"Length mismatch: matrix size={n_features}, feature_names={len(feature_names)}"
            )
        mask = np.zeros_like(corr_matrix, dtype=bool)
        if mask_diagonal:
            np.fill_diagonal(mask, True)
        if mask_upper:
            mask[np.triu_indices_from(mask, k=1)] = True
        masked_corr = np.ma.array(corr_matrix, mask=mask)
        im = ax.imshow(masked_corr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(np.arange(n_features))
        ax.set_yticks(np.arange(n_features))
        ax.set_xticklabels(feature_names, rotation=45, ha="right")
        ax.set_yticklabels(feature_names)
        if annot:
            for i in range(n_features):
                for j in range(n_features):
                    if not mask[i, j]:
                        text_color = (
                            "white" if abs(corr_matrix[i, j]) > 0.5 else "black"
                        )
                        ax.text(
                            j,
                            i,
                            format(corr_matrix[i, j], fmt),
                            ha="center",
                            va="center",
                            color=text_color,
                            fontsize=8,
                        )
        plt.colorbar(im, ax=ax, label="Correlation")
        self.prepare_axes(ax, grid=False)
        return im


@register_plotter("embedding")
class EmbeddingPlotter(BasePlotter):
    """埋め込み空間を可視化するプロッタークラス。

    高次元データを2次元または3次元に次元削減して可視化します。
    t-SNE、UMAP、PCAなどの次元削減手法をサポートします。

    Examples:
        >>> import numpy as np
        >>> from plotlib import EmbeddingPlotter, subplots
        >>>
        >>> # 高次元データの準備
        >>> data = np.random.randn(150, 64)  # 150サンプル, 64次元
        >>> labels = np.repeat([0, 1, 2], 50)
        >>>
        >>> # t-SNEで2次元に削減して可視化
        >>> plotter = EmbeddingPlotter()
        >>> fig, ax = subplots(figsize=(10, 8))
        >>> plotter.plot(ax, data, labels=labels, method="tsne")
        >>> ax.set_title("t-SNE Embedding")
    """

    plot_type = "embedding"
    supported_dims = (2,)
    is_3d = False

    def plot(
        self,
        ax: Axes,
        embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None,
        method: str = "tsne",
        n_components: int = 2,
        colors: Optional[Union[str, np.ndarray]] = None,
        sizes: Optional[Union[float, np.ndarray]] = None,
        alpha: float = 0.7,
        cmap: str = "tab10",
        **kwargs,
    ) -> Any:
        """埋め込み空間を可視化する。

        Args:
            ax: 描画先のAxesオブジェクト
            embeddings: 高次元データ
            labels: クラスラベル（省略可）
            method: 次元削減手法 ('tsne', 'umap', 'pca')
            n_components: 削減後の次元数 (2 or 3)
            colors: 色の指定
            sizes: マーカーサイズ
            alpha: 透明度 (0-1)
            cmap: カラーマップ
            **kwargs: 各次元削減手法に渡す追加引数

        Returns:
            PathCollectionオブジェクト

        Raises:
            ValueError: 無効な次元削減手法が指定された場合
            ImportError: 必要なライブラリがインストールされていない場合
        """
        embeddings = DataAdapter.to_numpy(embeddings)
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got {embeddings.ndim}D")
        if n_components not in [2, 3]:
            raise ValueError(f"n_components must be 2 or 3, got {n_components}")
        if embeddings.shape[1] == n_components:
            reduced = embeddings
        else:
            if method == "tsne":
                reducer = TSNE(
                    n_components=n_components, **kwargs.get("tsne_kwargs", {})
                )
                reduced = reducer.fit_transform(embeddings)
            elif method == "umap":
                if not HAS_UMAP:
                    raise ImportError("UMAP requires: pip install umap-learn")
                reducer = umap.UMAP(
                    n_components=n_components, **kwargs.get("umap_kwargs", {})
                )
                reduced = reducer.fit_transform(embeddings)
            elif method == "pca":
                reducer = PCA(n_components=n_components)
                reduced = reducer.fit_transform(embeddings)
            else:
                raise ValueError(
                    f"Unknown method: {method}. Available: tsne, umap, pca"
                )
        if sizes is None:
            sizes = 50
        if n_components == 2:
            if labels is not None:
                scatter = ax.scatter(
                    reduced[:, 0],
                    reduced[:, 1],
                    c=labels,
                    s=sizes,
                    alpha=alpha,
                    cmap=cmap,
                )
                unique_labels = np.unique(labels)
                if len(unique_labels) <= 10:
                    for label in unique_labels:
                        mask = labels == label
                        ax.scatter(
                            [],
                            [],
                            c=[plt.cm.get_cmap(cmap)(label / max(unique_labels))],
                            label=f"Class {int(label)}",
                        )
                    ax.legend()
            else:
                if colors is None:
                    colors = VizConfig.get_color(0)
                scatter = ax.scatter(
                    reduced[:, 0], reduced[:, 1], c=colors, s=sizes, alpha=alpha
                )
            ax.set_xlabel(f"{method.upper()} 1", fontsize=11)
            ax.set_ylabel(f"{method.upper()} 2", fontsize=11)
        else:
            from mpl_toolkits.mplot3d import Axes3D

            if labels is not None:
                scatter = ax.scatter(
                    reduced[:, 0],
                    reduced[:, 1],
                    reduced[:, 2],
                    c=labels,
                    s=sizes,
                    alpha=alpha,
                    cmap=cmap,
                )
            else:
                if colors is None:
                    colors = VizConfig.get_color(0)
                scatter = ax.scatter(
                    reduced[:, 0],
                    reduced[:, 1],
                    reduced[:, 2],
                    c=colors,
                    s=sizes,
                    alpha=alpha,
                )
            ax.set_xlabel(f"{method.upper()} 1", fontsize=11)
            ax.set_ylabel(f"{method.upper()} 2", fontsize=11)
            ax.set_zlabel(f"{method.upper()} 3", fontsize=11)
        self.prepare_axes(ax)
        return scatter


@register_plotter("decision_boundary")
class DecisionBoundaryPlotter(BasePlotter):
    """決定境界を描画するプロッタークラス。

    2次元特徴空間における分類器の決定境界を可視化します。
    訓練データの散布図と重ねて表示することで、
    モデルの振る舞いを直感的に理解できます。
    """

    plot_type = "decision_boundary"
    supported_dims = (2,)

    def plot(
        self,
        ax: Axes,
        predict_fn: callable,
        X: np.ndarray,
        y: np.ndarray,
        resolution: int = 200,
        alpha: float = 0.3,
        plot_proba: bool = False,
        scatter_train: bool = True,
        cmap: str = "RdYlBu",
        **kwargs,
    ) -> Any:
        """決定境界を描画する。

        Args:
            ax: 描画先のAxesオブジェクト
            predict_fn: 予測関数（2次元入力を受け取り予測を返す）
            X: 訓練データの特徴量 (n_samples, 2)
            y: 訓練データのラベル
            resolution: メッシュグリッドの解像度
            alpha: 決定境界の透明度
            plot_proba: 確率を表示するかどうか
            scatter_train: 訓練データをプロットするかどうか
            cmap: カラーマップ
            **kwargs: ax.contourfに渡す追加引数

        Returns:
            QuadContourSetオブジェクト

        Raises:
            ValueError: 特徴量が2次元でない場合や予測に失敗した場合
        """
        # Convert NumType to numpy array
        X = DataAdapter.to_numpy(X)
        y = DataAdapter.to_numpy(y)

        if X.shape[1] != 2:
            raise ValueError("DecisionBoundaryPlotter requires 2D features")
        x_min, x_max = (X[:, 0].min() - 1, X[:, 0].max() + 1)
        y_min, y_max = (X[:, 1].min() - 1, X[:, 1].max() + 1)
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
        )
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        try:
            if plot_proba:
                Z = predict_fn(mesh_points)
                if len(Z.shape) > 1 and Z.shape[1] > 1:
                    Z = np.max(Z, axis=1)
            else:
                Z = predict_fn(mesh_points)
                if len(Z.shape) > 1:
                    Z = Z.ravel()
        except Exception as e:
            raise ValueError(f"Prediction failed: {e}")
        Z = Z.reshape(xx.shape)
        if plot_proba:
            contour = ax.contourf(xx, yy, Z, levels=20, cmap=cmap, alpha=alpha)
            plt.colorbar(contour, ax=ax, label="Probability")
        else:
            n_classes = len(np.unique(y))
            contour = ax.contourf(
                xx, yy, Z, levels=n_classes - 1, cmap=cmap, alpha=alpha
            )
        if scatter_train:
            unique_classes = np.unique(y)
            for cls in unique_classes:
                mask = y == cls
                ax.scatter(
                    X[mask, 0],
                    X[mask, 1],
                    label=f"Class {cls}",
                    s=50,
                    edgecolors="black",
                    linewidth=1,
                )
            ax.legend(loc="best")
        ax.set_xlabel("Feature 1", fontsize=11)
        ax.set_ylabel("Feature 2", fontsize=11)
        ax.grid(True, alpha=0.3)
        self.prepare_axes(ax)
        return contour


@register_plotter("surface_3d")
class Surface3DPlotter(BasePlotter):
    """3次元曲面を描画するプロッタークラス。

    2次元入力と1次元出力の関数を3次元空間で可視化します。
    ニューラルネットワークの出力や損失関数の地形などの可視化に適しています。

    Examples:
        >>> import numpy as np
        >>> from plotlib import Surface3DPlotter, subplots
        >>>
        >>> # 3次元関数の可視化
        >>> def predict_fn(X):
        ...     return np.sin(X[:, 0]) * np.cos(X[:, 1])
        >>>
        >>> x = np.linspace(-3, 3, 50)
        >>> y = np.linspace(-3, 3, 50)
        >>> X, Y = np.meshgrid(x, y)
        >>>
        >>> plotter = Surface3DPlotter()
        >>> fig = plt.figure(figsize=(10, 8))
        >>> ax = fig.add_subplot(111, projection='3d')
        >>> plotter.plot(ax, predict_fn, x_range=(-3, 3), y_range=(-3, 3))
        >>> ax.set_title("3D Surface Plot")
    """

    plot_type = "surface_3d"
    supported_dims = (2,)
    is_3d = True

    def plot(
        self,
        ax,
        predict_fn: callable,
        x_range: Tuple[float, float] = (-1, 1),
        y_range: Tuple[float, float] = (-1, 1),
        resolution: int = 50,
        cmap: str = "viridis",
        alpha: float = 0.8,
        plot_wireframe: bool = False,
        scatter_points: Optional[np.ndarray] = None,
        scatter_values: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Any:
        """3次元曲面を描画する。

        Args:
            ax: 描画先の3DAxesオブジェクト
            predict_fn: 予測関数（2次元入力を受け取り1次元出力を返す）
            x_range: x軸の範囲 (min, max)
            y_range: y軸の範囲 (min, max)
            resolution: メッシュグリッドの解像度
            cmap: カラーマップ
            alpha: 透明度 (0-1)
            plot_wireframe: ワイヤーフレームで描画するかどうか
            scatter_points: 散布図としてプロットする点 (n_points, 2)
            scatter_values: 散布図の各点の値 (n_points,)
            **kwargs: plot_surfaceまたはplot_wireframeに渡す追加引数

        Returns:
            曲面オブジェクト

        Raises:
            ValueError: 予測に失敗した場合
        """
        # メッシュグリッドを作成
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)

        # 予測値を計算
        mesh_points = np.c_[X.ravel(), Y.ravel()]
        try:
            Z = predict_fn(mesh_points)
            if len(Z.shape) > 1:
                Z = Z.ravel()
        except Exception as e:
            raise ValueError(f"Prediction failed: {e}")

        Z = Z.reshape(X.shape)

        # 曲面を描画
        if plot_wireframe:
            surf = ax.plot_wireframe(X, Y, Z, cmap=cmap, alpha=alpha, **kwargs)
        else:
            surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=alpha, **kwargs)

        # 散布図を追加
        if scatter_points is not None:
            if scatter_values is None:
                # 値が指定されていない場合は予測
                scatter_values = predict_fn(scatter_points)
                if len(scatter_values.shape) > 1:
                    scatter_values = scatter_values.ravel()

            ax.scatter(
                scatter_points[:, 0],
                scatter_points[:, 1],
                scatter_values,
                c="red",
                s=100,
                edgecolors="black",
                linewidth=2,
                zorder=10,
                label="Training Points",
            )
            ax.legend()

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Output")

        return surf


class PublicationStyle:
    """出版物品質のプロットスタイルを管理するクラス。

    論文、プレゼンテーション、ポスターなど、
    用途に応じたスタイルプリセットを提供します。
    """

    PRESETS = {
        "publication": {
            "figure.figsize": (6, 4),
            "figure.dpi": 300,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
            "patch.linewidth": 0.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        },
        "presentation": {
            "figure.figsize": (8, 6),
            "figure.dpi": 150,
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.linewidth": 1.2,
            "grid.linewidth": 0.8,
            "lines.linewidth": 2.5,
            "lines.markersize": 8,
            "patch.linewidth": 0.8,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        },
        "poster": {
            "figure.figsize": (10, 8),
            "figure.dpi": 150,
            "font.size": 18,
            "axes.labelsize": 22,
            "axes.titlesize": 24,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 17,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.linewidth": 1.5,
            "grid.linewidth": 1.0,
            "lines.linewidth": 3.0,
            "lines.markersize": 10,
            "patch.linewidth": 1.0,
            "xtick.major.width": 1.5,
            "ytick.major.width": 1.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        },
    }

    @classmethod
    def apply(cls, style: str = "publication"):
        """指定されたスタイルを適用する。

        Args:
            style: スタイル名 ('publication', 'presentation', 'poster')

        Raises:
            ValueError: 無効なスタイル名が指定された場合
        """
        if style not in cls.PRESETS:
            raise ValueError(
                f"Unknown style: {style}. Available: {list(cls.PRESETS.keys())}"
            )
        plt.rcParams.update(cls.PRESETS[style])

    @classmethod
    def reset(cls):
        """スタイルをデフォルトにリセットする。"""
        plt.rcdefaults()


def set_style(style: str = "publication"):
    """プロットスタイルを設定する。

    Args:
        style: スタイル名
    """
    PublicationStyle.apply(style)


def get_style() -> str:
    """現在のスタイル設定を取得する。

    Returns:
        現在のスタイル名（推定）
    """
    current_figsize = plt.rcParams["figure.figsize"]
    for style_name, preset in PublicationStyle.PRESETS.items():
        if preset["figure.figsize"] == tuple(current_figsize):
            return style_name
    return "custom"


class VizEngine:
    """可視化エンジンの統合インターフェース。

    複数のプロッタークラスを統一的に管理し、
    自動的に適切なプロッタークラスを選択してプロットを作成します。
    """

    def __init__(self, style: str = "publication"):
        """VizEngineインスタンスを初期化する。

        Args:
            style: デフォルトのプロットスタイル
        """
        self.style = style
        PublicationStyle.apply(style)

    def plot(
        self,
        data: Any,
        plot_type: Optional[str] = None,
        **kwargs,
    ) -> PlotResult:
        """データを自動的にプロットする。

        Args:
            data: プロットするデータ
            plot_type: プロット種別（省略時は自動判定）
            **kwargs: プロッタークラスに渡す追加引数

        Returns:
            PlotResultオブジェクト

        Raises:
            ValueError: 適切なプロッタークラスが見つからない場合
        """
        data_np = DataAdapter.to_numpy(data)
        if plot_type is not None:
            plotter_class = PlotRegistry.get(plot_type)
            if plotter_class is None:
                raise ValueError(f"Unknown plot_type: {plot_type}")
        else:
            plotter_class = PlotRegistry.find_suitable_plotter(data_np)
            if plotter_class is None:
                raise ValueError(
                    f"No suitable plotter found for data with shape {data_np.shape}"
                )
        plotter = plotter_class(style=self.style)
        fig, ax = plt.subplots(figsize=VizConfig.get_default("figsize"))
        plot_obj = plotter.plot(ax, data_np, **kwargs)
        return PlotResult(fig, ax, plot_obj)


def plot(data, plot_type=None, **kwargs):
    """データを自動的にプロットする便利関数。

    Args:
        data: プロットするデータ
        plot_type: プロット種別（省略時は自動判定）
        **kwargs: プロッタークラスに渡す追加引数

    Returns:
        PlotResultオブジェクト
    """
    engine = VizEngine()
    return engine.plot(data, plot_type=plot_type, **kwargs)


def subplot(nrows, ncols, index, **kwargs):
    """サブプロットを作成する。

    Args:
        nrows: 行数
        ncols: 列数
        index: サブプロットのインデックス
        **kwargs: subplotに渡す追加引数

    Returns:
        Axesオブジェクト
    """
    return plt.subplot(nrows, ncols, index, **kwargs)


def subplots(nrows=1, ncols=1, **kwargs):
    """複数のサブプロットを作成する。

    Args:
        nrows: 行数
        ncols: 列数
        **kwargs: subplotsに渡す追加引数

    Returns:
        (Figure, Axes)のタプル
    """
    return plt.subplots(nrows, ncols, **kwargs)


def line(
    *args,
    figsize=(10, 6),
    title=None,
    xlabel=None,
    ylabel=None,
    show=True,
    save=None,
    dpi=300,
    **kwargs,
):
    """折れ線グラフを描画する便利関数。

    Args:
        *args: データ
        figsize: 図のサイズ
        title: タイトル
        xlabel: x軸ラベル
        ylabel: y軸ラベル
        show: 表示するかどうか
        save: 保存先ファイル名
        dpi: 解像度
        **kwargs: LinePlotterに渡す追加引数

    Returns:
        (Figure, Axes)のタプル

    Examples:
        基本的な使い方（コピペして実行可能）:

        ```python
        import plotlib as pl
        import numpy as np

        # 例1: 単純な折れ線グラフ（y値のみ）
        y = np.sin(np.linspace(0, 2*np.pi, 100))
        pl.line(y, title="Sine Wave", ylabel="sin(x)")

        # 例2: x, yを両方指定
        x = np.linspace(0, 10, 50)
        y = x ** 2
        pl.line(x, y, title="Quadratic Function", xlabel="X", ylabel="Y²")

        # 例3: 複数系列をプロット
        x = np.linspace(0, 4*np.pi, 200)
        y1 = np.sin(x)
        y2 = np.cos(x)
        y3 = np.sin(x) * np.cos(x)
        pl.line(x, np.column_stack([y1, y2, y3]),
                labels=["sin(x)", "cos(x)", "sin(x)cos(x)"],
                title="Trigonometric Functions")

        # 例4: 減衰振動
        x = np.linspace(0, 10, 100)
        y = np.exp(-x/5) * np.sin(x)
        pl.line(x, y, title="Damped Oscillation",
                xlabel="Time", ylabel="Amplitude")
        ```
    """
    return LinePlotter().quick_plot(
        *args,
        figsize=figsize,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        show=show,
        save=save,
        dpi=dpi,
        **kwargs,
    )


def scatter(
    x,
    y=None,
    figsize=(10, 6),
    title=None,
    xlabel=None,
    ylabel=None,
    show=True,
    save=None,
    dpi=300,
    **kwargs,
):
    """散布図を描画する便利関数。

    Args:
        x: x座標データまたは2次元データ
        y: y座標データ（省略可）
        figsize: 図のサイズ
        title: タイトル
        xlabel: x軸ラベル
        ylabel: y軸ラベル
        show: 表示するかどうか
        save: 保存先ファイル名
        dpi: 解像度
        **kwargs: ScatterPlotterに渡す追加引数

    Returns:
        (Figure, Axes)のタプル

    Examples:
        基本的な使い方（コピペして実行可能）:

        ```python
        import plotlib as pl
        import numpy as np

        # 例1: 単純な散布図
        np.random.seed(42)
        x = np.random.randn(100)
        y = 2 * x + np.random.randn(100) * 0.5
        pl.scatter(x, y, title="Linear Relationship",
                   xlabel="X", ylabel="Y", alpha=0.6)

        # 例2: クラスラベルで色分け
        from sklearn.datasets import make_blobs
        X, labels = make_blobs(n_samples=200, centers=3,
                               n_features=2, random_state=42)
        pl.scatter(X, labels=labels,
                   title="Clustered Data",
                   class_names=["Class A", "Class B", "Class C"])

        # 例3: サイズと色を変更
        x = np.random.randn(50)
        y = np.random.randn(50)
        sizes = np.random.rand(50) * 100
        pl.scatter(x, y, s=sizes, c='red', alpha=0.5,
                   title="Variable Size Points")

        # 例4: 相関関係の可視化
        n = 100
        x = np.linspace(0, 10, n)
        y = 3 * x + 2 + np.random.randn(n) * 2
        pl.scatter(x, y, title="Noisy Linear Data",
                   xlabel="Feature", ylabel="Target")
        ```
    """
    if y is not None:
        data = np.column_stack([x, y])
    else:
        data = x
    return ScatterPlotter().quick_plot(
        data,
        figsize=figsize,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        show=show,
        save=save,
        dpi=dpi,
        **kwargs,
    )


def heatmap(
    data,
    figsize=(10, 8),
    title=None,
    xlabel=None,
    ylabel=None,
    show=True,
    save=None,
    dpi=300,
    **kwargs,
):
    """ヒートマップを描画する便利関数。

    Args:
        data: 2次元データ
        figsize: 図のサイズ
        title: タイトル
        xlabel: x軸ラベル
        ylabel: y軸ラベル
        show: 表示するかどうか
        save: 保存先ファイル名
        dpi: 解像度
        **kwargs: HeatmapPlotterに渡す追加引数

    Returns:
        (Figure, Axes)のタプル

    Examples:
        基本的な使い方（コピペして実行可能）:

        ```python
        import plotlib as pl
        import numpy as np

        # 例1: ランダムデータのヒートマップ
        np.random.seed(42)
        data = np.random.randn(10, 12)
        pl.heatmap(data, title="Random Heatmap",
                   cmap='coolwarm', annot=True, fmt='.2f')

        # 例2: 相関行列の可視化
        n_features = 5
        X = np.random.randn(100, n_features)
        corr_matrix = np.corrcoef(X.T)
        pl.heatmap(corr_matrix,
                   title="Feature Correlation Matrix",
                   xticklabels=[f"F{i+1}" for i in range(n_features)],
                   yticklabels=[f"F{i+1}" for i in range(n_features)],
                   cmap='RdBu_r', vmin=-1, vmax=1, annot=True)

        # 例3: 混同行列の可視化
        confusion = np.array([[50, 2, 3],
                              [5, 45, 4],
                              [1, 3, 52]])
        pl.heatmap(confusion,
                   title="Confusion Matrix",
                   xticklabels=["Class A", "Class B", "Class C"],
                   yticklabels=["Class A", "Class B", "Class C"],
                   annot=True, fmt='d', cmap='Blues')

        # 例4: 時系列データのヒートマップ
        days = 30
        hours = 24
        activity = np.random.poisson(lam=5, size=(days, hours))
        pl.heatmap(activity,
                   title="Daily Activity Pattern",
                   xlabel="Hour of Day",
                   ylabel="Day",
                   cmap='YlOrRd')
        ```
    """
    return HeatmapPlotter().quick_plot(
        data,
        figsize=figsize,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        show=show,
        save=save,
        dpi=dpi,
        **kwargs,
    )


def parallel_coordinates(
    data,
    labels=None,
    feature_names=None,
    figsize=(12, 6),
    title=None,
    xlabel=None,
    ylabel=None,
    show=True,
    save=None,
    dpi=300,
    **kwargs,
):
    """並行座標プロットを描画する便利関数。

    Args:
        data: 2次元データ (n_samples, n_features)
        labels: クラスラベル配列（省略可）
        feature_names: 特徴量名のリスト
        figsize: 図のサイズ
        title: タイトル
        xlabel: x軸ラベル
        ylabel: y軸ラベル
        show: 表示するかどうか
        save: 保存先ファイル名
        dpi: 解像度
        **kwargs: ParallelCoordinatesPlotterに渡す追加引数

    Returns:
        (Figure, Axes)のタプル

    Examples:
        基本的な使い方（コピペして実行可能）:

        ```python
        import plotlib as pl
        import numpy as np

        # 例1: 基本的な並行座標プロット
        np.random.seed(42)
        n_samples, n_features = 100, 5
        data = np.random.randn(n_samples, n_features)
        feature_names = ["Age", "Income", "Education", "Experience", "Satisfaction"]
        pl.parallel_coordinates(data,
                               feature_names=feature_names,
                               title="Customer Profile Analysis")

        # 例2: クラスラベルで色分け
        # 3つの異なるグループを生成
        group1 = np.random.randn(30, n_features) + np.array([1, 2, 0, 1, 2])
        group2 = np.random.randn(30, n_features) + np.array([-1, 0, 2, -1, 0])
        group3 = np.random.randn(40, n_features) + np.array([0, -1, -1, 0, -1])
        data = np.vstack([group1, group2, group3])
        labels = np.array([0]*30 + [1]*30 + [2]*40)

        pl.parallel_coordinates(data, labels=labels,
                               feature_names=feature_names,
                               class_names=["High", "Medium", "Low"],
                               title="Customer Segments")

        # 例3: 正規化なしで描画
        pl.parallel_coordinates(data, labels=labels,
                               feature_names=feature_names,
                               normalize=False,
                               title="Non-normalized Data")

        # 例4: 特定サンプルをハイライト
        pl.parallel_coordinates(data,
                               feature_names=feature_names,
                               highlight_indices=[0, 10, 20, 30],
                               highlight_color="red",
                               highlight_linewidth=3,
                               alpha=0.3,
                               title="Highlighted Samples")

        # 例5: 実際のデータセット（sklearn iris）
        # from sklearn.datasets import load_iris
        # iris = load_iris()
        # pl.parallel_coordinates(iris.data, labels=iris.target,
        #                        feature_names=iris.feature_names,
        #                        class_names=iris.target_names,
        #                        title="Iris Dataset")
        ```
    """
    return ParallelCoordinatesPlotter().quick_plot(
        data,
        labels=labels,
        feature_names=feature_names,
        figsize=figsize,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        show=show,
        save=save,
        dpi=dpi,
        **kwargs,
    )


def learning_curve(
    train_values,
    val_values,
    figsize=(12, 7),
    title=None,
    xlabel=None,
    ylabel=None,
    show=True,
    save=None,
    dpi=300,
    **kwargs,
):
    """学習曲線を描画する便利関数。

    Args:
        train_values: 訓練データの値
        val_values: 検証データの値
        figsize: 図のサイズ
        title: タイトル
        xlabel: x軸ラベル
        ylabel: y軸ラベル
        show: 表示するかどうか
        save: 保存先ファイル名
        dpi: 解像度
        **kwargs: LearningCurvePlotterに渡す追加引数

    Returns:
        (Figure, Axes)のタプル

    Examples:
        基本的な使い方（コピペして実行可能）:

        ```python
        import plotlib as pl
        import numpy as np

        # 例1: 基本的な学習曲線（損失）
        epochs = 50
        train_loss = 2.0 * np.exp(-np.arange(epochs) / 10) + 0.1
        val_loss = 2.2 * np.exp(-np.arange(epochs) / 12) + 0.15
        pl.learning_curve(train_loss, val_loss,
                         title="Training Progress",
                         xlabel="Epoch",
                         ylabel="Loss")

        # 例2: 精度の学習曲線
        train_acc = 1 - np.exp(-np.arange(epochs) / 8) * 0.5
        val_acc = 1 - np.exp(-np.arange(epochs) / 10) * 0.6
        pl.learning_curve(train_acc, val_acc,
                         title="Model Accuracy",
                         xlabel="Epoch",
                         ylabel="Accuracy")

        # 例3: ノイズの多いデータを平滑化
        np.random.seed(42)
        train_noisy = train_loss + np.random.randn(epochs) * 0.1
        val_noisy = val_loss + np.random.randn(epochs) * 0.12
        pl.learning_curve(train_noisy, val_noisy,
                         smooth=5,
                         title="Smoothed Learning Curve",
                         ylabel="Loss")

        # 例4: 複数メトリクスの同時表示
        metrics = {
            'loss': (train_loss, val_loss),
            'accuracy': (train_acc, val_acc)
        }
        # 個別にプロット
        for name, (train, val) in metrics.items():
            pl.learning_curve(train, val,
                            title=f"Training {name.capitalize()}",
                            ylabel=name.capitalize())
        ```
    """
    return LearningCurvePlotter().quick_plot(
        train_values,
        val_values,
        figsize=figsize,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        show=show,
        save=save,
        dpi=dpi,
        **kwargs,
    )


def feature_importance(
    importances,
    feature_names=None,
    figsize=(10, 8),
    title=None,
    show=True,
    save=None,
    dpi=300,
    **kwargs,
):
    """特徴量重要度を描画する便利関数。

    Args:
        importances: 特徴量重要度の配列
        feature_names: 特徴量名のリスト
        figsize: 図のサイズ
        title: タイトル
        show: 表示するかどうか
        save: 保存先ファイル名
        dpi: 解像度
        **kwargs: FeatureImportancePlotterに渡す追加引数

    Returns:
        (Figure, Axes)のタプル

    Examples:
        基本的な使い方（コピペして実行可能）:

        ```python
        import plotlib as pl
        import numpy as np

        # 例1: 基本的な特徴量重要度
        importances = np.array([0.35, 0.25, 0.18, 0.12, 0.10])
        features = ["Age", "Income", "Education", "Experience", "Location"]
        pl.feature_importance(importances,
                            feature_names=features,
                            title="Feature Importance Analysis")

        # 例2: ランダムフォレストからの重要度（sklearn風）
        np.random.seed(42)
        n_features = 10
        importance_scores = np.random.exponential(scale=0.1, size=n_features)
        importance_scores = importance_scores / importance_scores.sum()
        feature_names = [f"Feature_{i+1}" for i in range(n_features)]
        pl.feature_importance(importance_scores,
                            feature_names=feature_names,
                            title="Random Forest Feature Importance",
                            top_n=7)

        # 例3: 標準偏差付き（複数モデルの平均）
        mean_importances = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
        std_importances = np.array([0.05, 0.04, 0.03, 0.02, 0.01])
        features = ["Temperature", "Humidity", "Pressure", "Wind", "Rain"]
        pl.feature_importance(mean_importances,
                            feature_names=features,
                            error_bars=std_importances,
                            title="Feature Importance (Mean ± Std)")

        # 例4: 多数の特徴量から上位のみ表示
        np.random.seed(123)
        all_features = 50
        all_importances = np.random.exponential(0.02, all_features)
        pl.feature_importance(all_importances,
                            top_n=15,
                            title="Top 15 Most Important Features",
                            color='steelblue')
        ```
    """
    return FeatureImportancePlotter().quick_plot(
        importances,
        feature_names=feature_names,
        figsize=figsize,
        title=title,
        show=show,
        save=save,
        dpi=dpi,
        **kwargs,
    )


def correlation_matrix(
    corr_matrix,
    feature_names=None,
    figsize=(12, 10),
    title=None,
    show=True,
    save=None,
    dpi=300,
    **kwargs,
):
    """相関係数行列を描画する便利関数。

    Args:
        corr_matrix: 相関係数行列
        feature_names: 特徴量名のリスト
        figsize: 図のサイズ
        title: タイトル
        show: 表示するかどうか
        save: 保存先ファイル名
        dpi: 解像度
        **kwargs: CorrelationMatrixPlotterに渡す追加引数

    Returns:
        (Figure, Axes)のタプル

    Examples:
        >>> import numpy as np
        >>> from plotlib import correlation_matrix
        >>>
        >>> # 相関係数行列の可視化
        >>> data = np.random.randn(100, 5)
        >>> corr = np.corrcoef(data.T)
        >>> names = ["F1", "F2", "F3", "F4", "F5"]
        >>> correlation_matrix(corr, feature_names=names,
        ...                    title="Correlation Matrix")
        >>>
        >>> # 値を表示
        >>> correlation_matrix(corr, feature_names=names,
        ...                    annot=True, fmt=".2f")
        >>>
        >>> # 上三角のみ表示
        >>> correlation_matrix(corr, feature_names=names,
        ...                    mask_upper=True, annot=True)
    """
    return CorrelationMatrixPlotter().quick_plot(
        corr_matrix,
        feature_names=feature_names,
        figsize=figsize,
        title=title,
        show=show,
        save=save,
        dpi=dpi,
        **kwargs,
    )


def embedding(
    embeddings,
    labels=None,
    method="tsne",
    figsize=(12, 10),
    title=None,
    xlabel=None,
    ylabel=None,
    show=True,
    save=None,
    dpi=300,
    **kwargs,
):
    """埋め込み空間を可視化する便利関数。

    Args:
        embeddings: 高次元データ
        labels: クラスラベル（省略可）
        method: 次元削減手法 ('tsne', 'umap', 'pca')
        figsize: 図のサイズ
        title: タイトル
        xlabel: x軸ラベル
        ylabel: y軸ラベル
        show: 表示するかどうか
        save: 保存先ファイル名
        dpi: 解像度
        **kwargs: EmbeddingPlotterに渡す追加引数

    Returns:
        (Figure, Axes)のタプル

    Examples:
        基本的な使い方（コピペして実行可能）:

        ```python
        import plotlib as pl
        import numpy as np

        # 例1: t-SNEで高次元データを可視化
        np.random.seed(42)
        # 3つのクラスタを持つ高次元データを生成
        n_samples, n_features = 150, 50
        X1 = np.random.randn(n_samples//3, n_features) + np.array([3]*n_features)
        X2 = np.random.randn(n_samples//3, n_features) + np.array([-3]*n_features)
        X3 = np.random.randn(n_samples//3, n_features)
        X = np.vstack([X1, X2, X3])
        labels = np.array([0]*(n_samples//3) + [1]*(n_samples//3) + [2]*(n_samples//3))

        pl.embedding(X, labels=labels, method="tsne",
                    title="t-SNE Embedding",
                    class_names=["Class A", "Class B", "Class C"])

        # 例2: PCAでの可視化（より高速）
        pl.embedding(X, labels=labels, method="pca",
                    title="PCA Projection",
                    class_names=["Class A", "Class B", "Class C"])

        # 例3: ラベルなしでの可視化
        data = np.random.randn(200, 30)
        pl.embedding(data, method="pca",
                    title="Unlabeled Data Embedding")

        # 例4: 実際のデータセット（sklearn digits）
        # from sklearn.datasets import load_digits
        # digits = load_digits()
        # pl.embedding(digits.data, labels=digits.target,
        #            method="tsne", perplexity=30,
        #            title="MNIST Digits t-SNE")

        # 例5: UMAP（umap-learnがインストールされている場合）
        # pl.embedding(X, labels=labels, method="umap",
        #            title="UMAP Embedding",
        #            n_neighbors=15, min_dist=0.1)
        ```
    """
    return EmbeddingPlotter().quick_plot(
        embeddings,
        labels=labels,
        method=method,
        figsize=figsize,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        show=show,
        save=save,
        dpi=dpi,
        **kwargs,
    )


def decision_boundary(
    predict_fn,
    X,
    y,
    figsize=(10, 10),
    title=None,
    xlabel=None,
    ylabel=None,
    show=True,
    save=None,
    dpi=300,
    **kwargs,
):
    """決定境界を描画する便利関数。

    Args:
        predict_fn: 予測関数
        X: 訓練データの特徴量
        y: 訓練データのラベル
        figsize: 図のサイズ
        title: タイトル
        xlabel: x軸ラベル
        ylabel: y軸ラベル
        show: 表示するかどうか
        save: 保存先ファイル名
        dpi: 解像度
        **kwargs: DecisionBoundaryPlotterに渡す追加引数

    Returns:
        (Figure, Axes)のタプル

    Examples:
        基本的な使い方（コピペして実行可能）:

        ```python
        import plotlib as pl
        import numpy as np

        # 例1: 線形分類器の決定境界
        np.random.seed(42)
        X = np.random.randn(200, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        def linear_predict(x):
            return (x[:, 0] + x[:, 1] > 0).astype(int)

        pl.decision_boundary(linear_predict, X, y,
                           title="Linear Decision Boundary",
                           xlabel="Feature 1", ylabel="Feature 2")

        # 例2: 円形の決定境界
        X = np.random.randn(300, 2) * 2
        y = ((X[:, 0]**2 + X[:, 1]**2) < 4).astype(int)

        def circle_predict(x):
            return ((x[:, 0]**2 + x[:, 1]**2) < 4).astype(int)

        pl.decision_boundary(circle_predict, X, y,
                           title="Circular Decision Boundary",
                           resolution=150)

        # 例3: より複雑な境界（XOR問題）
        n = 100
        X1 = np.random.randn(n, 2) + np.array([2, 2])
        X2 = np.random.randn(n, 2) + np.array([-2, -2])
        X3 = np.random.randn(n, 2) + np.array([2, -2])
        X4 = np.random.randn(n, 2) + np.array([-2, 2])
        X = np.vstack([X1, X2, X3, X4])
        y = np.array([0]*n + [0]*n + [1]*n + [1]*n)

        def xor_predict(x):
            return ((x[:, 0] * x[:, 1]) > 0).astype(int)

        pl.decision_boundary(xor_predict, X, y,
                           title="XOR Decision Boundary",
                           alpha=0.4)

        # 例4: sklearn互換（実際のモデルを使う場合）
        # from sklearn.svm import SVC
        # from sklearn.datasets import make_moons
        #
        # X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
        # model = SVC(kernel='rbf').fit(X, y)
        # pl.decision_boundary(model.predict, X, y,
        #                    title="SVM Decision Boundary")
        ```
    """
    return DecisionBoundaryPlotter().quick_plot(
        predict_fn,
        X,
        y,
        figsize=figsize,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        show=show,
        save=save,
        dpi=dpi,
        **kwargs,
    )


def surface_3d(
    predict_fn,
    x_range=(-1, 1),
    y_range=(-1, 1),
    resolution=50,
    scatter_points=None,
    scatter_values=None,
    figsize=(12, 9),
    title=None,
    xlabel="X",
    ylabel="Y",
    zlabel="Output",
    show=True,
    save=None,
    dpi=300,
    **kwargs,
):
    """3次元曲面を描画する便利関数。

    Args:
        predict_fn: 予測関数（2次元入力を受け取り1次元出力を返す）
        x_range: x軸の範囲 (min, max)
        y_range: y軸の範囲 (min, max)
        resolution: メッシュグリッドの解像度
        scatter_points: 散布図としてプロットする点 (n_points, 2)
        scatter_values: 散布図の各点の値 (n_points,)
        figsize: 図のサイズ
        title: タイトル
        xlabel: x軸ラベル
        ylabel: y軸ラベル
        zlabel: z軸ラベル
        show: 表示するかどうか
        save: 保存先ファイル名
        dpi: 解像度
        **kwargs: Surface3DPlotterに渡す追加引数

    Returns:
        (Figure, Axes)のタプル

    Examples:
        >>> import numpy as np
        >>> from plotlib import surface_3d
        >>>
        >>> # ニューラルネットワークの出力を3D表示
        >>> def predict(X):
        ...     return np.sin(X[:, 0]) * np.cos(X[:, 1])
        >>>
        >>> surface_3d(predict, x_range=(-3, 3), y_range=(-3, 3),
        ...            title="Network Output Surface")
        >>>
        >>> # 訓練データ点も一緒に表示
        >>> train_points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        >>> train_values = np.array([0, 1, 1, 0])
        >>> surface_3d(predict, x_range=(-0.5, 1.5), y_range=(-0.5, 1.5),
        ...            scatter_points=train_points,
        ...            scatter_values=train_values,
        ...            title="XOR Network Output")
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    plotter = Surface3DPlotter()
    surf = plotter.plot(
        ax,
        predict_fn,
        x_range=x_range,
        y_range=y_range,
        resolution=resolution,
        scatter_points=scatter_points,
        scatter_values=scatter_values,
        **kwargs,
    )

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    if zlabel:
        ax.set_zlabel(zlabel, fontsize=11)

    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()

    return (fig, ax)


# =============================================================================
# 最適化専用可視化関数
# =============================================================================


def optimization_history(
    objective_values,
    *,
    figsize=(10, 6),
    title="Optimization Convergence",
    xlabel="Iteration",
    ylabel="Objective Value",
    show=True,
    save=None,
    dpi=300,
    log_scale=False,
    best_marker=True,
    **kwargs,
):
    """最適化の収束履歴を可視化する。

    Args:
        objective_values: 各イテレーションでの目的関数値のリスト
        figsize: 図のサイズ
        title: タイトル
        xlabel: x軸ラベル
        ylabel: y軸ラベル
        show: 表示するかどうか
        save: 保存先ファイルパス
        dpi: 画像解像度
        log_scale: y軸を対数スケールにするか
        best_marker: 最良値にマーカーを表示するか
        **kwargs: その他のプロット引数

    Returns:
        (Figure, Axes)のタプル

    Examples:
        ```python
        import plotlib as pl
        import numpy as np

        # 最適化履歴のシミュレーション
        iterations = np.arange(100)
        obj_vals = 100 * np.exp(-iterations / 20) + np.random.normal(0, 1, 100)

        pl.optimization_history(obj_vals,
                               title="Gradient Descent Convergence",
                               ylabel="Loss")
        ```
    """
    fig, ax = plt.subplots(figsize=figsize)

    iterations = np.arange(len(objective_values))
    ax.plot(iterations, objective_values, "b-", linewidth=2, **kwargs)

    if log_scale:
        ax.set_yscale("log")

    if best_marker:
        best_idx = np.argmin(objective_values)
        best_val = objective_values[best_idx]
        ax.plot(
            best_idx,
            best_val,
            "r*",
            markersize=15,
            label=f"Best: {best_val:.4f} @ iter {best_idx}",
        )
        ax.legend()

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()

    return (fig, ax)


def parameter_sweep(
    param1_values,
    param2_values,
    objective_matrix,
    *,
    figsize=(10, 8),
    title="Parameter Sweep Results",
    param1_name="Parameter 1",
    param2_name="Parameter 2",
    objective_name="Objective Value",
    show=True,
    save=None,
    dpi=300,
    cmap="viridis",
    levels=20,
    **kwargs,
):
    """2Dパラメータスイープ結果を可視化する。

    Args:
        param1_values: 第1パラメータの値のリスト
        param2_values: 第2パラメータの値のリスト
        objective_matrix: 目的関数値の2D配列 (param2 x param1)
        figsize: 図のサイズ
        title: タイトル
        param1_name: 第1パラメータの名前
        param2_name: 第2パラメータの名前
        objective_name: 目的関数の名前
        show: 表示するかどうか
        save: 保存先ファイルパス
        dpi: 画像解像度
        cmap: カラーマップ
        levels: 等高線のレベル数
        **kwargs: その他のプロット引数

    Returns:
        (Figure, Axes)のタプル

    Examples:
        ```python
        import plotlib as pl
        import numpy as np

        # パラメータグリッド
        x = np.linspace(-2, 2, 50)
        y = np.linspace(-2, 2, 50)
        X, Y = np.meshgrid(x, y)

        # Rosenbrock関数
        Z = (1 - X)**2 + 100 * (Y - X**2)**2

        pl.parameter_sweep(x, y, Z,
                          param1_name="x",
                          param2_name="y",
                          objective_name="f(x,y)",
                          title="Rosenbrock Function")
        ```
    """
    fig, ax = plt.subplots(figsize=figsize)

    X, Y = np.meshgrid(param1_values, param2_values)

    # 等高線プロット
    contour = ax.contourf(X, Y, objective_matrix, levels=levels, cmap=cmap, **kwargs)
    ax.contour(
        X, Y, objective_matrix, levels=levels, colors="black", alpha=0.3, linewidths=0.5
    )

    # カラーバー
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label(objective_name, fontsize=11)

    # 最適値のマーカー
    min_idx = np.unravel_index(np.argmin(objective_matrix), objective_matrix.shape)
    opt_param1 = param1_values[min_idx[1]]
    opt_param2 = param2_values[min_idx[0]]
    opt_val = objective_matrix[min_idx]

    ax.plot(
        opt_param1,
        opt_param2,
        "r*",
        markersize=15,
        label=f"Optimum: ({opt_param1:.3f}, {opt_param2:.3f})",
    )
    ax.legend()

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(param1_name, fontsize=11)
    ax.set_ylabel(param2_name, fontsize=11)

    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()

    return (fig, ax)


def pareto_front(
    objectives,
    *,
    labels=None,
    figsize=(10, 8),
    title="Pareto Front",
    xlabel="Objective 1",
    ylabel="Objective 2",
    show=True,
    save=None,
    dpi=300,
    dominated_alpha=0.3,
    pareto_color="red",
    **kwargs,
):
    """多目的最適化のパレートフロントを可視化する。

    Args:
        objectives: 目的関数値の配列 (N x 2 または N x M)
        labels: 各点のラベル (オプション)
        figsize: 図のサイズ
        title: タイトル
        xlabel: x軸ラベル
        ylabel: y軸ラベル
        show: 表示するかどうか
        save: 保存先ファイルパス
        dpi: 画像解像度
        dominated_alpha: 支配される解の透明度
        pareto_color: パレート解の色
        **kwargs: その他のプロット引数

    Returns:
        (Figure, Axes, pareto_indices)のタプル

    Examples:
        ```python
        import plotlib as pl
        import numpy as np

        # 2目的最適化の例
        np.random.seed(42)
        n_points = 200

        # ランダムな解
        obj1 = np.random.uniform(0, 10, n_points)
        obj2 = np.random.uniform(0, 10, n_points)

        # パレート解を含むように調整
        pareto_indices = obj1 + obj2 < 8
        objectives = np.column_stack([obj1, obj2])

        pl.pareto_front(objectives,
                       xlabel="Minimize Cost",
                       ylabel="Minimize Time",
                       title="Multi-Objective Optimization")
        ```
    """
    objectives = np.asarray(objectives)
    if objectives.shape[1] != 2:
        raise ValueError("Current implementation supports only 2-objective problems")

    # パレート解を特定
    n_points = objectives.shape[0]
    pareto_indices = np.ones(n_points, dtype=bool)

    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                # j が i を支配するかチェック（最小化問題を想定）
                if np.all(objectives[j] <= objectives[i]) and np.any(
                    objectives[j] < objectives[i]
                ):
                    pareto_indices[i] = False
                    break

    fig, ax = plt.subplots(figsize=figsize)

    # 支配される解
    dominated = ~pareto_indices
    if np.any(dominated):
        ax.scatter(
            objectives[dominated, 0],
            objectives[dominated, 1],
            alpha=dominated_alpha,
            c="gray",
            label="Dominated Solutions",
            **kwargs,
        )

    # パレート解
    if np.any(pareto_indices):
        pareto_objectives = objectives[pareto_indices]
        ax.scatter(
            pareto_objectives[:, 0],
            pareto_objectives[:, 1],
            c=pareto_color,
            s=100,
            label="Pareto Front",
            edgecolor="black",
            linewidth=1,
            **kwargs,
        )

        # パレートフロントを線で結ぶ
        sorted_indices = np.argsort(pareto_objectives[:, 0])
        sorted_pareto = pareto_objectives[sorted_indices]
        ax.plot(
            sorted_pareto[:, 0],
            sorted_pareto[:, 1],
            c=pareto_color,
            alpha=0.7,
            linewidth=2,
            linestyle="--",
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()

    return (fig, ax, pareto_indices)


if __name__ == "__main__":
    print("nnlib.viz module loaded successfully")
    print(f"Version: {__version__}")
    print(f"Registered plotters: {list(PlotRegistry.list_plotters().keys())}")
