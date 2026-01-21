"""
CSVDataset

Generic dataset loader for CSV files with automatic column detection.
"""

import os
import csv as csv_module
import lemon.numlib as nm
from lemon.nnlib.data import Dataset


class CSVDataset(Dataset):
    """
    Load dataset from CSV file

    Generic dataset loader supporting both file-based data and direct data.
    Automatically detects columns starting with 'x' for inputs and 'y' for outputs.

    CSV format:
        x0,x1,...,y0,y1,...

    Parameters
    ----------
    root : str
        Root directory for files (default: './data')
    csv_file : str
        Path to CSV file
    xs : list of str, optional
        Column names for input data (default: None = auto-detect 'x*' columns)
    ys : list of str, optional
        Column names for labels/targets (default: None = auto-detect 'y*' columns)
    loader : callable, optional
        Function to load data from file path (filepath -> data).
        Required for file mode. If provided, xs[0] is treated as file paths.
        If None, xs columns are treated as raw data (data mode).
    transform : callable, optional
        Transform to apply to loaded data

    Examples
    --------
    # File mode (images) - user provides loader
    >>> def image_loader(filepath):
    ...     from PIL import Image
    ...     import lemon.numlib as nm
    ...     xp = nm.get_array_module(nm.zeros(1)._data)
    ...     img = Image.open(filepath).convert('RGB')
    ...     img_array = xp.array(img, dtype=xp.float32) / 255.0
    ...     return img_array.reshape(-1)
    >>>
    >>> from lemon.datasets.tabular import CSVDataset
    >>> dataset = CSVDataset(
    ...     root='./data',
    ...     csv_file='labels.csv',
    ...     loader=image_loader
    ... )

    # Data mode (single input) - auto-detect
    >>> dataset = CSVDataset(csv_file='data.csv')

    # Data mode (multiple inputs) - auto-detect
    >>> dataset = CSVDataset(csv_file='iris.csv')
    # CSV: x0,x1,x2,x3,y0

    # Explicit column specification
    >>> dataset = CSVDataset(
    ...     csv_file='iris.csv',
    ...     xs=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
    ...     ys=['species']
    ... )

    # Multiple outputs
    >>> dataset = CSVDataset(
    ...     csv_file='data.csv',
    ...     xs=['x0', 'x1'],
    ...     ys=['y0', 'y1']
    ... )

    CSV Examples
    ------------
    File mode CSV:
        x0,y0
        images/cat.jpg,0
        images/dog.jpg,1

    Data mode CSV (single input):
        x0,y0
        5.1,0
        4.9,0
        7.0,1

    Data mode CSV (multiple inputs):
        x0,x1,x2,x3,y0
        5.1,3.5,1.4,0.2,0
        4.9,3.0,1.4,0.2,0
        7.0,3.2,4.7,1.4,1

    Data mode CSV with metadata:
        x0,x1,y0,author,date
        5.1,3.5,0,Alice,2024-01-01
        4.9,3.0,0,Bob,2024-01-02

    Notes
    -----
    For file mode, loader function must be provided by the user.
    This keeps dataset dependency-free and allows maximum flexibility
    for loading images, audio, text, or any other file format.
    """

    def __init__(
        self,
        root="./data",
        csv_file="labels.csv",
        xs=None,
        ys=None,
        loader=None,
        transform=None,
    ):
        self.root = root
        self.xs = xs
        self.ys = ys
        self.loader = loader
        self.transform = transform

        # Determine mode
        self.mode = "file" if loader is not None else "data"

        # Load CSV
        self._load_csv(csv_file)

    def _load_csv(self, csv_file):
        """Load CSV file and extract data"""

        # Determine CSV path
        # If csv_file is absolute, use it as-is
        # If relative, treat it as relative to root directory
        csv_path = (
            csv_file if os.path.isabs(csv_file) else os.path.join(self.root, csv_file)
        )

        # Read CSV
        with open(csv_path, "r") as f:
            reader = csv_module.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames

        if len(rows) == 0:
            raise ValueError("CSV file is empty")

        # Auto-detect columns if not specified
        if self.xs is None:
            # Find all columns starting with 'x'
            x_cols = sorted([col for col in fieldnames if col.startswith("x")])
            if not x_cols:
                raise ValueError(
                    "No columns starting with 'x' found in CSV. "
                    "Please specify xs parameter explicitly."
                )
            self.xs = x_cols

        if self.ys is None:
            # Find all columns starting with 'y'
            y_cols = sorted([col for col in fieldnames if col.startswith("y")])
            if not y_cols:
                raise ValueError(
                    "No columns starting with 'y' found in CSV. "
                    "Please specify ys parameter explicitly."
                )
            self.ys = y_cols

        # Validate columns exist
        for col in self.xs:
            if col not in fieldnames:
                raise ValueError(f"Column '{col}' not found in CSV")

        for col in self.ys:
            if col not in fieldnames:
                raise ValueError(f"Column '{col}' not found in CSV")

        xp = nm.get_array_module(nm.zeros(1)._data)

        if self.mode == "file":
            # File mode: xs[0] contains file paths
            self.filenames = [row[self.xs[0]] for row in rows]

            # Extract labels
            labels_list = []
            for row in rows:
                y_values = [self._parse_value(row[col]) for col in self.ys]
                if len(y_values) == 1:
                    labels_list.append(y_values[0])
                else:
                    labels_list.append(y_values)

            # Convert to array if all labels are numeric
            if all(
                isinstance(label, (int, float))
                or (
                    isinstance(label, list)
                    and all(isinstance(v, (int, float)) for v in label)
                )
                for label in labels_list
            ):
                self.labels = xp.array(labels_list)
            else:
                self.labels = labels_list

        else:
            # Data mode: xs columns contain data
            data_list = []
            labels_list = []

            for row in rows:
                # Extract input features
                x_values = [float(row[col]) for col in self.xs]
                data_list.append(x_values)

                # Extract labels
                y_values = [self._parse_value(row[col]) for col in self.ys]
                if len(y_values) == 1:
                    labels_list.append(y_values[0])
                else:
                    labels_list.append(y_values)

            self.data = xp.array(data_list, dtype=xp.float32)

            # Convert labels to array if all numeric
            if all(
                isinstance(label, (int, float))
                or (
                    isinstance(label, list)
                    and all(isinstance(v, (int, float)) for v in label)
                )
                for label in labels_list
            ):
                self.labels = xp.array(labels_list)
            else:
                self.labels = labels_list

        # Store metadata (all rows)
        self.metadata = rows

    def _parse_value(self, value_str):
        """Parse value from string to appropriate type"""
        try:
            # Try as integer
            return int(value_str)
        except ValueError:
            try:
                # Try as float
                return float(value_str)
            except ValueError:
                # Keep as string
                return value_str

    def __len__(self):
        if self.mode == "file":
            return len(self.filenames)
        else:
            return len(self.data)

    def __getitem__(self, index):
        if self.mode == "file":
            # File mode: load from file
            filepath = os.path.join(self.root, self.filenames[index])

            if self.loader is None:
                raise RuntimeError(
                    "loader must be provided for file mode. "
                    "Example:\n"
                    "  def image_loader(path):\n"
                    "      from PIL import Image\n"
                    "      import lemon.numlib as nm\n"
                    "      ...\n"
                    "  dataset = CSVDataset(..., loader=image_loader)"
                )

            x = self.loader(filepath)
            y = self.labels[index]

        else:
            # Data mode: get from array
            x = self.data[index]
            y = self.labels[index]

        # Apply transform
        if self.transform:
            x = self.transform(x)

        # Ensure y has consistent shape (scalar -> 1D array)
        xp = nm.get_array_module(y)
        if xp.ndim(y) == 0:  # scalar
            y = xp.array([y])

        return x, y
