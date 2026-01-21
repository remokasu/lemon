from typing import Dict, Iterator, Tuple, Optional, Any
from lemon.nnlib.parameter import Parameter


class Module:
    """
    すべてのニューラルネットワークモジュールの基底クラス

    カスタムレイヤーを作る際は、このクラスを継承して forward() メソッドを実装する。

    Examples
    --------
    >>> import numlib as nm
    >>>
    >>> class Linear(Module):
    ...     def __init__(self, in_features, out_features):
    ...         super().__init__()
    ...         self.weight = Parameter(nm.randn(in_features, out_features))
    ...         self.bias = Parameter(nm.zeros(out_features))
    ...
    ...     def forward(self, x):
    ...         return x @ self.weight.data + self.bias.data
    >>>
    >>> layer = Linear(10, 5)
    >>> x = nm.randn(3, 10)
    >>> y = layer(x)
    """

    def __init__(self):
        self._parameters: Dict[str, Parameter] = {}
        self._modules: Dict[str, "Module"] = {}

    def forward(self, *args, **kwargs):
        """
        順伝播を定義する（サブクラスで実装）

        Raises
        ------
        NotImplementedError
            サブクラスで実装されていない場合
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement forward()")

    def __call__(self, *args, **kwargs):
        """forward() を呼び出す"""
        return self.forward(*args, **kwargs)

    def __setattr__(self, name: str, value):
        """
        属性設定時に Parameter と Module を自動登録
        """
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def parameters(self) -> Iterator[Parameter]:
        """
        すべてのパラメータを返す（再帰的）

        Yields
        ------
        Parameter
            このモジュールとサブモジュールのすべてのパラメータ

        Examples
        --------
        >>> for param in model.parameters():
        ...     print(param.shape)
        """
        # 自分のパラメータ
        for param in self._parameters.values():
            yield param

        # サブモジュールのパラメータ
        for module in self._modules.values():
            yield from module.parameters()

    def zero_grad(self):
        """
        すべてのパラメータの勾配をゼロにリセット

        Examples
        --------
        >>> model.zero_grad()
        >>> loss.backward()
        """
        for param in self.parameters():
            param.zero_grad()

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        """
        名前付きパラメータを返す（再帰的）

        Parameters
        ----------
        prefix : str, optional
            パラメータ名の接頭辞（内部使用、デフォルト: ''）
        recurse : bool, optional
            サブモジュールのパラメータも含めるか（デフォルト: True）

        Yields
        ------
        tuple
            (name, parameter) のタプル

        Examples
        --------
        >>> for name, param in model.named_parameters():
        ...     print(f"{name}: {param.shape}")
        conv1.weight: (64, 3, 3, 3)
        conv1.bias: (64,)
        fc.weight: (10, 64)
        fc.bias: (10,)

        >>> # パラメータごとの勾配確認
        >>> for name, param in model.named_parameters():
        ...     if param.grad is not None:
        ...         print(f"{name}: grad_norm={param.grad.norm()}")
        """
        # 自分のパラメータ
        for name, param in self._parameters.items():
            full_name = f"{prefix}.{name}" if prefix else name
            yield full_name, param

        # サブモジュールのパラメータ（再帰的）
        if recurse:
            for name, module in self._modules.items():
                submodule_prefix = f"{prefix}.{name}" if prefix else name
                yield from module.named_parameters(prefix=submodule_prefix, recurse=True)

    def named_modules(self, memo: Optional[set] = None, prefix: str = '') -> Iterator[Tuple[str, 'Module']]:
        """
        名前付きモジュールを返す（再帰的）

        Parameters
        ----------
        memo : set, optional
            訪問済みモジュールの記録（内部使用、デフォルト: None）
        prefix : str, optional
            モジュール名の接頭辞（内部使用、デフォルト: ''）

        Yields
        ------
        tuple
            (name, module) のタプル

        Examples
        --------
        >>> for name, module in model.named_modules():
        ...     print(f"{name}: {module.__class__.__name__}")
        : MyModel
        conv1: Conv2d
        fc: Linear

        >>> # 特定レイヤーの検索
        >>> for name, module in model.named_modules():
        ...     if isinstance(module, Conv2d):
        ...         print(f"Found Conv2d at: {name}")
        """
        if memo is None:
            memo = set()

        # 自分自身
        if self not in memo:
            memo.add(self)
            yield prefix, self

            # サブモジュール（再帰的）
            for name, module in self._modules.items():
                submodule_prefix = f"{prefix}.{name}" if prefix else name
                yield from module.named_modules(memo, submodule_prefix)

    def state_dict(self, destination: Optional[Dict[str, Any]] = None, prefix: str = '') -> Dict[str, Any]:
        """
        モジュールの状態を辞書形式で返す

        Parameters
        ----------
        destination : dict, optional
            結果を格納する辞書（内部使用、デフォルト: None）
        prefix : str, optional
            キーの接頭辞（内部使用、デフォルト: ''）

        Returns
        -------
        dict
            パラメータ名: パラメータ値（NumPy配列）の辞書

        Examples
        --------
        >>> state = model.state_dict()
        >>> print(state.keys())
        dict_keys(['conv1.weight', 'conv1.bias', 'fc.weight', 'fc.bias'])

        >>> # 保存
        >>> import pickle
        >>> with open('model_weights.pkl', 'wb') as f:
        ...     pickle.dump(state, f)

        >>> # モデルサイズの確認
        >>> import sys
        >>> total_size = sum(v.nbytes for v in state.values())
        >>> print(f"Model size: {total_size / 1024 / 1024:.2f} MB")
        """
        if destination is None:
            destination = {}

        import lemon.numlib as nm

        # 自分のパラメータを保存
        for name, param in self._parameters.items():
            key = f"{prefix}.{name}" if prefix else name
            # NumPy配列に変換して保存
            destination[key] = nm.as_numpy(param.data)

        # サブモジュールのパラメータ（再帰的）
        for name, module in self._modules.items():
            submodule_prefix = f"{prefix}.{name}" if prefix else name
            module.state_dict(destination, submodule_prefix)

        return destination

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        """
        state_dictからモジュールにパラメータを読み込む

        Parameters
        ----------
        state_dict : dict
            パラメータの辞書（state_dict()で取得したもの）
        strict : bool, optional
            厳密モード。Trueの場合、キーが完全一致する必要がある（デフォルト: True）

        Raises
        ------
        KeyError
            strict=True で、キーが一致しない場合

        Examples
        --------
        >>> # 読み込み
        >>> import pickle
        >>> with open('model_weights.pkl', 'rb') as f:
        ...     state = pickle.load(f)
        >>> model.load_state_dict(state)

        >>> # 部分的な読み込み（strict=False）
        >>> # モデル構造が少し変わった場合など
        >>> model.load_state_dict(state, strict=False)

        >>> # Transfer Learning: 特定レイヤー以外を読み込む
        >>> pretrained_state = torch.load('pretrained.pkl')
        >>> model_state = model.state_dict()
        >>> pretrained_state = {k: v for k, v in pretrained_state.items()
        ...                      if k in model_state and 'fc' not in k}
        >>> model.load_state_dict(pretrained_state, strict=False)
        """
        import lemon.numlib as nm

        # 現在のモデルのパラメータ名を取得
        current_keys = set()
        param_dict = {}
        for name, param in self.named_parameters():
            current_keys.add(name)
            param_dict[name] = param

        # state_dictのキーを取得
        loaded_keys = set(state_dict.keys())

        # strict=True の場合、キーの一致を確認
        if strict:
            missing_keys = current_keys - loaded_keys
            unexpected_keys = loaded_keys - current_keys

            if missing_keys:
                raise KeyError(f"Missing keys in state_dict: {missing_keys}")
            if unexpected_keys:
                raise KeyError(f"Unexpected keys in state_dict: {unexpected_keys}")

        # パラメータを読み込み
        for name, value in state_dict.items():
            if name in param_dict:
                # NumPy配列をTensorに変換して設定
                param_dict[name].data = nm.tensor(value)

    def __repr__(self):
        """モジュールの文字列表現"""
        lines = [f"{self.__class__.__name__}("]

        # パラメータ情報
        for name, param in self._parameters.items():
            lines.append(f"  ({name}): Parameter(shape={param.shape})")

        # サブモジュール情報
        for name, module in self._modules.items():
            module_str = repr(module)
            # インデント調整
            module_str = module_str.replace("\n", "\n  ")
            lines.append(f"  ({name}): {module_str}")

        lines.append(")")
        return "\n".join(lines)
