import sys
import os
import traceback

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import lemon.nnlib as nl
import lemon.numlib as nm


def test_layer_norm():
    print("Testing LayerNorm...")
    xp = nm.np

    # Test 1: output shape
    layer = nl.LayerNorm(8)
    x = nm.randn(4, 8)
    y = layer(x)
    assert y.shape == (4, 8), "Shape should be preserved"
    print("  ✅ output shape (2D)")

    # Test 2: 3D input (batch, seq, d_model)
    layer = nl.LayerNorm(16)
    x = nm.randn(2, 10, 16)
    y = layer(x)
    assert y.shape == (2, 10, 16), "Shape should be preserved (3D)"
    print("  ✅ output shape (3D)")

    # Test 3: normalized mean ≈ 0, std ≈ 1 over last dim
    x = nm.randn(4, 32)
    layer = nl.LayerNorm(32, elementwise_affine=False)
    y = layer(x)
    mean = xp.mean(y._data, axis=-1)
    std = xp.std(y._data, axis=-1)
    assert xp.allclose(mean, 0.0, atol=1e-5), "Mean should be ~0"
    assert xp.allclose(std, 1.0, atol=1e-3), "Std should be ~1"
    print("  ✅ normalization correctness (mean≈0, std≈1)")

    # Test 4: gradient
    nm.autograd.enable()
    layer = nl.LayerNorm(8)
    x = nm.randn(4, 8, requires_grad=True)
    y = layer(x)
    loss = nm.sum(y)
    loss.backward()
    assert x.grad is not None, "Input gradient should be computed"
    for p in layer.parameters():
        assert p.grad is not None, "Parameter gradients should be computed"
    print("  ✅ gradient (input + weight + bias)")

    # Test 5: repr
    assert "LayerNorm" in repr(nl.LayerNorm(64))
    print("  ✅ repr")

    print("✅ All LayerNorm tests passed!\n")


def test_embedding():
    print("Testing Embedding...")
    xp = nm.np

    # Test 1: output shape
    emb = nl.Embedding(100, 16)
    x = nm.tensor([0, 5, 3, 9])
    y = emb(x)
    assert y.shape == (4, 16), f"Shape should be (4, 16), got {y.shape}"
    print("  ✅ output shape")

    # Test 2: 2D index input
    x = nm.tensor([[0, 1, 2], [3, 4, 5]])
    y = emb(x)
    assert y.shape == (2, 3, 16), f"Shape should be (2, 3, 16), got {y.shape}"
    print("  ✅ 2D index input")

    # Test 3: same index returns same vector
    x1 = nm.tensor([3])
    x2 = nm.tensor([3])
    y1 = emb(x1)
    y2 = emb(x2)
    assert xp.allclose(y1._data, y2._data), "Same index should give same embedding"
    print("  ✅ deterministic lookup")

    # Test 4: gradient
    nm.autograd.enable()
    emb = nl.Embedding(10, 8)
    x = nm.tensor([0, 2, 0])  # index 0 appears twice
    y = emb(x)
    loss = nm.sum(y)
    loss.backward()
    assert emb.weight.grad is not None, "Weight gradient should be computed"
    print("  ✅ gradient")

    # Test 5: padding_idx gradient is zero
    nm.autograd.enable()
    emb = nl.Embedding(10, 8, padding_idx=0)
    x = nm.tensor([0, 1, 2])
    y = emb(x)
    loss = nm.sum(y)
    loss.backward()
    assert xp.allclose(emb.weight.grad._data[0], 0.0), (
        "padding_idx gradient should be 0"
    )
    print("  ✅ padding_idx gradient zeroed")

    # Test 6: repr
    assert "Embedding" in repr(nl.Embedding(100, 64))
    print("  ✅ repr")

    print("✅ All Embedding tests passed!\n")


def test_positional_encoding():
    print("Testing PositionalEncoding...")
    xp = nm.np

    # Test 1: output shape preserved
    pe = nl.PositionalEncoding(d_model=16, max_len=100)
    x = nm.randn(2, 10, 16)
    y = pe(x)
    assert y.shape == (2, 10, 16), f"Shape should be (2, 10, 16), got {y.shape}"
    print("  ✅ output shape")

    # Test 2: output != input (encoding is added)
    x = nm.zeros(1, 5, 16)
    y = pe(x)
    assert not xp.allclose(y._data, x._data), "PE should change the values"
    print("  ✅ encoding is added")

    # Test 3: gradient flows through
    nm.autograd.enable()
    x = nm.randn(2, 5, 16, requires_grad=True)
    y = pe(x)
    loss = nm.sum(y)
    loss.backward()
    assert x.grad is not None, "Gradient should flow through PE"
    print("  ✅ gradient flows through")

    # Test 4: repr
    assert "PositionalEncoding" in repr(pe)
    print("  ✅ repr")

    print("✅ All PositionalEncoding tests passed!\n")


def test_multi_head_attention():
    print("Testing MultiHeadAttention...")

    # Test 1: output shape (self-attention)
    nm.autograd.enable()
    attn = nl.MultiHeadAttention(d_model=32, num_heads=4)
    x = nm.randn(2, 10, 32)
    out = attn(x, x, x)
    assert out.shape == (2, 10, 32), f"Shape should be (2, 10, 32), got {out.shape}"
    print("  ✅ output shape (self-attention)")

    # Test 2: cross-attention (different seq lengths)
    q = nm.randn(2, 6, 32)
    kv = nm.randn(2, 10, 32)
    out = attn(q, kv, kv)
    assert out.shape == (2, 6, 32), f"Shape should be (2, 6, 32), got {out.shape}"
    print("  ✅ cross-attention (different seq lengths)")

    # Test 3: invalid num_heads
    try:
        nl.MultiHeadAttention(d_model=32, num_heads=5)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "divisible" in str(e)
    print("  ✅ invalid num_heads detection")

    # Test 4: gradient
    nm.autograd.enable()
    attn = nl.MultiHeadAttention(d_model=16, num_heads=2)
    x = nm.randn(2, 5, 16, requires_grad=True)
    out = attn(x, x, x)
    loss = nm.sum(out)
    loss.backward()
    assert x.grad is not None, "Input gradient should be computed"
    for p in attn.parameters():
        assert p.grad is not None, "Parameter gradients should be computed"
    print("  ✅ gradient (input + all parameters)")

    # Test 5: repr
    assert "MultiHeadAttention" in repr(attn)
    print("  ✅ repr")

    print("✅ All MultiHeadAttention tests passed!\n")


def test_transformer_stack():
    """Test a simple Transformer-like stack"""
    print("Testing Transformer-like stack...")

    nm.autograd.enable()

    d_model = 32
    batch, seq = 2, 8

    emb = nl.Embedding(50, d_model)
    pe = nl.PositionalEncoding(d_model)
    attn = nl.MultiHeadAttention(d_model, num_heads=4)
    norm1 = nl.LayerNorm(d_model)
    ff = nl.Sequential(
        nl.Linear(d_model, d_model * 2), nl.Relu(), nl.Linear(d_model * 2, d_model)
    )
    norm2 = nl.LayerNorm(d_model)

    x = nm.tensor([[i % 50 for i in range(seq)] for _ in range(batch)])

    # Forward pass
    h = emb(x)
    h = pe(h)
    h = norm1(h + attn(h, h, h))
    h = norm2(h + ff(h))

    assert h.shape == (batch, seq, d_model), (
        f"Output shape should be ({batch}, {seq}, {d_model})"
    )

    # Backward
    loss = nm.sum(h)
    loss.backward()
    assert emb.weight.grad is not None, "Embedding gradient should be computed"
    print("  ✅ full Transformer encoder block forward + backward")

    print("✅ All Transformer stack tests passed!\n")


def test_transformer_encoder_layer():
    print("Testing TransformerEncoderLayer...")

    nm.autograd.enable()
    d_model, num_heads = 32, 4

    # Test 1: output shape (Post-LN)
    layer = nl.TransformerEncoderLayer(d_model=d_model, num_heads=num_heads)
    x = nm.randn(2, 10, d_model)
    out = layer(x)
    assert out.shape == (2, 10, d_model), f"Shape mismatch: {out.shape}"
    print("  ✅ output shape (Post-LN)")

    # Test 2: Pre-LN variant
    layer_pre = nl.TransformerEncoderLayer(
        d_model=d_model, num_heads=num_heads, norm_first=True
    )
    out_pre = layer_pre(x)
    assert out_pre.shape == (2, 10, d_model), f"Pre-LN shape mismatch: {out_pre.shape}"
    print("  ✅ output shape (Pre-LN)")

    # Test 3: gelu activation
    layer_gelu = nl.TransformerEncoderLayer(
        d_model=d_model, num_heads=num_heads, activation="gelu"
    )
    out_gelu = layer_gelu(x)
    assert out_gelu.shape == (2, 10, d_model)
    print("  ✅ gelu activation")

    # Test 4: invalid activation
    try:
        nl.TransformerEncoderLayer(
            d_model=d_model, num_heads=num_heads, activation="tanh"
        )
        assert False, "Should raise ValueError"
    except ValueError:
        pass
    print("  ✅ invalid activation detection")

    # Test 5: with mask
    from lemon.nnlib.layer.transformer.mask import causal_mask

    mask = causal_mask(10)
    out_masked = layer(x, mask=mask)
    assert out_masked.shape == (2, 10, d_model)
    print("  ✅ with causal mask")

    # Test 6: gradient
    x = nm.randn(2, 5, d_model, requires_grad=True)
    out = layer(x)
    loss = nm.sum(out)
    loss.backward()
    assert x.grad is not None, "Input gradient should be computed"
    for p in layer.parameters():
        assert p.grad is not None, "Parameter gradients should be computed"
    print("  ✅ gradient")

    # Test 7: repr
    assert "TransformerEncoderLayer" in repr(layer)
    print("  ✅ repr")

    print("✅ All TransformerEncoderLayer tests passed!\n")


def test_transformer_encoder():
    print("Testing TransformerEncoder...")

    nm.autograd.enable()
    d_model, num_heads = 32, 4

    # Test 1: output shape
    enc_layer = nl.TransformerEncoderLayer(d_model=d_model, num_heads=num_heads)
    encoder = nl.TransformerEncoder(enc_layer, num_layers=3)
    x = nm.randn(2, 10, d_model)
    out = encoder(x)
    assert out.shape == (2, 10, d_model), f"Shape mismatch: {out.shape}"
    print("  ✅ output shape (3 layers)")

    # Test 2: with final norm
    final_norm = nl.LayerNorm(d_model)
    encoder_normed = nl.TransformerEncoder(enc_layer, num_layers=2, norm=final_norm)
    out_normed = encoder_normed(x)
    assert out_normed.shape == (2, 10, d_model)
    print("  ✅ with final norm")

    # Test 3: parameters collected from all layers
    params = encoder.parameters()
    assert len(params) > 0, "Should have parameters"
    print("  ✅ parameters collected")

    # Test 4: gradient through stack
    x = nm.randn(2, 5, d_model, requires_grad=True)
    out = encoder(x)
    loss = nm.sum(out)
    loss.backward()
    assert x.grad is not None, "Input gradient should flow through encoder"
    print("  ✅ gradient through stack")

    # Test 5: repr
    assert "TransformerEncoder" in repr(encoder)
    print("  ✅ repr")

    print("✅ All TransformerEncoder tests passed!\n")


def test_transformer_decoder_layer():
    print("Testing TransformerDecoderLayer...")

    nm.autograd.enable()
    d_model, num_heads = 32, 4

    # Test 1: output shape
    layer = nl.TransformerDecoderLayer(d_model=d_model, num_heads=num_heads)
    tgt = nm.randn(2, 8, d_model)
    memory = nm.randn(2, 12, d_model)
    out = layer(tgt, memory)
    assert out.shape == (2, 8, d_model), f"Shape mismatch: {out.shape}"
    print("  ✅ output shape")

    # Test 2: Pre-LN variant
    layer_pre = nl.TransformerDecoderLayer(
        d_model=d_model, num_heads=num_heads, norm_first=True
    )
    out_pre = layer_pre(tgt, memory)
    assert out_pre.shape == (2, 8, d_model)
    print("  ✅ Pre-LN variant")

    # Test 3: with causal mask on target
    from lemon.nnlib.layer.transformer.mask import causal_mask

    tgt_mask = causal_mask(8)
    out_masked = layer(tgt, memory, tgt_mask=tgt_mask)
    assert out_masked.shape == (2, 8, d_model)
    print("  ✅ with causal target mask")

    # Test 4: gradient
    tgt = nm.randn(2, 6, d_model, requires_grad=True)
    memory = nm.randn(2, 10, d_model, requires_grad=True)
    out = layer(tgt, memory)
    loss = nm.sum(out)
    loss.backward()
    assert tgt.grad is not None, "Target gradient should be computed"
    assert memory.grad is not None, "Memory gradient should be computed"
    print("  ✅ gradient (tgt + memory)")

    # Test 5: repr
    assert "TransformerDecoderLayer" in repr(layer)
    print("  ✅ repr")

    print("✅ All TransformerDecoderLayer tests passed!\n")


def test_transformer_decoder():
    print("Testing TransformerDecoder...")

    nm.autograd.enable()
    d_model, num_heads = 32, 4

    # Test 1: output shape
    dec_layer = nl.TransformerDecoderLayer(d_model=d_model, num_heads=num_heads)
    decoder = nl.TransformerDecoder(dec_layer, num_layers=3)
    tgt = nm.randn(2, 8, d_model)
    memory = nm.randn(2, 12, d_model)
    out = decoder(tgt, memory)
    assert out.shape == (2, 8, d_model), f"Shape mismatch: {out.shape}"
    print("  ✅ output shape (3 layers)")

    # Test 2: with final norm
    final_norm = nl.LayerNorm(d_model)
    decoder_normed = nl.TransformerDecoder(dec_layer, num_layers=2, norm=final_norm)
    out_normed = decoder_normed(tgt, memory)
    assert out_normed.shape == (2, 8, d_model)
    print("  ✅ with final norm")

    # Test 3: parameters collected
    params = decoder.parameters()
    assert len(params) > 0
    print("  ✅ parameters collected")

    # Test 4: gradient through stack
    tgt = nm.randn(2, 6, d_model, requires_grad=True)
    memory = nm.randn(2, 10, d_model, requires_grad=True)
    out = decoder(tgt, memory)
    loss = nm.sum(out)
    loss.backward()
    assert tgt.grad is not None, "Target gradient should flow through decoder"
    assert memory.grad is not None, "Memory gradient should flow through decoder"
    print("  ✅ gradient through stack")

    # Test 5: repr
    assert "TransformerDecoder" in repr(decoder)
    print("  ✅ repr")

    print("✅ All TransformerDecoder tests passed!\n")


def test_masks():
    print("Testing mask utilities...")
    xp = nm.np

    # causal_mask
    from lemon.nnlib.layer.transformer.mask import causal_mask, padding_mask

    mask = causal_mask(4)
    assert mask.shape == (4, 4), f"causal_mask shape: {mask.shape}"
    expected = xp.array(
        [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]], dtype=xp.float32
    )
    assert xp.allclose(mask._data, expected), "causal_mask values wrong"
    print("  ✅ causal_mask shape and values")

    # padding_mask
    mask = padding_mask([3, 5, 2], max_len=5)
    assert mask.shape == (3, 5), f"padding_mask shape: {mask.shape}"
    assert mask._data[0, 2] == 1.0 and mask._data[0, 3] == 0.0
    assert xp.all(mask._data[1] == 1.0)
    assert mask._data[2, 1] == 1.0 and mask._data[2, 2] == 0.0
    print("  ✅ padding_mask shape and values")

    # padding_mask auto max_len
    mask2 = padding_mask([2, 4])
    assert mask2.shape == (2, 4)
    print("  ✅ padding_mask auto max_len")

    print("✅ All mask tests passed!\n")


def run_all_tests():
    print("\n" + "=" * 70)
    print("RUNNING TRANSFORMER LAYER TESTS")
    print("=" * 70 + "\n")

    try:
        test_layer_norm()
        test_embedding()
        test_positional_encoding()
        test_multi_head_attention()
        test_transformer_stack()
        test_masks()
        test_transformer_encoder_layer()
        test_transformer_encoder()
        test_transformer_decoder_layer()
        test_transformer_decoder()

        print("=" * 70)
        print("✅ ALL TRANSFORMER LAYER TESTS PASSED!")
        print("=" * 70)
        return True

    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ TEST FAILED")
        print("=" * 70)
        print(f"\nError: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
