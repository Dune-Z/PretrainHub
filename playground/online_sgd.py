import torch
import triton
from typing import Optional, Tuple
from utils import tensor_cache, input_guard, autocast_custom_bwd, autocast_custom_fwd        
from wy_fast import fwd_prepare_wy_repr


def l2norm_fwd(x: torch.Tensor) -> torch.Tensor:
    pass


def l2norm_bwd(x: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
    pass


def prepare_chunk_indices(offsets: Optional[torch.LongTensor], chunk_size: int) -> torch.LongTensor:
    pass


@tensor_cache
def prepare_lens(offsets: torch.LongTensor) -> torch.LongTensor:
    return offsets[1:] - offsets[:-1]


@tensor_cache
def prepare_chunk_offsets(
    offsets: torch.LongTensor,
    chunk_size: int
) -> torch.LongTensor:
    return torch.cat([offsets.new_tensor([0]), triton.cdiv(prepare_lens(offsets), chunk_size)]).cumsum(-1)



def chunk_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor],
    initial_state: torch.Tensor,
    output_final_state: bool,
    offsets: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    chunk_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor]:
    if head_first:
        B, H, T, K, V = *k.shape, u.shape[-1]
    else:
        B, T, H, K, V = *k.shape, u.shape[-1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if offsets is None:
        N, NT, chunk_offset = B, torch.cdiv(T, BT), None
    else:
        N = len(offsets) - 1
        chunk_offset = prepare_chunk_offsets(offsets, BT)
        NT = chunk_offset[-1]
    BK = triton.next_power_of_2(K)
    assert BK <= 256, "current kernel does not support head dimension larger than 256."
    # H100 can have larger block size
    if torch.cuda.get_device_capability()[0] >= 9:
        BV = 64
        BC = 64
    # A100
    elif torch.cuda.get_device_capability() == (8, 0):
        BV = 32
        BC = 64
    else:
        BV = 32
        BC = 64 if K <= 128 else 32
    BC = min(BT, BC)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, 'NK > 1 is not supported because it involves time-consuming synchronization'

    if head_first:
        h = k.new_empty(B, H, NT, K, V)
    else:
        h = k.new_empty(B, NT, H, K, V)

    final_state = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    v_new = torch.empty_like(u)
    grid = (NK, NV, N * H)


def chunk_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    offsets: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    chunk_size: int = 64
):
    T = q.shape[2] if head_first else q.shape[1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    # obtain WY representation. u is actually the new v.
    w, u, A = fwd_prepare_wy_repr(
        k=k,
        v=v,
        beta=beta,
        offsets=offsets,
        indices=indices,
        head_first=head_first,
        chunk_size=BT
    )
    h, v_new, final_state = ...


def chunk_delta_rule_bwd():
    pass


class ChunkDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        offsets: Optional[torch.LongTensor] = None,
        head_first: bool = True,
        use_qk_l2norm_in_kernel: bool = True
    ):
        T = q.shape[2] if head_first else q.shape[1]
        chunk_size = min(64, max(triton.next_power_of_2(T), 16))

        q_orig = q
        k_orig = k

        if use_qk_l2norm_in_kernel:
            q = l2norm_fwd(q)
            k = l2norm_fwd(k)

        # 2-d indices denoting the offsets of chunks in each sequence
        # for example, if the passed `offsets` is [0, 100, 356] and `chunk_size` is 64,
        # then there are 2 and 4 chunks in the 1st and 2nd sequences respectively, and `indices` will be
        # [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [1, 3]]
        indices = prepare_chunk_indices(offsets, chunk_size) if offsets is not None else None

        o, A, final_state = chunk_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            offsets=offsets,
            indices=indices,
            head_first=head_first,
            chunk_size=chunk_size
        )
        ctx.save_for_backward(q_orig, k_orig, v, beta, A, initial_state)
        ctx.chunk_size = chunk_size
        ctx.scale = scale
        ctx.offsets = offsets
        ctx.indices = indices
        ctx.head_first = head_first
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor
    ):
        q, k, v, beta, A, initial_state = ctx.saved_tensors
        use_qk_l2norm_in_kernel = ctx.use_qk_l2norm_in_kernel
        if use_qk_l2norm_in_kernel:
            q, q_orig = l2norm_fwd(q), q
            k, k_orig = l2norm_fwd(k), k

        dq, dk, dv, db, dh0 = chunk_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            beta=beta,
            A=A,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            offsets=ctx.offsets,
            indices=ctx.indices,
            head_first=ctx.head_first,
            chunk_size=ctx.chunk_size
        )
        if use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q_orig, dq)
            dk = l2norm_bwd(k_orig, dk)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), db.to(beta.dtype), None, dh0, None, None, None, None, None, None



@torch.compiler.disable
def chunk_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    use_qk_l2norm_in_kernel: bool = False
):
    r"""
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `True`.
        use_qk_l2norm_in_kernel (Optional[bool]):
            Whether to use qk l2norm within the kernel for saving GPU memory.
            Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.delta_rule import chunk_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_delta_rule(q, k, v, beta,
                                     initial_state=h0,
                                     output_final_state=True,
                                     head_first=False)
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = chunk_delta_rule(q, k, v, beta,
                                             initial_state=h0,
                                             output_final_state=True,
                                             cu_seqlens=cu_seqlens,
                                             head_first=False)
    """
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, "ChunkDeltaRuleFunction does not support float32. Please use bfloat16."
    assert len(beta.shape) == 3, "beta must be of shape (batch size, num of head, seq len)."

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                             f"Please flatten variable-length inputs before processing.")
        if head_first:
            raise RuntimeError("Sequences with variable lengths are not supported for head-first mode")
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(f"The number of initial states is expected to be equal to the number of input sequences, "
                             f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.")
    scale = k.shape[-1] ** -0.5 if scale is None else scale
    o, final_state = ChunkDeltaRuleFunction.apply(
        q,
        k,
        v,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        head_first,
        use_qk_l2norm_in_kernel
    )
    return o, final_state


if __name__ == '__main__':
    offsets = torch.LongTensor([3, 100, 255, 4])
    print(offsets)
    offsets = prepare_chunk_offsets(offsets, 64)
    print(offsets)