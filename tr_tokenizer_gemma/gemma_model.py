import dataclasses
import enum
import os
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from tr_tokenizer import TRTokenizer


class AttentionType(enum.Enum):
    GLOBAL = 1
    LOCAL_SLIDING = 2


class Architecture(enum.Enum):
    GEMMA_1 = 1
    GEMMA_2 = 2
    GEMMA_3 = 3


@dataclasses.dataclass
class GemmaConfig:
    # The architecture of the model.
    architecture: Architecture = Architecture.GEMMA_3
    # The number of tokens in the vocabulary.
    vocab_size: int = 256000
    # The maximum sequence length that this model might ever be used with.
    max_position_embeddings: int = 8192
    # The number of blocks in the model.
    num_hidden_layers: int = 28
    # The number of attention heads used in the attention layers of the model.
    num_attention_heads: int = 16
    # The number of key-value heads for implementing attention.
    num_key_value_heads: int = 16
    # The hidden size of the model.
    hidden_size: int = 3072
    # The dimension of the MLP representations.
    intermediate_size: int = 24576
    # The number of head dimensions.
    head_dim: int = 256
    # The epsilon used by the rms normalization layers.
    rms_norm_eps: float = 1e-6
    # The dtype of the weights.
    dtype: str = 'float32'
    # Whether a quantized version of the model is used.
    quant: bool = False
    # The types of attention used in the layers of the model.
    attn_types: Optional[Sequence[AttentionType]] = None
    # The size of the sliding window used for local attention.
    sliding_window_size: Optional[int] = None
    # If provided, the final logits are softcapped to this value.
    final_logit_softcapping: Optional[float] = None
    # If provided, the attention logits are softcapped to this value.
    attn_logit_softcapping: Optional[float] = None
    # If provided, the query vector is normalized using the
    # inverse square root of this value instead of head_dim.
    query_pre_attn_scalar: Optional[int] = None
    # Whether to use pre mlp normalization.
    use_pre_ffw_norm: bool = False
    # Whether to use post mlp normalization.
    use_post_ffw_norm: bool = False
    # The wave length of the rotary embedding.
    rope_wave_length: dict[AttentionType, int] | None = None
    # Whether to use QK normalization in the attention blocks.
    use_qk_norm: bool = False
    # Vision model config.
    vision_config: None = None
    # The factor by which the rope wave length is divided for global layers.
    rope_scaling_factor: int| None = None

    def get_dtype(self) -> torch.dtype:
        """Convert string dtype to torch.dtype."""
        dtype_map = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'int8': torch.int8,
            'int16': torch.int16,
            'int32': torch.int32,
            'int64': torch.int64,
        }
        return dtype_map.get(self.dtype, torch.float32)



def precompute_freqs_cis(dim: int,
                         end: int,
                         theta: float = 10000.0,
                         rope_scaling_factor:int = 1) -> torch.Tensor:
    """Precomputes the frequency cis."""
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    freqs = freqs/rope_scaling_factor
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Applies the rotary embedding to the query and key tensors."""
    x_ = torch.view_as_complex(
        torch.stack(torch.chunk(x.transpose(1, 2).float(), 2, dim=-1),
                    dim=-1))
    x_out = torch.view_as_real(x_ * freqs_cis).type_as(x)
    x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2],
                          -1).transpose(1, 2)
    return x_out

class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        add_unit_offset: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # Llama does x.to(float16) * w whilst Gemma2 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = self._norm(x.float())
        if self.add_unit_offset:
            output = output * (1 + self.weight.float())
        else:
            output = output * self.weight.float()
        return output.type_as(x)


class GemmaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = F.gelu(gate, approximate="tanh")
        up = self.up_proj(x)
        fuse = gate * up
        outputs = self.down_proj(fuse)
        return outputs

class GemmaAttention(nn.Module):
    def __init__(
        self,
        config: GemmaConfig,
        attn_type: AttentionType,
    ):
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        if config.query_pre_attn_scalar is not None:
            self.scaling = config.query_pre_attn_scalar**-0.5
        else:
            self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        self.query_norm = (
            RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            if config.use_qk_norm
            else None
        )
        self.key_norm = (
            RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            if config.use_qk_norm
            else None
        )

        self.attn_type = attn_type
        self.sliding_window_size = config.sliding_window_size
        self.attn_logit_softcapping = config.attn_logit_softcapping

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
        local_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        hidden_states_shape = hidden_states.shape
        assert len(hidden_states_shape) == 3

        batch_size, input_len, _ = hidden_states_shape

        xq = self.q_proj(hidden_states)
        xk = self.k_proj(hidden_states)
        xv = self.v_proj(hidden_states)

        xq = xq.view(batch_size, -1, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, -1, self.num_kv_heads, self.head_dim)

        if self.query_norm is not None and self.key_norm is not None:
            xq = self.query_norm(xq)
            xk = self.key_norm(xk)

        # Positional embedding.
        xq = apply_rotary_emb(xq, freqs_cis=freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis=freqs_cis)

        # Write new kv cache.
        # [batch_size, input_len, n_local_kv_heads, head_dim]
        k_cache, v_cache = kv_cache
        k_cache.index_copy_(1, kv_write_indices, xk)
        v_cache.index_copy_(1, kv_write_indices, xv)

        key = k_cache
        value = v_cache
        if self.num_kv_heads != self.num_heads:
            # [batch_size, max_seq_len, n_local_heads, head_dim]
            key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=2)
            value = torch.repeat_interleave(value,
                                            self.num_queries_per_kv,
                                            dim=2)

        # [batch_size, n_local_heads, input_len, head_dim]
        q = xq.transpose(1, 2)
        # [batch_size, n_local_heads, max_seq_len, head_dim]
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        # [batch_size, n_local_heads, input_len, max_seq_len]
        q.mul_(self.scaling)
        scores = torch.matmul(q, k.transpose(2, 3))
        if (
            self.attn_type == AttentionType.LOCAL_SLIDING
            and self.sliding_window_size is not None
            and local_mask is not None
        ):
            mask = local_mask

        if self.attn_logit_softcapping is not None:
            scores = scores / self.attn_logit_softcapping
            scores = torch.tanh(scores)
            scores = scores * self.attn_logit_softcapping

        scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(q)

        # [batch_size, n_local_heads, input_len, head_dim]
        output = torch.matmul(scores, v)

        # [batch_size, input_len, hidden_dim]
        output = (output.transpose(1, 2).contiguous().view(
            batch_size, input_len, -1))
        output = self.o_proj(output)
        return output


class GemmaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: GemmaConfig,
        attn_type: AttentionType,
    ):
        super().__init__()
        self.attn_type = attn_type
        self.self_attn = GemmaAttention(
            config=config,
            attn_type=self.attn_type,
        )
        self.mlp = GemmaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.use_pre_ffw_norm
            else None
        )
        self.post_feedforward_layernorm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.use_post_ffw_norm
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
        local_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_cache=kv_cache,
            mask=mask,
            local_mask=local_mask,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        if self.pre_feedforward_layernorm is not None:
            hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.post_feedforward_layernorm is not None:
            hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            attn_type = (
                config.attn_types[i % len(config.attn_types)]
                if config.attn_types is not None
                else AttentionType.GLOBAL
            )
            self.layers.append(GemmaDecoderLayer(config, attn_type))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: Mapping[AttentionType, torch.Tensor],
        kv_write_indices: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
        local_mask: torch.Tensor,
    ) -> torch.Tensor:
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(
                hidden_states=hidden_states,
                freqs_cis=freqs_cis.get(layer.attn_type),
                kv_write_indices=kv_write_indices,
                kv_cache=kv_caches[i],
                mask=mask,
                local_mask=local_mask,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class GemmaForCausalLM(nn.Module):
  def __init__(
        self,
        config: GemmaConfig,
        tokenizer = None,
    ):
    super().__init__()
    self.config = config
    assert config.hidden_size % config.num_attention_heads == 0

    max_seq_len = config.max_position_embeddings
    head_dim = config.head_dim
    vocab_size = config.vocab_size

    self.tokenizer = tokenizer if tokenizer is not None else TRTokenizer()
    self.embedder = nn.Embedding(vocab_size, config.hidden_size)
    self.model = GemmaModel(config)

    # Pre-compute rotary embedding table.
    if config.rope_wave_length is None:
      raise ValueError('rope_wave_length must be provided for Gemma3.')

    rope_lengths = config.rope_wave_length
    defaults = {
              AttentionType.LOCAL_SLIDING: 10_000,
              AttentionType.GLOBAL: 10_000,
          }

    for attn_type, name in [
              (AttentionType.LOCAL_SLIDING, 'local_freqs_cis'),
              (AttentionType.GLOBAL, 'global_freqs_cis'),
          ]:
      theta = rope_lengths.get(
                  attn_type, defaults[attn_type]
              )
      self._register_freqs_cis(name, head_dim, max_seq_len, theta=theta)

  def _register_freqs_cis(
        self, name: str, head_dim: int, max_seq_len: int, theta: int = 10_000
    ):
    self.register_buffer(
            name, precompute_freqs_cis(head_dim, max_seq_len * 2, theta=theta)
        )

  @torch.no_grad()
  def forward(
        self,
        input_token_ids: torch.Tensor,
        input_positions: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
        output_positions: torch.Tensor,
        temperatures: Union[torch.Tensor, None],
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
        local_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs_cis = {}

    freqs_cis[AttentionType.LOCAL_SLIDING] = (
              self.local_freqs_cis.index_select(0, input_positions)
          )
    freqs_cis[AttentionType.GLOBAL] = (
              self.global_freqs_cis.index_select(0, input_positions)
          )

    kv_write_indices = input_positions

    # [batch_size, input_len, hidden_size]
    hidden_states = self.embedder(input_token_ids)
    # Gemma normalizes the embedding by sqrt(hidden_size).
    # Gemma2 downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
    # See https://github.com/huggingface/transformers/pull/29402
    normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype, device=hidden_states.device)
    hidden_states = hidden_states * normalizer

    hidden_states = self.model(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_caches=kv_caches,
            mask=mask,
            local_mask=local_mask,
        )
    embedder_weight = self.embedder.weight
    hidden_states = hidden_states.index_select(
    1, output_positions).squeeze(dim=1)
    logits = torch.matmul(hidden_states, embedder_weight.t())
    if temperatures is None:
      return torch.argmax(logits, dim=-1).squeeze(dim=-1), logits

    # Apply temperature scaling.
    logits.div_(temperatures.unsqueeze(dim=1))

    # Calculate probabilities with softmax.
    probs = torch.softmax(logits, dim=-1, dtype=torch.float)
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

    # Apply top-p, top-k.
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    top_ps_mask = (probs_sum - probs_sort) > top_ps.unsqueeze(dim=1)
    probs_sort = torch.where(top_ps_mask, 0, probs_sort)

    top_ks_mask = torch.arange(probs_idx.shape[-1],
                                device=probs_idx.device)
    top_ks_mask = top_ks_mask.expand(probs_idx.shape[0], -1)
    top_ks_mask = top_ks_mask >= top_ks.unsqueeze(dim=1)
    probs_sort = torch.where(top_ks_mask, 0, probs_sort)

    # Re-normalization.
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    probs = torch.gather(probs_sort,
                          dim=-1,
                          index=torch.argsort(probs_idx, dim=-1))

    next_token_ids = torch.multinomial(probs,
                                        num_samples=1,
                                        replacement=True).squeeze(dim=-1)

    return next_token_ids, logits

  def generate(
        self,
        prompts: Union[str, Sequence[str]],
        device: Any = "cpu",
        output_len: int = 100,
        temperature: Union[float, None] = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
    ) -> Union[str, Sequence[str]]:
    """Generates responses for given prompts using Gemma model."""
    # If a single prompt is provided, treat it as a batch of 1.
    is_str_prompt = isinstance(prompts, str)
    if is_str_prompt:
      prompts = [prompts]

    batch_size = len(prompts)
    prompt_tokens = [self.tokenizer.encode(prompt) for prompt in prompts]
    min_prompt_len = min(len(p) for p in prompt_tokens)
    max_prompt_len = max(len(p) for p in prompt_tokens)
    max_seq_len = max_prompt_len + output_len
    assert max_seq_len <= self.config.max_position_embeddings

    # build KV caches
    kv_caches = []
    for _ in range(self.config.num_hidden_layers):
      size = (batch_size, max_seq_len, self.config.num_key_value_heads,
                    self.config.head_dim)
      dtype = self.config.get_dtype()
      k_cache = torch.zeros(size=size, dtype=dtype, device=device)
      v_cache = torch.zeros(size=size, dtype=dtype, device=device)
      kv_caches.append((k_cache, v_cache))

    # prepare inputs
    token_ids_tensor = torch.full((batch_size, max_seq_len),
                                      self.tokenizer.pad_token_id, dtype=torch.int64)
    input_token_ids_tensor = torch.full((batch_size, min_prompt_len),
                                            self.tokenizer.pad_token_id,
                                            dtype=torch.int64)
    for i, p in enumerate(prompt_tokens):
      token_ids_tensor[i, :len(p)] = torch.tensor(p)
      input_token_ids_tensor[i, :min_prompt_len] = torch.tensor(
                p[:min_prompt_len])
    token_ids_tensor = token_ids_tensor.to(device)
    input_token_ids_tensor = input_token_ids_tensor.to(device)
    prompt_mask_tensor = token_ids_tensor != self.tokenizer.pad_token_id
    input_positions_tensor = torch.arange(0, min_prompt_len,
                                              dtype=torch.int64).to(device)
    mask_tensor = torch.full((1, 1, max_seq_len, max_seq_len),
                                 -2.3819763e38).to(torch.float)
    mask_tensor = torch.triu(mask_tensor, diagonal=1).to(device)
    local_mask_tensor = mask_tensor + torch.tril(
            torch.full((1, 1, max_seq_len, max_seq_len), -2.3819763e38, device=device),
            diagonal=-self.config.sliding_window_size,
        ) if self.config.sliding_window_size else None
    curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
    curr_local_mask_tensor = local_mask_tensor.index_select(
          2, input_positions_tensor
      ) if local_mask_tensor is not None else None
    output_positions_tensor = torch.LongTensor([min_prompt_len - 1]).to(device)
    temperatures_tensor = None if not temperature else torch.FloatTensor(
            [temperature] * batch_size).to(device)
    top_ps_tensor = torch.FloatTensor([top_p] * batch_size).to(device)
    top_ks_tensor = torch.LongTensor([top_k] * batch_size).to(device)
    output_index = torch.tensor(min_prompt_len, dtype=torch.int64).to(
            device)

    # Prefill up to min_prompt_len tokens, then treat other prefill as
    # decode and ignore output.
    for i in range(max_seq_len - min_prompt_len):
      next_token_ids, _ = self(
                input_token_ids=input_token_ids_tensor,
                input_positions=input_positions_tensor,
                kv_write_indices=None,
                kv_caches=kv_caches,
                mask=curr_mask_tensor,
                output_positions=output_positions_tensor,
                temperatures=temperatures_tensor,
                top_ps=top_ps_tensor,
                top_ks=top_ks_tensor,
                local_mask=curr_local_mask_tensor,
            )

      curr_prompt_mask = prompt_mask_tensor.index_select(
                1, output_index).squeeze(dim=1)
      curr_token_ids = token_ids_tensor.index_select(
                1, output_index).squeeze(dim=1)
      output_token_ids = torch.where(curr_prompt_mask, curr_token_ids,
                                           next_token_ids).unsqueeze(dim=1)
      token_ids_tensor.index_copy_(1, output_index, output_token_ids)

      input_token_ids_tensor = output_token_ids
      input_positions_tensor = output_index.unsqueeze(dim=-1)
      curr_mask_tensor = mask_tensor.index_select(2,
                                                        input_positions_tensor)
      curr_local_mask_tensor = local_mask_tensor.index_select(
                2, input_positions_tensor
            ) if local_mask_tensor is not None else None
      output_positions_tensor = torch.tensor(0, dtype=torch.int64).to(
                device)
      output_index = output_index + 1

    # Detokenization.
    token_ids = token_ids_tensor.tolist()
    results = []
    for i, tokens in enumerate(token_ids):
      trimmed_output = tokens[len(prompt_tokens[i]):len(prompt_tokens[i])
                                    + output_len]
      if self.tokenizer.eos_token_id in trimmed_output:
        eos_index = trimmed_output.index(self.tokenizer.eos_token_id)
        trimmed_output = trimmed_output[:eos_index]
      results.append(self.tokenizer.decode(trimmed_output))

    # If a string was provided as input, return a string as output.
    return results[0] if is_str_prompt else results
  
  def load_weights_from_hf(self, model_state_dict):
     """ 
     Loads weights from a Gemma3 model state dict into this GemmaForCausalLM model.
     
     Source keys: embed_tokens.weight, layers.X.*, norm.weight
     Target keys: embedder.weight, model.layers.X.*, model.norm.weight
     
     Key mappings:
     - embed_tokens.weight -> embedder.weight
     - layers.X.self_attn.q_norm.weight -> model.layers.X.self_attn.query_norm.weight
     - layers.X.self_attn.k_norm.weight -> model.layers.X.self_attn.key_norm.weight
     - layers.X.* -> model.layers.X.* (for all other layer weights)
     - norm.weight -> model.norm.weight
     """
     
     # Create a new state dict with mapped keys
     new_state_dict = {}
     
     for key, value in model_state_dict.items():
         if key == 'embed_tokens.weight':
             # Map embedding layer
             new_state_dict['embedder.weight'] = value
         elif key == 'norm.weight':
             # Map final layer norm
             new_state_dict['model.norm.weight'] = value
         elif key.startswith('layers.'):
             # Map layer weights
             if 'self_attn.q_norm' in key:
                 new_key = 'model.' + key.replace('self_attn.q_norm', 'self_attn.query_norm')
             elif 'self_attn.k_norm' in key:
                 new_key = 'model.' + key.replace('self_attn.k_norm', 'self_attn.key_norm')
             else:
                 new_key = 'model.' + key
             
             new_state_dict[new_key] = value
     
     # Load the mapped state dict
     missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)
     
     if missing_keys:
         print(f"Missing keys: {missing_keys[:5]}...")  # Show first 5 missing keys
     if unexpected_keys:
         print(f"Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5 unexpected keys
     
     print(f"Successfully loaded {len(new_state_dict)} weights")

  def save_pretrained(self, save_directory: str):
     # save model create directory if not exists
     os.makedirs(save_directory, exist_ok=True)
     torch.save(self.model.state_dict(), os.path.join(save_directory, "model.pth"))

  def from_pretrained(self, save_directory: str):
     # load model
     self.model.load_state_dict(torch.load(os.path.join(save_directory, "model.pth")))

def get_config_for_270m(dtype: str) -> GemmaConfig:
  return GemmaConfig(
      dtype=dtype,
      architecture=Architecture.GEMMA_3,
      num_hidden_layers=18,
      num_attention_heads=4,
      num_key_value_heads=1,
      hidden_size=640,
      intermediate_size=2048,
      use_pre_ffw_norm=True,
      use_post_ffw_norm=True,
      head_dim=256,
      attn_types=(
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.GLOBAL,
      ),
      sliding_window_size=512,
      rope_wave_length={
          AttentionType.LOCAL_SLIDING: 10_000,
          AttentionType.GLOBAL: 1_000_000,
      },
      vocab_size=262_144,
      max_position_embeddings=32_768,
      use_qk_norm=True,
      vision_config=None,
  )

def get_config_for_270m_tr_tokenizer(dtype: str) -> GemmaConfig:
  return GemmaConfig(
      dtype=dtype,
      architecture=Architecture.GEMMA_3,
      num_hidden_layers=18,
      num_attention_heads=4,
      num_key_value_heads=1,
      hidden_size=640,
      intermediate_size=2048,
      use_pre_ffw_norm=True,
      use_post_ffw_norm=True,
      head_dim=256,
      attn_types=(
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.LOCAL_SLIDING,
          AttentionType.GLOBAL,
      ),
      sliding_window_size=512,
      rope_wave_length={
          AttentionType.LOCAL_SLIDING: 10_000,
          AttentionType.GLOBAL: 1_000_000,
      },
      vocab_size=32_768,
      max_position_embeddings=32_768,
      use_qk_norm=True,
      vision_config=None,
  )

