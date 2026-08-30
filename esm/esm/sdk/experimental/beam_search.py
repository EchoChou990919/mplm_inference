# NOTE: This file was created for this mplm_inference repository

"""Generalized Beam Search for ESM3.

Unified framework for Best-of-N, SVDD, and Beam Search via (N, L, K) parameters:
- Best-of-N: N>1, L=1, K=T (independent trajectories, select best at end)
- SVDD:      N=1, L>1, K=1 (single trajectory, branch every step)
- Beam Search: N>1, L>1, K<T (multiple beams, branch every K steps)

Orthogonal to Layer 1 (vanilla) and Layer 2 (CFG) sampling — each step uses
whatever sampling strategy is configured (vanilla/CFG, any unmasking strategy).
"""

import random
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Sequence

import attr
import torch
from tqdm import tqdm

from esm.models.esm3 import ESM3
from esm.models.vqvae import StructureTokenDecoder
from esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    ESMProteinError,
    ESMProteinTensor,
    ForwardAndSampleOutput,
    ForwardTrackData,
    GenerationConfig,
    GenerationTrackConfig,
    LogitsConfig,
    LogitsOutput,
    MultiTrackGenerationConfig,
    SamplingConfig,
    SamplingTrackConfig,
)
from esm.tokenization import TokenizerCollectionProtocol
# reuse the same utility functions for vanilla sampling and CFG sampling
from esm.utils.generation import (
    _BatchedESMProteinTensor,
    _apply_cfg_to_logits,
    _apply_cfg_to_logits_cross_modal,
    _apply_cfg_to_logits_multitrack,
    _batch_forward,
    _build_cfg_uncond_batched_tokens,
    _build_cfg_uncond_batched_tokens_multitrack,
    _cat_batched_tokens,
    _get_annealed_temperature,
    _get_iterative_sampling_mask_for_prompt_and_step,
    _get_iterative_sampling_masks_for_multitrack,
    _get_masked_positions,
    _get_non_special_tokens,
    _sample_per_prompt,
    _slice_tensor_dataclass,
    _stack_protein_tensors,
    _trim_sequence_tensor_dataclass,
)
from esm.utils.noise_schedules import NOISE_SCHEDULE_REGISTRY


# ============================================================================
# Reward Functions
# ============================================================================


# the example reward function provided in guided_generation.py
class RewardFunction(ABC):
    """Abstract reward function: protein_tensor → scalar score."""

    @abstractmethod
    def __call__(
        self,
        protein_tensor: ESMProteinTensor,
        client: ESM3InferenceClient,
        tokenizers: TokenizerCollectionProtocol,
        forward_output: ForwardAndSampleOutput | None = None,
    ) -> float:
        """Compute reward for a (possibly partially masked) protein.

        Args:
            protein_tensor: The protein to score (may still have mask tokens;
                the reward function is responsible for jumpy-denoising if needed).
            client: ESM3 model for forward passes / decoding.
            tokenizers: Tokenizer collection.
            forward_output: If provided, reuse this forward output instead of
                doing a new forward pass (for Option A seq logprob).

        Returns:
            Scalar reward (higher is better).
        """
        ...


# 1. jumpy denoise to fill in all remaining masks with greedy decode
# 2. decode structure tokens to get pTM / pLDDT / PAE reward
class StructureReward(RewardFunction):
    """Reward based on model-predicted structure quality (pTM / pLDDT / pAE).

    Requires structure tokens to be available (after jumpy denoising).
    Uses the VQVAE structure decoder directly for efficient batched scoring.
    """

    def __init__(self, mode: str = "ptm"):
        assert mode in ("ptm", "plddt", "pae"), f"Unknown mode: {mode}"
        self.mode = mode

    def __call__(
        self,
        protein_tensor: ESMProteinTensor,
        client: ESM3InferenceClient,
        tokenizers: TokenizerCollectionProtocol,
        forward_output: ForwardAndSampleOutput | None = None,
    ) -> float:
        # Jumpy denoise: fill all remaining masks with greedy decode
        denoised_pt = _jumpy_denoise(protein_tensor, client)

        # Decode structure tokens → pTM / pLDDT / pAE
        assert isinstance(client, ESM3), "StructureReward requires local ESM3 model"
        structure_decoder = client.get_structure_decoder()
        structure_tokens = denoised_pt.structure
        assert structure_tokens is not None, "No structure tokens after jumpy denoise"

        # structure_decoder.decode expects [B, L]
        if structure_tokens.dim() == 1:
            structure_tokens = structure_tokens.unsqueeze(0)

        with torch.no_grad():
            decoder_output = structure_decoder.decode(structure_tokens)

        if self.mode == "ptm":
            return decoder_output["ptm"].item()
        elif self.mode == "plddt":
            plddt = decoder_output["plddt"]  # [B, L]
            # Exclude BOS/EOS positions
            return plddt[0, 1:-1].mean().item()
        elif self.mode == "pae":
            # Lower PAE is better, so negate
            pae = decoder_output["predicted_aligned_error"]  # [B, L, L]
            # Exclude BOS/EOS
            pae_inner = pae[0, 1:-1, 1:-1]
            return -pae_inner.mean().item()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

# 1. jumpy denoise to fill in all remaining masks with greedy decode
# 2. compute sequence log-probability of the denoised protein as reward (higher is better)  
class SequenceLogprobReward(RewardFunction):
    """Reward based on sequence log-probability.

    Option A (default): Use logprob from the jumpy-denoise forward pass
        (log P(token_i | partially masked context)).
    Option B (use_full_context_logprob=True): After jumpy denoise, do an extra
        forward pass on the fully decoded protein to get teacher-forcing logprob.
    """

    def __init__(self, use_full_context_logprob: bool = False):
        self.use_full_context_logprob = use_full_context_logprob

    def __call__(
        self,
        protein_tensor: ESMProteinTensor,
        client: ESM3InferenceClient,
        tokenizers: TokenizerCollectionProtocol,
        forward_output: ForwardAndSampleOutput | None = None,
    ) -> float:
        if self.use_full_context_logprob:
            # Option B: extra forward on fully decoded protein
            denoised_pt = _jumpy_denoise(protein_tensor, client)
            logprob = _compute_sequence_logprob(denoised_pt, client, tokenizers)
        else:
            # Option A: use forward_output from jumpy denoise
            denoised_pt, fwd_out = _jumpy_denoise_with_output(
                protein_tensor, client, tokenizers
            )
            logprob = _extract_sequence_logprob_from_output(
                denoised_pt, fwd_out, tokenizers
            )
        return logprob


# 1. jumpy denoise to fill sequence masks
# 2. clear structure tokens + coordinates (folding-only input)
# 3. single-step folding (temp=0) to predict structure tokens
# 4. VQVAE decode → pTM / pLDDT / pAE as reward
class SequenceFoldabilityReward(RewardFunction):
    """Reward based on sequence foldability — self-consistency check.

    Measures how confidently the model can fold the decoded sequence into
    a structure, using ONLY the sequence as input (structure/coords cleared).

    Pipeline:
        1. Jumpy denoise: fill all remaining sequence masks (temp=0)
        2. Construct folding-only input: keep sequence, clear structure + coords
        3. Single-step folding: forward_and_sample(structure, temp=0)
        4. VQVAE decode predicted structure tokens → pTM/pLDDT/pAE

    Stronger global signal than SequenceLogprobReward for inverse folding.
    Cost: 2× ESM3 forward + 1× VQVAE decode per evaluation.
    """

    def __init__(self, mode: str = "ptm"):
        assert mode in ("ptm", "plddt", "pae"), f"Unknown mode: {mode}"
        self.mode = mode

    def __call__(
        self,
        protein_tensor: ESMProteinTensor,
        client: ESM3InferenceClient,
        tokenizers: TokenizerCollectionProtocol,
        forward_output: ForwardAndSampleOutput | None = None,
    ) -> float:
        # Step 1: Jumpy denoise to fill all remaining masks
        denoised_pt = _jumpy_denoise(protein_tensor, client)

        # Step 2: Construct folding-only input (clear structure side)
        folding_pt = _build_folding_only_protein(denoised_pt)

        # Step 3: Single-step folding (temp=0) → predicted structure tokens
        fwd_out = client.forward_and_sample(
            folding_pt,
            SamplingConfig(
                structure=SamplingTrackConfig(temperature=0.0),
            ),
        )
        assert not isinstance(fwd_out, ESMProteinError)

        # Step 4: VQVAE decode → reward
        assert isinstance(client, ESM3)
        structure_decoder = client.get_structure_decoder()
        structure_tokens = fwd_out.protein_tensor.structure
        assert structure_tokens is not None

        if structure_tokens.dim() == 1:
            structure_tokens = structure_tokens.unsqueeze(0)

        with torch.no_grad():
            decoder_output = structure_decoder.decode(structure_tokens)

        if self.mode == "ptm":
            return decoder_output["ptm"].item()
        elif self.mode == "plddt":
            return decoder_output["plddt"][0, 1:-1].mean().item()
        elif self.mode == "pae":
            pae = decoder_output["predicted_aligned_error"][0, 1:-1, 1:-1]
            return -pae.mean().item()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


def _build_folding_only_protein(
    denoised_pt: ESMProteinTensor,
) -> ESMProteinTensor:
    """Construct a folding-only protein tensor from a denoised result.

    Keeps only the sequence track; clears structure tokens, coordinates,
    and all other tracks so the model must fold from sequence alone.
    """
    folding_pt = attr.evolve(denoised_pt)
    folding_pt.structure = None
    folding_pt.coordinates = None
    folding_pt.ss8 = None
    folding_pt.sasa = None
    folding_pt.function_annotations = None
    return folding_pt


# 1. if both StructureReward and SequenceLogprobReward are present, do a single jumpy denoise
# 2. compute each reward from the shared denoised result (structure decode for StructureReward, logprob extraction for SequenceLogprobReward)
# 3. return weighted combination of the rewards
class CombinedReward(RewardFunction):
    """Weighted combination of multiple rewards.

    Optimizes the common case (StructureReward + SequenceLogprobReward) by
    sharing a single jumpy denoise forward pass across sub-rewards.
    """

    def __init__(self, rewards: list[RewardFunction], weights: list[float]):
        assert len(rewards) == len(weights)
        self.rewards = rewards
        self.weights = weights

    def __call__(
        self,
        protein_tensor: ESMProteinTensor,
        client: ESM3InferenceClient,
        tokenizers: TokenizerCollectionProtocol,
        forward_output: ForwardAndSampleOutput | None = None,
    ) -> float:
        # Optimized path: if we have both StructureReward and SequenceLogprobReward,
        # share the jumpy denoise to avoid redundant forward passes.
        has_structure = any(isinstance(r, StructureReward) for r in self.rewards)
        has_seqlogprob = any(isinstance(r, SequenceLogprobReward) for r in self.rewards)

        denoised_pt = None
        fwd_out = None

        if has_structure and has_seqlogprob:
            # Do jumpy denoise once, share across sub-rewards
            denoised_pt, fwd_out = _jumpy_denoise_with_output(
                protein_tensor, client, tokenizers
            )

        total = 0.0
        for reward_fn, weight in zip(self.rewards, self.weights):
            if denoised_pt is not None and isinstance(reward_fn, StructureReward):
                # Use pre-denoised result directly
                score = _score_structure_from_denoised(
                    denoised_pt, client, reward_fn.mode
                )
            elif fwd_out is not None and isinstance(reward_fn, SequenceLogprobReward):
                if reward_fn.use_full_context_logprob:
                    assert denoised_pt is not None
                    score = _compute_sequence_logprob(denoised_pt, client, tokenizers)
                else:
                    assert denoised_pt is not None and fwd_out is not None
                    score = _extract_sequence_logprob_from_output(
                        denoised_pt, fwd_out, tokenizers
                    )
            else:
                score = reward_fn(protein_tensor, client, tokenizers, forward_output)
            total += weight * score
        return total


# ============================================================================
# Jumpy Denoise Helpers
# ============================================================================

# greedy single-step decode to fill in all remaining masks, returning the fully denoised protein tensor
def _jumpy_denoise(
    protein_tensor: ESMProteinTensor,
    client: ESM3InferenceClient,
) -> ESMProteinTensor:
    """Fill all remaining mask tokens with greedy (temp=0) single-step decode.

    Returns the fully decoded protein tensor (no mask tokens remaining).
    Does NOT modify the input.
    """
    fwd_out = client.forward_and_sample(
        protein_tensor,
        SamplingConfig(
            sequence=SamplingTrackConfig(temperature=0.0),
            structure=SamplingTrackConfig(temperature=0.0),
        ),
    )
    assert not isinstance(fwd_out, ESMProteinError)
    return fwd_out.protein_tensor

# greedy single-step decode & also return the ForwardAndSampleOutput for logprob extraction
def _jumpy_denoise_with_output(
    protein_tensor: ESMProteinTensor,
    client: ESM3InferenceClient,
    tokenizers: TokenizerCollectionProtocol,
) -> tuple[ESMProteinTensor, ForwardAndSampleOutput]:
    """Like _jumpy_denoise, but also returns the ForwardAndSampleOutput
    so we can extract logprob without an extra forward pass (Option A).

    Uses _batch_forward + _sample_per_prompt directly to get full output
    including logprob.
    """
    assert isinstance(client, ESM3)
    pt = attr.evolve(protein_tensor)

    # Fill in missing tracks with defaults (same as forward_and_sample does)
    default_pt = ESMProteinTensor.empty(
        len(pt) - 2, tokenizers=client.tokenizers, device=pt.device
    )
    for track in attr.fields(ESMProteinTensor):
        if getattr(pt, track.name, None) is None:
            setattr(pt, track.name, getattr(default_pt, track.name, None))

    batched = _BatchedESMProteinTensor.from_protein_tensor(pt)
    batched.to(pt.device)

    logits_output = _batch_forward(client, batched)

    sampling_config = SamplingConfig(
        sequence=SamplingTrackConfig(temperature=0.0),
        structure=SamplingTrackConfig(temperature=0.0),
    )
    fwd_sample_out = _sample_per_prompt(
        batched, logits_output, sampling_config, tokenizers
    )

    # Remove batch dim
    out = _slice_tensor_dataclass(fwd_sample_out, 0)
    return out.protein_tensor, out

# given the forward output, extract the sequence log-probability of the denoised protein
def _extract_sequence_logprob_from_output(
    denoised_pt: ESMProteinTensor,
    fwd_output: ForwardAndSampleOutput,
    tokenizers: TokenizerCollectionProtocol,
) -> float:
    """Extract total sequence log-probability from a ForwardAndSampleOutput.

    Uses the logprob field from sampling, summing over non-special positions.
    """
    if fwd_output.logprob is not None and fwd_output.logprob.sequence is not None:
        # logprob.sequence is [L] (after slice from batch)
        seq_logprob = fwd_output.logprob.sequence
        # Sum over interior positions (exclude BOS at 0, EOS at -1)
        return seq_logprob[1:-1].sum().item()

    # Fallback: compute from scratch
    return _compute_sequence_logprob(denoised_pt, None, tokenizers)

# given a denoised protein tensor, compute the structure reward (pTM / pLDDT / PAE) using the structure decoder
def _score_structure_from_denoised(
    denoised_pt: ESMProteinTensor,
    client: ESM3InferenceClient,
    mode: str,
) -> float:
    """Compute structure reward from an already-denoised protein tensor."""
    assert isinstance(client, ESM3)
    structure_decoder = client.get_structure_decoder()
    structure_tokens = denoised_pt.structure
    assert structure_tokens is not None

    if structure_tokens.dim() == 1:
        structure_tokens = structure_tokens.unsqueeze(0)

    with torch.no_grad():
        decoder_output = structure_decoder.decode(structure_tokens)

    if mode == "ptm":
        return decoder_output["ptm"].item()
    elif mode == "plddt":
        return decoder_output["plddt"][0, 1:-1].mean().item()
    elif mode == "pae":
        pae = decoder_output["predicted_aligned_error"][0, 1:-1, 1:-1]
        return -pae.mean().item()
    else:
        raise ValueError(f"Unknown mode: {mode}")

# given a denoised protein tensor, forward pass it through the model and compute the total sequence log-probability of the sequence
def _compute_sequence_logprob(
    protein_tensor: ESMProteinTensor,
    client: ESM3InferenceClient | None,
    tokenizers: TokenizerCollectionProtocol,
) -> float:
    """Forward pass on fully decoded protein, compute sum of sequence logprobs."""
    assert client is not None
    assert isinstance(client, ESM3)

    pt = attr.evolve(protein_tensor)
    # Fill defaults
    default_pt = ESMProteinTensor.empty(
        len(pt) - 2, tokenizers=client.tokenizers, device=pt.device
    )
    for track in attr.fields(ESMProteinTensor):
        if getattr(pt, track.name, None) is None:
            setattr(pt, track.name, getattr(default_pt, track.name, None))

    batched = _BatchedESMProteinTensor.from_protein_tensor(pt)
    batched.to(pt.device)

    logits_output = _batch_forward(client, batched)
    assert logits_output.logits is not None
    seq_logits = logits_output.logits.sequence  # [1, L, V]
    assert seq_logits is not None

    seq_tokens = pt.sequence  # [L]
    log_probs = seq_logits[0].log_softmax(dim=-1)  # [L, V]
    # Gather logprob for each token
    token_logprobs = log_probs[
        torch.arange(len(seq_tokens), device=seq_tokens.device), seq_tokens
    ]  # [L]
    # Sum interior positions
    return token_logprobs[1:-1].sum().item()


# ============================================================================
# Batched Structure Reward Computation
# ============================================================================

# compute structural reward for denoised candidates in a batched way for multiple candidates
def _batch_structure_reward(
    candidates: list[ESMProteinTensor],
    client: ESM3InferenceClient,
    mode: str = "ptm",
) -> list[float]:
    """Compute structure reward for a batch of candidates efficiently.

    Performs jumpy denoise on all candidates, then batches structure decode.
    """
    assert isinstance(client, ESM3)
    structure_decoder = client.get_structure_decoder()

    # Jumpy denoise all candidates
    denoised_list = []
    for cand in candidates:
        denoised_list.append(_jumpy_denoise(cand, client))

    # Batch structure decode
    return _batch_decode_structure_reward(denoised_list, structure_decoder, mode)

# pack muliple protein candidates into a batch, decode their 
# structure tokens together, and extract the rewards for each candidate
def _batch_decode_structure_reward(
    denoised_list: list[ESMProteinTensor],
    structure_decoder: StructureTokenDecoder,
    mode: str,
) -> list[float]:
    """Batch VQVAE structure decode and extract rewards."""
    # Collect structure tokens, pad to same length
    struct_tokens_list = []
    lengths = []
    for pt in denoised_list:
        assert pt.structure is not None
        struct_tokens_list.append(pt.structure)
        lengths.append(len(pt.structure))

    max_len = max(lengths)
    device = struct_tokens_list[0].device

    # Pad with PAD token (4098 per VQVAE special tokens)
    pad_token_id = structure_decoder.special_tokens.get("PAD", 4098)
    batched_tokens = torch.full(
        (len(struct_tokens_list), max_len), pad_token_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros(
        (len(struct_tokens_list), max_len), dtype=torch.bool, device=device
    )
    for i, (tokens, length) in enumerate(zip(struct_tokens_list, lengths)):
        batched_tokens[i, :length] = tokens
        attention_mask[i, :length] = True

    with torch.no_grad():
        decoder_output = structure_decoder.decode(
            batched_tokens, attention_mask=attention_mask
        )

    scores = []
    for i in range(len(denoised_list)):
        if mode == "ptm":
            scores.append(decoder_output["ptm"][i].item())
        elif mode == "plddt":
            plddt = decoder_output["plddt"][i, 1 : lengths[i] - 1]
            scores.append(plddt.mean().item())
        elif mode == "pae":
            pae = decoder_output["predicted_aligned_error"][i, 1 : lengths[i] - 1, 1 : lengths[i] - 1]
            scores.append(-pae.mean().item())
        else:
            raise ValueError(f"Unknown mode: {mode}")

    return scores


# compute foldability reward for denoised candidates:
# jumpy denoise → build folding-only protein → single-step fold → batch VQVAE decode
def _batch_foldability_reward(
    candidates: list[ESMProteinTensor],
    client: ESM3InferenceClient,
    mode: str = "ptm",
) -> list[float]:
    """Compute foldability reward for a batch of candidates.

    For each candidate:
    1. Jumpy denoise to fill sequence masks
    2. Construct folding-only input (clear structure/coords)
    3. Single-step folding forward (temp=0)
    Then batch VQVAE decode for all folded results.
    """
    assert isinstance(client, ESM3)
    structure_decoder = client.get_structure_decoder()

    # Steps 1-3 for each candidate
    folded_pts = []
    for cand in candidates:
        denoised_pt = _jumpy_denoise(cand, client)
        folding_pt = _build_folding_only_protein(denoised_pt)
        fwd_out = client.forward_and_sample(
            folding_pt,
            SamplingConfig(
                structure=SamplingTrackConfig(temperature=0.0),
            ),
        )
        assert not isinstance(fwd_out, ESMProteinError)
        folded_pts.append(fwd_out.protein_tensor)

    # Step 4: Batch VQVAE decode
    return _batch_decode_structure_reward(folded_pts, structure_decoder, mode)


# ============================================================================
# Batched Reward Computation (General)
# ============================================================================

# given a reward_funcion, compute rewards with the specified reward function for a batch of candidates
def _compute_rewards_batched(
    candidates: list[ESMProteinTensor],
    reward_fn: RewardFunction,
    client: ESM3InferenceClient,
    tokenizers: TokenizerCollectionProtocol,
) -> list[float]:
    """Compute rewards for a batch of candidates.

    For StructureReward, SequenceFoldabilityReward uses optimized batched paths.
    For CombinedReward or custom rewards, falls back to per-candidate evaluation.
    """
    if isinstance(reward_fn, StructureReward):
        return _batch_structure_reward(candidates, client, mode=reward_fn.mode)

    if isinstance(reward_fn, SequenceFoldabilityReward):
        return _batch_foldability_reward(candidates, client, mode=reward_fn.mode)

    # General fallback: evaluate each candidate individually
    scores = []
    for cand in candidates:
        scores.append(reward_fn(cand, client, tokenizers))
    return scores


def _log_final_scores(
    final_pts: list[ESMProteinTensor],
    final_scores: list[float],
    selected_idx: int,
    reward_fn: RewardFunction,
    client: ESM3InferenceClient,
    tokenizers: TokenizerCollectionProtocol,
) -> None:
    """Print detailed per-beam score breakdown at final selection."""
    if isinstance(reward_fn, CombinedReward):
        # Compute each sub-reward separately for breakdown
        sub_scores: dict[str, list[float]] = {}
        for sub_fn, weight in zip(reward_fn.rewards, reward_fn.weights):
            if isinstance(sub_fn, StructureReward):
                name = f"StructReward({sub_fn.mode})"
            elif isinstance(sub_fn, SequenceFoldabilityReward):
                name = f"FoldabilityReward({sub_fn.mode})"
            elif isinstance(sub_fn, SequenceLogprobReward):
                name = "SeqLogprobReward"
            else:
                name = type(sub_fn).__name__
            sub_scores[name] = _compute_rewards_batched(
                final_pts, sub_fn, client, tokenizers
            )

        for i in range(len(final_pts)):
            marker = " <-- SELECTED" if i == selected_idx else ""
            parts = ", ".join(
                f"{name}={sub_scores[name][i]:.4f}" for name in sub_scores
            )
            print(f"  Beam {i}: {parts}, combined={final_scores[i]:.4f}{marker}", flush=True)
    else:
        for i in range(len(final_pts)):
            marker = " <-- SELECTED" if i == selected_idx else ""
            print(f"  Beam {i}: score={final_scores[i]:.4f}{marker}", flush=True)


def _threshold_random_select(
    final_pts: list[ESMProteinTensor],
    final_scores: list[float],
    reward_fn: RewardFunction,
    client: ESM3InferenceClient,
    tokenizers: TokenizerCollectionProtocol,
    struct_threshold: float,
    seq_threshold: float,
    verbose: bool,
) -> int:
    """Select a random beam whose sub-rewards both exceed thresholds.

    For CombinedReward: check each sub-reward (StructureReward >= struct_threshold,
    SequenceFoldabilityReward/SequenceLogprobReward >= seq_threshold).
    Falls back to argmax of combined score if no beam qualifies.

    Returns the index of the selected beam.
    """
    if not isinstance(reward_fn, CombinedReward):
        # Single reward: just use the combined threshold (struct_threshold)
        eligible = [i for i, s in enumerate(final_scores) if s >= struct_threshold]
        if not eligible:
            selected = max(range(len(final_scores)), key=lambda i: final_scores[i])
            if verbose:
                print(f"  [threshold_random] No beam >= {struct_threshold:.4f}, "
                      f"fallback to best (beam {selected})", flush=True)
            return selected
        selected = random.choice(eligible)
        if verbose:
            print(f"  [threshold_random] {len(eligible)}/{len(final_scores)} beams eligible, "
                  f"selected beam {selected}", flush=True)
        return selected

    # CombinedReward: compute each sub-reward separately
    struct_scores = None
    seq_scores = None
    for sub_fn in reward_fn.rewards:
        if isinstance(sub_fn, StructureReward):
            struct_scores = _compute_rewards_batched(final_pts, sub_fn, client, tokenizers)
        elif isinstance(sub_fn, (SequenceFoldabilityReward, SequenceLogprobReward)):
            seq_scores = _compute_rewards_batched(final_pts, sub_fn, client, tokenizers)

    # Filter beams passing both thresholds
    eligible = []
    for i in range(len(final_pts)):
        s_ok = (struct_scores[i] >= struct_threshold) if struct_scores is not None else True
        q_ok = (seq_scores[i] >= seq_threshold) if seq_scores is not None else True
        if s_ok and q_ok:
            eligible.append(i)

    if verbose:
        print(f"  [threshold_random] struct_thrd={struct_threshold:.4f}, "
              f"seq_thrd={seq_threshold:.4f}", flush=True)
        for i in range(len(final_pts)):
            s_val = f"{struct_scores[i]:.4f}" if struct_scores is not None else "N/A"
            q_val = f"{seq_scores[i]:.4f}" if seq_scores is not None else "N/A"
            marker = " [eligible]" if i in eligible else ""
            print(f"  Beam {i}: struct={s_val}, seq={q_val}, "
                  f"combined={final_scores[i]:.4f}{marker}", flush=True)

    if not eligible:
        selected = max(range(len(final_scores)), key=lambda i: final_scores[i])
        if verbose:
            print(f"  [threshold_random] No beam passes both thresholds, "
                  f"fallback to best (beam {selected})", flush=True)
        return selected

    selected = random.choice(eligible)
    if verbose:
        print(f"  [threshold_random] {len(eligible)}/{len(final_pts)} beams eligible, "
              f"selected beam {selected}", flush=True)
    return selected


def _threshold_random_select_n(
    candidate_pts: list[ESMProteinTensor],
    scores: list[float],
    n: int,
    reward_fn: RewardFunction,
    client: ESM3InferenceClient,
    tokenizers: TokenizerCollectionProtocol,
    struct_threshold: float,
    seq_threshold: float,
) -> list[int]:
    """Select N indices from candidates using threshold-based random selection.

    For intermediate scoring steps: pick N beams from N×L candidates.
    Filters by sub-reward thresholds, randomly selects N from eligible.
    If fewer than N eligible, fills remaining with top-scoring non-eligible.

    Returns list of N selected indices.
    """
    if not isinstance(reward_fn, CombinedReward):
        # Single reward: threshold on combined score
        eligible = [i for i, s in enumerate(scores) if s >= struct_threshold]
        non_eligible = [i for i in range(len(scores)) if i not in set(eligible)]
    else:
        # CombinedReward: compute sub-rewards for threshold check
        struct_scores = None
        seq_scores = None
        for sub_fn in reward_fn.rewards:
            if isinstance(sub_fn, StructureReward):
                struct_scores = _compute_rewards_batched(candidate_pts, sub_fn, client, tokenizers)
            elif isinstance(sub_fn, (SequenceFoldabilityReward, SequenceLogprobReward)):
                seq_scores = _compute_rewards_batched(candidate_pts, sub_fn, client, tokenizers)

        eligible = []
        non_eligible = []
        for i in range(len(candidate_pts)):
            s_ok = (struct_scores[i] >= struct_threshold) if struct_scores is not None else True
            q_ok = (seq_scores[i] >= seq_threshold) if seq_scores is not None else True
            if s_ok and q_ok:
                eligible.append(i)
            else:
                non_eligible.append(i)

    if len(eligible) >= n:
        return random.sample(eligible, n)
    else:
        # Not enough eligible: take all eligible + fill with top-scoring non-eligible
        fill_needed = n - len(eligible)
        non_eligible_sorted = sorted(non_eligible, key=lambda i: scores[i], reverse=True)
        return eligible + non_eligible_sorted[:fill_needed]


# ============================================================================
# Single-Step Sampling (Core Building Block)
# ============================================================================

# record the state of a beam during the search
# entrop is specifically used for decoding position selection
class _BeamState:
    """State for a single beam in the search.

    Tracks the current protein tensor and the entropy from the last forward pass
    (needed for entropy/stochastic position selection strategies).
    """

    __slots__ = ("protein_tensor", "entropy", "initial_mask")

    def __init__(
        self,
        protein_tensor: ESMProteinTensor,
        entropy: ForwardTrackData | None = None,
        initial_mask: torch.Tensor | None = None,
    ):
        self.protein_tensor = protein_tensor
        self.entropy = entropy
        self.initial_mask = initial_mask

# modified from iterative_sampling_tokens()
def _single_track_sample_step(
    beam: _BeamState,
    step: int,
    config: GenerationConfig,
    client: ESM3InferenceClient,
    tokenizers: TokenizerCollectionProtocol,
    total_to_sample: int,
    input_tokens: ESMProteinTensor | None = None,
) -> _BeamState:
    """Execute one step of single-track iterative sampling on a beam.

    This replicates the inner loop of iterative_sampling_tokens() for one prompt,
    one step, using the same utility functions.
    """
    pt = beam.protein_tensor
    seq_len = len(pt)
    device = pt.device

    # Wrap in batched form
    batched = _BatchedESMProteinTensor.from_protein_tensor(pt)

    # --- Forward pass (with optional CFG) ---
    cfg_enabled = config.cfg_scale > 0
    if cfg_enabled:
        # Build single-prompt config/tokens lists for CFG helpers
        configs_list = [config]
        input_tokens_list = [input_tokens] if input_tokens is not None else None
        uncond = _build_cfg_uncond_batched_tokens(
            batched, configs_list, tokenizers, [seq_len],
            input_tokens=input_tokens_list,
        )
        combined = _cat_batched_tokens(batched, uncond)
        forward_out = _batch_forward(client, combined)
    else:
        forward_out = _batch_forward(client, batched)

    per_prompt_forward_out = _slice_tensor_dataclass(forward_out, 0, keep_dim=True)

    if cfg_enabled:
        per_prompt_uncond_out = _slice_tensor_dataclass(forward_out, 1, keep_dim=True)
        per_prompt_forward_out = _apply_cfg_to_logits(
            per_prompt_forward_out, per_prompt_uncond_out,
            config.cfg_scale, config.track,
        )

    # Trim to sequence length
    per_prompt_forward_out = _trim_sequence_tensor_dataclass(
        per_prompt_forward_out, seq_len
    )

    # --- Temperature annealing ---
    if config.temperature_annealing:
        temperature = _get_annealed_temperature(
            step, config.num_steps, config.temperature
        )
    else:
        temperature = config.temperature

    # --- Sample all positions ---
    track_sample_config = SamplingTrackConfig()
    track_sample_config.invalid_ids = config.invalid_ids
    track_sample_config.temperature = temperature
    track_sample_config.top_p = config.top_p
    sampling_config = SamplingConfig(**{config.track: track_sample_config})

    fwd_sample_out = _sample_per_prompt(
        batched,
        per_prompt_forward_out,
        sampling_config,
        tokenizers,
        decode_sasa_tokens=False,
        client=client,
        enable_sequence_resample=config.enable_sequence_resample,
        resample_ratio=config.resample_ratio,
        resample_temperature=config.resample_temperature,
        allow_remask=config.allow_remask,
    )

    new_sampled = fwd_sample_out.protein_tensor
    assert fwd_sample_out.entropy is not None

    # --- Determine which positions to unmask this step ---
    # Use entropy from CURRENT forward pass (not the beam's stored entropy)
    # for position selection, matching iterative_sampling_tokens behavior
    # that always uses fresh entropy from the current step's forward.
    where_to_sample = _get_iterative_sampling_mask_for_prompt_and_step(
        batched,
        torch.tensor(seq_len),
        torch.tensor(total_to_sample),
        step,
        fwd_sample_out.entropy,
        config,
        tokenizers,
        initial_mask=beam.initial_mask,
    )
    where_to_sample = where_to_sample.to(device)

    # --- Apply unmasking ---
    old_track = getattr(batched, config.track)
    new_track = getattr(new_sampled, config.track)

    if config.allow_remask and beam.initial_mask is not None:
        is_mask = _get_masked_positions(
            config.track, old_track, getattr(tokenizers, config.track).mask_token_id
        )
        updated = torch.where(where_to_sample & is_mask, new_track, old_track)
        updated = torch.where(
            where_to_sample, updated, getattr(tokenizers, config.track).mask_token_id
        )
        updated = torch.where(beam.initial_mask, updated, old_track)
    else:
        updated = torch.where(where_to_sample, new_track, old_track)

    # Build output protein tensor
    out_pt = batched.slice(0)
    setattr(out_pt, config.track, updated[0])

    # Restore non-sampled tracks from original
    setattr(out_pt, "coordinates", pt.coordinates)
    for f in attr.fields(SamplingConfig):
        if "embedding" in f.name or f.name == "return_hidden_states":
            continue
        if f.name != config.track:
            setattr(out_pt, f.name, getattr(pt, f.name))

    new_entropy = fwd_sample_out.entropy
    return _BeamState(out_pt, new_entropy, beam.initial_mask)

# modified from iterative_sampling_multitrack_tokens()
def _multi_track_sample_step(
    beam: _BeamState,
    step: int,
    config: MultiTrackGenerationConfig,
    client: ESM3InferenceClient,
    tokenizers: TokenizerCollectionProtocol,
    total_to_sample: dict[str, int],
    input_tokens: ESMProteinTensor | None = None,
) -> _BeamState:
    """Execute one step of multi-track iterative sampling on a beam.

    Replicates the inner loop of iterative_sampling_multitrack_tokens() for one
    prompt, one step.
    """
    pt = beam.protein_tensor
    seq_len = len(pt)
    device = pt.device

    batched = _BatchedESMProteinTensor.from_protein_tensor(pt)

    # --- Determine CFG mode ---
    cfg_enabled = False
    is_cross_modal = False
    active_cfg_mode = None
    track_cfg_scales = {}
    for track_name in config.get_all_tracks():
        tc = config.get_track(track_name)
        track_cfg_scales[track_name] = tc.cfg_scale
        if tc.cfg_scale > 0:
            cfg_enabled = True
            if tc.cfg_mode in ("cross_modal", "motif_anchored_cross_modal", "drop_motif_cross_modal"):
                is_cross_modal = True
            if active_cfg_mode is None:
                active_cfg_mode = tc.cfg_mode

    # --- Forward pass ---
    if cfg_enabled:
        input_tokens_list = [input_tokens] if input_tokens is not None else None
        uncond_result = _build_cfg_uncond_batched_tokens_multitrack(
            batched, [config], tokenizers, [seq_len],
            input_tokens=input_tokens_list,
            cfg_mode=active_cfg_mode,
        )
        if is_cross_modal:
            uncond_for_seq, uncond_for_struct = uncond_result
            combined = _BatchedESMProteinTensor()
            for f in attr.fields(ESMProteinTensor):
                if f.name == "potential_sequence_of_concern":
                    continue
                c = getattr(batched, f.name)
                u1 = getattr(uncond_for_seq, f.name)
                u2 = getattr(uncond_for_struct, f.name)
                if c is not None and u1 is not None and u2 is not None:
                    setattr(combined, f.name, torch.cat([c, u1, u2], dim=0))
                else:
                    setattr(combined, f.name, None)
            forward_out = _batch_forward(client, combined)
        else:
            combined = _cat_batched_tokens(batched, uncond_result)
            forward_out = _batch_forward(client, combined)
    else:
        forward_out = _batch_forward(client, batched)

    per_prompt_forward_out = _slice_tensor_dataclass(forward_out, 0, keep_dim=True)

    # --- Apply CFG ---
    if cfg_enabled:
        has_any_cfg = any(s > 0 for s in track_cfg_scales.values())
        if has_any_cfg:
            if is_cross_modal:
                uncond_seq_out = _slice_tensor_dataclass(forward_out, 1, keep_dim=True)
                uncond_struct_out = _slice_tensor_dataclass(forward_out, 2, keep_dim=True)
                uncond_seq_out = _trim_sequence_tensor_dataclass(uncond_seq_out, seq_len)
                uncond_struct_out = _trim_sequence_tensor_dataclass(uncond_struct_out, seq_len)
                per_prompt_forward_out = _trim_sequence_tensor_dataclass(
                    per_prompt_forward_out, seq_len
                )
                per_prompt_forward_out = _apply_cfg_to_logits_cross_modal(
                    per_prompt_forward_out,
                    uncond_seq_out,
                    uncond_struct_out,
                    track_cfg_scales,
                )
            else:
                uncond_out = _slice_tensor_dataclass(forward_out, 1, keep_dim=True)
                uncond_out = _trim_sequence_tensor_dataclass(uncond_out, seq_len)
                per_prompt_forward_out = _trim_sequence_tensor_dataclass(
                    per_prompt_forward_out, seq_len
                )
                per_prompt_forward_out = _apply_cfg_to_logits_multitrack(
                    per_prompt_forward_out,
                    uncond_out,
                    track_cfg_scales,
                )

    per_prompt_forward_out = _trim_sequence_tensor_dataclass(
        per_prompt_forward_out, seq_len
    )

    # --- Build sampling config for all tracks ---
    sampling_config_dict = {}
    for track_name, track_config in config.tracks.items():
        if track_config.temperature_annealing:
            temperature = _get_annealed_temperature(
                step, track_config.num_steps, track_config.temperature
            )
        else:
            temperature = track_config.temperature

        tsc = SamplingTrackConfig()
        tsc.invalid_ids = track_config.invalid_ids
        tsc.temperature = temperature
        tsc.top_p = track_config.top_p
        sampling_config_dict[track_name] = tsc

    sampling_config = SamplingConfig(**sampling_config_dict)

    allow_remask_for_call = any(
        config.get_track(tn).allow_remask for tn in config.get_all_tracks()
    )

    fwd_sample_out = _sample_per_prompt(
        batched,
        per_prompt_forward_out,
        sampling_config,
        tokenizers,
        decode_sasa_tokens=False,
        allow_remask=allow_remask_for_call,
    )

    new_sampled = fwd_sample_out.protein_tensor
    assert fwd_sample_out.entropy is not None

    # --- Get masks for each track ---
    initial_masks_dict = None
    if beam.initial_mask is not None and isinstance(beam.initial_mask, dict):
        initial_masks_dict = beam.initial_mask

    where_to_sample_dict = _get_iterative_sampling_masks_for_multitrack(
        batched,
        torch.tensor(seq_len),
        {tn: torch.tensor(v) for tn, v in total_to_sample.items()},
        step,
        fwd_sample_out.entropy,
        config,
        tokenizers,
        initial_masks=initial_masks_dict,
    )

    # --- Apply unmasking for each track ---
    out_pt = batched.slice(0)

    for track_name, where_to_sample in where_to_sample_dict.items():
        where_to_sample = where_to_sample.to(device)
        old_track = getattr(batched, track_name)
        new_track = getattr(new_sampled, track_name)

        track_config = config.get_track(track_name)
        initial_mask_for_track = (
            initial_masks_dict.get(track_name) if initial_masks_dict else None
        )

        if track_config.allow_remask and initial_mask_for_track is not None:
            is_mask = _get_masked_positions(
                track_name, old_track, getattr(tokenizers, track_name).mask_token_id
            )
            updated = torch.where(where_to_sample & is_mask, new_track, old_track)
            updated = torch.where(
                where_to_sample, updated, getattr(tokenizers, track_name).mask_token_id
            )
            updated = torch.where(initial_mask_for_track, updated, old_track)
        else:
            updated = torch.where(where_to_sample, new_track, old_track)

        setattr(out_pt, track_name, updated[0])

    # Restore coordinates
    setattr(out_pt, "coordinates", pt.coordinates)
    # Restore non-sampled tracks
    sampled_tracks = set(config.get_all_tracks())
    for f in attr.fields(ESMProteinTensor):
        if f.name == "coordinates":
            continue
        if f.name not in sampled_tracks:
            setattr(out_pt, f.name, getattr(pt, f.name))

    new_entropy = fwd_sample_out.entropy
    return _BeamState(out_pt, new_entropy, beam.initial_mask)


# ============================================================================
# Generalized Beam Search
# ============================================================================

# the beam search itself
# we should provide: ESM3 model, reward function, ESM3 tokenizers
# we should specify: beam width N, branching factor L, scoring interval K
class GeneralizedBeamSearch:
    """Generalized Beam Search for ESM3.

    Unifies Best-of-N, SVDD, and Beam Search into a single framework
    parameterized by (N, L, K):
      - N (beam_width): number of beams maintained in parallel
      - L (branching_factor): candidates generated per beam at scoring steps
      - K (scoring_interval): score and prune every K steps
    """

    def __init__(
        self,
        client: ESM3InferenceClient,
        reward_fn: RewardFunction,
        tokenizers: TokenizerCollectionProtocol,
    ):
        assert isinstance(client, ESM3), "GeneralizedBeamSearch requires local ESM3 model"
        self.client = client
        self.reward_fn = reward_fn
        self.tokenizers = tokenizers

    def search(
        self,
        protein: ESMProtein | ESMProteinTensor,
        config: GenerationConfig | MultiTrackGenerationConfig,
        beam_width: int = 1,
        branching_factor: int = 4,
        scoring_interval: int = 1,
        input_tokens: ESMProteinTensor | None = None,
        verbose: bool = True,
        selection_mode: str = "best",
        struct_reward_threshold: float = 0.0,
        seq_reward_threshold: float = 0.0,
    ) -> ESMProtein:
        """Run generalized beam search.

        Args:
            protein: Input protein (will be encoded if ESMProtein).
            config: Generation config (single-track or multi-track).
            beam_width: N — number of beams.
            branching_factor: L — candidates per beam at scoring steps.
            scoring_interval: K — score every K steps.
            input_tokens: Original input tokens (needed for CFG drop_motif mode).
            verbose: Show progress bar.
            selection_mode: "best" (argmax) or "threshold_random" (random among
                beams passing both struct and seq reward thresholds).
            struct_reward_threshold: Minimum structure reward for threshold_random.
            seq_reward_threshold: Minimum sequence reward for threshold_random.

        Returns:
            Best decoded ESMProtein.
        """
        assert selection_mode in ("best", "threshold_random"), \
            f"Unknown selection_mode: {selection_mode}"

        # Encode if needed
        if isinstance(protein, ESMProtein):
            protein_tensor = self.client.encode(protein)
            assert not isinstance(protein_tensor, ESMProteinError)
        else:
            protein_tensor = attr.evolve(protein)

        is_multitrack = isinstance(config, MultiTrackGenerationConfig)

        if is_multitrack:
            return self._search_multitrack(
                protein_tensor, config, beam_width, branching_factor,
                scoring_interval, input_tokens, verbose,
                selection_mode, struct_reward_threshold, seq_reward_threshold,
            )
        else:
            return self._search_single_track(
                protein_tensor, config, beam_width, branching_factor,
                scoring_interval, input_tokens, verbose,
                selection_mode, struct_reward_threshold, seq_reward_threshold,
            )

    def _search_single_track(
        self,
        protein_tensor: ESMProteinTensor,
        config: GenerationConfig,
        beam_width: int,
        branching_factor: int,
        scoring_interval: int,
        input_tokens: ESMProteinTensor | None,
        verbose: bool,
        selection_mode: str = "best",
        struct_reward_threshold: float = 0.0,
        seq_reward_threshold: float = 0.0,
    ) -> ESMProtein:
        N, L, K = beam_width, branching_factor, scoring_interval
        T = config.num_steps
        track = config.track

        if config.condition_on_coordinates_only and protein_tensor.coordinates is not None:
            protein_tensor.structure = None

        # Compute total tokens to sample
        if getattr(protein_tensor, track) is None:
            total_to_sample = _get_non_special_tokens(protein_tensor, self.tokenizers)
        else:
            masked = _get_masked_positions(
                track,
                getattr(protein_tensor, track),
                getattr(self.tokenizers, track).mask_token_id,
            )
            total_to_sample = torch.sum(masked).item()

        if total_to_sample > 0 and total_to_sample < config.num_steps:
            config = attr.evolve(config)
            config.num_steps = int(total_to_sample)
            T = config.num_steps

        # Fill None tracks with mask tokens, matching iterative_sampling_tokens behavior.
        # _stack_protein_tensors fills None tracks with [BOS, MASK, ..., MASK, EOS].
        # We must do this BEFORE creating beams so all forward passes see consistent inputs.
        filled_batched = _stack_protein_tensors(
            [protein_tensor], [len(protein_tensor)], self.tokenizers, protein_tensor.device
        )
        protein_tensor_filled = filled_batched.slice(0, sequence_len=len(protein_tensor))
        # Restore coordinates (not handled by _stack_protein_tensors slicing)
        protein_tensor_filled.coordinates = protein_tensor.coordinates

        # Compute initial mask for allow_remask support
        # IMPORTANT: use filled_batched (after None tracks filled with masks)
        initial_mask = None
        if config.allow_remask:
            track_data = getattr(filled_batched, track)
            if track_data is not None:
                initial_mask = _get_masked_positions(
                    track, track_data[0:1], getattr(self.tokenizers, track).mask_token_id
                )

        # Initialize N identical beams (using filled protein tensor)
        beams: list[_BeamState] = []
        for _ in range(N):
            beams.append(
                _BeamState(
                    protein_tensor=attr.evolve(protein_tensor_filled),
                    entropy=None,
                    initial_mask=initial_mask,
                )
            )

        if verbose:
            pbar = tqdm(range(T), desc="Beam Search")
        else:
            pbar = range(T)

        for t in pbar:
            is_scoring_step = (t > 0) and (t % K == 0)

            if is_scoring_step:
                # --- Scoring step: branch L candidates from each beam ---
                all_candidates: list[_BeamState] = []
                for b in range(N):
                    for l in range(L):
                        candidate = _single_track_sample_step(
                            beams[b], t, config, self.client, self.tokenizers,
                            total_to_sample, input_tokens,
                        )
                        all_candidates.append(candidate)

                # Compute rewards via jumpy denoise
                candidate_pts = [c.protein_tensor for c in all_candidates]
                scores = _compute_rewards_batched(
                    candidate_pts, self.reward_fn, self.client, self.tokenizers
                )

                # Select N beams
                if selection_mode == "threshold_random":
                    top_indices = _threshold_random_select_n(
                        candidate_pts, scores, N, self.reward_fn,
                        self.client, self.tokenizers,
                        struct_reward_threshold, seq_reward_threshold,
                    )
                else:
                    top_indices = sorted(
                        range(len(scores)), key=lambda i: scores[i], reverse=True
                    )[:N]
                beams = [all_candidates[i] for i in top_indices]

                if verbose:
                    best_score = max(scores[i] for i in top_indices)
                    pbar.set_description(f"Beam Search (best={best_score:.3f})")
            else:
                # --- Non-scoring step: advance each beam independently ---
                new_beams = []
                for b in range(N):
                    new_beam = _single_track_sample_step(
                        beams[b], t, config, self.client, self.tokenizers,
                        total_to_sample, input_tokens,
                    )
                    new_beams.append(new_beam)
                beams = new_beams

        # --- Final selection ---
        final_pts = [b.protein_tensor for b in beams]
        final_scores = _compute_rewards_batched(
            final_pts, self.reward_fn, self.client, self.tokenizers
        )

        if selection_mode == "threshold_random":
            selected_idx = _threshold_random_select(
                final_pts, final_scores, self.reward_fn,
                self.client, self.tokenizers,
                struct_reward_threshold, seq_reward_threshold, verbose,
            )
        else:
            selected_idx = max(range(len(final_scores)), key=lambda i: final_scores[i])

        best_pt = final_pts[selected_idx]

        if verbose:
            _log_final_scores(
                final_pts, final_scores, selected_idx,
                self.reward_fn, self.client, self.tokenizers,
            )

        # Decode
        decoded = self.client.decode(best_pt)
        assert not isinstance(decoded, ESMProteinError)
        return decoded

    def _search_multitrack(
        self,
        protein_tensor: ESMProteinTensor,
        config: MultiTrackGenerationConfig,
        beam_width: int,
        branching_factor: int,
        scoring_interval: int,
        input_tokens: ESMProteinTensor | None,
        verbose: bool,
        selection_mode: str = "best",
        struct_reward_threshold: float = 0.0,
        seq_reward_threshold: float = 0.0,
    ) -> ESMProtein:
        N, L, K = beam_width, branching_factor, scoring_interval
        T = config.get_max_num_steps()

        if config.condition_on_coordinates_only and protein_tensor.coordinates is not None:
            protein_tensor.structure = None

        # Compute total tokens to sample per track
        total_to_sample: dict[str, int] = {}
        for track_name in config.get_all_tracks():
            track_tokens = getattr(protein_tensor, track_name)
            if track_tokens is None:
                num = _get_non_special_tokens(protein_tensor, self.tokenizers)
            else:
                masked = _get_masked_positions(
                    track_name, track_tokens,
                    getattr(self.tokenizers, track_name).mask_token_id,
                )
                num = torch.sum(masked).item()
            total_to_sample[track_name] = int(num)

            tc = config.get_track(track_name)
            if num > 0 and num < tc.num_steps:
                tc.num_steps = int(num)

        T = config.get_max_num_steps()

        # Fill None tracks with mask tokens, matching iterative_sampling_multitrack_tokens.
        filled_batched = _stack_protein_tensors(
            [protein_tensor], [len(protein_tensor)], self.tokenizers, protein_tensor.device
        )
        protein_tensor_filled = filled_batched.slice(0, sequence_len=len(protein_tensor))
        protein_tensor_filled.coordinates = protein_tensor.coordinates

        # Compute initial masks for allow_remask support (using filled batched)
        initial_masks_dict = None
        any_remask = any(
            config.get_track(tn).allow_remask for tn in config.get_all_tracks()
        )
        if any_remask:
            initial_masks_dict = {}
            for track_name in config.get_all_tracks():
                tc = config.get_track(track_name)
                if tc.allow_remask:
                    track_data = getattr(filled_batched, track_name)
                    if track_data is not None:
                        initial_masks_dict[track_name] = _get_masked_positions(
                            track_name, track_data[0:1],
                            getattr(self.tokenizers, track_name).mask_token_id,
                        )

        # Initialize N identical beams (using filled protein tensor)
        beams: list[_BeamState] = []
        for _ in range(N):
            beams.append(
                _BeamState(
                    protein_tensor=attr.evolve(protein_tensor_filled),
                    entropy=None,
                    initial_mask=initial_masks_dict,
                )
            )

        if verbose:
            pbar = tqdm(range(T), desc="Beam Search (multitrack)")
        else:
            pbar = range(T)

        for t in pbar:
            is_scoring_step = (t > 0) and (t % K == 0)

            if is_scoring_step:
                all_candidates: list[_BeamState] = []
                for b in range(N):
                    for l in range(L):
                        candidate = _multi_track_sample_step(
                            beams[b], t, config, self.client, self.tokenizers,
                            total_to_sample, input_tokens,
                        )
                        all_candidates.append(candidate)

                candidate_pts = [c.protein_tensor for c in all_candidates]
                scores = _compute_rewards_batched(
                    candidate_pts, self.reward_fn, self.client, self.tokenizers
                )

                if selection_mode == "threshold_random":
                    top_indices = _threshold_random_select_n(
                        candidate_pts, scores, N, self.reward_fn,
                        self.client, self.tokenizers,
                        struct_reward_threshold, seq_reward_threshold,
                    )
                else:
                    top_indices = sorted(
                        range(len(scores)), key=lambda i: scores[i], reverse=True
                    )[:N]
                beams = [all_candidates[i] for i in top_indices]

                if verbose:
                    best_score = max(scores[i] for i in top_indices)
                    pbar.set_description(
                        f"Beam Search multitrack (best={best_score:.3f})"
                    )
            else:
                new_beams = []
                for b in range(N):
                    new_beam = _multi_track_sample_step(
                        beams[b], t, config, self.client, self.tokenizers,
                        total_to_sample, input_tokens,
                    )
                    new_beams.append(new_beam)
                beams = new_beams

        # --- Final selection ---
        final_pts = [b.protein_tensor for b in beams]
        final_scores = _compute_rewards_batched(
            final_pts, self.reward_fn, self.client, self.tokenizers
        )

        if selection_mode == "threshold_random":
            selected_idx = _threshold_random_select(
                final_pts, final_scores, self.reward_fn,
                self.client, self.tokenizers,
                struct_reward_threshold, seq_reward_threshold, verbose,
            )
        else:
            selected_idx = max(range(len(final_scores)), key=lambda i: final_scores[i])

        best_pt = final_pts[selected_idx]

        if verbose:
            _log_final_scores(
                final_pts, final_scores, selected_idx,
                self.reward_fn, self.client, self.tokenizers,
            )

        decoded = self.client.decode(best_pt)
        assert not isinstance(decoded, ESMProteinError)
        return decoded
