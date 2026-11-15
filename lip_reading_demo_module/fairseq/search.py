

import math
from typing import List, Optional

import torch
import torch.nn as nn
from fairseq.token_generation_constraints import (
    ConstraintState,
    OrderedConstraintState,
    UnorderedConstraintState,
)
from torch import Tensor

class Search(nn.Module):
    def __init__(self, tgt_dict):
        super().__init__()
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.src_lengths = torch.tensor(-1)
        self.supports_constraints = False
        self.stop_on_max_len = False

    def step(
        self, step, lprobs, scores, prev_output_tokens=None, original_batch_idxs=None
    ):
        """Take a single search step.

        Args:
            step: the current search step, starting at 0
            lprobs: (bsz x input_beam_size x vocab_size)
                the model's log-probabilities over the vocabulary at the current step
            scores: (bsz x input_beam_size x step)
                the historical model scores of each hypothesis up to this point
            prev_output_tokens: (bsz x step)
                the previously generated oputput tokens
            original_batch_idxs: (bsz)
                the tensor with the batch indices, in the range [0, bsz)
                this is useful in case there has been applied a re-ordering
                and we need to know the orignal indices

        Return: A tuple of (scores, indices, beams) where:
            scores: (bsz x output_beam_size)
                the scores of the chosen elements; output_beam_size can be
                larger than input_beam_size, e.g., we may return
                2*input_beam_size to account for EOS
            indices: (bsz x output_beam_size)
                the indices of the chosen elements
            beams: (bsz x output_beam_size)
                the hypothesis ids of the chosen elements, in the range [0, input_beam_size)
        """
        raise NotImplementedError

    @torch.jit.export
    def set_src_lengths(self, src_lengths):
        self.src_lengths = src_lengths

    @torch.jit.export
    def init_constraints(self, batch_constraints: Optional[Tensor], beam_size: int):
        """Initialize constraint states for constrained decoding (if supported).

        Args:
            batch_constraints: (torch.Tensor, optional)
                the list of constraints, in packed form
            beam_size: (int)
                the beam size
        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        pass

    def prune_sentences(self, batch_idxs: Tensor):
        """
        Removes constraint states for completed sentences (if supported).
        This is called from sequence_generator._generate() when sentences are
        deleted from the batch.

        Args:
            batch_idxs: Indices of *sentences* whose constraint state should be *kept*.
        """
        pass

    def update_constraints(self, active_hypos: Tensor):
        """
        Updates the constraint states by selecting the beam items that are retained.
        This is called at each time step of sequence_generator._generate() when
        the set of 2 * {beam_size} candidate hypotheses are reduced to the beam size.

        Args:
            active_hypos: (batch size, beam size)
              list of integers denoting, for each sentence, which beam candidate items
              should be kept.
        """
        pass

class BeamSearch(Search):
    def __init__(self, tgt_dict):
        super().__init__(tgt_dict)
        self.constraint_states = None

    @torch.jit.export
    def step(
        self,
        step: int,
        lprobs,
        scores: Optional[Tensor],
        prev_output_tokens: Optional[Tensor] = None,
        original_batch_idxs: Optional[Tensor] = None,
    ):
        bsz, beam_size, vocab_size = lprobs.size()

        if step == 0:

            lprobs = lprobs[:, ::beam_size, :].contiguous()
        else:

            assert scores is not None
            lprobs = lprobs + scores[:, :, step - 1].unsqueeze(-1)

        top_prediction = torch.topk(
            lprobs.view(bsz, -1),
            k=min(

                beam_size * 2,
                lprobs.view(bsz, -1).size(1) - 1,
            ),
        )
        scores_buf = top_prediction[0]
        indices_buf = top_prediction[1]

        beams_buf = indices_buf // vocab_size
        indices_buf = indices_buf.fmod(vocab_size)

        return scores_buf, indices_buf, beams_buf

class PrefixConstrainedBeamSearch(Search):
    def __init__(self, tgt_dict, prefix_allowed_tokens_fn):
        super().__init__(tgt_dict)
        self.prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self.stop_on_max_len = True

    @torch.jit.export
    def apply_mask(self, x, prev_output_tokens, original_batch_idxs):
        beam_size = x.shape[0] // original_batch_idxs.shape[0]
        original_batch_idxs = (
            original_batch_idxs.unsqueeze(-1).repeat((1, beam_size)).flatten().tolist()
        )

        mask = torch.full_like(x, -math.inf)
        for sent_i, (sent, batch_i) in enumerate(
            zip(prev_output_tokens, original_batch_idxs)
        ):
            mask[sent_i, :, self.prefix_allowed_tokens_fn(batch_i, sent)] = 0

        return mask

    @torch.jit.export
    def step(
        self,
        step: int,
        lprobs: Tensor,
        scores: Tensor,
        prev_output_tokens: Tensor,
        original_batch_idxs: Tensor,
    ):
        bsz, beam_size, vocab_size = lprobs.size()

        lprobs += self.apply_mask(
            lprobs.view(bsz * beam_size, 1, vocab_size),
            prev_output_tokens,
            original_batch_idxs,
        ).view(bsz, beam_size, vocab_size)

        if step == 0:

            lprobs = lprobs[:, ::beam_size, :].contiguous()
        else:

            assert scores is not None
            lprobs = lprobs + scores[:, :, step - 1].unsqueeze(-1)

        top_prediction = torch.topk(
            lprobs.view(bsz, -1),
            k=min(

                beam_size,
                lprobs.view(bsz, -1).size(1) - 1,
            ),
        )
        scores_buf = top_prediction[0]
        indices_buf = top_prediction[1]
        beams_buf = indices_buf // vocab_size
        indices_buf = indices_buf.fmod(vocab_size)
        return scores_buf, indices_buf, beams_buf

class LexicallyConstrainedBeamSearch(Search):
    """Implements lexically constrained beam search as described in

        Fast Lexically Constrained Decoding with Dynamic Beam
        Allocation for Neural Machine Translation.  Post & Vilar,
        NAACL 2018.  https://www.aclweb.org/anthology/N18-1119/

    and

        Improved Lexically Constrained Decoding for Translation and
        Monolingual Rewriting. Hu et al, NAACL
        2019. https://www.aclweb.org/anthology/N19-1090/

    This is accomplished by maintaining, for each beam hypothesis, a
    ConstraintState object (see constraints.py) that tracks which
    constraints have been generated and using this information to
    shape the beam for each input sentence.
    """

    def __init__(self, tgt_dict, representation):
        super().__init__(tgt_dict)
        self.representation = representation
        self.vocab_size = len(tgt_dict)
        self.num_cands = 0
        self.supports_constraints = True

    @torch.jit.export
    def init_constraints(self, batch_constraints: Optional[Tensor], beam_size: int):
        self.constraint_states = []
        for constraint_tensor in batch_constraints:
            if self.representation == "ordered":
                constraint_state = OrderedConstraintState.create(constraint_tensor)
            elif self.representation == "unordered":
                constraint_state = UnorderedConstraintState.create(constraint_tensor)

            self.constraint_states.append([constraint_state for i in range(beam_size)])

    @torch.jit.export
    def prune_sentences(self, batch_idxs: Tensor):
        self.constraint_states = [
            self.constraint_states[i] for i in batch_idxs.tolist()
        ]

    @torch.jit.export
    def update_constraints(self, active_hypos: Tensor):
        if self.constraint_states:
            batch_size = active_hypos.size(0)
            for sentid in range(batch_size):
                self.constraint_states[sentid] = [
                    self.constraint_states[sentid][i] for i in active_hypos[sentid]
                ]

    @torch.jit.export
    def step(
        self,
        step: int,
        lprobs: Tensor,
        scores: Optional[Tensor],
        prev_output_tokens: Optional[Tensor] = None,
        original_batch_idxs: Optional[Tensor] = None,
    ):
        """
        A constrained step builds a large candidates list from the following:
        - the top 2 * {beam_size} items over the whole beam
        - for each item in the beam
          - the top {each_k} (default 1)
          - all next constraints
        We then compute the constrained state of each beam item, and assign
        stripe codes: 0 to the best in each bank, 1 to the 2nd-best, and so
        on. We then sort by (stripe, score), and truncate the list at
        2 * beam size.

        Args:
            step: the decoder step
            lprobs: (batch size, beam size, target vocab)
                the target-vocab distributions for each item in the beam.
        Retrun: A tuple of (scores, indices, beams, constraints) where:
            scores: (batch, output beam size)
                the scores of the chosen elements
            indices: (batch, output beam size)
                the target vocab indices of the chosen elements
            beams: (batch, output beam size)
                the 0-indexed hypothesis ids of the chosen elements
            constraints: (batch, output beam size)
                the new constraint states
        """
        each_k = 1
        device = lprobs.device

        batch_size, beam_size, vocab_size = lprobs.size()

        self.num_cands = min(

            beam_size * 2,
            lprobs.view(batch_size, -1).size(1) - 1,
        )

        constraint_states = self.constraint_states
        if constraint_states and step > 0:
            not_finished_indices = []
            for sentno, sent_constraints in enumerate(constraint_states):
                for beamno, state in enumerate(sent_constraints):
                    index = sentno * beam_size + beamno
                    if not state.finished:
                        not_finished_indices.append(index)
            not_finished_indices = torch.tensor(not_finished_indices)
            if not_finished_indices.numel() > 0:
                lprobs.view(batch_size * beam_size, -1)[
                    not_finished_indices, self.eos
                ] = -math.inf

        if step == 0:

            lprobs = lprobs[:, ::beam_size, :].contiguous()
        else:

            assert scores is not None
            lprobs = lprobs + scores[:, :, step - 1].unsqueeze(-1)

        top_prediction = torch.topk(
            lprobs.view(batch_size, -1),
            self.num_cands,
        )
        scores_buf, indices_buf = top_prediction

        beams_buf = indices_buf // vocab_size
        indices_buf = indices_buf.fmod(vocab_size)

        if not constraint_states:
            return scores_buf, indices_buf, beams_buf

        if step > 0:
            top_scores, top_indices = torch.topk(
                lprobs.view(batch_size * beam_size, -1),
                k=each_k,
                dim=1,
            )
            top_scores = top_scores.view(batch_size, -1)
            top_indices = top_indices.view(batch_size, -1)
            scores_buf = torch.cat((scores_buf, top_scores), dim=1)
            indices_buf = torch.cat((indices_buf, top_indices), dim=1)
            new_beams = torch.arange(0, beam_size, device=device).repeat(batch_size, 1)
            beams_buf = torch.cat((beams_buf, new_beams), dim=1)

        new_scores_buf = torch.zeros((batch_size, 2 * beam_size), device=device)
        new_indices_buf = torch.zeros((batch_size, 2 * beam_size), device=device).long()
        new_beams_buf = torch.zeros((batch_size, 2 * beam_size), device=device).long()
        for sentno, states in enumerate(constraint_states):
            scores, indices, beams, new_states = self.step_sentence(
                step,
                sentno,
                lprobs[sentno],
                constraint_states[sentno],
                beams_buf[sentno].clone(),
                indices_buf[sentno].clone(),
                scores_buf[sentno].clone(),
            )
            new_scores_buf[sentno] = scores
            new_indices_buf[sentno] = indices
            new_beams_buf[sentno] = beams
            self.constraint_states[sentno] = new_states

        return new_scores_buf, new_indices_buf, new_beams_buf

    @torch.jit.export
    def step_sentence(
        self,
        step: int,
        sentno: int,
        lprobs: Tensor,
        constraint_states: List[List[ConstraintState]],
        beams_buf: Tensor,
        indices_buf: Tensor,
        scores_buf: Tensor,
    ):
        """Does per-sentence processing. Adds all constraints for each
        hypothesis to the list of candidates; then removes duplicates,
        sorts, and dynamically stripes across the banks. All tensor inputs
        are collapsed to those pertaining to a single input sentence.
        """
        device = lprobs.device

        for beamno, state in enumerate(constraint_states):
            next_tokens = torch.tensor(list(state.next_tokens()), device=device).long()
            if next_tokens.numel() != 0:
                indices_buf = torch.cat((indices_buf, next_tokens))
                next_beams = (
                    torch.tensor(beamno, device=device)
                    .repeat(next_tokens.size(0))
                    .long()
                )
                beams_buf = torch.cat((beams_buf, next_beams))
                next_values = lprobs[beamno].take(next_tokens.view(-1))
                scores_buf = torch.cat((scores_buf, next_values))

            if step == 0:
                break

        cands_size = indices_buf.size(0)
        constraint_states = [
            constraint_states[beams_buf[i]].advance(indices_buf[i])
            for i in range(cands_size)
        ]

        banks = torch.tensor([state.bank for state in constraint_states], device=device)

        num_constraint_tokens = len(state.tokens)

        MAX_SCORE = -100
        sort_key = (num_constraint_tokens - banks) * MAX_SCORE + scores_buf
        sort_values, sort_indices = sort_key.sort(dim=0, descending=True)
        scores_buf = scores_buf[sort_indices]
        indices_buf = indices_buf[sort_indices]
        beams_buf = beams_buf[sort_indices]
        banks = banks[sort_indices]

        constraint_states = [constraint_states[i] for i in sort_indices]

        def roll(t):
            """Rolls a 1d tensor left by 1.

            [0, 1, 2, 3, 4] becomes [4, 0, 1, 2, 3]
            """
            return torch.cat((t[-1].unsqueeze(0), t[0:-1]), dim=0)

        uniques_mask = beams_buf * (self.vocab_size + 1) + indices_buf
        uniques_mask = roll(uniques_mask) != uniques_mask

        scores_buf = torch.masked_select(scores_buf, uniques_mask)
        indices_buf = torch.masked_select(indices_buf, uniques_mask)
        beams_buf = torch.masked_select(beams_buf, uniques_mask)
        banks = torch.masked_select(banks, uniques_mask)
        i = 1
        for mask in uniques_mask[1:]:
            if not mask:
                constraint_states.pop(i)
            i += mask

        stripe_offsets = [offset * (len(banks) + 1) for offset in range(len(banks) + 1)]
        stripes = torch.zeros_like(banks)
        cur_bank_count = -1
        cur_bank = banks[0]
        for i, bank in enumerate(banks):
            if bank != cur_bank:
                cur_bank_count = 0
                cur_bank = bank
            else:
                cur_bank_count += 1
            stripes[i] = num_constraint_tokens - bank + stripe_offsets[cur_bank_count]

        sort_values, sort_indices = stripes.sort(dim=0)
        scores_buf = scores_buf[sort_indices]
        indices_buf = indices_buf[sort_indices]
        beams_buf = beams_buf[sort_indices]
        constraint_states = [constraint_states[i] for i in sort_indices]

        scores_buf = scores_buf[: self.num_cands]
        indices_buf = indices_buf[: self.num_cands]
        beams_buf = beams_buf[: self.num_cands]

        return scores_buf, indices_buf, beams_buf, constraint_states

class LengthConstrainedBeamSearch(Search):
    def __init__(self, tgt_dict, min_len_a, min_len_b, max_len_a, max_len_b):
        super().__init__(tgt_dict)
        self.min_len_a = min_len_a
        self.min_len_b = min_len_b
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.beam = BeamSearch(tgt_dict)
        self.needs_src_lengths = True

    def step(
        self,
        step: int,
        lprobs,
        scores,
        prev_output_tokens: Optional[Tensor] = None,
        original_batch_idxs: Optional[Tensor] = None,
    ):
        min_lens = self.min_len_a * self.src_lengths + self.min_len_b
        max_lens = self.max_len_a * self.src_lengths + self.max_len_b
        lprobs[step < min_lens, :, self.eos] = -math.inf
        lprobs[step >= max_lens, :, self.eos] = 0
        return self.beam.step(step, lprobs, scores)

class DiverseBeamSearch(Search):
    """Diverse Beam Search.

    See "Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence
    Models" for details.

    We only implement the Hamming Diversity penalty here, which performed best
    in the original paper.
    """

    def __init__(self, tgt_dict, num_groups, diversity_strength):
        super().__init__(tgt_dict)
        self.num_groups = num_groups
        self.diversity_strength = -diversity_strength
        self.beam = BeamSearch(tgt_dict)

    @torch.jit.export
    def step(
        self,
        step: int,
        lprobs,
        scores,
        prev_output_tokens: Optional[Tensor] = None,
        original_batch_idxs: Optional[Tensor] = None,
    ):
        bsz, beam_size, vocab_size = lprobs.size()
        if beam_size % self.num_groups != 0:
            raise ValueError(
                "DiverseBeamSearch requires --beam to be divisible by the number of groups"
            )

        diversity_buf = torch.zeros(lprobs[:, 0, :].size()).to(lprobs)

        scores_G, indices_G, beams_G = [], [], []
        for g in range(self.num_groups):
            lprobs_g = lprobs[:, g :: self.num_groups, :]
            scores_g = scores[:, g :: self.num_groups, :] if step > 0 else None

            if g > 0:
                lprobs_g = torch.add(
                    lprobs_g,
                    other=diversity_buf.unsqueeze(1),
                    alpha=self.diversity_strength,
                )
            else:
                lprobs_g = lprobs_g.contiguous()

            scores_buf, indices_buf, beams_buf = self.beam.step(
                step, lprobs_g, scores_g
            )
            beams_buf.mul_(self.num_groups).add_(g)

            scores_G.append(scores_buf.clone())
            indices_G.append(indices_buf.clone())
            beams_G.append(beams_buf.clone())

            diversity_buf.scatter_add_(
                1, indices_buf, torch.ones(indices_buf.size()).to(diversity_buf)
            )

        scores_buf = torch.stack(scores_G, dim=2).view(bsz, -1)
        indices_buf = torch.stack(indices_G, dim=2).view(bsz, -1)
        beams_buf = torch.stack(beams_G, dim=2).view(bsz, -1)
        return scores_buf, indices_buf, beams_buf

class Sampling(Search):
    sampling_topk: int
    sampling_topp: float

    def __init__(self, tgt_dict, sampling_topk=-1, sampling_topp=-1.0):
        super().__init__(tgt_dict)
        self.sampling_topk = sampling_topk
        self.sampling_topp = sampling_topp

    def _sample_topp(self, lprobs):
        """Sample among the smallest set of elements whose cumulative probability mass exceeds p.

        See `"The Curious Case of Neural Text Degeneration"
        (Holtzman et al., 2019) <https://arxiv.org/abs/1904.09751>`_.

        Args:
            lprobs: (bsz x input_beam_size x vocab_size)
                the model's log-probabilities over the vocabulary at the current step

        Return: A tuple of (trimed_probs, truncated_indices) where:
            trimed_probs: (bsz x input_beam_size x ?)
                the model's probabilities over the elements selected to sample from. The
                width of the third dimension is determined by top-P.
            truncated_indices: (bsz x input_beam_size x ?)
                the indices of the chosen elements.
        """
        probs = lprobs.exp_()

        sorted_probs, sorted_indices = probs.sort(descending=True)

        cumsum_probs = sorted_probs.cumsum(dim=2)
        mask = cumsum_probs.lt(self.sampling_topp)

        cumsum_mask = mask.cumsum(dim=2)
        last_included = cumsum_mask[:, :, -1:]
        last_included.clamp_(0, mask.size()[2] - 1)
        mask = mask.scatter_(2, last_included, 1)

        max_dim = last_included.max()
        truncated_mask = mask[:, :, : max_dim + 1]
        truncated_probs = sorted_probs[:, :, : max_dim + 1]
        truncated_indices = sorted_indices[:, :, : max_dim + 1]

        trim_mask = ~truncated_mask
        trimed_probs = truncated_probs.masked_fill_(trim_mask, 0)
        return trimed_probs, truncated_indices

    @torch.jit.export
    def step(
        self,
        step: int,
        lprobs,
        scores,
        prev_output_tokens: Optional[Tensor] = None,
        original_batch_idxs: Optional[Tensor] = None,
    ):
        bsz, beam_size, vocab_size = lprobs.size()

        if step == 0:

            lprobs = lprobs[:, ::beam_size, :].contiguous()

        if self.sampling_topp > 0:

            probs, top_indices = self._sample_topp(lprobs)
        elif self.sampling_topk > 0:

            lprobs, top_indices = lprobs.topk(self.sampling_topk)
            probs = lprobs.exp_()
        else:
            probs = lprobs.exp_()

            top_indices = torch.empty(0).to(probs)

        if step == 0:
            indices_buf = torch.multinomial(
                probs.view(bsz, -1),
                beam_size,
                replacement=True,
            ).view(bsz, beam_size)
        else:
            indices_buf = torch.multinomial(
                probs.view(bsz * beam_size, -1),
                1,
                replacement=True,
            ).view(bsz, beam_size)

        if step == 0:

            probs = probs.expand(bsz, beam_size, -1)

        scores_buf = torch.gather(probs, dim=2, index=indices_buf.unsqueeze(-1))
        scores_buf = scores_buf.log_().view(bsz, -1)

        if self.sampling_topk > 0 or self.sampling_topp > 0:
            indices_buf = torch.gather(
                top_indices.expand(bsz, beam_size, -1),
                dim=2,
                index=indices_buf.unsqueeze(-1),
            ).squeeze(2)

        if step == 0:
            beams_buf = indices_buf.new_zeros(bsz, beam_size)
        else:
            beams_buf = torch.arange(0, beam_size).to(indices_buf).repeat(bsz, 1)

            scores_buf.add_(
                torch.gather(scores[:, :, step - 1], dim=1, index=beams_buf)
            )

        return scores_buf, indices_buf, beams_buf

class DiverseSiblingsSearch(Search):
    """
    Beam search with diverse siblings.

    See "A Simple, Fast Diverse Decoding Algorithm for Neural Generation" for details.
    https://arxiv.org/abs/1611.08562

    1/ Calculate hypotheses for each beam
    2/ Intra-sibling ordering
    3/ Rewrite scores
    4/ Choose top K hypotheses

    if diversity_rate == 0 is equivalent to BeamSearch
    """

    def __init__(self, tgt_dict, diversity_rate):
        super().__init__(tgt_dict)
        self.diversity_rate = diversity_rate
        self.beam = BeamSearch(tgt_dict)

    def step(
        self,
        step: int,
        lprobs,
        scores,
        prev_output_tokens: Optional[Tensor] = None,
        original_batch_idxs: Optional[Tensor] = None,
    ):
        bsz, beam_size, vocab_size = lprobs.size()
        k = min(

            beam_size * 2,
            lprobs.view(bsz, -1).size(1) - 1,
        )
        s_list: List[Tensor]
        i_list: List[Tensor]
        s_list = [torch.empty(0).to(lprobs) for i in range(beam_size)]
        i_list = [torch.LongTensor().to(device=lprobs.device) for i in range(beam_size)]
        sibling_score = torch.arange(1, k + 1).to(lprobs) * self.diversity_rate

        if step == 0:
            return self.beam.step(step, lprobs, scores)
        lprobs.add_(scores[:, :, step - 1].unsqueeze(-1))

        for i in range(beam_size):
            torch.topk(lprobs[:, i, :].view(bsz, -1), k, out=(s_list[i], i_list[i]))
            i_list[i].fmod_(vocab_size)

            s_list[i].sub_(sibling_score)

        indices = torch.stack(i_list, dim=1).view(bsz, -1)

        final_scores = torch.empty(0).to(lprobs)
        final_indices = torch.LongTensor().to(device=lprobs.device)
        final_beams = torch.LongTensor().to(device=lprobs.device)
        (final_scores, final_indices) = torch.topk(
            torch.stack(s_list, dim=1).view(bsz, -1),
            k,
        )

        final_beams = final_indices // k

        for i in range(bsz):
            final_indices[i] = indices[i][final_indices[i]]

        return final_scores, final_indices, final_beams
