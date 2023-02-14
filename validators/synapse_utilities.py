from numpy import zeros_like
import bittensor
import threading
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from concurrent.futures import Future
import queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from bittensor.utils.tokenizer_utils import phrase_cross_entropy, topk_tokens_to_vocab_size
from typing import List, Tuple, Callable, Dict, Any, Union, Set
from loguru import logger
from validators.log_utilities import ValidatorLogger

def calc_loss_fct(loss_fct, logits, labels):
    r""" Calculates self.loss_fct with logits and labels that are expected to be aligned already.
    """
    _logits = logits.contiguous()
    _labels = labels.contiguous()
    loss = loss_fct(_logits.view(-1, _logits.size(-1)), _labels.view(-1))
    return loss

def scaling_law_loss_to_params(loss):
    r""" (OpenAI scaling laws) Kaplan, Jared, et al. "Scaling laws for neural language models." arXiv:2001.08361 (2020)
    """
    num_params = torch.exp(torch.log(torch.tensor(8.8e13).to(loss.device)) -
                           torch.log(torch.clamp(loss, 1.69)) / 0.076)  # loss lower bound 1.69 is entropy of natural text
    return num_params

def textcausallm(uids: torch.Tensor, query_responses: List[List[torch.FloatTensor]], return_ops: List[torch.LongTensor],
                 times: List[torch.FloatTensor], routing_score: torch.FloatTensor,
                 inputs: torch.FloatTensor, validation_len: int, loss_fct: Callable,                 
                 scaling_law_power: float, synergy_scaling_law_power: float, vlogger: ValidatorLogger,
                 logits_divergence_penalty: float,logging, synapse: 'bittensor.TextCausalLM' = None, index_s: int = 0
                 ) -> Tuple[torch.FloatTensor, Dict]:
    r"""
    Calculate Shapley values and neuron response validation measure statistics, given TextCausalLM synapse responses.
        Args:
            uids (:obj:`torch.Tensor`, `required`): [num_neurons]
                Neuron UIDs.
            query_responses (:obj:`List[List[torch.FloatTensor]]`, `required`):
                List of outputs from synapses, each a list of size num_endpoints of tensors with relevant size. Non-responses are zeroes of relevant
                synapse shape. Shape num_synapses * ( num_endpoints * ( -1, -1, -1 ) )
            return_ops (:obj:`List[torch.LongTensor]` of shape :obj:`[num_endpoints]`, `required`):
                Return code per call per synapse.
            times (:obj:`List [torch.FloatTensor]` of shape :obj:`[num_endpoints]`, `required`):
                Times per call per synapse.
            routing_score (:obj:`torch.FloatTensor`, `required`):
                [metagraph.n] Predictive routing score per endpoint in the metagraph, mean over the batch.
            inputs (:obj:`torch.FloatTensor`, `required`):
                [batch_size, sequence_len + validation_len] Token batch of original inputs with validation tokens.
            validation_len (:obj:`int`, `required`):
                Number of held-out phrase token batch for extended validation, not sent to neurons.
            loss_fct (:obj:`Callable`, `required`):
                CrossEntropy loss function to use.
            scaling_law_power (:obj:`float`, `required`):
                Power for modified scaling law, powered down to improve dynamic range, e.g. 3 → 6 nats for 0.5.
            synergy_scaling_law_power (:obj:`float`, `required`):
                Power for synergy modified scaling law, powered down to improve dynamic range, e.g. 3 → 6 nats for 0.5.
            logits_divergence_penalty (:obj:`float`, `required`):
                Penalty scaling for logits divergence.
            vlogger (:obj:`ValidatorLogger`, `required`):
                Logger for validator.
            logging (:obj:`bool`, `required`):
                Log tables to console.
            synapse (:obj:`bittensor.TextCausalLM`, `optional`):
                TextCausalLM synapse object.
            index_s (:obj:`int`, `optional`):
                Index of synapse to extract responses.

        Returns:
            loss (:obj:`torch.FloatTensor`):
                Loss for training validator nucleus and dendrite backward to endpoints.
            stats (:obj:`Dict`, `required`):
                Statistics per endpoint for this batch.
    """
    inputs_seq = inputs[..., :-validation_len]  # input sequence without last token [batch_size, sequence_len]
    inputs_val = inputs[..., -validation_len]  # input validation with next token [batch_size]

    def _base_params(_stats, query_response):
        _stats.update({'logits': query_response[:, :-1, :],
                       'logits_val': query_response[:, -1:, :]})

        for target, _ext in [(inputs_seq[:, 1:], ''), (inputs_val, '_val')]:
            _loss = calc_loss_fct(loss_fct, _stats['logits' + _ext], target)  # CausalLM loss
            if _loss.isnan() or _loss.isinf():
                _loss = 20  # assign large loss

            # estimate the effective number of model parameters, modified with the scaling_law_power
            _num_params = scaling_law_loss_to_params(_loss)

            # powered down number of params, e.g. dynamic range 3 → 6 nats for scaling_law_power=0.5
            _pow_num_params = torch.pow(_num_params, scaling_law_power)

            _stats.update({'loss' + _ext: _loss,
                           'est_params' + _ext: _num_params, 'base_params' + _ext: _pow_num_params,
                           'synergy' + _ext: 0, 'synergy_loss_diff' + _ext: 0})

    def _synergy(first, second, target, _ext):
        # Combined logits: log of average probabilities per token between responses
        combined_logits = torch.log((torch.softmax(first['logits' + _ext], dim=-1) +
                                     torch.softmax(second['logits' + _ext], dim=-1)) / 2 + 1e-40)
        measured_loss = calc_loss_fct(loss_fct, combined_logits, target)  # actual measured loss

        return measured_loss

    shapley_start_time = time.time()

    loss, stats, unsuccessful = shapley_base(uids, query_responses, return_ops, times, routing_score,
                                             _base_params, index_s, ext='')

    logger.info(f'{str(synapse)} \t| Shapley base values (power={scaling_law_power:.1f}) '
                f'<dim>[{time.time() - shapley_start_time:.3g}s]</dim>')

    synergy_start_time = time.time()

    syn_loss_diff = shapley_synergy(stats, _synergy, ext='', target=inputs_seq[:, 1:],
                                    scaling_law_power=synergy_scaling_law_power)
    syn_loss_diff_val = shapley_synergy(stats, _synergy, ext='_val', target=inputs_val,
                                        scaling_law_power=synergy_scaling_law_power)

    # === Shapley value combination ===
    # Combine base values with synergy approximation to get final Shapley values.
    for s in stats.values():
        for ext in ['', '_val']:
            if 'base_params' + ext in s and 'synergy' + ext in s:
                s['shapley_values' + ext] = (s['base_params' + ext] + s['synergy' + ext])

            if 'logits' + ext in s:
                del s['logits' + ext]  # remove logits - not needed for stats anymore

        if 'shapley_values' in s and 'shapley_values_val' in s:
            s['shapley_values_min'] = torch.min(s['shapley_values'], s['shapley_values_val'])

        for key in s:
            if hasattr(s[key], 'item'):
                s[key] = s[key].item()

    logger.info(f'{str(synapse)} \t| Shapley synergy values (power={synergy_scaling_law_power:.1f}) '
                f'<dim>[{time.time() - synergy_start_time:.3g}s]</dim>')

    if logging:
        # === Synergy table ===
        # Prints the synergy loss diff matrix with pairwise loss reduction due to synergy (original loss on diagonal)
        vlogger.print_synergy_table(stats, syn_loss_diff, 'shapley_values_min')

        # === Neuron responses (table) ===
        # Prints the evaluation of the neuron responses to the validator request
        vlogger.print_synapse_table(str(synapse), stats, 'shapley_values_min', shapley_start_time)

    # === Unsuccessful responses ===
    # Prints the return codes and response times of unsuccessful responses
    unsuccess(str(synapse), unsuccessful)

    return loss, stats

def textcausallmnext(uids: torch.Tensor, query_responses: List[List[torch.FloatTensor]], return_ops: List[torch.LongTensor],
                     times: List[torch.FloatTensor], routing_score: torch.FloatTensor,
                     inputs: torch.FloatTensor, validation_len: int, loss_fct: Callable,                     
                     scaling_law_power: float, synergy_scaling_law_power: float, vlogger:ValidatorLogger,
                     logits_divergence_penalty: float,logging, synapse: 'bittensor.TextCausalLMNext' = None, index_s: int = 0
                     ) -> Tuple[torch.FloatTensor, Dict]:
    r"""
    Calculate Shapley values and neuron response validation measure statistics, given TextCausalLMNext synapse responses.
        Args:
            uids (:obj:`torch.Tensor`, `required`): [num_neurons]
                Neuron UIDs.
            query_responses (:obj:`List[List[torch.FloatTensor]]`, `required`):
                List of outputs from synapses, each a list of size num_endpoints of tensors with relevant size. Non-responses are zeroes of relevant
                synapse shape. Shape num_synapses * ( num_endpoints * ( -1, -1, -1 ) )
            return_ops (:obj:`List[torch.LongTensor]` of shape :obj:`[num_endpoints]`, `required`):
                Return code per call per synapse.
            times (:obj:`List [torch.FloatTensor]` of shape :obj:`[num_endpoints]`, `required`):
                Times per call per synapse.
            routing_score (:obj:`torch.FloatTensor`, `required`):
                [metagraph.n] Predictive routing score per endpoint in the metagraph, mean over the batch.
            inputs (:obj:`torch.FloatTensor`, `required`):
                [batch_size, sequence_len + validation_len] Token batch of original inputs with validation tokens.
            validation_len (:obj:`int`, `required`):
                Number of held-out phrase token batch for extended validation, not sent to neurons.
            loss_fct (:obj:`Callable`, `required`):
                CrossEntropy loss function to use.
            scaling_law_power (:obj:`float`, `required`):
                Power for modified scaling law, powered down to improve dynamic range, e.g. 3 → 6 nats for 0.5.
            synergy_scaling_law_power (:obj:`float`, `required`):
                Power for synergy modified scaling law, powered down to improve dynamic range, e.g. 3 → 6 nats for 0.5.
            logits_divergence_penalty (:obj:`float`, `required`):
                Penalty scaling for logits divergence.
            vlogger (:obj:`ValidatorLogger`, `required`):
                Logger for validator.
            logging (:obj:`bool`, `required`):
                Log tables to console.
            synapse (:obj:`bittensor.TextCausalLMNext`, `optional`):
                TextCausalLMNext Synapse object.
            index_s (:obj:`int`, `optional`):
                Index of synapse to extract responses.

        Returns:
            loss (:obj:`torch.FloatTensor`):
                Loss for training validator nucleus and dendrite backward to endpoints.
            stats (:obj:`Dict`, `required`):
                Statistics per endpoint for this batch.
    """
    inputs_nxt = inputs[..., -validation_len:]  # input validation with next token target phrase [batch_size, val_len]

    def _base_params(_stats, query_response):
        # topk_tensor = unravel_topk_token_phrases(query_response, topk=synapse.topk)  # [batch_size, topk + 1, max_len]
        _losses_val, _losses = phrase_cross_entropy(inputs_nxt, query_response, reduce=False)
        _losses_val[_losses_val.isnan()] = 20  # assign large loss
        _losses[_losses.isnan()] = 20  # assign large loss
        _loss_val = _losses_val.mean()
        _loss = _losses.mean()

        _stats.update({'loss_val_nxt': _loss_val, 'losses_nxt': _losses, 'loss_nxt': _loss,
                       'synergy_nxt': 0, 'synergy_loss_diff_nxt': 0})

    def _synergy(first, second, target, ext):
        # average first + second probabilities per batch item, convert to loss
        measured_loss = -torch.log((torch.exp(-first['losses_nxt']) +
                                    torch.exp(-second['losses_nxt'])) / 2 + 1e-40).mean()

        return measured_loss

    shapley_start_time = time.time()
    loss, stats, unsuccessful = shapley_base(uids, query_responses, return_ops, times, routing_score,
                                             _base_params, index_s, ext='_nxt')
    logger.info(f'{str(synapse)} \t| Shapley base values (power={scaling_law_power:.1f}) '
                f'<dim>[{time.time() - shapley_start_time:.3g}s]</dim>')

    divergence_start_time = time.time()
    with torch.no_grad():
        logits_divergence(stats, uids, query_responses, return_ops, times, index_s, ext='_nxt')
    logger.info(f'{str(synapse)} \t| Logits divergences (penalty={logits_divergence_penalty}) '
                f'<dim>[{time.time() - divergence_start_time:.3g}s]</dim>')

    synergy_start_time = time.time()
    syn_loss_diff = shapley_synergy(stats, _synergy, '_nxt', scaling_law_power=synergy_scaling_law_power)
    logger.info(f'{str(synapse)} \t| Shapley synergy values (power={synergy_scaling_law_power:.1f}) '
                f'<dim>[{time.time() - synergy_start_time:.3g}s]</dim>')

    # === Shapley value combination ===
    # Combine base values with synergy approximation to get final Shapley values.
    for s in stats.values():
        if 'losses_nxt' in s:
            del s['losses_nxt']  # remove batch losses - not needed for stats anymore

        for key in s:
            if hasattr(s[key], 'item'):
                s[key] = s[key].item()

    if logging:
        # === Response table ===
        # Prints the query response table: top prediction probabilities and texts for batch tasks
        batch_predictions = format_predictions(uids, query_responses, return_ops, inputs, validation_len, index_s)
        vlogger.print_response_table(batch_predictions, stats, sort_col='loss_nxt')

        # === Synergy table ===
        # Prints the synergy loss diff matrix with pairwise loss reduction due to synergy (original loss on diagonal)
        vlogger.print_synergy_table(stats, syn_loss_diff, 'loss_nxt')

        # === Neuron responses (table) ===
        # Prints the evaluation of the neuron responses to the validator request
        vlogger.print_synapse_table(str(synapse), stats, 'loss_nxt', shapley_start_time)

    # === Unsuccessful responses ===
    # Prints the return codes and response times of unsuccessful responses
    unsuccess(str(synapse), unsuccessful)

    return loss, stats


def shapley_base(uids: torch.Tensor, query_responses: List[List[torch.FloatTensor]], return_ops: List[torch.LongTensor],
                 times: List[torch.FloatTensor], routing_score: torch.FloatTensor,
                 base_params: Callable, index_s: int = 0, ext: str = None) -> Tuple[Union[float, torch.FloatTensor],
                                                                                    Dict,
                                                                                    List]:
    r"""
    Calculate Shapley base values and neuron response validation measure statistics, given responses from a synapse.
        Args:
            uids (:obj:`torch.Tensor`, `required`): [num_neurons]
                Neuron UIDs.
            query_responses (:obj:`List[List[torch.FloatTensor]]`, `required`):
                List of outputs from synapses, each a list of size num_endpoints of tensors with relevant size. Non-responses are zeroes of relevant
                synapse shape. Shape num_synapses * ( num_endpoints * ( -1, -1, -1 ) )
            return_ops (:obj:`List[torch.LongTensor]` of shape :obj:`[num_endpoints]`, `required`):
                Return code per call per synapse.
            times (:obj:`List [torch.FloatTensor]` of shape :obj:`[num_endpoints]`, `required`):
                Times per call per synapse.
            routing_score (:obj:`torch.FloatTensor`, `required`):
                [metagraph.n] Predictive routing score per endpoint in the metagraph, mean over the batch.
            base_params (:obj:`Callable`, `required`):
                CrossEntropy loss function to use.
            index_s (:obj:`int`, `optional`):
                Index of synapse to extract responses.
            ext (:obj:`str`, `optional`):
                Extension to parameter string for stats key.

        Returns:
            loss (:obj:`torch.FloatTensor`):
                Loss for training validator nucleus and dendrite backward to endpoints.
            stats (:obj:`Dict`, `required`):
                Statistics per endpoint for this batch.
            unsuccessful (:obj:`List`, `required`):
                Unsuccessful endpoints [(uid, return_op, time)].
    """
    stats = {}
    unsuccessful = []
    neuron_loss = 0.  # neuron losses to accumulate to then backward() via dendrite
    routing_loss = 0.  # validator routing loss for local model update

    # === Base parameter estimation ===
    # Shapley values - base level - coalition size 1
    # Collect successful neuron responses, calculate base Shapley values.
    # Measured in effective number of model parameters, according to OpenAI scaling laws.
    for index, _uid in enumerate(uids.tolist()):
        if return_ops[index][index_s] == bittensor.proto.ReturnCode.Success:
            _stats = {'uid': _uid,
                      'response_time' + ext: times[index][index_s],
                      'routing_score': routing_score[_uid]}

            try:
                base_params(_stats, query_responses[index][index_s])

                neuron_loss += _stats['loss' + ext]  # add sequence loss to be backward() to neuron

                # === Add routing loss ===
                # MSE loss between predicted routing score and ideal target routing score.
                # The Bayes risk approx. 1.69, i.e. the minimal loss achievable for next-token
                # prediction on the full distribution 𝑃, a.k.a the "entropy of natural text"
                # Hoffmann, Jordan, et al. "Training Compute-Optimal Large Language Models." arXiv:2203.15556 (2022).
                routing_score_target = torch.exp(-torch.clamp(_stats['loss' + ext].detach() - 1.69, 0))
                _routing_loss = (routing_score[_uid] - routing_score_target) ** 2  # MSE loss
                routing_loss += _routing_loss
                _stats.update({'routing_score_target' + ext: routing_score_target, 'routing_loss' + ext: _routing_loss})

                stats[_uid] = _stats
            except Exception as e:
                logger.warning(f'Synapse {index_s} error (shapley_base)\t| '
                               f'UID {_uid} <dim>[{times[index][index_s]:.2f}s]</dim>: {e}')
                stats[_uid] = _stats
                unsuccessful += [(_uid, return_ops[index][index_s], times[index][index_s])]
        else:
            stats[_uid] = {'uid': _uid,
                           'response_time' + ext: times[index][index_s],
                           'routing_score': routing_score[_uid]}
            unsuccessful += [(_uid, return_ops[index][index_s], times[index][index_s])]

    return neuron_loss + routing_loss, stats, unsuccessful


def logits_divergence(stats: Dict, uids: torch.Tensor, query_responses: List[List[torch.FloatTensor]],
                      return_ops: List[torch.LongTensor], times: List[torch.FloatTensor],
                      index_s: int = 0, ext: str = None):
    r"""
    Calculate each logits divergence per neuron per task from the average logits over all neurons per task,
    given responses from a synapse.
        Args:
            stats (:obj:`Dict`, `required`):
                Statistics per endpoint for this batch.
            uids (:obj:`torch.Tensor`, `required`): [num_neurons]
                Neuron UIDs.
            query_responses (:obj:`List[List[torch.FloatTensor]]`, `required`):
                List of outputs from synapses, each a list of size num_endpoints of tensors with relevant size.
                Non-responses are zeroes of relevant synapse shape.
                Shape num_synapses * ( num_endpoints * ( -1, -1, -1 ) )
            return_ops (:obj:`List[torch.LongTensor]` of shape :obj:`[num_endpoints]`, `required`):
                Return code per call per synapse.
            times (:obj:`List [torch.FloatTensor]` of shape :obj:`[num_endpoints]`, `required`):
                Times per call per synapse.
            index_s (:obj:`int`, `optional`):
                Index of synapse to extract responses.
            ext (:obj:`str`, `optional`):
                Extension to parameter string for stats key.
    """
    probs_k = 0
    probs_avg = None

    # === Probs averaging ===
    # Calculate the average token distribution for each batch task.
    for index, _uid in enumerate(uids.tolist()):
        if return_ops[index][index_s] == bittensor.proto.ReturnCode.Success:
            try:
                probs = topk_tokens_to_vocab_size(query_responses[index][index_s],
                                                  bittensor.__vocab_size__)  # [batch_size, vocab_size]
                if probs_avg is None:
                    probs_avg = probs
                else:
                    probs_avg += probs
                probs_k += 1

            except Exception as e:
                logger.warning(f'Synapse {index_s} error (logits_divergence)\t| '
                               f'UID {_uid} <dim>[{times[index][index_s]:.2f}s]</dim>: {e}')

    if probs_avg is not None:
        probs_avg /= probs_k
        probs_avg_sqrt = probs_avg.sqrt()
        batch_divergences = []

        # === Distribution divergence ===
        # Calculate the Hellinger distance (f-divergence) from the average probability distribution for each batch task.
        for index, _uid in enumerate(uids.tolist()):
            if return_ops[index][index_s] == bittensor.proto.ReturnCode.Success:
                try:
                    probs = topk_tokens_to_vocab_size(query_responses[index][index_s],
                                                      bittensor.__vocab_size__)  # [batch_size, vocab_size]
                    divergences = 0.5 * torch.pow(probs.sqrt() - probs_avg_sqrt, 2).sum(dim=1)  # [batch_size] in [0, 1]
                    divergences = divergences.sqrt()
                    stats[_uid]['logits_divergences' + ext] = divergences  # [batch_size]
                    stats[_uid]['logits_divergence' + ext] = divergences.mean()  # scalar
                    batch_divergences += [divergences]

                except Exception as e:
                    logger.warning(f'Synapse {index_s} error (logits_divergence)\t| '
                                   f'UID {_uid} <dim>[{times[index][index_s]:.2f}s]</dim>: {e}')

        batch_divergences = torch.stack(batch_divergences)  # [uids_len, batch_size]
        avg = batch_divergences.mean(dim=0)  # [batch_size]
        std = batch_divergences.std(dim=0)  # [batch_size]

        # logger.info(f"Logits divergences: "
        #             f"avg={', '.join([f'{i}:{v:.3g}' for i, v in enumerate(avg)])}")
        # logger.info(f"std={', '.join([f'{i}:{v:.3g}' for i, v in enumerate(std)])}")

        # === Calculate divergence excess ===
        # For each batch task, calculate excess deviation above a single stddev, in terms of stddev,
        # and apply power to increase score above two stddev, and decrease between one and two stddev.
        # This will effectively allow zero excess below one stddev, and minimal excess below two stddev,
        # but amplify any excess above two stddev (only 2.1% of population for normal dist).
        for _uid, _stats in stats.items():
            if 'logits_divergences' + ext in _stats:
                try:
                    excess = torch.clamp(_stats['logits_divergences' + ext] - (avg + std), 0)  # divergence > avg + std
                    excess /= std + 1e-9  # stddev multiples above 1 stddev
                    excess = torch.pow(excess, 3)  # reduce < 2std, increase > 2std
                    excess = torch.clamp(excess, 0, 10)  # maximum excess ratio of 10

                    _stats['logits_excess' + ext] = excess.mean()  # in [0, 10]
                    del _stats['logits_divergences' + ext]  # keep only scalar stats beyond this

                    # logger.info(f"UID{uid} divergences [{_stats['logits_divergences' + ext].mean():.4g}]: "
                    #             f"{', '.join([f'{i}:{dist:.3g}' for i, dist in enumerate(_stats['logits_divergences' + ext])])}")
                    # logger.info(f"UID{uid} excess [{excess.mean():.3g}]: "
                    #             f"{', '.join([f'{i}:{exc:.3g}' for i, exc in enumerate(excess)])}")

                except Exception as e:
                    logger.warning(f'Synapse {index_s} error (logits_divergence)\t| UID {_uid}: {e}')


def shapley_synergy(stats: Dict, synergy: Callable, ext: str, target: torch.Tensor = None, scaling_law_power: float = 0.5):
    r"""
    Calculates Shapley synergy for coalition size 2, measured performance above expected performance.
    Measured in effective number of model parameters, just like base Shapley values.
        Args:
            stats (:obj:`Dict`, `required`):
                Statistics per endpoint for this batch.
            synergy (:obj:`Callable`, `required`)
                Function to calculate measured loss.
            ext (:obj:`str`, `optional`):
                Extension to parameter string for stats key.
            target (:obj:`torch.Tensor`, `optional`):
                Target to measure loss against.
            scaling_law_power (:obj:`float`, `optional`):
                Power for modified scaling law, powered down to improve dynamic range, e.g. 3 → 6 nats for 0.5.

        Returns:
            syn_loss_diff (:obj:`Dict`, `required`):
                Dictionary table of pairwise synergies as loss reductions, with direct loss on diagonal.
    """
    # === Shapley synergy approximation ===
    # Shapley values - second level - coalition size 2
    # Synergy = measured performance above expected performance
    # Measured in effective number of model parameters, just like base Shapley values.
    syn_loss_diff = {}  # expected_loss - measured_loss (where > 0)
    responsives = [uid for uid, stat in stats.items() if 'loss' + ext in stat]
    for _first, first in stats.items():
        if 'loss' + ext not in first:
            continue
        first_diff = syn_loss_diff.setdefault(_first, {})
        first_diff[_first] = first['loss' + ext]  # diagonal keeps direct loss

        for _second, second in stats.items():
            if 'loss' + ext not in second or _second <= _first:
                continue
            second_diff = syn_loss_diff.setdefault(_second, {})

            with torch.no_grad():
                expected_loss = torch.min(first['loss' + ext], second['loss' + ext])  # expecting min loss

                measured_loss = synergy(first, second, target, ext)

                loss_diff_share = torch.clamp(expected_loss - measured_loss, 0) / 2  # record direct loss diff
                loss_diff_share /= len(responsives)  # average over responsives
                first['synergy_loss_diff' + ext] += loss_diff_share
                second['synergy_loss_diff' + ext] += loss_diff_share

                # pairwise loss reduction of expected to measured loss due to synergy between first and second
                first_diff[_second] = loss_diff_share
                second_diff[_first] = loss_diff_share

                measured_params = scaling_law_loss_to_params(measured_loss)
                expected_params = scaling_law_loss_to_params(expected_loss)

                # powered down number of params, e.g. dynamic range 3 → 6 nats for scaling_law_power=0.5
                pow_measured_params = torch.pow(measured_params, scaling_law_power)
                pow_expected_params = torch.pow(expected_params, scaling_law_power)

                synergy_share = torch.clamp(pow_measured_params - pow_expected_params, 0) / 2
                synergy_share /= len(responsives)  # average over responsives
                first['synergy' + ext] += synergy_share  # share synergy amongst coalition members
                second['synergy' + ext] += synergy_share

    return syn_loss_diff


def format_predictions(uids: torch.Tensor, query_responses: List[List[torch.FloatTensor]],
                       return_ops: List[torch.LongTensor], inputs: torch.FloatTensor,
                       validation_len: int, index_s: int = 0, number_of_predictions: int = 3) -> List:
    r""" Format batch task topk predictions for rich table print of query responses.
    """
    batch_predictions = []
    std_tokenizer = bittensor.tokenizer()

    # === Batch iteration ===
    for batch_item in range(inputs.shape[0]):
        # === Task formatting ===
        context = inputs[batch_item][:-validation_len]
        answer = inputs[batch_item][-validation_len:]

        context = repr(std_tokenizer.decode(context))[1:-1][-30:]  # strip '' and truncate
        answer = repr(std_tokenizer.decode(answer))[1:-1][:15]  # strip '' and truncate

        task = f"[reverse]{context}[/reverse][bold]{answer}[/bold]"

        # === Prediction formatting ===
        predictions = {}
        for index, uid in enumerate(uids.tolist()):
            if return_ops[index][index_s] == bittensor.proto.ReturnCode.Success:
                topk_tensor = query_responses[index][index_s]  # [batch_size, (topk + 1), max_len] (prob_k) + floor_prob
                topk_tokens = topk_tensor[batch_item, :-1, 1:].int()  # [batch_size, topk, max_len - 1] Phrase tokens with ignore_index token for padding.
                topk_probs = topk_tensor[batch_item, :-1, 0]  # [batch_size, topk] Probabilities for each phrase in topk

                # === Topk iteration ===
                topk_predictions = ''
                for i in range(number_of_predictions):
                    phrase = topk_tokens[i]
                    phrase = phrase[phrase >= 0]  # strip negative ignore_index = -100
                    phrase_str = repr(std_tokenizer.decode(phrase))[:15]  # decode, escape and truncate
                    prob = f'{topk_probs[i]:.3f}'.lstrip('0').replace('1.000', '1.00')
                    topk_predictions += f"[green]{prob}[/green]: {phrase_str} "

                predictions[uid] = topk_predictions[:-1]  # strip trailing space

        batch_predictions += [(task, predictions)]

    return batch_predictions


def unsuccess(_name, _unsuccessful):
    r""" Prints the return codes and response times of unsuccessful responses
    """
    # === Unsuccessful responses ===
    unsuccess_txt = f'{_name} \t| Unsuccessful <cyan>UID</cyan>[<red>return_op</red> <yellow>time</yellow>]: '
    for _uid, _return_op, _time in _unsuccessful:
        unsuccess_txt += f'{_uid}[<red>{_return_op}</red> <yellow>{_time:.2f}</yellow>] '
    logger.info(unsuccess_txt)
