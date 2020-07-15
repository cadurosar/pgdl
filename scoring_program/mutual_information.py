# Copyright 2020 The PGDL Competition organizers.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Scoring program for the PGDL competition
# Main contributor: Yiding Jiang, July 2020-October 2020

# Code for computing the conditional mutual information.

import copy
import numpy as np


def create_combination_name(hparams):
    combination = []
    for ch, chv in hparams.items():
        single_hparam_combination = []
        for v in chv:
            for w in chv:
                single_hparam_combination.append(str(v) + str(w))
        if len(combination) == 0:
            combination = single_hparam_combination
        else:
            new_combination = []
            for ch in combination:
                for sh in single_hparam_combination:
                    new_combination.append(ch + sh)
            combination = new_combination
    return combination


def make_key_from_models(model_1, model_2, conditioning_hparam):
  key = ''
  for ch in conditioning_hparam:
    key += str(model_1['hparams'][ch]['current_value']) + str(model_2['hparams'][ch]['current_value'])
  return key


def mi_empty_conditioning_set_table(prediction, model_specs):
    a0 = {'+gap': 0, '-gap': 1}
    a1 = {'+measure': 0, '-measure': 1}
    table = np.zeros((2, 2, 1))
    # fill the table
    for mid_1 in model_specs:
        for mid_2 in model_specs:
            try:
                k0 = '+gap' if model_specs[mid_1]['gen_gap'] > all_model_repr[mid_2]['gen_gap'] else '-gap'
                k1 = '+measure' if prediction[mid_1] > prediction[mid_2] else '-measure'
                all_table[a0[k0], a1[k1], 0] += 1
            except KeyError as e:
                continue
    axis_meaning = [a0, a1, None]
    return all_table, axis_meaning


def build_mi_table(prediction, model_specs, conditioning_hparams):
    if len(conditioning_hparams) == 0:
        return mi_empty_conditioning_set_table(prediction, model_specs)
    third_axis_len = 1
    for hparam in conditioning_hparams:
        values = conditioning_hparams[hparam]
        third_axis_len *= len(values)**2
    table = np.zeros((2, 2, third_axis_len))
    combinations = create_combination_name(conditioning_hparams)
    assert len(combinations) == third_axis_len, 'combination length and axis length do not match'
    # map to index mapping
    a0 = {'+gap': 0, '-gap': 1}
    a1 = {'+measure': 0, '-measure': 1}
    a2 = {k: i for i, k in enumerate(combinations)}
    # fill the table
    count = 0
    for mid_1 in model_specs:
      for mid_2 in model_specs:
        try:
          gap_positive = model_specs[mid_1]['metrics']['gen_gap'] > model_specs[mid_2]['metrics']['gen_gap']
          measure_diff_positive = prediction[mid_1] > prediction[mid_2]
          k0 = '+gap' if gap_positive else '-gap'
          k1 = '+measure' if measure_diff_positive else '-measure'
          # assert not gap_positive^measure_diff_positive, 'gap: {:.3f} {:.3f} measure: {:.3f} {:.3f}'.format(model_specs[mid_1]['metrics']['gen_gap'], model_specs[mid_2]['metrics']['gen_gap'], prediction[mid_1], prediction[mid_2])
          k2 = make_key_from_models(model_specs[mid_1], model_specs[mid_2], conditioning_hparams)
          table[a0[k0], a1[k1], a2[k2]] += 1
          count += 1
        except KeyError as e:
          print(e)
          continue
    axis_meaning = [a0, a1, a2]
    return table, axis_meaning


def entropy(p):
  total = p.sum()
  normalized_p = p / total
  log = np.log2(normalized_p + 1e-12)
  if np.isnan(-np.sum(normalized_p * log)):
    print('p: ', p, 'logp: ', log, 'H: ', -np.sum(normalized_p * log))
  return -np.sum(normalized_p * log)

def mi_from_table(table):
    total_entry = table.sum()
    a0_sum = np.sum(table, axis=1)
    a1_sum = np.sum(table, axis=0)
    a0_entropy = entropy(a0_sum)
    a1_entropy = entropy(a1_sum)
    mi = 0.0
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            joint = table[i, j] / total_entry
            if joint == 0:
                continue
            p_i = a0_sum[i] / total_entry
            p_j = a1_sum[j] / total_entry
            mi += joint * np.log2(joint / (p_i * p_j))
    return mi, a0_entropy, a1_entropy


def total_mi_from_table(table):
    expected_mi = 0
    expected_ce_0 = 0
    expected_ce_1 = 0
    for i in range(table.shape[-1]):
        mi, ce_0, ce_1 = mi_from_table(table[:, :, i])
        expected_mi += mi
        expected_ce_0 += ce_0
        expected_ce_1 += ce_1
    expected_mi /= table.shape[-1]
    expected_ce_0 /= table.shape[-1]
    expected_ce_1 /= table.shape[-1]
    return expected_mi / expected_ce_0


def conditional_mutual_information(prediction, model_specs):
    #model_specs = copy.deepcopy(model_specs)
    model_specs_copy = {}
    for mid in model_specs:
        if 'model' not in mid:
            model_specs_copy['model_{}'.format(mid)] = model_specs[mid]
        else:
            model_specs_copy[mid] = model_specs[mid]
    model_specs = model_specs_copy
    # compute and store generalization gap
    reference = {}
    for mid in model_specs:
        model_metrics = model_specs[mid]['metrics']
        model_metrics['gen_gap'] = model_metrics['train_acc'] - model_metrics['test_acc']
        reference[mid] = model_metrics['gen_gap']
    # get names of the hyerparameters
    all_mid = list(model_specs.keys())
    hp_names = list(model_specs[all_mid[0]]['hparams'].keys())
    hp_values = {k: model_specs[all_mid[0]]['hparams'][k]['possible_values'] for k in hp_names}
    # loop over possible hyperparam combinations
    min_mi = np.inf
    for h1 in hp_names:
        for h2 in hp_names:
            if h1 == h2:
                continue
            table, axis_meaning = build_mi_table(
                prediction, model_specs, {h1: hp_values[h1], h2: hp_values[h2]})
            mi = total_mi_from_table(table)
            min_mi = min(mi, min_mi)
    return min_mi
