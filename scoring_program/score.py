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

#!/usr/bin/env python

# Scoring program for the PGDL competition at NeurIPS 2020
# Modified from the AutoDL competition scoring program
# Yiding Jiang, July 2020-October 2020

# Some libraries and options
import os
from sys import argv
import json
import collections
import itertools

import my_metric
import libscores
import sklearn.metrics as metrics
import yaml
from libscores import *
from mutual_information import conditional_mutual_information


# Default I/O directories:
root_dir = "../"
default_solution_dir = root_dir + "sample_data"
default_prediction_dir = root_dir + "sample_result_submission"
default_score_dir = root_dir + "scoring_output"

# Debug flag 0: no debug, 1: show all scores, 2: also show version amd listing of dir
debug_mode = 0

# Constant used for a missing score
missing_score = -0.999999

# Version number
scoring_version = 1.0

# Names for filtering
filter_filenames = ['.DS_Store', '__MACOSX']

def name_filter(name):
    for fn in filter_filenames:
        if fn in name:
            return True
    return False

def _HERE(*args):
    h = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(h, *args)

def get_metric():
    with open(_HERE('metric.txt'), 'r') as f:
        metric_name = f.readline().strip()
    scoring_function = getattr(my_metric, metric_name)
    return metric_name, scoring_function

def check_data_validity(model_specs):
  sample_mid = list(model_specs.keys())[0]
  sample_config = model_specs[sample_mid]['hparams']
  ordered_hpnames = sorted(list(sample_config.keys()))
  possible_values = [sample_config[hp]["possible_values"] for hp in ordered_hpnames]
  def get_index(config):
      index = []
      for hp, hpv in zip(ordered_hpnames, possible_values):
          index.append(hpv.index(config[hp]["current_value"]))
      return index

  shape = [len(pv) for pv in possible_values]
  grid = np.zeros(shape)

  def fill_grid(index, grid):
      curr_level = grid
      for idx in index[:-1]:
          curr_level = curr_level[idx]
      curr_level[index[-1]] += 1

  for mid in model_specs:
      fill_grid(get_index(model_specs[mid]['hparams']), grid)

  complete = np.sum(1.0-np.float32(grid > 0))
  unique = np.sum(1.0-np.float32(grid == 1))

  if complete != 0:
    raise ValueError("The data are not complete!")

  if unique != 0:
    raise ValueError("The data are not unique!")

  print("Data is valid :)")

def check_data_validity_v2(cfg):
    params = next(iter(cnf.values()))['hparams']
    expected_names = itertools.product(*[[(hparam_name, value) for value in params[hparam_name]['possible_values']] for hparam_name in params])
    expected = collections.Counter(expected_names)
    found_names = [tuple((hparam_name, e['current_value']) for hparam_name, e in v['hparams'].items()) for v in cnf.values()]
    found = collections.Counter(found_names)
    if len(found-expected):
      print('Difference between expected and found: {}'.format(found-expected))
      raise Value('Data are invalid!')

# =============================== MAIN ========================================

if __name__ == "__main__":

    #### INPUT/OUTPUT: Get input and output directory names
    if len(argv) == 1:  # Use the default data directories if no arguments are provided
        solution_dir = default_solution_dir
        prediction_dir = default_prediction_dir
        score_dir = default_score_dir
    elif len(argv) == 3: # The current default configuration of Codalab
        solution_dir = os.path.join(argv[1], 'ref')
        prediction_dir = os.path.join(argv[1], 'res')
        score_dir = argv[2]
    elif len(argv) == 4:
        solution_dir = argv[1]
        prediction_dir = argv[2]
        score_dir = argv[3]
    else:
        swrite('\n*** WRONG NUMBER OF ARGUMENTS ***\n\n')
        exit(1)

    # Going into reference data
    solution_dir_content = os.listdir(solution_dir)
    if 'reference_data' in solution_dir_content:
      solution_dir = os.path.join(solution_dir, 'reference_data')

    print('\nsolution_dir: ', solution_dir)
    print('prediction_dir: ', prediction_dir)
    print('score_dir: ', score_dir, '\n')
    # Create the output directory, if it does not already exist and open output files
    mkdir(score_dir)
    score_file = open(os.path.join(score_dir, 'scores.txt'), 'w')
    html_file = open(os.path.join(score_dir, 'scores.html'), 'w')

    # Get the metric
    metric_name, scoring_function = get_metric()

    # Get all the solution files from the solution directory
    # data_names = [name for name in os.listdir(solution_dir) if '__MACOSX' not in name and '.DS_Store' not in name]
    data_names = [name for name in os.listdir(solution_dir) if not name_filter(name)]
    print(data_names)

    time_exceeded = False
    task_scores = []
    # Loop over files in solution directory and search for predictions with extension .predict having the same basename
    for i, basename in enumerate(data_names):
        set_num = i + 1  # 1-indexed
        # score_name = 'set%s_score' % set_num
        score_name = 'task_{}_score'.format(basename)
        
        score = 0.0
       
        try:
            # Get the last prediction from the res subdirectory (must end with '.predict')
            predict_file = os.path.join(prediction_dir, basename + '.predict')
            with open(predict_file, 'r') as f:
                prediction = json.load(f)
            print('Read prediction from: {}'.format(predict_file))
        
            for mid in prediction:
                if prediction[mid] == 'EXCEEDED':
                    time_exceeded = True

            if time_exceeded:
              continue

            model_specs_file = os.path.join(solution_dir, basename, 'model_configs.json') 
            with open(model_specs_file, 'r') as f:
                model_specs = json.load(f)
            print('Read model configs from: {}'.format(model_specs_file))
            check_data_validity(model_specs)

            if len(model_specs) != len(prediction):
            	raise ValueError("Prediction shape={} instead of Solution shape={}".format(len(prediction), len(model_specs)))
            try:
                # Compute the score prescribed by the metric file
                print('Start computing score for {}'.format(basename))
                score = scoring_function(prediction, model_specs)
                print('Score computation finished...')
                print(
                    "======= Set %d" % set_num + " (" + basename.capitalize() + "): " + metric_name + "(" + score_name + ")=%0.12f =======" % score)
                html_file.write(
                    "<pre>======= Set %d" % set_num + " (" + basename.capitalize() + "): " + metric_name + "(" + score_name + ")=%0.12f =======\n" % score)
            except Exception as inst:
                raise Exception('Error in calculation of the specific score of the task: \n {}'.format(inst))
            # record score for individual tasks
            task_scores.append(score)
            i#if debug_mode > 0:
             #   scores = compute_all_scores(solution, prediction)
             #   write_scores(html_file, scores)

        except Exception as inst:
            score = missing_score
            print(
                "======= Set %d" % set_num + " (" + basename.capitalize() + "): " + metric_name + "(" + score_name + ")=ERROR =======")
            html_file.write(
                "======= Set %d" % set_num + " (" + basename.capitalize() + "): " + metric_name + "(" + score_name + ")=ERROR =======\n")
            print(inst)

        # Write score corresponding to selected task and metric to the output file
        score_file.write(score_name + ": %0.12f\n" % score)

    # End loop for solution_file in solution_names

    task_average_score = sum(task_scores)/float(len(task_scores))
    # Solution exceeding time budget receives lowest score of 0.0
    if time_exceeded:
      task_average_score = 0.0
    task_average_score *= 100.0
    score_file.write("task_average_score" + ": %0.12f\n" % task_average_score)
    print("Task average: %0.12f" % task_average_score)

    # Read the execution time and add it to the scores:
    try:
        metadata = yaml.load(open(os.path.join(input_dir, 'res', 'metadata'), 'r'))
        score_file.write("Duration: %0.6f\n" % metadata['elapsedTime'])
    except:
        score_file.write("Duration: 0\n")
        html_file.close()

    score_file.close()

    # Lots of debug stuff
    if debug_mode > 1:
        swrite('\n*** SCORING PROGRAM: PLATFORM SPECIFICATIONS ***\n\n')
        show_platform()
        show_io(prediction_dir, score_dir)
        show_version(scoring_version)
