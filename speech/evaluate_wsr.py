#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import itertools
import json

from multiprocessing import cpu_count

import numpy as np
import progressbar
import tensorflow as tf

from ds_ctcdecoder import ctc_beam_search_decoder_batch, Scorer
from six.moves import zip

from util.config import Config, initialize_globals
from util.evaluate_tools import calculate_report, get_cm
from util.feeding import create_dataset
from util.flags import create_flags, FLAGS
from util.logging import log_error, log_progress, create_progressbar

from tensorflow.compat.v1 import InteractiveSession
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def sparse_tensor_value_to_texts(value, alphabet):
    r"""
    Given a :class:`tf.SparseTensor` ``value``, return an array of Python strings
    representing its values, converting tokens to strings using ``alphabet``.
    """
    return sparse_tuple_to_texts((value.indices, value.values, value.dense_shape), alphabet)


def sparse_tuple_to_texts(sp_tuple, alphabet):
    indices = sp_tuple[0]
    values = sp_tuple[1]
    results = [''] * sp_tuple[2][0]
    for i, index in enumerate(indices):
        results[index[0]] += alphabet.string_from_label(values[i])
    # List of strings
    return results


def evaluate(test_csvs, create_model, try_loading):
    scorer = Scorer(FLAGS.lm_alpha, FLAGS.lm_beta,
                    FLAGS.lm_binary_path, FLAGS.lm_trie_path,
                    Config.alphabet)

    test_csvs = FLAGS.test_files.split(',')
    test_sets = [create_dataset([csv], batch_size=FLAGS.test_batch_size) for csv in test_csvs]
    iterator = tf.data.Iterator.from_structure(test_sets[0].output_types,
                                               test_sets[0].output_shapes,
                                               output_classes=test_sets[0].output_classes)
    test_init_ops = [iterator.make_initializer(test_set) for test_set in test_sets]

    (batch_x, batch_x_len), batch_y = iterator.get_next()

    # One rate per layer
    no_dropout = [None] * 6
    logits, _ = create_model(batch_x=batch_x,
                             seq_length=batch_x_len,
                             dropout=no_dropout)

    # Transpose to batch major and apply softmax for decoder
    transposed = tf.nn.softmax(tf.transpose(logits, [1, 0, 2]))

    loss = tf.nn.ctc_loss(labels=batch_y,
                          inputs=logits,
                          sequence_length=batch_x_len)

    tf.train.get_or_create_global_step()

    # Get number of accessible CPU cores for this process
    try:
        num_processes = cpu_count()
    except NotImplementedError:
        num_processes = 1

    # Create a saver using variables from the above newly created graph
    saver = tf.train.Saver()

    with tf.Session(config=Config.session_config) as session:
        # Restore variables from training checkpoint
        loaded = try_loading(session, saver, 'best_dev_checkpoint', 'best validation')
        if not loaded:
            loaded = try_loading(session, saver, 'checkpoint', 'most recent')
        if not loaded:
            log_error('Checkpoint directory ({}) does not contain a valid checkpoint state.'.format(FLAGS.checkpoint_dir))
            exit(1)

        def run_test(init_op, dataset):
            losses = []
            predictions = []
            ground_truths = []

            bar = create_progressbar(prefix='Test epoch | ',
                                     widgets=['Steps: ', progressbar.Counter(), ' | ', progressbar.Timer()]).start()
            log_progress('Test epoch...')

            step_count = 0

            # Initialize iterator to the appropriate dataset
            session.run(init_op)

            # First pass, compute losses and transposed logits for decoding
            while True:
                try:
                    batch_logits, batch_loss, batch_lengths, batch_transcripts = \
                        session.run([transposed, loss, batch_x_len, batch_y])
                except tf.errors.OutOfRangeError:
                    break

                decoded = ctc_beam_search_decoder_batch(batch_logits, batch_lengths, Config.alphabet, FLAGS.beam_width,
                                                        num_processes=num_processes, scorer=scorer)
                predictions.extend(d[0][1] for d in decoded)
                ground_truths.extend(sparse_tensor_value_to_texts(batch_transcripts, Config.alphabet))
                losses.extend(batch_loss)

                step_count += 1
                bar.update(step_count)

            bar.finish()

            wer, cer, wsr, wdr, wir, samples = calculate_report(ground_truths, predictions, losses)
            mean_loss = np.mean(losses)

            get_cm(ground_truths, predictions, losses, , cm_dir)

            # Take only the first report_count items 显示定义数量的翻译 按wer排序
            report_samples = itertools.islice(samples, FLAGS.report_count)

            print('Test on %s - WER: %f, CER: %f, loss: %f, WSR: %f, WDR: %f, WIR: %f' %
                  (dataset, wer, cer, mean_loss, wsr, wdr, wir))
            print('-' * 80)
            for sample in report_samples:
                print('WER: %f, CER: %f, loss: %f, WSR: %f, WDR: %f, WIR: %f' %
                      (sample.wer, sample.cer, sample.loss, sample.wsr, sample.wdr, sample.wir))
                print(' - src: "%s"' % sample.src)
                print(' - res: "%s"' % sample.res)
                print('-' * 80)

            return samples

        samples = []
        for csv, init_op in zip(test_csvs, test_init_ops):
            print('Testing model on {}'.format(csv))
            samples.extend(run_test(init_op, dataset=csv))
        return samples


def main(_):
    initialize_globals()

    if not FLAGS.test_files:
        log_error('You need to specify what files to use for evaluation via '
                  'the --test_files flag.')
        exit(1)

    from DeepSpeech import create_model, try_loading # pylint: disable=cyclic-import
    samples = evaluate(FLAGS.test_files.split(','), create_model, try_loading, FLAGS.cm_dir)

    if FLAGS.test_output_file:
        # Save decoded tuples as JSON, converting NumPy floats to Python floats
        json.dump(samples, open(FLAGS.test_output_file, 'w'), default=float)


if __name__ == '__main__':
    create_flags()
    tf.app.flags.DEFINE_string('test_output_file', '', 'path to a file to save all src/decoded/distance/loss tuples')
    tf.app.run(main)
