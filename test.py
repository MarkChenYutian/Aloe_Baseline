# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example code for running model on CLEVRER."""
import json

from absl import app
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf
  
from colorama import Fore, Back, Style

from object_attention_for_reasoning import model as modellib

from tqdm.auto import tqdm
import os




BATCH_SIZE = 1
NUM_FRAMES = 25
NUM_OBJECTS = 8

_BASE_DIR = flags.DEFINE_string(
        "base_dir", "./clevrer_monet_latents",
        "Directory containing checkpoints and MONet latents.")
_SCENE_IDX = flags.DEFINE_integer(
        "scene_idx", 10000, "Scene index of CLEVRER video.")


def load_monet_latents(base_dir, scene_index):
    filename = f"{base_dir}/valid/{scene_index}.npz"
    with open(filename, "rb") as f:
        return np.load(f)


def _split_string(s):
    """Splits string to words and standardize alphabet."""
    return s.lower().replace("?", "").split()


def _pad(array, length):
    """Pad an array to desired length."""
    if len(array.shape) == 1:
        return np.pad(array, [(0, length - array.shape[0])], mode="constant")
    else:
        return np.pad(array, [(0, length - array.shape[0]), (0, 0)], mode="constant")


def encode_sentence(token_map, sentence, pad_length):
    """Encode CLEVRER question/choice sentences as sequence of token ids."""
    ret = np.array(
            [token_map["question_vocab"][w] for w in _split_string(sentence)],
            np.int32)
    return _pad(ret, pad_length)


def encode_choices(token_map, choices):
    """Encode CLEVRER choices."""
    arrays = [encode_sentence(token_map, choice["choice"],
                                                        modellib.MAX_CHOICE_LENGTH)
                        for choice in choices]
    return _pad(np.stack(arrays, axis=0), modellib.NUM_CHOICES)


def main(unused_argv):
    os.environ['KMP_WARNINGS'] = 'False'
    base_dir = _BASE_DIR.value
    with open(f"{base_dir}/vocab.json", "rb") as f:
        token_map = json.load(f)

    reverse_answer_lookup = {v: k for k, v in token_map["answer_vocab"].items()}

    with open(f"{base_dir}/valid.json", "rb") as f:
        questions_data = json.load(f)

    tf.reset_default_graph()
    model = modellib.ClevrerTransformerModel(**modellib.PRETRAINED_MODEL_CONFIG)

    inputs_descriptive = {
            "monet_latents": tf.placeholder(
                    tf.float32,
                    [BATCH_SIZE, NUM_FRAMES, NUM_OBJECTS, modellib.EMBED_DIM]),
            "question": tf.placeholder(
                    tf.int32, [BATCH_SIZE, modellib.MAX_QUESTION_LENGTH]),
    }

    inputs_mc = {
            "monet_latents": tf.placeholder(
                    tf.float32,
                    [BATCH_SIZE, NUM_FRAMES, NUM_OBJECTS, modellib.EMBED_DIM]),
            "question": tf.placeholder(tf.int32,
                                                                  [BATCH_SIZE, modellib.MAX_QUESTION_LENGTH]),
            "choices": tf.placeholder(
                    tf.int32, [BATCH_SIZE, modellib.NUM_CHOICES,
                                          modellib.MAX_CHOICE_LENGTH]),
    }

    output_descriptive = model.apply_model_descriptive(inputs_descriptive)
    output_mc = model.apply_model_mc(inputs_mc)

    # Restore from checkpoint
    saver = tf.train.Saver()
    checkpoint_dir = f"{base_dir}/checkpoints/"
    sess = tf.train.SingularMonitoredSession(checkpoint_dir=checkpoint_dir)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)

    def eval_descriptive(monet_latents, question_json):
        # CLEVRER provides videos with 128 frames. In our model, we subsample 25
        # frames (as was done in Yi et al (2020)).
        # For training, we randomize the choice of 25 frames, and for evaluation, we
        # sample the 25 frames as evenly as possible.
        # We do that by doing strided sampling of the frames.
        stride, rem = divmod(monet_latents.shape[0], NUM_FRAMES)
        monet_latents = monet_latents[None, :-rem:stride]
        assert monet_latents.shape[1] == NUM_FRAMES
        question = encode_sentence(token_map, question_json["question"],
                                                              modellib.MAX_QUESTION_LENGTH)
        batched_question = np.expand_dims(question, axis=0)
        logits = sess.run(output_descriptive, feed_dict={
                inputs_descriptive["monet_latents"]: monet_latents,
                inputs_descriptive["question"]: batched_question,
        })
        descriptive_answer = np.argmax(logits)
        return reverse_answer_lookup[descriptive_answer]

    def eval_mc(monet_latents, question_json):
        stride, rem = divmod(monet_latents.shape[0], NUM_FRAMES)
        monet_latents = monet_latents[None, :-rem:stride]
        assert monet_latents.shape[1] == NUM_FRAMES
        question = encode_sentence(
                token_map, question_json["question"], modellib.MAX_QUESTION_LENGTH)
        choices = encode_choices(
                token_map, question_json["choices"])
        mc_answer = sess.run(output_mc, feed_dict={
                inputs_mc["monet_latents"]: monet_latents,
                inputs_mc["question"]: np.expand_dims(question, axis=0),
                inputs_mc["choices"]: np.expand_dims(choices, axis=0),
        })
        return mc_answer >= 0

    sample_scene_idx = _SCENE_IDX.value

    NUM_SCENES = 5000

    pred_map = {True: 'correct', False: 'wrong'}

    total, correct = 0, 0
    total_per_q, correct_per_q = 0, 0
    total_expl, correct_expl = 0, 0
    total_expl_per_q, correct_expl_per_q = 0, 0
    total_pred, correct_pred = 0, 0
    total_pred_per_q, correct_pred_per_q = 0, 0
    total_coun, correct_coun = 0, 0
    total_coun_per_q, correct_coun_per_q = 0, 0
    total_desc, correct_desc = 0, 0

    progress_bar = tqdm(total=NUM_SCENES, dynamic_ncols=True, leave=False, position=0, desc='test')

    for sample_scene_idx in range(NUM_SCENES):
        scene_json = questions_data[sample_scene_idx]
        num_questions = scene_json["questions"][-1]["question_id"]

        for question_id in range(num_questions):
            question_json = scene_json["questions"][question_id]

            if question_json["question_type"] == "descriptive":
                ans = eval_descriptive(load_monet_latents(base_dir, sample_scene_idx + 10000), question_json)
                if ans == question_json["answer"]:
                    correct_desc += 1
                    correct += 1
                    correct_per_q += 1
                total_desc += 1
                total += 1
                total_per_q += 1
            else:
                correct_question = True
                anss = eval_mc(load_monet_latents(base_dir, sample_scene_idx + 10000), question_json)[0]
                for choice_id, choice_json in enumerate(question_json["choices"]):
                    ans = anss[choice_id]
                    correct_choice = False
                    if pred_map[ans] == choice_json["answer"]:
                        correct += 1
                        correct_choice = True
                    else:
                        correct_question = False
                    total += 1
                    if question_json['question_type'].startswith('explanatory'):
                        if correct_choice:
                            correct_expl += 1
                        total_expl += 1

                    if question_json['question_type'].startswith('predictive'):
                        if correct_choice:
                            correct_pred += 1
                        total_pred += 1

                    if question_json['question_type'].startswith('counterfactual'):
                        if correct_choice:
                            correct_coun += 1
                        total_coun += 1

                if correct_question:
                    correct_per_q += 1
                total_per_q += 1

                if question_json['question_type'].startswith('explanatory'):
                    if correct_question:
                        correct_expl_per_q += 1
                    total_expl_per_q += 1

                if question_json['question_type'].startswith('predictive'):
                    if correct_question:
                        correct_pred_per_q += 1
                    total_pred_per_q += 1

                if question_json['question_type'].startswith('counterfactual'):
                    if correct_question:
                        correct_coun_per_q += 1
                    total_coun_per_q += 1
                      
        progress_bar.update()
    
    print('============ results ============')
    print('overall accuracy per option: %f %%' % (float(correct) * 100.0 / total))
    print('overall accuracy per question: %f %%' % (float(correct_per_q) * 100.0 / total_per_q))
    print('descriptive accuracy per question: %f %%' % (float(correct_desc) * 100.0 / total_desc))
    print('explanatory accuracy per option: %f %%' % (float(correct_expl) * 100.0 / total_expl))
    print('explanatory accuracy per question: %f %%' % (float(correct_expl_per_q) * 100.0 / total_expl_per_q))
    print('predictive accuracy per option: %f %%' % (float(correct_pred) * 100.0 / total_pred))
    print('predictive accuracy per question: %f %%' % (float(correct_pred_per_q) * 100.0 / total_pred_per_q))
    print('counterfactual accuracy per option: %f %%' % (float(correct_coun) * 100.0 / total_coun))
    print('counterfactual accuracy per question: %f %%' % (float(correct_coun_per_q) * 100.0 / total_coun_per_q))
    print('============ results ============')
    print(total, total_per_q, total_desc, total_expl, total_expl_per_q, total_pred, total_pred_per_q, total_coun, total_coun_per_q)


if __name__ == "__main__":
    app.run(main)
