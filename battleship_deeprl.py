import gym
import gym_pomdp
import numpy as np
import tensorflow as tf
import argparse

BOARD_SIZE = 5
# ship_sizes = [4,3,2,2,1]
ship_sizes = [3]
INIT = False


def play_game(env, sess, probabilities,input_positions,training=False, render = True):
    done = False
    board_position_log = []
    action_log = []
    hit_log = []
    current_board = [[-1 for i in range(BOARD_SIZE*BOARD_SIZE)]]

    while not done:
        board_position_log.append([[i for i in current_board[0]]])
        probs = sess.run([probabilities], feed_dict={input_positions:current_board})[0][0]
        probs = [p * (index not in action_log) for index, p in enumerate(probs)]
        print(probs)
        probs = [p / sum(probs) for p in probs]
        if training:
            bomb_index = np.random.choice(BOARD_SIZE*BOARD_SIZE, p=probs)
        else:
            bomb_index = np.argmax(probs)
        ob, rw, done, state = env.step(bomb_index)
        if render:
            env.render()
        hit_log.append(ob)
        action_log.append(bomb_index)
        current_board[0][bomb_index] = ob
    return board_position_log, action_log, hit_log


def rewards_calculator(hit_log, gamma=0.5):
    """ Discounted sum of future hits over trajectory"""

    hit_log_weighted = [(item -
                         float(sum(ship_sizes) - sum(hit_log[:index])) / float(BOARD_SIZE*BOARD_SIZE - index)) * (
                                gamma ** index) for index, item in enumerate(hit_log)]
    return [((gamma) ** (-i)) * sum(hit_log_weighted[i:]) for i in range(len(hit_log))]


def define_model():
    hidden_units = BOARD_SIZE*BOARD_SIZE
    output_units = BOARD_SIZE*BOARD_SIZE
    input_positions = tf.placeholder(tf.float32, shape=(1, BOARD_SIZE*BOARD_SIZE))
    labels = tf.placeholder(tf.int64)
    learning_rate = tf.placeholder(tf.float32, shape=[])
    # Generate hidden layer
    W1 = tf.Variable(tf.truncated_normal([BOARD_SIZE*BOARD_SIZE, hidden_units],
                                         stddev=0.1 / np.sqrt(float(BOARD_SIZE*BOARD_SIZE))))
    b1 = tf.Variable(tf.zeros([1, hidden_units]))
    h1 = tf.tanh(tf.matmul(input_positions, W1) + b1)
    # Second layer -- linear classifier for action logits
    W2 = tf.Variable(tf.truncated_normal([hidden_units, output_units],
                                         stddev=0.1 / np.sqrt(float(hidden_units))))
    b2 = tf.Variable(tf.zeros([1, output_units]))
    logits = tf.matmul(h1, W2) + b2
    probabilities = tf.nn.softmax(logits)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='xentropy')
    train_step = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate).minimize(cross_entropy)

    return probabilities,input_positions,labels,learning_rate, train_step


def train(probabilities, input_positions, labels, learning_rate, train_step,  model_file, INIT=False):
    saver = tf.train.Saver()

    with tf.Session() as sess:

        if INIT:
            init = tf.initialize_all_variables()
            sess.run(init)
        else:
            saver.restore(sess, model_file)
        game_lengths = []

        ALPHA = 0.03
        for game in range(10000):
            env = gym.make("Battleship-v0", board_size=(BOARD_SIZE,BOARD_SIZE), ship_sizes=ship_sizes)
            env.reset()
            board_position_log, action_log, hit_log = play_game(env,sess,probabilities,input_positions, True, True)
            game_lengths.append(len(action_log))
            rewards_log = rewards_calculator(hit_log)
            for reward, current_board, action in zip(rewards_log, board_position_log, action_log):
                # Take step along gradient
                    sess.run([train_step],feed_dict={input_positions:current_board, labels:[action], learning_rate:ALPHA * reward})
        print(game_lengths)
        saver.save(sess,model_file)


def play(probabilities, input_positions, model_file):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_file)

        env = gym.make("Battleship-v0", board_size=(BOARD_SIZE,BOARD_SIZE), ship_sizes=ship_sizes)
        env.reset()
        play_game(env,sess,probabilities,input_positions)
        input("press enter to finish")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="either train or play",
                        type=str)
    parser.add_argument("--model_file", help="model file to use to train or play",
                        type=str, default="tf_deep_rl_model_5_2.ckpt")

    parser.add_argument("--init", help="initialize a new model",
                        action="store_true")

    args = parser.parse_args()
    model_file = args.model_file
    mode = args.mode
    init = args.init
    probabilities,input_positions,labels,learning_rate, train_step = define_model()
    if mode == "train":
        train(probabilities,input_positions,labels,learning_rate, train_step, model_file, init)
    else:
        play(probabilities, input_positions, model_file)



