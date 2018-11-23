import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from datetime import datetime as dt
import os
import tensorflow as tf


class SuccessiveTransitionSimulator:

    def __init__(self, init_minima_E, init_delta_E, Q, NUM_ITER=20000, NUM_PARTICLE = 100, temperature=6):
        self.NUM_PARTICLE = NUM_PARTICLE  # 粒子の数
        self.NUM_ITER = NUM_ITER  # イテレーション回数
        self.minima_E = init_minima_E  # 各エネルギー極小値
        self.delta_E = init_delta_E  # 各極小値のを隔てるエネルギー壁の高さ
        self.maxima_E = self.minima_E + self.delta_E  # 各エネルギー極大値
        self.Q = Q  # 各エネルギー極小値状態に対応する物理量
        self.particles_pos = np.zeros(self.NUM_PARTICLE)  # 各粒子の位置。0(一番左の極小値)で初期化。
        self.temperature = temperature  # 温度
        self.rates_history = []  # 各状態にある粒子の割合(rates[i]=極小値iにいる粒子数/全粒子数)の時系列
        self.q_history = []
        self.P_DIR = dt.now().strftime('%Y_%m_%d_%H_%M_%S')

    def set_temperature(self, temperature):
        self.temperature = temperature
        
    def get_temperature(self):
        return self.temperature
        
    def simulate(self, save_fig=True):
        for _ in range(self.NUM_ITER):

            for i, particle_pos in enumerate(self.particles_pos):
                left_p = 0  # 左に移る確率
                right_p = 0  # 右に移る確率

                rand = np.random.rand()

                # 遷移方向(右/左)をランダムに決める
                if np.random.rand() < 0.5:
                    # 一番左にいる場合は何もしない
                    if particle_pos > 0:
                        # 左に移る確率を計算
                        left_p = np.exp(-max(0, self.maxima_E[int(particle_pos-1)] - self.minima_E[int(particle_pos)])/self.temperature) / self.temperature
                        # left_pの確率で左に移る
                        if left_p > rand:
                            self.particles_pos[i] -= 1
                else:
                    # 一番右にいる場合は何もしない
                    if particle_pos < len(self.minima_E)-1:
                        # 右に移る確率を計算
                        right_p = np.exp(-max(0, self.maxima_E[int(particle_pos)] - self.minima_E[int(particle_pos)])/self.temperature) / self.temperature
                        # right_pの確率で右に移る
                        if right_p > rand:
                            self.particles_pos[i] += 1

            # 各準安定状態にある粒子の割合ratesと全体として観測される物理量qを計算
            q = 0
            rates = []
            for j in range(len(self.minima_E)):
                rate = len(*np.where(self.particles_pos == j)) / self.NUM_PARTICLE
                q += self.Q[j] * rate
                rates.append(rate)
            self.rates_history.append(rates)
            self.q_history.append(q)

        if save_fig:
            self._save_fig()


    def _save_fig(self):
        out_dir = 'Transition/{}/{}'.format(self.P_DIR, dt.now().strftime('%Y_%m_%d_%H_%M_%S'))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # 極小値、極大値、補間した曲線をプロット
        plt.subplots(1, 1)
        points = []
        for minima, maxima in zip(self.minima_E, self.maxima_E):
            points.append(minima)
            points.append(maxima)
        interpolate = interp1d([i for i in range(len(points))], points, kind='quadratic')
        xlist = np.linspace(0, len(points)-1, num=101)
        plt.ylabel("Energy")
        plt.plot(xlist, interpolate(xlist))
        plt.scatter([2*x for x in range(len(self.minima_E))], self.minima_E, color='red')
        plt.scatter([2*x+1 for x in range(len(self.maxima_E))], self.maxima_E, color='green')
        plt.savefig(out_dir + '/energy_curve.png')
        
        # 観測される物理量の時系列変化をプロット 
        plt.subplots(1, 1)
        plt.ylim(-0.1, 10.1)
        plt.xlabel("Iteration")
        plt.ylabel("Physical Quantity")
        plt.plot([x for x in range(len(self.q_history))], self.q_history)
        plt.savefig(out_dir + '/q-t_curve.png')

        # 各状態の粒子の割合の時系列変化をプロット
        self.rates_history = np.array(self.rates_history)
        plt.subplots(1, 1)
        plt.ylim(-0.1, 1.1)
        plt.xlabel("Iteration")
        plt.ylabel("Particle Rate")
        for k in range(len(self.Q)):
            plt.plot(self.rates_history[:, k], label='State {}'.format(k))
        plt.legend()
        plt.savefig(out_dir + '/rates-t_curve.png')

        # 各状態の粒子の割合最終状態を棒グラフでプロット
        plt.subplots(1, 1)
        plt.ylim(-0.1, 1.1)
        plt.xlabel("State")
        plt.ylabel("Particle Rate")
        plt.bar([i for i in range(len(self.rates_history[-1, :]))], self.rates_history[-1, :])
        plt.savefig(out_dir + '/final_rates.png')


    # エネルギー壁をランダムに初期化、粒子の位置を0で初期化
    def random_init(self):
        # 極小値、極大値をランダム値で初期化する。※両端の極小値は固定
        self.delta_E = np.random.randint(2, 40, len(self.delta_E))
        for i in range(len(self.delta_E)-2):
            self.maxima_E[i] = self.minima_E[i] + self.delta_E[i]
            self.minima_E[i+1] = np.random.randint(self.minima_E[-1] + 20, self.maxima_E[i], 1)
        self.maxima_E[len(self.delta_E)-2] = self.minima_E[len(self.delta_E)-2] + self.delta_E[len(self.delta_E)-2]

        # 粒子の位置を初期化
        self.particles_pos = np.zeros(self.NUM_PARTICLE)
        self.particles_pos[:] = 0
        self.q_history = []
        self.rates_history = []


    # エネルギー極値を左から並べたリストを取得
    def get_extremes(self):
        extrems = []
        for minima, maxima in zip(self.minima_E, self.maxima_E):
            extrems.append(minima)
            extrems.append(maxima)
        return np.array(extrems)

    #
    def get_rates_history(self):
        return np.array(self.rates_history)
    
    def get_particles_pos(self):
        return self.particles_pos


class CNNMultiRegressionModel(object):
    """
    CNN多対多回帰モデル
    """
    def __init__(
      self, num_classes, row_size, col_size, filter_sizes, num_filters):

        self.input_x = tf.placeholder(tf.float32, [None, row_size, col_size], name="input_x")
        self.input_x_ = tf.reshape(self.input_x, [-1, row_size, col_size, 1])
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):

                filter_shape = [filter_size, col_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.input_x_,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, row_size - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        num_filters_total = num_filters * len(filter_sizes)
        #self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b_out")
            self.scores_ = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.scores = tf.nn.softmax(self.scores_)

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.square(self.scores - self.input_y))

        self.prepare_session()

    def prepare_session(self):
        sess = tf.InteractiveSession()
        #sess.run(tf.initialize_all_variables())
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        self.sess = sess
        self.saver = saver

class NN1DRegressionModel:
    """
    1次元全結合ニューラルネットワーク
    """
    def __init__(self, input_size, output_size):
        # input layer (input_size)
        self.input_x = tf.placeholder(tf.float32, [None, input_size])
        #self.input_x_ = tf.reshape(self.input_x, [-1, input_size, 1])

        # fully connected layer (32)
        W_fc1 = tf.Variable(tf.truncated_normal([input_size, 64], stddev=0.01))
        b_fc1 = tf.Variable(tf.zeros([64]))
        h_fc1 = tf.nn.relu(tf.matmul(self.input_x, W_fc1) + b_fc1)

        # output layer (n_actions)
        W_out = tf.Variable(tf.truncated_normal([64, output_size], stddev=0.01))
        b_out = tf.Variable(tf.zeros([output_size]))
        self.y = tf.matmul(h_fc1, W_out) + b_out
        self.scores = tf.nn.softmax(self.y)

        # loss function
        self.input_y = tf.placeholder(tf.float32, [None, output_size])
        self.loss = tf.reduce_mean(tf.square(self.input_y - self.scores))
        

    
if __name__ == '__main__':
    sts = SuccessiveTransitionSimulator(
        init_minima_E=np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10 , 0]), # エネルギー極小値初期状態
        init_delta_E=np.array([20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]),
        Q=np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]),
        NUM_ITER=100000
    )

    DATA_LEN = len(sts.get_extremes())

    extremes_series = []
    rates_series = []
    
    temperature = sts.get_temperature()

    for _ in range(200):
        print(_)
        sts.simulate(save_fig=True)
        extremes = sts.get_extremes()
        extremes_series.append(extremes/temperature)
        rates_history = sts.get_rates_history()
        rates_series.append(np.mean(rates_history[-100:, :], axis=0))
        print(sts.get_particles_pos())
        sts.random_init()

    #print(extremes_series)
    #print(rates_series)
    
    train_output = np.array(rates_series)
    train_input = []
    for extremes in extremes_series:
        train_input.append([[extreme] for extreme in extremes])
    train_input = np.array(train_input).reshape(-1, DATA_LEN)
    print(train_output)
    print(train_input)
    
    filter_sizes = [2, 3, 4]#, 4, 8]
    num_filters = 100
    
    COL_DIM = 1 # 1次元のデータを入力するため列数は1
    INPUT_DATA_LEN = len(train_input[0]) # データ長
    print(INPUT_DATA_LEN)
    INPUT_DATA_NUM = len(train_input) # データ数
    print(INPUT_DATA_NUM)
    DIM_OUTPUT = len(train_output[0]) # 出力層次元
    print(DIM_OUTPUT)

    #sess = tf.InteractiveSession()
    #sess = tf.Session()

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    #saver = tf.train.Saver()

    #with sess.as_default():

    #model = CNNMultiRegressionModel(num_classes = DIM_OUTPUT, row_size = INPUT_DATA_LEN, col_size = COL_DIM, filter_sizes = filter_sizes, num_filters = num_filters)
    model = NN1DRegressionModel(input_size=INPUT_DATA_LEN, output_size=DIM_OUTPUT)
    
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-3)
    grads_and_vars = optimizer.compute_gradients(model.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    sess.run(tf.global_variables_initializer())
    #saver = tf.train.Saver()

    dropout_keep_prob = 0.5
    TRAIN_STEPS = 30000

    loss_history = []
    # 学習開始
    print("Started Training..")
    local_step = 0
    while True:
        local_step += 1

        x_batch = train_input
        y_batch = train_output
        feed_dict = {
          model.input_x: x_batch,
          model.input_y: y_batch,
          #model.dropout_keep_prob: dropout_keep_prob
        }
        _, step, loss, scores = sess.run(
            [train_op, global_step, model.loss, model.scores],
            feed_dict)


        if step % 3000 == 0:
            print('Step:{}, loss:{}'.format(step, loss))
            print('Step:{}, scores:{}'.format(step, scores))
            loss_history.append(loss)

        """
        if (step + 1) % 1000 == 0 or step == TRAIN_STEPS:
            # モデルの保存
            saver.save(sess, "TimeseriesCNN_{}".format(step+1), global_step=step)
        """

        if step >= TRAIN_STEPS:
            break

    plt.subplots(1, 1)
    plt.plot(loss_history)
    for score, out in zip(scores, train_output):
        plt.subplots(1, 1)
        plt.bar([i for i in range(len(score))], score, width=0.3, align="center")
        plt.bar([i+0.3 for i in range(len(score))], out, width=0.3, align="center")
        
        
    # 予測用シミュレーション
    for _ in range(30):
        sts.simulate(save_fig=False)
        extremes = sts.get_extremes() / temperature
        
        rates_history = sts.get_rates_history()
        final_rates = np.mean(rates_history[-100:, :], axis=0)
        
        predicted_scores = sess.run([model.scores], {model.input_x: np.array([[extreme] for extreme in extremes]).reshape(1, DATA_LEN)})
        #predicted_scores = sess.run([model.scores], {model.input_x: np.array([[extreme] for extreme in extremes]).reshape(1, DATA_LEN, 1), model.dropout_keep_prob: dropout_keep_prob})
        print('predicted_scores : ', predicted_scores)
        
        plt.subplots(1, 1)
        plt.bar([i for i in range(len(predicted_scores[0][0]))], predicted_scores[0][0], width=0.3, align="center", color='red', label='Predicted Rate')
        plt.bar([i+0.3 for i in range(len(final_rates))], final_rates, width=0.3, align="center", color='blue', label='Actual Rate')
        plt.xlabel("State")
        plt.ylabel("Particle Rate")
        plt.legend()
        plt.savefig('Transition/predicted_rate{}.png'.format(_))
        sts.random_init()