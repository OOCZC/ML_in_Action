# tensorflow 1.4 version
import tensorflow as tf

TrainPath = '../logistic-regression/horseColic/horseColicTraining.txt'
TestPath = '../logistic-regression/horseColic/horseColicTest.txt'
batch_size = 16
hidden_layer = 4  
epochs = 100

w1 = tf.Variable(tf.random_normal([21, hidden_layer], stddev = 1, seed = 1)) # stddev 是标准差
w2 = tf.Variable(tf.random_normal([hidden_layer, 1], stddev = 1, seed = 1))

X = tf.placeholder(tf.float32, shape=(None, 21), name = 'x-input') 
Y = tf.placeholder(tf.float32, shape=(None, 1), name = 'y-input')

h = tf.matmul(X, w1)
h = tf.tanh(h)
h = tf.matmul(h, w2)
y_hat = tf.sigmoid(h)

# 损失函数
cross_entropy = -tf.reduce_mean(
    Y * tf.log(tf.clip_by_value(y_hat, 1e-10, 1.0)) +
    (1 - Y) * tf.log(tf.clip_by_value((1 - y_hat), 1e-10, 1.0))) # reduce_mean 是求平均值
# L2 正则化
# tf.contrib.layers.l2_regularizer(lambda)(w)
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# read file
with open(TrainPath, 'r') as trainFile, open(TestPath, 'r') as testFile:
    trainFile_list = trainFile.readlines()
    testFile_list = testFile.readlines()

X_train = [i.split() for i in trainFile_list]
X_test = [i.split() for i in testFile_list]
Y_train = []; Y_test = []
for i in X_train:
    Y_train.append([i[21]])
for i in X_test:
    Y_test.append([i[21]])
'''
Y_train = [i[20] for i in X_train]
Y_test = [i[21] for i in X_test]
Y_train = Y_train.reshape(-1,1)
Y_test = Y_test.reshape(-1,1)
'''
X_train = [i[:21] for i in X_train]
X_test = [i[:21] for i in X_test]
print('shape X_train : ', len(X_train), len(X_train[0]))
print('X_train like ', X_train[0])
print('shape Y_train : ', len(Y_train), len(Y_train[0]))
print('Y_train like ', Y_train[0])
#print('Y_train is ', Y_train)

'''
# forecast
y_hat_ = tf.greater(y_hat, 0.5).eval()
correct_prediction = tf.equal(y_hat_, Y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
'''

with tf.Session() as sess:
    init_var = tf.global_variables_initializer() # 初始化变量
    sess.run(init_var)

    for i in range(epochs):
        start = 0
        while start < len(X_train):
            end = start + batch_size
            sess.run(train_step,
                feed_dict={X: X_train[start: end], Y: Y_train[start: end]})
            start = start + batch_size


        total_cross_entropy = sess.run(cross_entropy, 
            feed_dict={X: X_train, Y: Y_train})
        print("after ", i + 1, " epochs, cross_entropy is ", total_cross_entropy)

    # forecast
    test_acc = sess.run(y_hat, feed_dict={X: X_test, Y: Y_test})
    # print("testset y_hat ", test_acc)
    test_acc = sum(test_acc >= 0.5) / len(test_acc)
    
    print("acc is ", test_acc)


    #print("w1 : ", sess.run(w1))
    #print("w2 : ", sess.run(w2))

