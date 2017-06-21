def homework(train_X, train_y, test_X, test_y):
    #正解ラベルのエンコーディング
    def encode(y):
        onehot_y = np.zeros((len(y), 10))
        onehot_y[np.arange(len(y)), y] = 1
        return onehot_y
    int_y = test_y
    train_y, test_y = encode(train_y), encode(test_y)

    from sklearn.utils import shuffle
    tf.reset_default_graph() # グラフのリセット

    # Step1. プレースホルダーと変数の定義
    ## Placeholders
    with tf.name_scope("placeholders"):
        x = tf.placeholder(tf.float32, [None, 784])
        t = tf.placeholder(tf.float32, [None, 10])

    ## 変数
    with tf.name_scope("layers"):
        W1 = tf.Variable(np.random.uniform(low=-0.1, high=0.1, size=(784, 200)).astype('float32'), name='W1')
        b1 = tf.Variable(np.zeros(200).astype('float32'), name='b1')
        W2 = tf.Variable(np.random.uniform(low=-0.1, high=0.1, size=(200, 100)).astype('float32'), name='W2')
        b2 = tf.Variable(np.zeros(100).astype('float32'), name='b2')
        W3 = tf.Variable(np.random.uniform(low=-0.1, high=0.1, size=(100, 10)).astype('float32'), name='W3')
        b3 = tf.Variable(np.zeros(10).astype('float32'), name='b3')
    params = [W1, b1, W2, b2, W3, b3]

    # Step2. グラフの定義
    with tf.name_scope("graph"):
        u1 = tf.matmul(x, W1) + b1
        z1 = tf.nn.relu(u1)
        u2 = tf.matmul(z1, W2) + b2
        z2 = tf.nn.relu(u2)
        u3 = tf.matmul(z2, W3) + b3
        y = tf.nn.softmax(u3)

    # Step3. 誤差関数の定義
    # cost = -tf.reduce_mean(tf.reduce_sum(t*tf.log(y)))
    with tf.name_scope("optimize"):
        cost = -tf.reduce_mean(tf.reduce_sum(t*tf.log(tf.clip_by_value(y, 1e-10, 1.0)))) # tf.log(0)によるnanを防ぐ

    # Step4. 更新則の設定
        gW1, gb1, gW2, gb2, gW3, gb3 = tf.gradients(cost, params)
        updates = [
            W1.assign(W1 - 0.01 * gW1),
            b1.assign(b1 - 0.01 * gb1),
            W2.assign(W2 - 0.01 * gW2),
            b2.assign(b2 - 0.01 * gb2),
            W3.assign(W3 - 0.01 * gW3),
            b3.assign(b3 - 0.01 * gb3),
        ]

    #Dropoutの設定
    #keep_prob = tf.placeholder(tf.float32)
    #z1_drop = tf.nn.dropout(z1, keep_prob)

    train = tf.group(*updates)

    valid = tf.argmax(y, 1)

    n_epochs = 50
    batch_size = 2000
    n_batches = train_X.shape[0] // batch_size

    # Step5. 学習
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(n_epochs):
            train_X, train_y = shuffle(train_X, train_y)
            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size
                #sess.run(train, feed_dict={x: train_X[start:end], t: train_y[start:end], keep_prob:0.5})
                sess.run(train, feed_dict={x: train_X[start:end], t: train_y[start:end]})
            pred_y, valid_cost = sess.run([valid, cost], feed_dict={x: test_X, t: test_y})
            #print(np.equal(pred_y, int_y))
            print('EPOCH:: %i, Validation cost: %.3f, Validation F1: %.3f' % (epoch + 1, valid_cost, f1_score(np.argmax(test_y, 1).astype('int32'), pred_y, average='macro')))

        #テンソルボードの表示
        #tb.show_graph(sess.graph)
        return sess.run(valid, feed_dict={x: test_X})
