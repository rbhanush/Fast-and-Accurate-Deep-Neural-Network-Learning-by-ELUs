"""
MIT License:
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

#Took help from TensorFlow Tutorial Website
#https://www.tensorflow.org/get_started/mnist/beginners

import tensorflow as tf
import math
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

lrate = 0.001

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def plotAccuracy(final_out):    
    fig = plt.figure()
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    axes.plot(final_out[0][0], final_out[0][1], 'r', label='Sigmoid')
    axes.plot(final_out[1][0], final_out[1][1], 'g', label='ReLU')
    axes.plot(final_out[2][0], final_out[2][1], 'b', label='ELU')
    axes.legend()
    axes.set_xlabel('Training Epoch')
    axes.set_ylabel('Averaged Training Accuracy over 5 iterations')
    plt.show() 

def plotLoss(final_out): 
    fig = plt.figure()
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    axes.plot(final_out[0][0], final_out[0][1], 'r', label='Sigmoid')
    axes.plot(final_out[1][0], final_out[1][1], 'g', label='ReLU')
    axes.plot(final_out[2][0], final_out[2][1], 'b', label='ELU')
    axes.legend()
    axes.set_xlabel('Training Epoch')
    axes.set_ylabel('Averaged Training Loss over 5 iterations')
    plt.show()
    
def plotAccuracy1(final_out):    
    fig = plt.figure()
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    axes.plot(final_out[0][0], final_out[0][1], 'r', label='Sigmoid')
    axes.plot(final_out[1][0], final_out[1][1], 'g', label='ReLU')
    axes.plot(final_out[2][0], final_out[2][1], 'b', label='ELU')
    axes.legend()
    axes.set_xlabel('Training Epoch')
    axes.set_ylabel('Training Accuracy')
    plt.show() 

def plotLoss1(final_out): 
    fig = plt.figure()
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    axes.plot(final_out[0][0], final_out[0][2], 'r', label='Sigmoid')
    axes.plot(final_out[1][0], final_out[1][2], 'g', label='ReLU')
    axes.plot(final_out[2][0], final_out[2][2], 'b', label='ELU')
    axes.legend()
    axes.set_xlabel('Training Epoch')
    axes.set_ylabel('Training Loss')
    plt.show()

def plotTestAccuracy(final_out): 
    fig = plt.figure()
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    axes.plot(final_out[0][0], final_out[0][1], 'r', label='Sigmoid')
    axes.plot(final_out[1][0], final_out[1][1], 'g', label='ReLU')
    axes.plot(final_out[2][0], final_out[2][1], 'b', label='ELU')
    axes.legend()
    axes.set_xlabel('Different Training Epochs and Different Hidden Neurons over 5 iterations')
    axes.set_ylabel('Testing Accuracy')
    plt.show()
    
def sigmoidnn(n, num):
    X = tf.placeholder(tf.float32, [None, 784])
    Y = tf.placeholder(tf.float32, [None, 10])
    
    #Number of neurons per hidden layer
    h1 = n
    h2 = n
    h3 = n
    h4 = n
    h5 = n
    h6 = n
    
    #Weight and Bias initialization
    w1 = tf.Variable(tf.truncated_normal([784, h1], stddev=1.0 / math.sqrt(float(h2))))
    b1 = tf.Variable(tf.zeros([h1]))
    w2 = tf.Variable(tf.truncated_normal([h1, h2], stddev=1.0 / math.sqrt(float(h2))))
    b2 = tf.Variable(tf.zeros([h2]))
    w3 = tf.Variable(tf.truncated_normal([h2, h3], stddev=1.0 / math.sqrt(float(h2))))
    b3 = tf.Variable(tf.zeros([h3]))
    w4 = tf.Variable(tf.truncated_normal([h3, h4], stddev=1.0 / math.sqrt(float(h2))))
    b4 = tf.Variable(tf.zeros([h4]))
    w5 = tf.Variable(tf.truncated_normal([h4, h5], stddev=1.0 / math.sqrt(float(h2))))
    b5 = tf.Variable(tf.zeros([h5]))
    w6 = tf.Variable(tf.truncated_normal([h5, h6], stddev=1.0 / math.sqrt(float(h2))))
    b6 = tf.Variable(tf.zeros([h6]))
    w7 = tf.Variable(tf.truncated_normal([h6, 10], stddev=1.0 / math.sqrt(float(h2))))
    b7 = tf.Variable(tf.zeros([10]))
    
    #Creating our Model 
    y1 = tf.nn.sigmoid(tf.matmul(X, w1) + b1)
    y2 = tf.nn.sigmoid(tf.matmul(y1, w2) + b2)
    y3 = tf.nn.sigmoid(tf.matmul(y2, w3) + b3)
    y4 = tf.nn.sigmoid(tf.matmul(y3, w4) + b4)
    y5 = tf.nn.sigmoid(tf.matmul(y4, w5) + b5)
    y6 = tf.nn.sigmoid(tf.matmul(y5, w6) + b6)
    out_layer = tf.matmul(y6, w7) + b7
    result = out_layer
    
    loss = tf.nn.softmax_cross_entropy_with_logits(labels = Y,logits = result)
    loss = tf.reduce_mean(loss)
    
    training_step = tf.train.AdamOptimizer(learning_rate = lrate).minimize(loss)
    
    prediction = tf.equal(tf.argmax(result, 1), tf.argmax(Y, 1))
    prediction = tf.cast(prediction, tf.float32)
    accuracy = tf.reduce_mean(prediction)
    
    output = []
    iterations = []
    train_acc = []
    train_loss = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(num):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={X: batch[0], Y: batch[1]})
                #print('Step %d, Training Accuracy: %g' % (i, train_accuracy))
                iterations.append(i)
                train_acc.append(train_accuracy)
            #training_step.run(feed_dict={X: batch[0], Y: batch[1]}
            _, c = sess.run([training_step, loss], feed_dict={X: batch[0], Y: batch[1]})  
            if i% 100 == 0:
                train_loss.append(c)
            
        test_accuracy = accuracy.eval(feed_dict={X: mnist.test.images, Y: mnist.test.labels})
        #print('Test Accuracy: %g' % (test_accuracy))
        output.append(iterations)
        output.append(train_acc)
        output.append(train_loss)
        output.append(test_accuracy)
        return output
      
def relunn(n, num):    
    X = tf.placeholder(tf.float32, [None, 784])
    Y = tf.placeholder(tf.float32, [None, 10])
    
    #Number of neurons per hidden layer
    h1 = n
    h2 = n
    h3 = n
    h4 = n
    h5 = n
    h6 = n
    
    #Weight and Bias initialization
    w1 = tf.Variable(tf.truncated_normal([784, h1], stddev=1.0 / math.sqrt(float(h2))))
    b1 = tf.Variable(tf.zeros([h1]))
    w2 = tf.Variable(tf.truncated_normal([h1, h2], stddev=1.0 / math.sqrt(float(h2))))
    b2 = tf.Variable(tf.zeros([h2]))
    w3 = tf.Variable(tf.truncated_normal([h2, h3], stddev=1.0 / math.sqrt(float(h2))))
    b3 = tf.Variable(tf.zeros([h3]))
    w4 = tf.Variable(tf.truncated_normal([h3, h4], stddev=1.0 / math.sqrt(float(h2))))
    b4 = tf.Variable(tf.zeros([h4]))
    w5 = tf.Variable(tf.truncated_normal([h4, h5], stddev=1.0 / math.sqrt(float(h2))))
    b5 = tf.Variable(tf.zeros([h5]))
    w6 = tf.Variable(tf.truncated_normal([h5, h6], stddev=1.0 / math.sqrt(float(h2))))
    b6 = tf.Variable(tf.zeros([h6]))
    w7 = tf.Variable(tf.truncated_normal([h6, 10], stddev=1.0 / math.sqrt(float(h2))))
    b7 = tf.Variable(tf.zeros([10]))
    
    #Creating our Model 
    y1 = tf.nn.relu(tf.matmul(X, w1) + b1)
    y2 = tf.nn.relu(tf.matmul(y1, w2) + b2)
    y3 = tf.nn.relu(tf.matmul(y2, w3) + b3)
    y4 = tf.nn.relu(tf.matmul(y3, w4) + b4)
    y5 = tf.nn.relu(tf.matmul(y4, w5) + b5)
    y6 = tf.nn.relu(tf.matmul(y5, w6) + b6)
    out_layer = tf.matmul(y6, w7) + b7
    result = out_layer
    
    loss = tf.nn.softmax_cross_entropy_with_logits(labels = Y,logits = result)
    loss = tf.reduce_mean(loss)
    
    training_step = tf.train.AdamOptimizer(learning_rate = lrate).minimize(loss)
    
    prediction = tf.equal(tf.argmax(result, 1), tf.argmax(Y, 1))
    prediction = tf.cast(prediction, tf.float32)
    accuracy = tf.reduce_mean(prediction)
    
    output = []
    iterations = []
    train_acc = []
    train_loss = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(num):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={X: batch[0], Y: batch[1]})
                #print('Step %d, Training Accuracy: %g' % (i, train_accuracy))
                iterations.append(i)
                train_acc.append(train_accuracy)
            #training_step.run(feed_dict={X: batch[0], Y: batch[1]}
            _, c = sess.run([training_step, loss], feed_dict={X: batch[0], Y: batch[1]})  
            if i% 100 == 0:
                train_loss.append(c)
            
        test_accuracy = accuracy.eval(feed_dict={X: mnist.test.images, Y: mnist.test.labels})
        #print('Test Accuracy: %g' % (test_accuracy))
        output.append(iterations)
        output.append(train_acc)
        output.append(train_loss)
        output.append(test_accuracy)
        return output

def elunn(n, num):    
    X = tf.placeholder(tf.float32, [None, 784])
    Y = tf.placeholder(tf.float32, [None, 10])
    
    #Number of neurons per hidden layer
    h1 = n
    h2 = n
    h3 = n
    h4 = n
    h5 = n
    h6 = n
    
    #Weight and Bias initialization
    w1 = tf.Variable(tf.truncated_normal([784, h1], stddev=1.0 / math.sqrt(float(h2))))
    b1 = tf.Variable(tf.zeros([h1]))
    w2 = tf.Variable(tf.truncated_normal([h1, h2], stddev=1.0 / math.sqrt(float(h2))))
    b2 = tf.Variable(tf.zeros([h2]))
    w3 = tf.Variable(tf.truncated_normal([h2, h3], stddev=1.0 / math.sqrt(float(h2))))
    b3 = tf.Variable(tf.zeros([h3]))
    w4 = tf.Variable(tf.truncated_normal([h3, h4], stddev=1.0 / math.sqrt(float(h2))))
    b4 = tf.Variable(tf.zeros([h4]))
    w5 = tf.Variable(tf.truncated_normal([h4, h5], stddev=1.0 / math.sqrt(float(h2))))
    b5 = tf.Variable(tf.zeros([h5]))
    w6 = tf.Variable(tf.truncated_normal([h5, h6], stddev=1.0 / math.sqrt(float(h2))))
    b6 = tf.Variable(tf.zeros([h6]))
    w7 = tf.Variable(tf.truncated_normal([h6, 10], stddev=1.0 / math.sqrt(float(h2))))
    b7 = tf.Variable(tf.zeros([10]))
    
    #Creating our Model 
    y1 = tf.nn.elu(tf.matmul(X, w1) + b1)
    y2 = tf.nn.elu(tf.matmul(y1, w2) + b2)
    y3 = tf.nn.elu(tf.matmul(y2, w3) + b3)
    y4 = tf.nn.elu(tf.matmul(y3, w4) + b4)
    y5 = tf.nn.elu(tf.matmul(y4, w5) + b5)
    y6 = tf.nn.elu(tf.matmul(y5, w6) + b6)
    out_layer = tf.matmul(y6, w7) + b7
    result = out_layer
    
    loss = tf.nn.softmax_cross_entropy_with_logits(labels = Y,logits = result)
    loss = tf.reduce_mean(loss)
    
    training_step = tf.train.AdamOptimizer(learning_rate = lrate).minimize(loss)
    
    prediction = tf.equal(tf.argmax(result, 1), tf.argmax(Y, 1))
    prediction = tf.cast(prediction, tf.float32)
    accuracy = tf.reduce_mean(prediction)
    
    output = []
    iterations = []
    train_acc = []
    train_loss = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(num):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={X: batch[0], Y: batch[1]})
                #print('Step %d, Training Accuracy: %g' % (i, train_accuracy))
                iterations.append(i)
                train_acc.append(train_accuracy)
            #training_step.run(feed_dict={X: batch[0], Y: batch[1]}
            _, c = sess.run([training_step, loss], feed_dict={X: batch[0], Y: batch[1]})  
            if i% 100 == 0:
                train_loss.append(c)
            
        test_accuracy = accuracy.eval(feed_dict={X: mnist.test.images, Y: mnist.test.labels})
        #print('Test Accuracy: %g' % (test_accuracy))
        output.append(iterations)
        output.append(train_acc)
        output.append(train_loss)
        output.append(test_accuracy)
        return output
    
def main():
    #First, program is executed with 64 hidden neurons in each layer and with 20000 iterations
    final_out = []    
    
    final_out.append(sigmoidnn(64, 20000))
    final_out.append(relunn(64, 20000))
    final_out.append(elunn(64, 20000))
    
    print('Sigmod Test Accuracy: %g' %(final_out[0][3]))
    print('Relu Test Accuracy: %g' %(final_out[1][3]))
    print('Elu Test Accuracy: %g' %(final_out[2][3]))  
      
    final_out1 = []
    final_out2 = []
    final_out3 = []
    final_out4 = []
    final_out5 = []
            
    #Program is executed with different number of hidden neurons in each layer with 1000 iterations
    final_out1.append(sigmoidnn(16, 1000))
    final_out1.append(relunn(16, 1000))
    final_out1.append(elunn(16, 1000))
    print('16 Hidden Neurons Done')
    final_out2.append(sigmoidnn(32, 1000))
    final_out2.append(relunn(32, 1000))
    final_out2.append(elunn(32, 1000))
    print('32 Hidden Neurons Done')
    final_out3.append(sigmoidnn(64, 1000))
    final_out3.append(relunn(64, 1000))
    final_out3.append(elunn(64, 1000))
    print('64 Hidden Neurons Done')
    final_out4.append(sigmoidnn(128, 1000))
    final_out4.append(relunn(128, 1000))
    final_out4.append(elunn(128, 1000))
    print('128 Hidden Neurons Done')
    final_out5.append(sigmoidnn(256, 1000))
    final_out5.append(relunn(256, 1000))
    final_out5.append(elunn(256, 1000))
    print('256 Hidden Neurons Done')
    
    #Calculating average training accuracies over the different number of times functions called
    sample = []
    trainingaccuracy = []
    average1 = []
    for i in range(len(final_out1[0][0])):
        avg1 = 0
        avg1 = final_out1[0][1][i] + final_out2[0][1][i] + final_out3[0][1][i] + final_out4[0][1][i] + final_out5[0][1][i]
        avg1 = avg1 / 5
        average1.append(avg1)
    sample.append(final_out1[0][0])
    sample.append(average1)
    trainingaccuracy.append(sample)
    
    sample = []
    average1 = []
    for i in range(len(final_out1[0][0])):
        avg1 = 0
        avg1 = final_out1[1][1][i] + final_out2[1][1][i] + final_out3[1][1][i] + final_out4[1][1][i] + final_out5[1][1][i]
        avg1 = avg1 / 5
        average1.append(avg1)
    sample.append(final_out1[0][0])
    sample.append(average1)
    trainingaccuracy.append(sample)
    
    
    sample = []
    average1 = [] 
    for i in range(len(final_out1[0][0])):
        avg1 = 0
        avg1 = final_out1[2][1][i] + final_out2[2][1][i] + final_out3[2][1][i] + final_out4[2][1][i] + final_out5[2][1][i]
        avg1 = avg1 / 5
        average1.append(avg1)
    sample.append(final_out1[0][0])
    sample.append(average1)
    trainingaccuracy.append(sample)
    
    #Calculating average loss over the different number of times functions called
    sample = []
    losscalculation = []
    average1 = []
    for i in range(len(final_out1[0][0])):
        avg1 = 0
        avg1 = final_out1[0][2][i] + final_out2[0][2][i] + final_out3[0][2][i] + final_out4[0][2][i] + final_out5[0][2][i]
        avg1 = avg1 / 5
        average1.append(avg1)
    sample.append(final_out1[0][0])
    sample.append(average1)
    losscalculation.append(sample)
    
    sample = []
    average1 = []
    for i in range(len(final_out1[0][0])):
        avg1 = 0
        avg1 = final_out1[1][2][i] + final_out2[1][2][i] + final_out3[1][2][i] + final_out4[1][2][i] + final_out5[1][2][i]
        avg1 = avg1 / 5
        average1.append(avg1)
    sample.append(final_out1[0][0])
    sample.append(average1)
    losscalculation.append(sample)
    
    sample = []
    average1 = [] 
    for i in range(len(final_out1[0][0])):
        avg1 = 0
        avg1 = final_out1[2][2][i] + final_out2[2][2][i] + final_out3[2][2][i] + final_out4[2][2][i] + final_out5[2][2][i]
        avg1 = avg1 / 5
        average1.append(avg1)
    sample.append(final_out1[0][0])
    sample.append(average1)
    losscalculation.append(sample)
    
    #Test Accuracies for different number of times program executed
    test1 = []
    test2 = []
    test3 = []
    iterations = []
    sigmoidtest = []
    relutest = []
    elutest = []
    testaccuracy = []
    iterations = [1,2,3,4,5]
    
    test1.append(final_out1[0][3])
    test1.append(final_out2[0][3])
    test1.append(final_out3[0][3])
    test1.append(final_out4[0][3])
    test1.append(final_out5[0][3])
    sigmoidtest.append(iterations)
    sigmoidtest.append(test1)

    test2.append(final_out1[1][3])
    test2.append(final_out2[1][3])
    test2.append(final_out3[1][3])
    test2.append(final_out4[1][3])
    test2.append(final_out5[1][3])
    relutest.append(iterations)
    relutest.append(test2)    
    
    test3.append(final_out1[2][3])
    test3.append(final_out2[2][3])
    test3.append(final_out3[2][3])
    test3.append(final_out4[2][3])
    test3.append(final_out5[2][3])
    elutest.append(iterations)
    elutest.append(test3)
    
    testaccuracy.append(sigmoidtest)
    testaccuracy.append(relutest)
    testaccuracy.append(elutest)

    plotAccuracy1(final_out)
    plotLoss1(final_out)
    plotAccuracy(trainingaccuracy)
    plotLoss(losscalculation)
    plotTestAccuracy(testaccuracy)
    
if __name__ == '__main__':
    main()