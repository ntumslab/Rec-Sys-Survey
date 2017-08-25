import os
import sys
import random
import argparse
import numpy as np
from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description = 'Like like alike')
    parser.add_argument('--userFeat' , type = str, default = '', help = 'User feature file (optional)')
    parser.add_argument('--itemFeat' , type = str, default = '', help = 'Item feature file (optional)')
    parser.add_argument('--train' , type = str, default = '', help = 'Training file')
    parser.add_argument('--test' , type = str, default = '', help = 'Testing file')
    parser.add_argument('--output', type = str, default = '', help = 'Output file')
    
    parser.add_argument('--maxIter', type = int, default = 10, help = 'Max training iterations')
    parser.add_argument('--batchSize', type = int, default = 200, help = 'Training instances for each iteration')
    parser.add_argument('--K', type = int, default = 16, help = 'Dimension of latent fectors')
    parser.add_argument('--lr', type = float, default = 1e-2, help = 'Learning rate for sgd')
    parser.add_argument('--lrw', type = float, default = 1e-6, help = 'Learning rate for W for sgd')
    parser.add_argument('--Cw', type = float, default = 0.1, help = 'Regularization for W')
    parser.add_argument('--C', type = float, default = 0.1, help = 'Regularization for U and I')
    return parser.parse_args()


def preprocess(args):
    userFeat, itemFeat, train, test = None, None, [], []
    userFeat_d, itemFeat_d = dict(), dict()
    user_map, item_map = {}, {}
    userCount, itemCount = 0, 0

    if not args.userFeat == '':
        userFeat = []
        with open(args.userFeat, 'r') as fp_r:
            fp_r.readline()
            for line in fp_r:
                line = line.strip().split(',')
                user = int(line[0])
                if not user in user_map:
                    user_map[user] = userCount
                    userCount += 1
                userFeat_d[user_map[user]] = np.array([float(x) for x in line[1:]])
        for i in range(len(userFeat_d)):
            userFeat.append(userFeat_d[i])
        userFeat = np.array(userFeat)

    if not args.itemFeat == '':
        itemFeat = []
        with open(args.itemFeat, 'r') as fp_r:
            fp_r.readline()
            for line in fp_r:
                line = line.strip().split(',')
                item = int(line[0])
                if not item in item_map:
                    item_map[item] = itemCount
                    itemCount += 1
                itemFeat_d[item_map[item]] = np.array([float(x) for x in line[1:]])
        for i in range(len(itemFeat_d)):
            itemFeat.append(itemFeat_d[i])
        itemFeat = np.array(itemFeat)

    with open(args.train, 'r') as fp_r:
        fp_r.readline()
        for line in fp_r:
            line = line.strip().split(',')

            user = int(line[0])
            if (args.userFeat == '') and (not user in user_map):
                user_map[user] = userCount
                userCount += 1
            user = user_map[user]

            item = int(line[1])
            if (args.itemFeat == '') and (not item in item_map):
                item_map[item] = itemCount
                itemCount += 1
            item = item_map[item]
            
            rating = float(line[2])
            
            train.append(np.array([user, item, rating]))

    with open(args.test, 'r') as fp_r:
        fp_r.readline()
        for line in fp_r:
            line = line.strip().split(',')

            user = int(line[0])
            if (args.userFeat == '') and (not user in user_map):
                user_map[user] = userCount
                userCount += 1
            user = user_map[user]
            
            item = int(line[1])
            if (args.itemFeat == '') and (not item in item_map):
                item_map[item] = itemCount
                itemCount += 1
            item = item_map[item]

            rating = float(line[2])
            
            test.append(np.array([user, item, rating]))
    return userCount, itemCount, userFeat, itemFeat, train, test, user_map, item_map


def train(Nu, Ni, userFeat, itemFeat, X, maxIter, batchSize, K, lr, lrw, C, Cw):
    hasUserFeat = False if userFeat is None else True
    hasItemFeat = False if itemFeat is None else True
    X = np.array(X)[:, 0:3]
    U = np.random.random((Nu, K))
    I = np.random.random((Ni, K))
    W, Wu, Wi = None, None, None
    if hasUserFeat and hasItemFeat:
        Du, Di = len(userFeat[0]), len(itemFeat[0])
        W = np.random.random((Du, Di)) / (Du * Di)
    elif hasUserFeat and (not hasItemFeat):
        Du = len(userFeat[0])
        Wu = np.random.random((Nu, Du))
    elif (not hasUserFeat) and hasItemFeat:
        Di = len(itemFeat[0])
        Wi = np.random.random((Ni, Di))

    for currIter in range(maxIter):
        loss = 0.0
        np.random.shuffle(X)
        for currBatch in range(int(len(X) / batchSize) + 1):
            x = X[(batchSize * currBatch):(batchSize * (currBatch + 1)), :]
            users = x[:, 0].flatten().astype(int)
            items = x[:, 1].flatten().astype(int)
            userInt, itemInt = U[users], I[items]
            if hasUserFeat:
                userExt = userFeat[users]
            if hasItemFeat:
                itemExt = itemFeat[items]
            y = x[:, 2].flatten()

            # ===== calculate loss ===== #
            r = np.multiply(userInt, itemInt).sum(axis = 1)
            if hasUserFeat and hasItemFeat:
                r += np.multiply(np.dot(userExt, W), itemExt).sum(axis = 1)
            elif hasUserFeat and (not hasItemFeat):
                r += np.multiply(userExt, Wu[users]).sum(axis = 1)
            elif (not hasUserFeat) and hasItemFeat:
                r += np.multiply(itemExt, Wi[items]).sum(axis = 1)
            diff = r - y
            loss += np.multiply(diff, diff).sum()

            # ===== calculate differential ===== #
            dr_du = np.multiply(itemInt, diff.reshape((len(x), 1))) + C * userInt
            dr_di = np.multiply(userInt, diff.reshape((len(x), 1))) + C * itemInt
            if hasUserFeat and hasItemFeat:
                dr_dW = Cw * np.dot(np.multiply(userExt, diff.reshape((len(x), 1))).T, itemExt)
            elif hasUserFeat and (not hasItemFeat):
                dr_dWu = Cw * np.multiply(userExt, diff.reshape((len(x), 1)))
            elif (not hasUserFeat) and hasItemFeat:
                dr_dWi = Cw * np.multiply(itemExt, diff.reshape((len(x), 1)))

            # ===== update ===== #
            for i in range(len(x)):
                U[users[i]] -= lr * dr_du[i] / len(x)
                I[items[i]] -= lr * dr_di[i] / len(x)
            if hasUserFeat and hasItemFeat:
                W -= lrw * dr_dW / len(x)
            elif hasUserFeat and (not hasItemFeat):
                for i in range(len(x)):
                    Wu[users[i]] -= lrw * dr_dWu[i] / len(x)
            elif (not hasUserFeat) and hasItemFeat:
                for i in range(len(x)):
                    Wi[items[i]] -= lrw * dr_dWi[i] / len(x)
        print('[FIP] Iter {}/{}. Training loss = {:.2f}'.format(currIter+1, maxIter, loss))
    return U, I, W, Wu, Wi


def predict(user_map, item_map, output, X, U, I, userFeat, itemFeat, W, Wu, Wi):
    pred = list()
    orin = list()
    for x in X:
        user, item, value = int(x[0]), int(x[1]), int(x[2])
        r = np.dot(U[user], I[item])
        if not (W is None):
            r += np.dot(np.dot(userFeat[user], W), itemFeat[item])
        elif not (Wu is None):
            r += np.dot(userFeat[user], Wu[user])
        elif not (Wi is None):
            r += np.dot(itemFeat[item], Wi[item])
        pred.append(r)
        orin.append(value)
    return np.array(pred), np.array(orin)

def main():
    args = parse_args()
    Nu, Ni, userFeat, itemFeat, trainData, testData, user_map, item_map = preprocess(args)
    print('[Settings] maxIter = {}. batchSize = {}. K = {}. lr = {}. lrw = {}. C = {}. Cw = {}'.format(args.maxIter, args.batchSize, args.K, args.lr, args.lrw, args.C, args.Cw))
    U, I, W, Wu, Wi = train(Nu, Ni, userFeat, itemFeat, trainData, args.maxIter, args.batchSize, args.K, args.lr, args.lrw, args.C, args.Cw)
    pred, orin = predict(user_map, item_map, args.output, testData, U, I, userFeat, itemFeat, W, Wu, Wi)
    rmse = RMSE(orin, pred)
    save_result(args, rmse)
    print('[Result] RMSE = {:.4f}'.format(rmse))
    return

if __name__ == '__main__':
    main()
