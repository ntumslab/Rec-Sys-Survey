#  Friendship and Interest Propagation
This is a python implementation of "Joint Friendship and Interest Propagation".

## Input Data Format
Rating matrix is stored in coordinate format (user, item, rating). Feature matrix is stored in dense format (id, feature 1, feauture2, ..., feature n). See **data/ml-1m/** for examples.

## Quick Start
```
$ python FIP.py --train data/ml-1m/train.txt --test data/ml-1m/test.txt --userFeat data/ml-1m/user.txt --itemFeat data/ml-1m/item.txt --batchSize 128 --maxIter 10 --K 8 --lr 0.1 --lrw 0.1 --C 0.1 --Cw 0.1
```
You can type **python FIP.py --help** for more details about the parameters.

## Reference
```
* Yang, Shuang-Hong, et al. "Like like alike: joint friendship and interest propagation in social networks." Proceedings of the 20th international conference on World wide web. ACM, 2011.
```
