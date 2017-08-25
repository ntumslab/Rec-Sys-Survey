import numpy as np
import os.path

def RMSE(X, Y):
    assert( len(X)==len(Y) ), "RMSE: Two arrays must be the same size."
    return np.sqrt( sum( pow(X-Y, 2) ) / len(X) ) 

def save_result(args, rmse):
    if args.userFeat != '' and args.itemFeat != '':
        fip_type = 'useritem'
    elif args.userFeat == '' and args.itemFeat != '':
        fip_type = 'item'
    elif args.userFeat != '' and args.itemFeat == '':
        fip_type = 'user'
    elif args.userFeat == '' and args.itemFeat == '':
        fip_type = 'none'

    if args.output != '':
        if os.path.exists(args.output) == True:
            with open(args.output, 'a') as fp_w:
                fp_w.write('{},{},{},{},{:f},{:f},{:f},{:f},{:.4f}\n'.format(fip_type, args.maxIter, args.batchSize, args.K, args.lr, args.lrw, args.C, args.Cw, rmse))
        else:
            with open(args.output, 'w') as fp_w:
                fp_w.write('type,maxIter,batchSize,K,lr,lrw,C,Cw,rmse\n')
                fp_w.write('{},{},{},{},{:f},{:f},{:f},{:f},{:.4f}\n'.format(fip_type, args.maxIter, args.batchSize, args.K, args.lr, args.lrw, args.C, args.Cw, rmse))
