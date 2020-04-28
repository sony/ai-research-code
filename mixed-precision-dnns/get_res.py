import csv

expe = {'U1 WQ' :'PARAMETRIC_FP_WQ_DELTA_B_INIT_ADAM',
        'U2 WQ' :'PARAMETRIC_FP_WQ_B_XMAX_INIT_ADAM',
        'U3 WQ' :'PARAMETRIC_FP_WQ_DELTA_XMAX_INIT_ADAM',
        'U1 WAQ':'PARAMETRIC_FP_WAQ_DELTA_B_INIT_ADAM',
        'U2 WAQ':'PARAMETRIC_FP_WAQ_B_XMAX_INIT_ADAM',
        'U3 WAQ':'PARAMETRIC_FP_WAQ_DELTA_XMAX_INIT_ADAM'}
for k, exp in expe.items():
    with open(exp + '/results.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        errs = [] 
        for row in reader:
            errs += [row['val_err']]
        print(k, ' : ', min(errs))
