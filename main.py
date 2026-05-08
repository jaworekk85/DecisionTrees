import Node
import numpy as np
from Generator import Generator
from Tools import Tools
from Criterion import Criterion
from Statistics import Statistics, Extended_Statistics
from Tree import Tree
import matplotlib.pyplot as plt
import pickle
import math

from multiprocessing import Process

MAIN_FOLDER = 'D:\\mj\\'

def create_a_problem(process, genID, gen, preq_l, W, N, holdout_dataset, holdout_steps):

    approach = 'hoeffding'
    meas = 'gini'
    pref = approach[0]+meas[0]

    list_of_attributes, classes = gen.get_list_of_attributes()
    statistics = Statistics(list_of_attributes, classes)

    #deltas
    #cr_hgd2 = Criterion(2, 0.01, approach, meas, False, 0.05)
    #cr_hgd3 = Criterion(2, 0.001, approach, meas, False, 0.05)
    #cr_hgd4 = Criterion(2, 0.0001, approach, meas, False, 0.05)
    #cr_hgd5 = Criterion(2, 0.00001, approach, meas, False, 0.05)
    #cr_hgd6 = Criterion(2, 0.000001, approach, meas, False, 0.05)


    #taus
    #cr_hgt010 = Criterion(2, 0.001, approach, meas, False, 0.01)
    #cr_hgt025 = Criterion(2, 0.001, approach, meas, False, 0.025)
    #cr_hgt050 = Criterion(2, 0.001, approach, meas, False, 0.05)
    #cr_hgt075 = Criterion(2, 0.001, approach, meas, False, 0.075)
    #cr_hgt100 = Criterion(2, 0.001, approach, meas, False, 0.1)

    # fractional Hoeffding
    cr_hf1 = Criterion(2, 0.001, approach, meas, 1.0, False, 0.0)
    cr_hf075 = Criterion(2, 0.001, approach, meas, 0.75, False, 0.0)
    cr_hf05 = Criterion(2, 0.001, approach, meas, 0.5, False, 0.0)
    cr_hf025 = Criterion(2, 0.001, approach, meas, 0.25, False, 0.0)
    cr_hf01 = Criterion(2, 0.001, approach, meas, 0.1, False, 0.0)



    tree_list = []

    #tree_list.append(Tree(pref+'d2_proc' + str(process) + '_gid' + str(genID), statistics, [cr_hgd2], False, preq_l, W))
    #tree_list.append(Tree(pref+'d3_proc' + str(process) + '_gid' + str(genID), statistics, [cr_hgd3], False, preq_l, W))
    #tree_list.append(Tree(pref+'d4_proc' + str(process) + '_gid' + str(genID), statistics, [cr_hgd4], False, preq_l, W))
    #tree_list.append(Tree(pref+'d5_proc' + str(process) + '_gid' + str(genID), statistics, [cr_hgd5], False, preq_l, W))
    #tree_list.append(Tree(pref+'d6_proc' + str(process) + '_gid' + str(genID), statistics, [cr_hgd6], False, preq_l, W))

    #tree_list.append(Tree(pref+'t010_proc' + str(process) + '_gid' + str(genID), statistics, [cr_hgt010], False, preq_l, W))
    #tree_list.append(Tree(pref+'t025_proc' + str(process) + '_gid' + str(genID), statistics, [cr_hgt025], False, preq_l, W))
    #tree_list.append(Tree(pref+'t050_proc' + str(process) + '_gid' + str(genID), statistics, [cr_hgt050], False, preq_l, W))
    #tree_list.append(Tree(pref+'t075_proc' + str(process) + '_gid' + str(genID), statistics, [cr_hgt075], False, preq_l, W))
    #tree_list.append(Tree(pref+'t100_proc' + str(process) + '_gid' + str(genID), statistics, [cr_hgt100], False, preq_l, W))

    tree_list.append(Tree(pref + 'hf1_proc' + str(process) + '_gid' + str(genID), statistics, [cr_hf1], False, preq_l, W))
    tree_list.append(Tree(pref + 'hf075_proc' + str(process) + '_gid' + str(genID), statistics, [cr_hf075], False, preq_l, W))
    tree_list.append(Tree(pref + 'hf05_proc' + str(process) + '_gid' + str(genID), statistics, [cr_hf05], False, preq_l, W))
    tree_list.append(Tree(pref + 'hf025_proc' + str(process) + '_gid' + str(genID), statistics, [cr_hf025], False, preq_l, W))
    tree_list.append(Tree(pref + 'hf01_proc' + str(process) + '_gid' + str(genID), statistics, [cr_hf01], False, preq_l, W))

    #tree_list.append(Tree(pref+'d2_proc' + str(process) + '_gid' + str(genID), statistics, [cr_hgd2], True, preq_l, W))
    #tree_list.append(Tree(pref+'d3_proc' + str(process) + '_gid' + str(genID), statistics, [cr_hgd3], True, preq_l, W))
    #tree_list.append(Tree(pref+'d4_proc' + str(process) + '_gid' + str(genID), statistics, [cr_hgd4], True, preq_l, W))
    #tree_list.append(Tree(pref+'d5_proc' + str(process) + '_gid' + str(genID), statistics, [cr_hgd5], True, preq_l, W))
    #tree_list.append(Tree(pref+'d6_proc' + str(process) + '_gid' + str(genID), statistics, [cr_hgd6], True, preq_l, W))

    #tree_list.append(Tree(pref+'t010_proc' + str(process) + '_gid' + str(genID), statistics, [cr_hgt010], True, preq_l, W))
    #tree_list.append(Tree(pref+'t025_proc' + str(process) + '_gid' + str(genID), statistics, [cr_hgt025], True, preq_l, W))
    #tree_list.append(Tree(pref+'t050_proc' + str(process) + '_gid' + str(genID), statistics, [cr_hgt050], True, preq_l, W))
    #tree_list.append(Tree(pref+'t075_proc' + str(process) + '_gid' + str(genID), statistics, [cr_hgt075], True, preq_l, W))
    #tree_list.append(Tree(pref+'t100_proc' + str(process) + '_gid' + str(genID), statistics, [cr_hgt100], True, preq_l, W))

    tree_list.append(Tree(pref + 'hf1_proc' + str(process) + '_gid' + str(genID), statistics, [cr_hf1], True, preq_l, W))
    tree_list.append(Tree(pref + 'hf075_proc' + str(process) + '_gid' + str(genID), statistics, [cr_hf075], True, preq_l, W))
    tree_list.append(Tree(pref + 'hf05_proc' + str(process) + '_gid' + str(genID), statistics, [cr_hf05], True, preq_l, W))
    tree_list.append(Tree(pref + 'hf025_proc' + str(process) + '_gid' + str(genID), statistics, [cr_hf025], True, preq_l, W))
    tree_list.append(Tree(pref + 'hf01_proc' + str(process) + '_gid' + str(genID), statistics, [cr_hf01], True, preq_l, W))

    #tree_list.append(Tree('hef3_'+str(process), statistics, [cr_hef3], False, preq_l, W))
    #tree_list.append(Tree('hef5_'+str(process), statistics, [cr_hef5], False, preq_l, W))
    #tree_list.append(Tree('hgf3_'+str(process), statistics, [cr_hgf3], False, preq_l, W))
    #tree_list.append(Tree('hgf5_'+str(process), statistics, [cr_hgf5], False, preq_l, W))
    #tree_list.append(Tree('mgf3_'+str(process), statistics, [cr_mgf3], False, preq_l, W))
    #tree_list.append(Tree('mgf5_'+str(process), statistics, [cr_mgf5], False, preq_l, W))
    #tree_list.append(Tree('mgt3_'+str(process), statistics, [cr_mgt3], False, preq_l, W))
    #tree_list.append(Tree('mgt5_'+str(process), statistics, [cr_mgt5], False, preq_l, W))
    #tree_list.append(Tree('gmf3_'+str(process), statistics, [cr_gmf3], False, preq_l, W))
    #tree_list.append(Tree('gmf5_'+str(process), statistics, [cr_gmf5], False, preq_l, W))
    #tree_list.append(Tree('mgf3_gmf3_'+str(process), statistics, [cr_mgf3, cr_gmf3], False, preq_l, W))
    #tree_list.append(Tree('mgf5_gmf5_'+str(process), statistics, [cr_mgf5, cr_gmf5], False, preq_l, W))
    #tree_list.append(Tree('hef3_es_'+str(process), statistics, [cr_hef3], True, preq_l, W))
    #tree_list.append(Tree('hef5_es_'+str(process), statistics, [cr_hef5], True, preq_l, W))
    #tree_list.append(Tree('hgf3_es_'+str(process), statistics, [cr_hgf3], True, preq_l, W))
    #tree_list.append(Tree('hgf5_es_'+str(process), statistics, [cr_hgf5], True, preq_l, W))
    #tree_list.append(Tree('mgf3_es_'+str(process), statistics, [cr_mgf3], True, preq_l, W))
    #tree_list.append(Tree('mgf5_es_'+str(process), statistics, [cr_mgf5], True, preq_l, W))
    #tree_list.append(Tree('mgt3_es_'+str(process), statistics, [cr_mgt3], True, preq_l, W))
    #tree_list.append(Tree('mgt5_es_'+str(process), statistics, [cr_mgt5], True, preq_l, W))
    #tree_list.append(Tree('gmf3_es_'+str(process), statistics, [cr_gmf3], True, preq_l, W))
    #tree_list.append(Tree('gmf5_es_'+str(process), statistics, [cr_gmf5], True, preq_l, W))
    #tree_list.append(Tree('mgf3_gmf3_es_'+str(process), statistics, [cr_mgf3, cr_gmf3], True, preq_l, W))
    #tree_list.append(Tree('mgf5_gmf5_es_'+str(process), statistics, [cr_mgf5, cr_gmf5], True, preq_l, W))


    for i in range(N):
        if i % 1000 == 0:
            print('process ', process, ', i = ', i)
        dv = gen.generate()
        for tree in tree_list:
            tree.pass_data(dv)
        if i in holdout_steps:
            for tree in tree_list:
                tree.holdout_test(holdout_dataset, i)

    t = 0
    for tree in tree_list:
        tree.toFile()
        tree.save_tree('proc_' + str(process)+'genID_'+str(genID)+'_drzewo_powstale_'+str(t)+'.txt')
        t = t+1


def make_average_file(files_prefix, range_of_proc_numbers, es_or_not, genID):
    dicts = []
    final_dict_avg = {}
    final_dict_std = {}
    eon = ''
    if es_or_not:
        eon = '_ES'
    print('avg')
    for pn in range_of_proc_numbers:
        print('process number: ', pn)
        print('file: ', files_prefix+'_proc'+str(pn)+'_gid'+str(genID)+eon+'.pkl')
        with open(MAIN_FOLDER+files_prefix+'_proc'+str(pn)+'_gid'+str(genID)+eon+'.pkl', 'rb') as f:
            dicts.append(pickle.load(f))
    print(len(dicts))
    print(type(dicts[0]))
    for keys in dicts[0]:
        print('key: ', keys)
        temp_avg = []
        temp_std = []
        for n in range(len(dicts[0][keys])):
            avg = 0.0
            std = 0.0
            for i in range(len(dicts)):
                avg = avg + dicts[i][keys][n]
            avg = avg / len(dicts)
            for i in range(len(dicts)):
                std = std + math.pow(dicts[i][keys][n] - avg, 2.0)
            std = math.sqrt(std / (len(dicts)*(len(dicts)-1)))
            temp_avg.append(avg)
            temp_std.append(std)
        final_dict_avg[keys] = temp_avg
        final_dict_std[keys] = temp_std
    print('save file: ', files_prefix + '_gid' + str(genID) + eon +'_' + '_avg.pkl')
    with open(MAIN_FOLDER+files_prefix + '_gid' + str(genID) + eon +'_' + '_avg.pkl', 'wb') as f:
        pickle.dump(final_dict_avg, f)
    with open(MAIN_FOLDER+files_prefix + '_gid' + str(genID) + eon +'_' + '_std.pkl', 'wb') as f:
        pickle.dump(final_dict_std, f)


def plot(list_of_cores, list_of_legends, measure_x, measure_y, output_file, std=True, step=1):
    xs = []
    avgs = []
    stds = []
    lowers = []
    uppers = []
    colors = ['k', 'c', 'g', 'r', 'm', 'b', 'y']
    print('measure x: ', measure_x)
    print('measure_y: ', measure_y)
    for core in list_of_cores:
        print('core: ', core)
        core2 = core
        if std:
            core2 = core2+'_avg'
        with open(MAIN_FOLDER+core2+'.pkl', 'rb') as f:
            loaded_obj = pickle.load(f)
            avgs.append(np.array(loaded_obj[measure_y]))
            if measure_x is None:
                xs.append([i for i in range(len(avgs[-1]))])
            else:
                xs.append(np.array(loaded_obj[measure_x]))
        if std:
            with open(MAIN_FOLDER+core+'_std.pkl', 'rb') as f:
                stds.append(np.array(pickle.load(f)[measure_y]))
            #OPTIONAL!!!!!!
        #print('optional operation')
        #for y in range(len(stds[-1])):
            #stds[-1][y] = math.sqrt(stds[-1][y] / math.sqrt(8*7))
            #END OF OPTIONAL!!!!!
            lowers.append(np.array(avgs[-1]) - np.array(stds[-1]))
            uppers.append(np.array(avgs[-1]) + np.array(stds[-1]))
    print('plotting...')
    for scale in ['log', 'linear']:
        plt.clf()
        if measure_x is not None:
            if measure_x == 'holdout_n':
                plt.xlabel('number of data elements')
            elif measure_x == 'leaves':
                plt.xlabel('number of leaves')
            else:
                plt.xlabel(measure_x)
        else:
            plt.xlabel('number of data elements')
        if measure_y == 'holdout_m' or measure_y == 'holdout_b' or measure_y == 'prequential_m' or measure_y == 'prequential_b':
            plt.ylabel('classification accuracy')
        elif measure_y == 'leaves':
            plt.ylabel('nuber of leaves')
        else:
            plt.ylabel(measure_y)
        plt.xscale(scale)
        for i in range(len(avgs)):
            beg_a = max(0, len(avgs[i]) - len(xs[i]))
            beg_x = max(0, len(xs[i]) - len(avgs[i]))
            plt.plot(xs[i][beg_x::step], avgs[i][beg_a::step], color=colors[i], label=list_of_legends[i])
            if std:
                plt.fill_between(xs[i][beg_x::step], lowers[i][beg_a::step], uppers[i][beg_a::step], color=colors[i], alpha=.2)
        plt.legend()
        xstring = 'xN'
        if measure_x is not None:
            xstring = 'x'+measure_x
        plt.savefig(MAIN_FOLDER+output_file + '_' + xstring + '_' + measure_y + '_scale_' + scale + '.png')




def plot_N(list_of_list_of_cores, list_of_legends, xs, label_x, measure_y, output_file, std, Ns_list, scale):
    avgs = []
    stds = []
    lowers = []
    uppers = []
    colors = ['k', 'c', 'g', 'r', 'm', 'b', 'y']
    for list_of_cores in list_of_list_of_cores:
        avgs.append([])
        stds.append([])
        lowers.append([])
        uppers.append([])
        for core in list_of_cores:
            print('core: ', core)
            core2 = core
            if std:
                core2 = core2+'_avg'
            with open(MAIN_FOLDER+core2+'.pkl', 'rb') as f:
                avgs[-1].append(np.array(pickle.load(f)[measure_y]))
            if std:
                with open(MAIN_FOLDER+core+'_std.pkl', 'rb') as f:
                    stds[-1].append(np.array(pickle.load(f)[measure_y]))
                lowers[-1].append(np.array(avgs[-1][-1]) - np.array(stds[-1][-1]))
                uppers[-1].append(np.array(avgs[-1][-1]) + np.array(stds[-1][-1]))
        print('plotting...')
    for Nn in Ns_list:
        plt.clf()
        plt.xlabel(label_x)
        plt.ylabel(measure_y)
        plt.xscale(scale)
        for plts in range(len(avgs)):
            values = []
            lws = []
            ups = []
            for i in range(len(avgs[plts])):
                values.append(avgs[plts][i][Nn])
                if std:
                    lws.append(lowers[plts][i][Nn])
                    ups.append(uppers[plts][i][Nn])
            plt.plot(xs, values, color=colors[plts], label=list_of_legends[plts])
            if std:
                plt.fill_between(xs, lws, ups, color=colors[plts], alpha=.2)
        plt.legend()
        plt.savefig(MAIN_FOLDER+output_file + '_' + measure_y + '_N'+str(Nn)+'.png')






if __name__ == '__main__':


    #********** SIMULATION! **********

    N = 2000000

    lambdaa = 0.999
    W = 1000

    generator_repetitions = 4
    mind = 3
    maxd = 18
    d = [20, 50]
    w = [0.15, 0.2, 0.25, 0.3]
    '''
    generators = []
    for di in range(len(d)):
        for wi in range(len(w)):
            generators.append(Generator())
            generators[-1].create(d[di], mind, maxd, w[wi])
            #generators[-1].load_tree('gentree_'+str(di*len(w)+wi)+'.txt')

    print(generators)

    #for ig in range(len(d)*len(w)):
    #    generators[ig].save_tree('gentree_'+str(ig)+'.txt')

    hN = 2000
    holdouts = []
    for ih in range(len(d)*len(w)):
        for gr in range(generator_repetitions):
            holdouts.append([])
            hidx = len(holdouts)-1
            for nh in range(hN):
                holdouts[hidx].append(generators[ih].generate())

    hstep = 10000
    logaritmic_holdout = True
    hcheck = N // hstep

    if logaritmic_holdout:
        hstep = (math.log(N) / math.log(10.0)) / hcheck
        holdout_steps = [int(math.pow(10.0, (dl + 1) * hstep)) for dl in range(hcheck)]
        filtter = 50
        holdout_steps = [x for x in holdout_steps if x > filtter]
    else:
        holdout_steps = [(x + 1) * hstep for x in range(hcheck)]
        
        
        
    list_of_processes = []      
    for proc in range(32):
        gID = proc // 4
        list_of_processes.append(Process(target=create_a_problem, args=(proc, gID, generators[gID], lambdaa, W, N, holdouts[proc], holdout_steps)))

    for process in list_of_processes:
        process.start()
    '''
    # ********** END OF SIMULATION! **********



    # ********* CREATING AVERAGING (and STD) RESULT FILES ! ***************
    '''
    list_of_cores = []
    #for delta in [2, 3, 4, 5, 6]:
    #for delta in [3]:
    #    list_of_cores.append('hgd' + str(delta))
    #for tau in ['t010', 't025', 't050', 't075', 't100']:
        #list_of_cores.append('hg' + tau)
    for frac in ['hf01', 'hf025', 'hf05', 'hf075', 'hf1']:
        list_of_cores.append('hg' + frac)


    for files_prefix in list_of_cores:
        for gID in range(8):
            for es_or_not in [False, True]:
                process_numbers = [(4 * gID + pp) for pp in range(4)]
                make_average_file(files_prefix, process_numbers, es_or_not, gID)
    '''

    # ********* END of CREATING AVERAGING (and STD) RESULT FILES ! ***************




    # ************ PLOTTING !!!! ************************************************

    list_of_legends = ['sufficient statistics', 'extended_statistics']

    #param = ['d2','d3','d4','d5','d6','t010', 't025', 't050', 't075', 't100']
    #param = ['t010', 't025', 't050', 't075', 't100']
    param = ['hf01', 'hf025', 'hf05', 'hf075', 'hf1']

    measures_y = ['time', 'leaves', 'prequential_b', 'prequential_m']
    '''
    for gid in range(8):
        for d in param:
            core = 'hg'+d+'_gid'+str(gid)+'_'
            list_of_cores = [core, core+'ES_']
            for my in measures_y:
                plot(list_of_cores, list_of_legends, None, my, core, std=True, step=100)
            plot(list_of_cores, list_of_legends, 'holdout_n', 'holdout_m', core, std=True, step=1)
            plot(list_of_cores, list_of_legends, 'holdout_n', 'holdout_b', core, std=True, step=1)
            plot(list_of_cores, list_of_legends, 'leaves', 'prequential_m', core, std=True, step=100)
    '''
    '''
    
    for gid in range(8):
        list1 = ['hg' + core + '_gid' + str(gid) + '_' for core in param[:5]]
        list2 = [core + 'ES_' for core in list1]
       # xs = [0.01, 0.001, 0.0001, 0.00001, 0.000001]
       # for my in measures_y:
       #     plot_N([list1, list2], list_of_legends, xs, 'delta', my, 'delta_gid'+str(gid), True, [100, 1000, 10000, 100000, 1000000], 'log')
       # list1 = ['hg' + core + '_gid' + str(gid) + '_' for core in param[5:]]
       # list2 = [core + 'ES_' for core in list1]
       # xs = [0.01, 0.025, 0.05, 0.075, 0.1]
        xs = [0.1, 0.25, 0.5, 0.75, 1]
        for my in measures_y:
            plot_N([list1, list2], list_of_legends, xs, 'fraction of original Hoeffding\'s bound', my, 'hf_gid' + str(gid), True, [100, 1000, 10000, 100000, 1000000], 'linear')
    
    '''
    list_of_legends =['f=0.1', 'f=0.25', 'f=0.5', 'f=0.75', 'f=1']
    for gid in range(8):
        for suffix in ['', 'ES_']:
            coref = 'fraction_cmparison_'+str(gid)+'_'+suffix
            list_of_cores = ['hg' + d + '_gid' + str(gid) + '_' + suffix for d in param]
            for my in measures_y:
                plot(list_of_cores, list_of_legends, None, my, coref, std=True, step=100)
            plot(list_of_cores, list_of_legends, 'holdout_n', 'holdout_m', coref, std=True, step=1)
            plot(list_of_cores, list_of_legends, 'holdout_n', 'holdout_b', coref, std=True, step=1)
            plot(list_of_cores, list_of_legends, 'leaves', 'prequential_m', coref, std=True, step=100)

    # ************* END OF PLOTTING *****************************************************