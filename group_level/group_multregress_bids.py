"""
ROUGH DRAFT

for help:
contact: annepark@mit.edu


Example call:

python /git/gopenfmri/group_level/group_multregress_bids.py \
-m 124 \
-t 5 \
-d /om/project/voice/bids/data/ \
-l1 /om/project/voice/processedData/l1analysis/l1output_20160901a/ \
-o /om/project/voice/processedData/groupAnalysis/l2output_20160901a/ \
-w /om/scratch/Fri/user/group_voice`date +%Y%m%d%H%M%S`/ \
-p 'SLURM' --plugin_args "{'sbatch_args':'-p om_all_nodes'}" \
-s /om/project/voice/processedData/openfmri/groups/user/20160904/participant_key.txt \
-b /om/project/voice/processedData/openfmri/groups/user/20160904/behav.txt \
-g /om/project/voice/processedData/openfmri/groups/user/20160904/contrasts.txt

## Required files
#  participant_key.txt
#  behav.txt
#  contrasts.txt

## Additional documentation
https://github.mit.edu/MGHPCC/OpenMind/wiki/Lab-Specific-Cookbook:-Gabrieli-lab#grouplevelscripts

"""

import os
from nipype import config
config.enable_provenance()

from nipype import Workflow, Node, MapNode, Function
from nipype import DataGrabber, DataSink
from nipype.interfaces.fsl import (Merge, FLAMEO, ContrastMgr,
                                   SmoothEstimate, Cluster, ImageMaths, MultipleRegressDesign)
import nipype.interfaces.fsl as fsl
import nipype.interfaces.utility as util
from nipype.interfaces.fsl.maths import BinaryMaths

get_len = lambda x: len(x)

def l1_contrasts_num(model_id, task_name, dataset_dir):
    import numpy as np
    import os
    contrast_def = []
    contrasts = 0
    contrast_file = os.path.join(dataset_dir, 'code', 'model', 'model%03d' % model_id,
                                 'task_contrasts.txt')
    if os.path.exists(contrast_file):
        with open(contrast_file, 'rt') as fp:
            contrast_def.extend([np.array(row.split()) for row in fp.readlines() if row.strip()])
    for row in contrast_def:
        if row[0] != "task-%s" % task_name:
            continue
        contrasts = contrasts + 1
    cope_id = range(1, contrasts + 1)
    return cope_id

def get_taskname(base_dir, task_id):
    import os
    task_key = os.path.join(base_dir, 'code', 'task_key.txt')
    if not os.path.exists(task_key):
        return
    with open(task_key, 'rt') as fp:
        for line in fp:
            info = line.strip().split()
            if 'task%03d'%(task_id) in info:
                return info[1]

def get_sub_vars(dataset_dir, task_name, model_id, sub_list_file, behav_file, group_contrast_file):
    import numpy as np
    import os
    import pandas as pd
    import re  

    # Read in all subjects in participant_key ("sub_list_file")
    # Process only subjects with a nonzero number
    # Check to make sure every subject to be processed has a line in behav.txt ("behav_file")
    subs_list = pd.read_table(sub_list_file, index_col=0)['task-%s' % task_name]
    subs_needed = subs_list.index[np.nonzero(subs_list)[0]]
    behav_info = pd.read_table(behav_file,  delim_whitespace=True, index_col=0)

    missing_subjects = np.setdiff1d(subs_needed, behav_info.index.tolist())
    if len(missing_subjects) > 0:
        raise ValueError('Subjects %s are missing from participant key' % ' '.join(missing_subjects))

    contrast_defs=[]
    with open(group_contrast_file, 'rt') as fp:
        contrast_defs = fp.readlines()

    contrasts = []
    for row in contrast_defs:
        print row
        if 'task-%s' % task_name not in row:
            continue
        regressor_names =  re.search("\[([\w\s',]+)\]", row).group(1)
        regressor_names = eval('[' + regressor_names + ']')

        for val in regressor_names:
            if val not in behav_info.keys():
                raise ValueError('Regressor %s not in behav.txt file' % val)
        contrast_name = row.split()[1]
        contrast_vector = np.array(re.search("\]([\s\d.-]+)", row).group(1).split()).astype(float).tolist()
        con = [tuple([contrast_name, 'T', regressor_names, contrast_vector])]
        contrasts.append(con)

    regressors_needed = []
    for idx, con in enumerate(contrasts):
        model_regressor = {}
        for cond in con[0][2]:
            values = behav_info.ix[subs_needed, cond].values
            if tuple(np.unique(values).tolist()) not in [(1,), (0, 1)]:
                values = values - values.mean()
            model_regressor[cond] = values.tolist()
        regressors_needed.append(model_regressor)
    groups = [1 for val in subs_needed]
    return regressors_needed, contrasts, groups, subs_needed.values.tolist()


def run_palm(cope_file, design_file, contrast_file, group_file, mask_file, 
             cluster_threshold=3.09):
    import os
    from glob import glob
    from nipype.interfaces.base import CommandLine
    cmd = ("palm -i {cope_file} -m {mask_file} -d {design_file} -t {contrast_file} -eb {group_file} -T " 
           "-C {cluster_threshold} -Cstat extent -fdr -noniiclass -twotail -logp -zstat")
    cl = CommandLine(cmd.format(cope_file=cope_file, mask_file=mask_file, design_file=design_file, 
                                contrast_file=contrast_file,
                                group_file=group_file, cluster_threshold=cluster_threshold))
    results = cl.run(terminal_output='file')
    return [os.path.join(os.getcwd(), val) for val in sorted(glob('palm*'))]


def group_multregress_openfmri(dataset_dir, model_id=None, task_id=None, l1output_dir=None, out_dir=None, 
                               no_reversal=False, plugin=None, plugin_args=None, flamemodel='flame1',
                               nonparametric=False, use_spm=False,
                               sub_list_file=None, behav_file=None, group_contrast_file=None):

    meta_workflow = Workflow(name='mult_regress')
    meta_workflow.base_dir = work_dir
    for task in task_id:
        task_name = get_taskname(dataset_dir, task)
        cope_ids = l1_contrasts_num(model_id, task_name, dataset_dir)
        regressors_needed, contrasts, groups, subj_list = get_sub_vars(dataset_dir, task_name, model_id,
                                                                      sub_list_file, behav_file, group_contrast_file)
        for idx, contrast in enumerate(contrasts):
            wk = Workflow(name='model_%03d_task_%03d_contrast_%s' % (model_id, task, contrast[0][0]))

            info = Node(util.IdentityInterface(fields=['model_id', 'task_id', 'dataset_dir', 'subj_list']),
                        name='infosource')
            info.inputs.model_id = model_id
            info.inputs.task_id = task
            info.inputs.dataset_dir = dataset_dir
            
            dg = Node(DataGrabber(infields=['model_id', 'task_id', 'cope_id'],
                                  outfields=['copes', 'varcopes']), name='grabber')
            dg.inputs.template = os.path.join(l1output_dir,
                                              'model%03d/task%03d/%s/%scopes/%smni/%scope%02d.nii%s')
            if use_spm:
                dg.inputs.template_args['copes'] = [['model_id', 'task_id', subj_list, '', 'spm/',
                                                     '', 'cope_id', '']]
                dg.inputs.template_args['varcopes'] = [['model_id', 'task_id', subj_list, 'var', 'spm/',
                                                        'var', 'cope_id', '.gz']]
            else:
                dg.inputs.template_args['copes'] = [['model_id', 'task_id', subj_list, '', '', '', 
                                                     'cope_id', '.gz']]
                dg.inputs.template_args['varcopes'] = [['model_id', 'task_id', subj_list, 'var', '',
                                                        'var', 'cope_id', '.gz']]
            dg.iterables=('cope_id', cope_ids)
            dg.inputs.sort_filelist = False

            wk.connect(info, 'model_id', dg, 'model_id')
            wk.connect(info, 'task_id', dg, 'task_id')

            print '------------'
            print dg
            
            model = Node(MultipleRegressDesign(), name='l2model')
            model.inputs.groups = groups
            model.inputs.contrasts = contrasts[idx]
            model.inputs.regressors = regressors_needed[idx]
            
            mergecopes = Node(Merge(dimension='t'), name='merge_copes')
            wk.connect(dg, 'copes', mergecopes, 'in_files')
            
            if flamemodel != 'ols':
                mergevarcopes = Node(Merge(dimension='t'), name='merge_varcopes')
                wk.connect(dg, 'varcopes', mergevarcopes, 'in_files')
            
            mask_file = fsl.Info.standard_image('MNI152_T1_2mm_brain_mask.nii.gz')
            flame = Node(FLAMEO(), name='flameo')
            flame.inputs.mask_file =  mask_file
            flame.inputs.run_mode = flamemodel
            #flame.inputs.infer_outliers = True

            wk.connect(model, 'design_mat', flame, 'design_file')
            wk.connect(model, 'design_con', flame, 't_con_file')
            wk.connect(mergecopes, 'merged_file', flame, 'cope_file')
            if flamemodel != 'ols':
                wk.connect(mergevarcopes, 'merged_file', flame, 'var_cope_file')
            wk.connect(model, 'design_grp', flame, 'cov_split_file')
            
            if nonparametric:
                palm = Node(Function(input_names=['cope_file', 'design_file', 'contrast_file', 
                                                  'group_file', 'mask_file', 'cluster_threshold'],
                                     output_names=['palm_outputs'],
                                     function=run_palm),
                            name='palm')
                palm.inputs.cluster_threshold = 3.09
                palm.inputs.mask_file = mask_file
                palm.plugin_args = {'sbatch_args': '-p om_all_nodes -N1 -c2 --mem=10G', 'overwrite': True}
                wk.connect(model, 'design_mat', palm, 'design_file')
                wk.connect(model, 'design_con', palm, 'contrast_file')
                wk.connect(mergecopes, 'merged_file', palm, 'cope_file')
                wk.connect(model, 'design_grp', palm, 'group_file')
                
            smoothest = Node(SmoothEstimate(), name='smooth_estimate')
            wk.connect(flame, 'zstats', smoothest, 'zstat_file')
            smoothest.inputs.mask_file = mask_file
        
            cluster = Node(Cluster(), name='cluster')
            wk.connect(smoothest,'dlh', cluster, 'dlh')
            wk.connect(smoothest, 'volume', cluster, 'volume')
            cluster.inputs.connectivity = 26
            cluster.inputs.threshold = 2.3
            cluster.inputs.pthreshold = 0.05
            cluster.inputs.out_threshold_file = True
            cluster.inputs.out_index_file = True
            cluster.inputs.out_localmax_txt_file = True
            
            wk.connect(flame, 'zstats', cluster, 'in_file')
    
            ztopval = Node(ImageMaths(op_string='-ztop', suffix='_pval'),
                           name='z2pval')
            wk.connect(flame, 'zstats', ztopval,'in_file')
            
            sinker = Node(DataSink(), name='sinker')
            sinker.inputs.base_directory = os.path.join(out_dir, 'task%03d' % task, contrast[0][0])
            sinker.inputs.substitutions = [('_cope_id', 'contrast'),
                                           ('_maths_', '_reversed_')]
            
            wk.connect(flame, 'zstats', sinker, 'stats')
            wk.connect(cluster, 'threshold_file', sinker, 'stats.@thr')
            wk.connect(cluster, 'index_file', sinker, 'stats.@index')
            wk.connect(cluster, 'localmax_txt_file', sinker, 'stats.@localmax')
            if nonparametric:
                wk.connect(palm, 'palm_outputs', sinker, 'stats.palm')

            if not no_reversal:
                zstats_reverse = Node( BinaryMaths()  , name='zstats_reverse')
                zstats_reverse.inputs.operation = 'mul'
                zstats_reverse.inputs.operand_value = -1
                wk.connect(flame, 'zstats', zstats_reverse, 'in_file')
                
                cluster2=cluster.clone(name='cluster2')
                wk.connect(smoothest, 'dlh', cluster2, 'dlh')
                wk.connect(smoothest, 'volume', cluster2, 'volume')
                wk.connect(zstats_reverse, 'out_file', cluster2, 'in_file')
                
                ztopval2 = ztopval.clone(name='ztopval2')
                wk.connect(zstats_reverse, 'out_file', ztopval2, 'in_file')
                
                wk.connect(zstats_reverse, 'out_file', sinker, 'stats.@neg')
                wk.connect(cluster2, 'threshold_file', sinker, 'stats.@neg_thr')
                wk.connect(cluster2, 'index_file',sinker, 'stats.@neg_index')
                wk.connect(cluster2, 'localmax_txt_file', sinker, 'stats.@neg_localmax')
            meta_workflow.add_nodes([wk])
    return meta_workflow

if __name__ == '__main__':
    import argparse
    defstr = ' (default %(default)s)'
    parser = argparse.ArgumentParser(prog='group_multregress_openfmri.py',
                                     description=__doc__)
    parser.add_argument('-m', '--model', default=1, type=int,
                        help="Model index" + defstr)
    parser.add_argument('-t', '--task', default=[1], nargs='+',
                        type=int, help="Task index" + defstr)
    parser.add_argument("-o", "--output_dir", dest="outdir",
                        help="Output directory base")
    parser.add_argument('-d', '--datasetdir', required=True)
    parser.add_argument("-l1", "--l1_output_dir", dest="l1out_dir",
                        help="l1_output directory ")
    parser.add_argument("-w", "--work_dir", dest="work_dir",
                        help="Output directory base")
    parser.add_argument("-p", "--plugin", dest="plugin",
                        default='Linear',
                        help="Plugin to use" + defstr)
    parser.add_argument("--plugin_args", dest="plugin_args",
                        help="Plugin arguments")
    parser.add_argument("--norev",action='store_true',
                        help="do not generate reverse contrasts")
    parser.add_argument("--use_spm",action='store_true', default=False,
                        help="use spm estimation results from 1st level")
    parser.add_argument("--nonparametric", action='store_true', default=False,
                        help="Run non-parametric estimation using palm" + defstr)
    parser.add_argument('-f','--flame', dest='flamemodel', default='flame1',
                        choices=('ols', 'flame1', 'flame12'),
                        help='tool to use for dicom conversion' + defstr)
    parser.add_argument("--sleep", dest="sleep", default=60., type=float,
                        help="Time to sleep between polls" + defstr)
    parser.add_argument("-s", "--sub_list_file", dest="sub_list_file",
                        default=None,
                        help="sub list file" + defstr)
    parser.add_argument("-b", "--behav_file", dest="behav_file",
                        default=None,
                        help="behav_file " + defstr)
    parser.add_argument("-g", "--group_contrast_file", dest="group_contrast_file",
                        default=None,
                        help="group_contrast_file" + defstr)    
    parser.add_argument("--crashdump_dir", dest="crashdump_dir",
                        help="Crashdump dir", default=None)    
                        
    args = parser.parse_args()
    outdir = args.outdir
    work_dir = os.getcwd()

    if args.work_dir:
        work_dir = os.path.abspath(args.work_dir)
    if args.outdir:
        outdir = os.path.abspath(outdir)
    if args.l1out_dir:
        l1_outdir=os.path.abspath(args.l1out_dir)
    else:
        l1_outdir=os.path.join(args.datasetdir, 'l1output')

    outdir = os.path.join(outdir, 'model%03d' % args.model)

    wf = group_multregress_openfmri(model_id=args.model,
                                    task_id=args.task,
                                    l1output_dir=l1_outdir,
                                    out_dir=outdir,
                                    dataset_dir=os.path.abspath(args.datasetdir),
                                    no_reversal=args.norev,
                                    flamemodel=args.flamemodel,
                                    nonparametric=args.nonparametric,
                                    use_spm=args.use_spm,
                                    sub_list_file=args.sub_list_file, 
                                    behav_file=args.behav_file, 
                                    group_contrast_file=args.group_contrast_file)
    wf.config['execution']['poll_sleep_duration'] = args.sleep
    
    if not (args.crashdump_dir is None):
        wf.config['execution']['crashdump_dir'] = args.crashdump_dir    

    if args.plugin_args:
        wf.run(args.plugin, plugin_args=eval(args.plugin_args))
    else:
        wf.run(args.plugin)
