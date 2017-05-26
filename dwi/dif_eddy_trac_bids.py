# vi: set ft=python sts=4 ts=4 sw=4 et:
""" Process BIDS-format diffusion data combining eddy and tracula
using Nipype/Freesurfer.
Assumes you have a tracula_config file in the same directory
---
gablab
"""

from nipype import config
#config.enable_provenance()

import os
from glob import glob
import argparse

from nipype import (Node, Workflow, MapNode, Function,
                    IdentityInterface, DataSink)
from nipype.interfaces.fsl import (Eddy, TOPUP, BET)
from nipype.interfaces.io import SelectFiles

from bids.grabbids import BIDSLayout

def get_diff_info(layout, files):
    """ Return phase encoding direction for each dwi """
    pe_key = []
    for dwi in files:
        pe_key.append(layout.get_metadata(dwi)['PhaseEncodingDirection'])
        try:
            readout = layout.get_metadata(dwi)['TotalReadoutTime']
        except KeyError:
            readout = ((layout.get_metadata(dwi)['dcmmeta_shape'][0] - 1)
                        * layout.get_metadata(dwi)['EffectiveEchoSpacing'])
    return pe_key, readout

def create_diffusion_workflow(bids_dir, work_dir, trac_dir, config_dir,
                              session=None, subjects=None):
    """
    Creates metaworkflow and iterates through subjects - grabs diffusion
    weighted images, bvals, and bvecs from each subject (using pybids) and
    analyzes at subject level
    """

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    subjs_to_analyze = []
    layout = BIDSLayout(bids_dir)
    if subjects:
        subjs_to_analyze = ['sub-{}'.format(val) for val in subjects]
    else:
        subjs_to_analyze = ['sub-{}'.format(val) for val in layout.get_subjects()]

    print(subjs_to_analyze)

    trac_config = os.path.join(config_dir, 'tracula_config')

    meta_wf = Workflow(name='meta_level')

    for subj_label in subjs_to_analyze:
        subj = subj_label.replace('sub-','')
        print(subj)
        if session:
            dwis = sorted([f.filename for f in
                           layout.get(subject=subj, type='dwi',
                                      session=session_id,
                                      extensions=['nii.gz','nii'])])
            bvals = sorted([f.filename for f in
                            layout.get(subject=subj, type='dwi',
                                       session=session_id,
                                       extensions=['bval'])])
            bvecs = sorted([f.filename for f in
                            layout.get(subject=subj, type='dwi',
                                       session=session_id,
                                       extensions=['bvec'])])
        else:
            dwis = sorted([f.filename for f in
                           layout.get(subject=subj, type='dwi',
                                      extensions=['nii.gz','nii'])])
            bvals = sorted([f.filename for f in
                            layout.get(subject=subj, type='dwi',
                                       extensions=['bval'])])
            bvecs = sorted([f.filename for f in
                            layout.get(subject=subj, type='dwi',
                                       extensions=['bvec'])])

        name = '{}_diffusion'.format(subj_label)
        pe_key, readout = get_diff_info(layout, dwis)
        kwargs = dict(sid=subj_label,
                      dwis=dwis,
                      bvals=bvals,
                      bvecs=bvecs,
                      pe_key=pe_key,
                      readout=readout,
                      tracula_config=trac_config,
                      tracula_dir=os.path.join(trac_dir, subj_label),
                      name=name)
        wf = analyze_diffusion(**kwargs)
        meta_wf.add_nodes([wf])

    return meta_wf


def analyze_diffusion(sid, dwis, bvals, bvecs, pe_key, readout,
                      tracula_config, tracula_dir, name='eddy_trac_csd'):
    """ annotate when finished """

    if not os.path.exists(tracula_dir):
        os.makedirs(tracula_dir)
    wf = Workflow(name=name)

    infosource = Node(IdentityInterface(fields=['dwis', 'bvals', 'bvecs']),
                      name='inputnode')
    infosource.inputs.dwis = dwis
    infosource.inputs.bvals = bvals
    infosource.inputs.bvecs = bvecs

    wf.connect(infosource, 'dwis', preproc, 'in_files')
    wf.connect(infosource, 'bvals', preproc, 'bval_files')
    wf.connect(infosource, 'bvecs', preproc, 'bvec_files')


    # Find readout
    def create_files(in_files, bval_files, bvec_files, order, readout):
        """ Set up for processing """
        import numpy as np
        import os
        from nipype.interfaces.fsl import Merge, Split
        bvecs = []
        bvals = []
        indices = []
        acqparams = []
        b0indices = []
        for idx, fname in enumerate(bval_files):
            vals = np.genfromtxt(fname).flatten()
            bvals.extend(vals.tolist())
            min_val = min(vals)
            b0idx = np.nonzero(vals==min_val)[0]
            b0indices.extend(len(indices) + b0idx)
            index = np.zeros(vals.shape)
            index[b0idx] = 1
            index = np.cumsum(index)
            indices.extend(len(acqparams) + index)
            acqp = {'j': [0, -1, 0, '{:.4f}'.format(readout)],
                    'j-': [0, 1, 0, '{:.4f}'.format(readout)]}[order[idx]]
            for _ in range(len(b0idx)):
                acqparams.append(acqp)
            vals = np.genfromtxt(bvec_files[idx])
            if vals.shape[0] == 3:
                vals = vals.T
            bvecs.extend(vals.tolist())
        merged_file = os.path.join(os.getcwd(), 'merged.nii.gz')
        Merge(in_files=in_files, dimension='t', output_type='NIFTI_GZ', merged_file=merged_file).run()
        merged_bvals = os.path.join(os.getcwd(), 'merged.bvals')
        np.savetxt(merged_bvals, bvals, '%.1f')
        merged_bvecs = os.path.join(os.getcwd(), 'merged.bvecs')
        np.savetxt(merged_bvecs, bvecs, '%.10f %.10f %.10f')
        merged_index = os.path.join(os.getcwd(), 'merged.index')
        np.savetxt(merged_index, indices, '%d')
        acq_file = os.path.join(os.getcwd(), 'b0_acq.txt')
        np.savetxt(acq_file, acqparams, '%d %d %d %f')
        b0file = os.path.join(os.getcwd(), 'b0_merged.nii.gz')
        res = Split(in_file=merged_file, dimension='t').run()
        Merge(in_files=np.array(res.outputs.out_files)[b0indices].tolist(), dimension='t',
              output_type='NIFTI_GZ', merged_file=b0file).run()
        return merged_file, merged_bvals, merged_bvecs, merged_index, acq_file, b0file

    preproc = Node(Function(input_names=['in_files', 'bval_files',
                                         'bvec_files', 'order'],
                            output_names=['merged_file', 'merged_bvals',
                                          'merged_bvecs', 'merged_index',
                                          'acq_file', 'b0file'],
                            function=create_files),
                   name='preproc')
    preproc.inputs.order = pe_key

    def rotate_bvecs(bvec_file, par_file):
        """ Rotates bvecs """
        import os
        import numpy as np
        pars = np.genfromtxt(par_file)
        bvecs = np.genfromtxt(bvec_file)
        new_bvecs = []
        rotfunc = lambda x: np.array([[np.cos(x), np.sin(x)],
                                      [-np.sin(x), np.cos(x)]])
        for idx, vector in enumerate(bvecs):
            par = pars[idx]
            Rx = np.eye(3)
            Rx[1:3, 1:3] = rotfunc(par[3])
            Ry = np.eye(3)
            Ry[(0, 0, 2, 2), (0, 2, 0, 2)] = rotfunc(par[4]).ravel()
            Rz = np.eye(3)
            Rz[0:2, 0:2] = rotfunc(par[5])
            R = np.linalg.inv(Rx.dot(Ry.dot(Rz)))
            new_bvecs.append(R.dot(vector.T).tolist())
        new_bvec_file = os.path.join(os.getcwd(), 'rotated.bvecs')
        np.savetxt(new_bvec_file, new_bvecs, '%.10f %.10f %.10f')
        return new_bvec_file

    rotate = Node(Function(input_names=['bvec_file', 'par_file'],
                            output_names=['bvec_file'],
                            function=rotate_bvecs), name='rotate')

    wf.connect(preproc, 'merged_bvecs', rotate, 'bvec_file')
    wf.connect(eddy, 'out_parameter', rotate, 'par_file')

    """
    TRACULA
    """

    def run_prep(sid, template, nifti, bvec, bval, tracula_dir):
        """ Runs trac-all from command line, could convert into interface """
        from glob import glob
        import oss
        from string import Template
        with open(template, 'rt') as fp:
            tpl = Template(fp.read())
        out = tpl.substitute(subjects=sid, bvec=bvec, bval=bval, niftis=nifti)
        config_file = os.path.join(os.getcwd(), 'config_%s' % sid)
        with open(config_file, 'wt') as fp:
            fp.write(out)
        from nipype.interfaces.base import CommandLine
        from nipype.pipeline.engine import Node
        node = Node(CommandLine('trac-all -prep -c %s -no-isrunning -noqa' % config_file,
                                terminal_output='allatonce'),
                    name='trac-prep-%s' % sid)
        node.base_dir = os.getcwd()
        node.run()
        dwi_file = os.path.join(tracula_dir, sid, 'dmri', 'dwi.nii.gz')
        return sid, config_file, dwi_file

    node1 = Node(Function(input_names=['sid', 'template', 'nifti', 'bvec', 'bval', 'tracula_dir'],
                          output_names=['sid', 'config_file', 'dwi_file'],
                          function=run_prep),
                 name='trac-prep')

    def run_bedpost(sid, tracula_dir, dwi_file):
        """ """
        import os
        import shutil
        from nipype.interfaces.base import CommandLine
        pwd = os.getcwd()
        os.chdir(os.path.join(tracula_dir, sid))
        if scanner == 'trio':
            NJOBS, model = [1, 2]
        elif scanner == 'prisma':
            NJOBS, model = []
        bedpost = CommandLine('bedpostx_gpu dmri -NJOBS 1 --model=2 --rician', terminal_output='allatonce')
        if os.path.exists(os.path.join(os.getcwd(), 'dmri.bedpostX')):
            shutil.rmtree(os.path.join(os.getcwd(), 'dmri.bedpostX'))
        bedpost.run()
        bedpost_file = os.path.join(os.getcwd(), 'dmri.bedpostX', 'dyads2.nii.gz')
        os.chdir(pwd)
        return sid, bedpost_file

    #bedpost = create_bedpostx_pipeline()

    node2 = Node(Function(input_names=['sid', 'tracula_dir', 'dwi_file'],
                         output_names=['sid', 'bedpost_file'], function=run_bedpost),
                 name='trac-bedp')
    node2.inputs.tracula_dir = tracula_dir
    node2.plugin_args = {'sbatch_args': '--gres=gpu:1',
                          'overwrite': True}
    wf.connect(node1, 'sid', node2, 'sid')
    wf.connect(node1, 'dwi_file', node2, 'dwi_file')



    def run_path(sid, config_file, bedpost_file):
        import os
        from nipype.interfaces.base import CommandLine
        from nipype.pipeline.engine import Node
        node = Node(CommandLine('trac-all -path -c %s -no-isrunning' % config_file, terminal_output='file'),
                    name='trac-path-%s' % sid)
        node.base_dir = os.getcwd()
        node.run()
        return sid

    """
    dipy CSD reconstruction
    """

    def dmri_recon(sid, tracula_dir, dwi_file, recon='csd', num_threads=1):
        import os
        oldval = None
        if 'MKL_NUM_THREADS' in os.environ:
            oldval = os.environ['MKL_NUM_THREADS']
        os.environ['MKL_NUM_THREADS'] = '%d' % num_threads
        ompoldval = None
        if 'OMP_NUM_THREADS' in os.environ:
            ompoldval = os.environ['OMP_NUM_THREADS']
        os.environ['OMP_NUM_THREADS'] = '%d' % num_threads
        import nibabel as nib
        import numpy as np
        from glob import glob


        fimg = os.path.abspath(glob(os.path.join(tracula_dir, '%s/dmri/dwi.nii.gz' % sid))[0])
        fbvec = os.path.abspath(glob(os.path.join(tracula_dir, '%s/dmri/bvecs' % sid))[0])
        fbval = os.path.abspath(glob(os.path.join(tracula_dir, '%s/dmri/bvals' % sid))[0])
        img = nib.load(fimg)
        data = img.get_data()

        prefix = sid

        from dipy.io import read_bvals_bvecs
        from dipy.core.gradients import vector_norm
        bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
        b0idx = []
        for idx, val in enumerate(bvals):
            if val > 1:
                b0idx.append(idx)
        bvecs[b0idx, :] = bvecs[b0idx, :]/vector_norm(bvecs[b0idx])[:, None]

        from dipy.core.gradients import gradient_table
        gtab = gradient_table(bvals, bvecs)

        from dipy.reconst.csdeconv import auto_response
        response, ratio = auto_response(gtab, data, roi_radius=10, fa_thr=0.7)

        #from dipy.segment.mask import median_otsu
        #b0_mask, mask = median_otsu(data[:, :, :, b0idx].mean(axis=3).squeeze(), 4, 4)

        fmask1 = os.path.abspath(glob(os.path.join(tracula_dir,
                                                  '%s/dlabel/diff/aparc+aseg_mask.bbr.nii.gz' % sid))[0])
        fmask2 = os.path.abspath(glob(os.path.join(tracula_dir,
                                                  '%s/dlabel/diff/notventricles.bbr.nii.gz' % sid))[0])
        mask = (nib.load(fmask1).get_data() > 0.5) * nib.load(fmask2).get_data()

        useFA = True
        if recon == 'csd':
            from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
            model = ConstrainedSphericalDeconvModel(gtab, response)
            useFA = True
        elif recon == 'csa':
            from dipy.reconst.shm import CsaOdfModel, normalize_data
            model = CsaOdfModel(gtab, 4)
            useFA = False
        else:
            raise ValueError('only csd, csa supported currently')
            from dipy.reconst.dsi import (DiffusionSpectrumDeconvModel,
                                          DiffusionSpectrumModel)
            model = DiffusionSpectrumDeconvModel(gtab)
        #fit = model.fit(data)

        from dipy.data import get_sphere
        sphere = get_sphere('symmetric724')
        #odfs = fit.odf(sphere)

        from dipy.reconst.peaks import peaks_from_model
        peaks = peaks_from_model(model=model,
                                 data=data,
                                 sphere=sphere,
                                 mask=mask,
                                 return_sh=True,
                                 return_odf=False,
                                 normalize_peaks=True,
                                 npeaks=5,
                                 relative_peak_threshold=.5,
                                 min_separation_angle=25,
                                 parallel=num_threads > 1,
                                 nbr_processes=num_threads)

        from dipy.reconst.dti import TensorModel
        tenmodel = TensorModel(gtab)
        tenfit = tenmodel.fit(data, mask)

        from dipy.reconst.dti import fractional_anisotropy
        FA = fractional_anisotropy(tenfit.evals)
        FA[np.isnan(FA)] = 0
        fa_img = nib.Nifti1Image(FA, img.get_affine())
        tensor_fa_file = os.path.abspath('%s_tensor_fa.nii.gz' % (prefix))
        nib.save(fa_img, tensor_fa_file)

        evecs = tenfit.evecs
        evec_img = nib.Nifti1Image(evecs, img.get_affine())
        tensor_evec_file = os.path.abspath('%s_tensor_evec.nii.gz' % (prefix))
        nib.save(evec_img, tensor_evec_file)

        #from dipy.reconst.dti import quantize_evecs
        #peak_indices = quantize_evecs(tenfit.evecs, sphere.vertices)
        #eu = EuDX(FA, peak_indices, odf_vertices = sphere.vertices, a_low=0.2, seeds=10**6, ang_thr=35)

        fa_img = nib.Nifti1Image(peaks.gfa, img.get_affine())
        model_gfa_file = os.path.abspath('%s_%s_gfa.nii.gz' % (prefix, recon))
        nib.save(fa_img, model_gfa_file)

        from dipy.tracking.eudx import EuDX
        if useFA:
            eu = EuDX(FA, peaks.peak_indices[..., 0], odf_vertices = sphere.vertices,
                      a_low=0.1, seeds=10**6, ang_thr=45)
        else:
            eu = EuDX(peaks.gfa, peaks.peak_indices[..., 0], odf_vertices = sphere.vertices,
                      a_low=0.1, seeds=10**6, ang_thr=45)

        #import dipy.tracking.metrics as dmetrics
        streamlines = ((sl, None, None) for sl in eu) # if dmetrics.length(sl) > 15)

        hdr = nib.trackvis.empty_header()
        hdr['voxel_size'] = fa_img.get_header().get_zooms()[:3]
        hdr['voxel_order'] = 'LAS'
        hdr['dim'] = FA.shape[:3]

        sl_fname = os.path.abspath('%s_%s_streamline.trk' % (prefix, recon))

        nib.trackvis.write(sl_fname, streamlines, hdr, points_space='voxel')
        if oldval:
            os.environ['MKL_NUM_THREADS'] = oldval
        else:
            del os.environ['MKL_NUM_THREADS']
        if ompoldval:
            os.environ['OMP_NUM_THREADS'] = ompoldval
        else:
            del os.environ['OMP_NUM_THREADS']
        return tensor_fa_file, tensor_evec_file, model_gfa_file, sl_fname


    topup = Node(TOPUP(out_corrected='b0correct.nii.gz', numprec='float', output_type='NIFTI_GZ'), name='topup')
    wf.connect(preproc, 'acq_file', topup, 'encoding_file')
    wf.connect(preproc, 'b0file', topup, 'in_file')
    masker = Node(BET(mask=True), name='mask')
    wf.connect(topup, 'out_corrected', masker, 'in_file')

    eddy = Node(Eddy(), name='eddy')
    eddy._interface._cmd = 'eddy_openmp'
    eddy.inputs.num_threads = 4
    eddy.plugin_args = {'sbatch_args': '--mem=10G -c 4'}

    wf.connect(masker, 'mask_file', eddy, 'in_mask')
    wf.connect(preproc, 'merged_file', eddy, 'in_file')
    wf.connect(preproc, 'merged_bvals', eddy, 'in_bval')
    wf.connect(preproc, 'merged_bvecs', eddy, 'in_bvec')
    wf.connect(preproc, 'merged_index', eddy, 'in_index')
    wf.connect(preproc, 'acq_file', eddy, 'in_acqp')
    wf.connect(topup, 'out_fieldcoef', eddy, 'in_topup_fieldcoef')
    wf.connect(topup, 'out_movpar', eddy, 'in_topup_movpar')

    rotate = Node(Function(input_names=['bvec_file', 'par_file'],
                            output_names=['bvec_file'],
                            function=rotate_bvecs), name='rotate')

    wf.connect(preproc, 'merged_bvecs', rotate, 'bvec_file')
    wf.connect(eddy, 'out_parameter', rotate, 'par_file')

    node1 = Node(Function(input_names=['sid', 'template', 'nifti', 'bvec', 'bval', 'tracula_dir'],
                          output_names=['sid', 'config_file', 'dwi_file'], function=run_prep),
                name='trac-prep')
    node1.inputs.template = tracula_config
    node1.inputs.tracula_dir = tracula_dir

    wf.connect(infosource, 'subject_id', node1, 'sid')
    wf.connect(eddy, 'out_corrected', node1, 'nifti')
    wf.connect(preproc, 'merged_bvals', node1, 'bval')
    wf.connect(rotate, 'bvec_file', node1, 'bvec')

    node2 = Node(Function(input_names=['sid', 'tracula_dir', 'dwi_file'],
                         output_names=['sid', 'bedpost_file'], function=run_bedpost),
                name='trac-bedp')
    node2.inputs.tracula_dir = tracula_dir
    node2.plugin_args = {'sbatch_args': '--gres=gpu:1',
                          'overwrite': True}
    wf.connect(node1, 'sid', node2, 'sid')
    wf.connect(node1, 'dwi_file', node2, 'dwi_file')

    node3 = Node(Function(input_names=['sid', 'config_file', 'bedpost_file'],
                         output_names=['sid'], function=run_path),
                name='trac-path')
    wf.connect(node2, 'sid', node3, 'sid')
    wf.connect(node2, 'bedpost_file', node3, 'bedpost_file')
    wf.connect(node1, 'config_file', node3, 'config_file')

    tracker = Node(Function(input_names=['sid', 'tracula_dir', 'dwi_file', 'recon', 'num_threads'],
                            output_names=['tensor_fa_file', 'tensor_evec_file', 'model_gfa_file',
                                          'model_track_file'],
                            function=dmri_recon), name='tracker')
    tracker.inputs.recon = 'csd'
    tracker.inputs.tracula_dir = tracula_dir
    num_threads = 4
    tracker.inputs.num_threads = num_threads
    tracker.plugin_args = {'sbatch_args': '--mem=%dG -N 1 -c %d' % (3 * num_threads,
                                                                    num_threads),
                           'overwrite': True}

    wf.connect(node1, 'sid', tracker, 'sid')
    wf.connect(node1, 'dwi_file', tracker, 'dwi_file')

    ds = Node(DataSink(), name='sinker')
    ds.inputs.base_directory = tracula_dir
    ds.plugin_args = {'sbatch_args': '-p om_interactive -N 1 -c 2',
                      'overwrite': True}

    wf.connect(node1, 'sid', ds, 'container')
    wf.connect(preproc, 'merged_bvecs', ds, 'pre.@bvec_file')
    wf.connect(preproc, 'merged_bvals', ds, '@bval_file')
    wf.connect(rotate, 'bvec_file', ds, '@rot_bvec_file')
    wf.connect(eddy, 'out_corrected', ds, '@out_file')
    wf.connect(tracker, 'tensor_fa_file', ds, 'recon.@fa')
    wf.connect(tracker, 'tensor_evec_file', ds, 'recon.@evec')
    wf.connect(tracker, 'model_gfa_file', ds, 'recon.@gfa')
    wf.connect(tracker, 'model_track_file', ds, 'recon.@track')

    return wf

def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    defstr = ' default %(default)s'
    parser.add_argument('-d', dest='datadir', required=True,
                        help="BIDS dataset directory")
    parser.add_argument('-s', '--subjects', default=[],
                        type=str, nargs='+',
                        help="Specific subjects to run (minus sub- prefix)")
    parser.add_argument('-ss', dest='session', default=None,
                        help="Session ID (ses-[input])" + defstr)
    parser.add_argument('-c', dest='config',
                        help="Directory where tracula config file is located")
    parser.add_argument('-w', dest='workdir',
                        help="Working directory")
    parser.add_argument('-t', dest='tracdir',
                        help="Directory where tracula will output")
    parser.add_argument('-p', dest='plugin', default='MultiProc',
                        help="Plugin to use" + defstr)
    parser.add_argument('-o', dest='outdir',
                        help="Output directory")
    parser.add_argument('--plugin_args',
                        help="Plugin arguments")
    parser.add_argument('--debug', action='store_true',
                        help="Activate nipype debug mode")
    args = parser.parse_args(args)

    bids_dir = os.path.abspath(args.datadir)
    if args.workdir:
        workdir = os.path.abspath(args.workdir)
    else:
        workdir = os.path.join(os.getcwd(), 'workdir')
    if args.outdir:
        outdir = os.path.abspath(args.outdir)
    else:
        outdir = os.path.join(os.getcwd(), 'output')
    if args.tracdir:
        tracdir = os.path.abspath(args.tracdir)
    else:
        tracdir = os.path.join(bids_dir, 'derivatives', 'diffusion')
    if args.config:
        config_dir = os.path.abspath(args.config)
    else:
        config_dir = os.getcwd()
    if args.debug:
        from nipype import logging
        config.enable_debug_mode()
        logging.update_logging(config)

    wf = create_diffusion_workflow(bids_dir, workdir, tracdir,
                                   config_dir, args.session,
                                   args.subjects)
    wf.base_dir = workdir

    # Configurations
    wf.config['execution']['parameterize_dirs'] = False

    # run the workflow
    if args.plugin_args:
        wf.run(args.plugin, plugin_args=eval(args.plugin_args))
    else:
        wf.run(args.plugin)

if __name__ == '__main__':
    main()
