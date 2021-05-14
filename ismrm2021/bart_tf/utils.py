import numpy as np
from numpy.matlib import repmat
import os
import mmap
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

def gen_weights(nSpokes, N):
    """
    generate radial compensation weight
    """
    rho = np.linspace(-0.5,0.5,N).astype('float32')
    w = abs(rho)/0.5
    w = np.transpose(repmat(w, nSpokes, 1), [1, 0])
    w = np.reshape(w, [1, N, nSpokes])
    return np.sqrt(w)

def writecfl(name, array):
    with open(name + ".hdr", "wt") as h:
        h.write('# Dimensions\n')
        for i in (array.shape):
                h.write("%d " % i)
        h.write('\n')

    size = np.prod(array.shape) * np.dtype(np.complex64).itemsize

    with open(name + ".cfl", "a+b") as d:
        os.truncate(d.fileno(), size)
        with mmap.mmap(d.fileno(), size, flags=mmap.MAP_SHARED, prot=mmap.PROT_WRITE) as mm:
            mm.write(array.astype(np.complex64).tobytes(order='F'))

def readcfl(name):
    # get dims from .hdr
    with open(name + ".hdr", "rt") as h:
        h.readline() # skip
        l = h.readline()
    dims = [int(i) for i in l.split()]

    # remove singleton dimensions from the end
    n = np.prod(dims)
    dims_prod = np.cumprod(dims)
    dims = dims[:np.searchsorted(dims_prod, n)+1]

    # load data and reshape into dims
    with open(name + ".cfl", "rb") as d:
        a = np.fromfile(d, dtype=np.complex64, count=n);
    return a.reshape(dims, order='F') # column-major

def write_tf_model(saver, sess, path, name, as_text=False, gpu_id=None):
    if saver is not None:
        saver.save(sess, os.path.join(path, name))
    tf.train.write_graph(sess.graph, path, name+'.pb', as_text)
    if gpu_id is not None:
        with open(os.path.join(path, name+'_gpu_id'), 'w+') as fs:
            for i in range(len(gpu_id)):
                fs.write(gpu_id[i])
                fs.write('\t')

def export_model(model_path, exported_path, name, as_text, use_gpu):
    
    
    gpu_id=None
    if use_gpu:
        gpu_options= tf.GPUOptions(allow_growth=True, visible_device_list='0') 
        config_proto = tf.ConfigProto(gpu_options=gpu_options)
        config_proto.gpu_options.allow_growth=True
        serialized = config_proto.SerializeToString()
        gpu_id = list(map(hex, serialized))

    sess = tf.Session(config=config_proto if use_gpu else None)
    sess.run(tf.global_variables_initializer()) # TF_GraphOperationByName(graph, "init");
    saver = tf.train.Saver()
    if model_path is not None:
        saver.restore(sess, model_path) #TF_GraphOperationByName(graph, "save/restore_all");
    write_tf_model(saver, sess, exported_path, name, as_text=as_text,gpu_id=None if not use_gpu else gpu_id)
    
    print('Exported')
