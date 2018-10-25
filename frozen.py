import tensorflow as tf
import os
from im2txt import inference_wrapper
from im2txt import configuration

# dir = os.path.dirname(os.path.realpath(__file__))
# checkpoint = tf.train.get_checkpoint_state(dir + '/output')
# input_checkpoint = checkpoint.model_checkpoint_path
# print(input_checkpoint)

# absolute_model = '/'.join(input_checkpoint.split('/')[:-1])
# print(absolute_model)

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "/output",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")



g = tf.Graph()
with g.as_default():
  model = inference_wrapper.InferenceWrapper()
  restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                             FLAGS.checkpoint_path)
g.finalize()
with tf.Session(graph=g) as sess:
    restore_fn(sess)
    # saver = tf.train.import_meta_graph(input_checkpoint + '.meta',
    #                                    clear_devices=True)

    # saver.restore(sess, input_checkpoint)

    # 打印图中的变量，查看要保存的
    # for op in tf.get_default_graph().get_operations():
    #     print(op.name, op.values())
    # print([n.name for n in tf.get_default_graph().as_graph_def().node])

    global_step=tf.train.global_step(sess,g.get_tensor_by_name('global_step:0'))
    output_grap_def = tf.graph_util.convert_variables_to_constants(sess,
                                                                   tf.get_default_graph().as_graph_def(),
                                                                   output_node_names=['lstm/initial_state','lstm/state','softmax'])
    
    output_grap=FLAGS.checkpoint_path+'/frozen_model_'+str(global_step)+'.pb'
    with tf.gfile.GFile(output_grap, 'wb') as f:
        f.write(output_grap_def.SerializeToString())
    print("%d ops in the final graph." % len(output_grap_def.node))