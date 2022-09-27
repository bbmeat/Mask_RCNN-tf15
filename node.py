import tensorflow.compat.v1 as tf

# model_path = "./Model/frozen_inference_graph.pb";
model_path = "./logs/frozen_inference_graph_converted.pb"
model_pbTXT = "./Model/mask_rcnn.pbtxt"
# 读取节点
def create_graph():
    with tf.gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

create_graph()
tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
for tensor_name in tensor_name_list:
    print(tensor_name,'\n')
