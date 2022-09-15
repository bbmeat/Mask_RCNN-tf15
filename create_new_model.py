import tensorflow as tf
from tensorflow.tools.graph_transforms import TransformGraph
'''
转化frozen_inference_graph.pb到新的.pb,以解决assert(graph_def.node[0].op == 'Placeholder')的问题
'''
with tf.gfile.FastGFile('G:/Python/Mask_RCNN-tf15/Model/mask_rcnn_landing.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    graph_def = TransformGraph(graph_def, ['image_tensor'], ['num_detections', 'detection_scores', 'detection_boxes', 'detection_classes', 'detection_masks'], ['sort_by_execution_order'])
    with tf.gfile.FastGFile('G:/Python/Mask_RCNN-tf15/Model/frozen_inference_graph_converted.pb', 'wb') as f:
        f.write(graph_def.SerializeToString())#保存新的pb