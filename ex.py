import pyrealsense2 as rs
import numpy as np
import cv2
import tensorflow as tf
import os
import uuid
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3)

#
W = 848
H = 480
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

# save video
SAVE_VIDEO_ROOT="serialize/videos"
SAVE_VIDEO_PATH = SAVE_VIDEO_ROOT + os.sep + 'strawberryDetection%s.avi' % str(uuid.uuid4())[-10:]
four_cc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(SAVE_VIDEO_PATH, four_cc, 10.0, (W, H), True)

print("[INFO] start streaming...")
pipeline.start(config)

aligned_stream = rs.align(rs.stream.color) # alignment between color and depth
point_cloud = rs.pointcloud()

print("[INFO] loading model...")
PATH_TO_CKPT = r"model/strawberry_detection/frozen_inference_graph2-2.pb"
# download model from: https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API#run-network-in-opencv

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v1.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.compat.v1.import_graph_def(od_graph_def, name='')
    # sess = tf.compat.v1.Session(graph=detection_graph)
    sess = tf.compat.v1.Session(graph=detection_graph,config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
# code source of tensorflow model loading: https://www.geeksforgeeks.org/ml-training-image-classifier-using-tensorflow-object-detection-api/

try:
    while True:
        frames = pipeline.wait_for_frames()
        frames = aligned_stream.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        points = point_cloud.calculate(depth_frame)
        verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, W, 3)  # xyz

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        scaled_size = (int(W), int(H))
        # expand image dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        image_expanded = np.expand_dims(color_image, axis=0)
        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
                                                 feed_dict={image_tensor: image_expanded})
        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)

        print("[INFO] drawing bounding box on detected objects...")
        print("[INFO] each detected object has a unique color")
        class_type = ('START', 'strawberry', 'other2', "other3")
        countTarget=0   # -----chiry
        for idx in range(int(num)):
            class_ = classes[idx]
            score = scores[idx]
            box = boxes[idx]
            print("[DEBUG box:]",box)
            print(" [DEBUG] class : ", class_, "idx : ", idx, "num : ", num)
            # count object
            if score > 0.8 and class_ == 1: # 1 for rice
                countTarget+=1   # -----chiry
                left = box[1] * W
                top = box[0] * H
                right = box[3] * W
                bottom = box[2] * H

                width = right - left
                height = bottom - top
                bbox = (int(left), int(top), int(width), int(height))
                p1 = (int(bbox[0]), int(bbox[1]+18)) # y roi 区域缩小15
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]-18)) # y roi 区域缩小10
                # draw box
                cv2.rectangle(color_image, p1, p2, (255,0,0), 2, 1)

                # x,y,z of bounding box
                obj_points = verts[int(bbox[1]+18):int(bbox[1] + bbox[3]-18), int(bbox[0]):int(bbox[0] + bbox[2])].reshape(-1, 3)
                zs = obj_points[:, 2]

                z = np.median(zs)

                ys = obj_points[:, 1]
                ys = np.delete(ys, np.where(
                    (zs < z - 1) | (zs > z + 1)))  # take only y for close z to prevent including background

                my = np.amin(ys, initial=1)
                My = np.amax(ys, initial=-1)

                height = (My - my)  # add next to rectangle print of height using cv library
                height = float("{:.5f}".format(height*100))
                if(height>10):
                    height='processing'
                print("[INFO] object height is: ", height, "[cm]")
                height_txt = class_type[class_]+" : "+str(height) + "[cm]"

                # Write some Text
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (p1[0], p1[1] + 20)
                fontScale = 1
                fontColor = (255, 255, 255)
                lineType = 2
                cv2.putText(color_image, height_txt,
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            lineType)
                # Write some Text      -----chiry
                if cv2.waitKey(1) & 0xff == ord('p') :
                    tmpFilename="tmp/strawberry/1/{}.png".format(uuid.uuid4().__str__()[-5:])
                    cv2.imwrite(tmpFilename,color_image)
                    print(tmpFilename)
        point2 = (10, 30)
        fontColor2 = (255, 255,0)
        # cv2.putText(color_image, "The number of strawberry:{}".format(countTarget),
        #             point2,
        #             0,
        #             1,
        #             fontColor2,
        #             2)

        # Show imagesq
        # cv2.namedWindow('Strawberry-monitor', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense-monitor', color_image)
        # save videos
        video_writer.write(color_image)

        # cv2.waitKey(1)
        if cv2.waitKey(1) & 0xff == ord('q') or cv2.waitKey(1)== 27:
            cv2.destroyAllWindows()
            break
finally:
    print("Application streaming closed...")
    # Stop streaming
    video_writer.release()
    pipeline.stop()