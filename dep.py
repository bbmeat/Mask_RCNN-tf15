# coding=UTF-8

import os
import random
import time
import schedule
import depthai as dai
import numpy as np
import strawberry_config
import cv2 as cv
import json
import requests
# from utility import *
from detect_mask import Mask
from requests_toolbelt.multipart.encoder import MultipartEncoder
import datetime
import shutil

config = strawberry_config.StrawberryConfig()


def fun_timer():
    print("start timer")
    # schedule.every().minute.do(main())
    # schedule.every().minute.do(main)
    schedule.every(3).seconds.do(main)
    # schedule
    # schedule.every(5).hours.do(main)

    while True:
        schedule.run_pending()
        time.sleep(1)


# get now time
def time_now():
    time_lo = time.strftime("%Y_%m_%d_%H%M%S", time.localtime())
    now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("start time:", now_time)
    return time_lo


# create local result dir_file
def create_file_path():
    local_time = time_now()
    file_path = "./img_result/strawberry_" + local_time
    folder = os.path.exists(file_path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(file_path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
        print("---  OK  ---")
        # print(file_path)
    else:
        print("---  There is this folder!  ---")
    return file_path


def clear_file():
    dest_dir = "./img_result"
    all_dir = []
    for f in list(os.listdir(dest_dir)):
        dir = "{}\\{}".format(dest_dir, f)
        if os.path.isdir(dir):
            all_dir.append(dir)

    for i in range(len(all_dir)):
        dir_create_time = time.strftime("%Y%m%d", time.localtime(os.path.getctime(all_dir[i])))
        now_time = time.strftime("%Y%m%d", time.localtime())
        del_time = datetime.date.today() - datetime.timedelta(days=60)
        del_time_str = del_time.strftime("%Y%m%d")
        if int(dir_create_time) < int(del_time_str):
            shutil.rmtree(all_dir[i])
            print("已删除文件夹 {}".format(all_dir[i]))


# save result to video
def save_video(dep_frame, rgb_frame):
    width = 640
    height = 400
    # video setting

    local_time = time_now()
    file_path = "./video/strawberry_" + local_time

    folder = os.path.exists(file_path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(file_path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
        print("---  OK  ---")
    else:
        print("---  There is this folder!  ---")

    # excel_path = file_path + "/strawberry_result.xls"
    video_path = file_path + "/strawberry_result.mp4"
    rvideo_path = file_path + "/strawberry_rgb.mp4"
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # 指定视频视频编解码器格式
    depresult = cv.VideoWriter(video_path, fourcc, 10, (width, height), True)
    rgbresult = cv.VideoWriter(rvideo_path, fourcc, 10, (width, height), True)
    depresult.write(dep_frame)
    rgbresult.write(rgb_frame)


# save img to local and return serving path
def save_img(file_path, result_frame, rgb_frame, color_dep_frame):
    dfile_name = "strawberry_result.jpg"
    rfile_name = "strawberry_image.jpg"
    cdfile_name = "strawberry_depth.jpg"
    dfile_path = file_path + "/" + dfile_name
    rfile_path = file_path + "/" + rfile_name
    cdfile_path = file_path + "/" + cdfile_name
    # print("result:", dfile_path, "\nrgb:", rfile_path, "\ndepth:", cdfile_name)
    cv.imwrite(dfile_path, result_frame, [cv.IMWRITE_JPEG_QUALITY, 100])
    cv.imwrite(rfile_path, rgb_frame, [cv.IMWRITE_JPEG_QUALITY, 100])
    cv.imwrite(cdfile_path, color_dep_frame, [cv.IMWRITE_JPEG_QUALITY, 100])
    time.sleep(2)
    rgb_res = upload_img(rfile_name, rfile_path)
    dep_res = upload_img(dfile_name, dfile_path)
    cdep_res = upload_img(cdfile_name, cdfile_path)
    return rgb_res, dep_res, cdep_res


# timer


# read json change to a dict type
def read_json_data(json_data):
    json_dict = json.loads(json_data)
    # json_str = json_dict.replace("'", '"')
    print("dict", json_dict)
    return json_dict


# zip result data to new data
def zip_json(sdict):
    # d ={}
    IDs = []
    count = 1
    for i in sdict:
        num = "fruit" + str(count)
        IDs.append(num)
        count += 1
    fruit_data = dict(zip(IDs, sdict))
    print("fruit_data", fruit_data)
    # json_data = {"image_data": sdata, "measure_data": fruit_data}
    # print("jsd", json_data)
    return fruit_data


# create random id
def generate_random_str():
    random_str = ''
    base_str = 'abcdefghigklmnopqrstuvwxyz0123456789'
    length = len(base_str) - 1
    for i in range(8):
        random_str += base_str[random.randint(0, length)]
    for j in range(2):
        random_str = random_str + "-"
        for i in range(4):
            random_str = random_str + base_str[random.randint(0, length)]
    print("str", random_str)
    return random_str


# save json data to local file
def creat_json(dev_info, img_data, strawberry_data, json_path):
    now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    batch = generate_random_str()
    lst = list(strawberry_data.values())
    json_data = {
        "device_id": dev_info,
        "create_time": now_time,
    }
    data = {
        "device_id": dev_info,
        "batch_id": batch,
        "image_path": img_data["image_path"],
        "result_path": img_data["result_path"],
        "dep_path": img_data["dep_path"],
        "obj_data": lst
    }
    json_data["data"] = data
    print("js_d", json_data)
    json_str = json.dumps(json_data, indent=4)
    with open(json_path, 'w') as f:
        f.write(json_str)


# save json_data
def save_data(dict, file_path, img_res, device_info):
    file_name = "dep_result.txt"
    json_path = file_path + "/" + file_name
    rgb_res = read_json_data(img_res[0])
    dep_res = read_json_data(img_res[1])
    cdep_res = read_json_data(img_res[2])
    img_data = {
        "image_path": rgb_res["data"],
        "result_path": dep_res["data"],
        "dep_path": cdep_res["data"]
    }
    json_data = zip_json(dict)
    creat_json(device_info, img_data, json_data, json_path)
    time.sleep(2)
    upload_json(file_name, json_path)


# upload json data to serving
def upload_json(file_name, file_path):
    strawberry_data = open(file_path, mode='rb')
    file = {
        "json_file": (file_name, strawberry_data, 'file/txt')
    }
    m = MultipartEncoder(file)
    api_url = "http://106.38.31.47:10007/edge-common/edge/jsonData"
    res = requests.post(url=api_url, data=m, headers={'Content-Type': m.content_type})
    print(res.text)


# upload img to serving
def upload_img(img_name, img_file):
    strawberry_img = open(img_file, 'rb')
    print(strawberry_img)
    m = MultipartEncoder(fields={
        'bucket_name': 'obj-measure',
        'file': (img_name, strawberry_img, 'image/jpg')
    })
    api_url = "http://106.38.31.47:10007/edge-common/edge/fileUpload"
    res = requests.request(method="post", url=api_url, data=m, headers={"Content-Type": m.content_type})
    print(res.text)
    return res.text


# create pipeline
def createPipeline():
    print("creat pipeline")
    # 可选的。如果设置(True)， ColorCamera从1080p缩小到720p。否则(False)，对齐深度自动放大到1080p
    downscaleColor = True
    fps = 30
    monoResolution = dai.MonoCameraProperties.SensorResolution.THE_720_P
    pipeline = dai.Pipeline()
    queueNames = []
    # 设置相机
    camRgb = pipeline.create(dai.node.ColorCamera)
    left = pipeline.create(dai.node.MonoCamera)
    right = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)

    lrcheck = True
    subpixel = True

    rgbOut = pipeline.create(dai.node.XLinkOut)
    disparityOut = pipeline.create(dai.node.XLinkOut)

    rgbOut.setStreamName("rgb")
    queueNames.append("rgb")
    disparityOut.setStreamName("disp")
    queueNames.append("disp")

    # Properties
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setFps(fps)
    if downscaleColor: camRgb.setIspScale(2, 3)
    # 目前，RGB需要固定的焦点来正确地与深度对齐。
    # 此值在校准过程中使用
    camRgb.initialControl.setManualFocus(130)

    left.setResolution(monoResolution)
    left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    left.setFps(fps)
    right.setResolution(monoResolution)
    right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    right.setFps(fps)

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

    # LR-check is required for depth alignment
    # stereo.setLeftRightCheck(True)
    stereo.setLeftRightCheck(lrcheck)
    stereo.setSubpixel(subpixel)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

    # Linking
    camRgb.isp.link(rgbOut.input)
    left.out.link(stereo.left)
    right.out.link(stereo.right)
    stereo.disparity.link(disparityOut.input)

    return pipeline, stereo


def main():
    pipeline, stereo = createPipeline()
    with dai.Device(pipeline) as device:
        rgb = device.getOutputQueue('rgb', maxSize=1, blocking=False)
        depthQueue = device.getOutputQueue(name="disp", maxSize=1, blocking=False)
        # text = TextHelper()
        # Mask(device)
        frames = 0
        while True:
            previrew = rgb.get()
            inDepth = depthQueue.get()
            frame = previrew.getCvFrame()
            depthFrame = inDepth.getFrame()
            gray_dep_frame = (depthFrame * (255 / stereo.initialConfig.getMaxDisparity())).astype(np.uint8)
            color_dep_frame = cv.applyColorMap(gray_dep_frame, cv.COLORMAP_AUTUMN)

            frame_num = 1
            if frames == 25:
                frame_num += 1
                file_path = create_file_path()

                data_all = Mask(device).detect_with_h5(frame, depthFrame, color_dep_frame)
                result = data_all[0][0]
                dep_frame = data_all[0][-1]
                strawberry_data = data_all[-1]
                print("sdict", strawberry_data)
                masked_image = cv.cvtColor(result, cv.COLOR_RGB2BGR)
                img_res_path = save_img(file_path, masked_image, frame, dep_frame)
                print("res", img_res_path)

                save_data(strawberry_data, file_path, img_res_path, device_info)
                break

            frames += 1
            # key = cv.waitKey(1)
            # if key == ord('q'):
            #     break
    print("work done!")


if __name__ == '__main__':
    time.sleep(10)
    (res, info) = dai.DeviceBootloader.getFirstAvailableDevice()
    if res == True:
        print('Found device with name:', info.mxid)
        device_info = info.mxid
    else:
        print('No devices found')
    # main()
    clear_file()
    fun_timer()
