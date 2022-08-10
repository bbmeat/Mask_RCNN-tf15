// annotations
        for json_file in imglist:
            with json_file.open() as f:
                json_result = json.load(f)
            if type(json_result['shapes']) is dict:
                polygons = [r['points'] for r in json_result['shapes'].values()]
                shapes=[r['label'] for r in json_result['shapes']]
            else:
                polygons = [r['points'] for r in json_result['shapes']]
                shapes=[r['label'] for r in json_result['shapes']]
                #shapes=

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            # labelme include the height and weight

            image_path = os.path.join(dataset_dir, json_result['imagePath'])
            height=json_result['imageHeight']
            width = json_result['imageWidth']

            self.add_image(
                "shapes",
                image_id=json_result['imagePath'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                shapes=shapes)

def load_mask(self, image_id):
    """Generate instance masks for an image.
   Returns:
    masks: A bool array of shape [height, width, instance count] with
        one mask per instance.
    class_ids: a 1D array of class IDs of the instance masks.
    """
    # If not a balloon dataset image, delegate to parent class.
    # image_info = self.image_info[image_id]
    # if image_info["source"] != "balloon":
    #    return super(self.__class__, self).load_mask(image_id)
    info = self.image_info[image_id]
    shapes = info['shapes']
    count = len(shapes)  # number of object

    # Convert polygons to a bitmap mask of shape
    # [height, width, instance_count]

    mask = np.zeros([info["height"], info["width"], count],
                    dtype=np.uint8)
    for i, p in enumerate(info['polygons']):
        p_y = []
        p_x = []
        for point in p:
            p_y.append(point[1])
            p_x.append(point[0])
        rr, cc = skimage.draw.polygon(p_y, p_x)
        mask[rr, cc, i:i + 1] = 1

    # Handle occlusions
    occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
    for i in range(count - 2, -1, -1):
        mask[:, :, i] = mask[:, :, i] * occlusion
        occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))

    # Map class names to class IDs.
    class_ids = np.array([self.class_names.index(s) for s in shapes])
    return mask.astype(np.bool), class_ids.astype(np.int32)
