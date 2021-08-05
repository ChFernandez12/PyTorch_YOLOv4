# nuScenes dev-kit.
# Code written by Holger Caesar, 2020.

import argparse
import gc
import os
import random
from typing import List
from collections import defaultdict

import cv2
import tqdm
import base64

import json
import os.path as osp
import sys
import time
from collections import defaultdict
from typing import Any, List, Dict, Optional, Tuple, Callable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

import base64
from pycocotools import mask as cocomask



PYTHON_VERSION = sys.version_info[0]

if not PYTHON_VERSION == 3:
    raise ValueError("nuScenes dev-kit only supports Python version 3.")

# Define dictionary    
#All classes dict
#classes_dict ={'movable_object.barrier': 0, 'vehicle.construction': 1, 'movable_object.trafficcone': 2, 'vehicle.car': 3, 'human.pedestrian.construction_worker': 4, 'vehicle.truck': 5, 'human.pedestrian.adult': 6, 'vehicle.motorcycle': 7, 'vehicle.bus.rigid': 8, 'static_object.bicycle_rack': 9, 'vehicle.bicycle': 10, 'movable_object.pushable_pullable': 11, 'movable_object.debris': 12, 'animal': 13, 'vehicle.trailer': 14, 'vehicle.emergency.ambulance': 15, 'human.pedestrian.stroller': 16, 'human.pedestrian.personal_mobility': 17, 'human.pedestrian.child': 18, 'vehicle.bus.bendy': 19, 'human.pedestrian.wheelchair': 20, 'human.pedestrian.police_officer': 21, 'vehicle.emergency.police': 22, 'vehicle.ego': 23}

classes_dict ={'vehicle.construction': 0,  'vehicle.car': 1, 'human.pedestrian.construction_worker': 2, 'vehicle.truck': 3, 'human.pedestrian.adult': 2, 'vehicle.motorcycle': 4, 'vehicle.bus.rigid': 5, 'vehicle.bicycle': 6, 'vehicle.trailer': 7, 'vehicle.emergency.ambulance': 0, 'human.pedestrian.stroller': 2, 'human.pedestrian.personal_mobility': 2, 'human.pedestrian.child': 2, 'vehicle.bus.bendy': 5, 'human.pedestrian.wheelchair': 2, 'human.pedestrian.police_officer': 2, 'vehicle.emergency.police': 0}
custom_dict = True  # If True just classes on the dictionary will be used, if False all classes will be exported

#image size NuScenes
image_h = 1600
image_w = 900


def annotation_name(attributes: List[dict],
                    category_name: str,
                    with_attributes: bool = False):
    """
    Returns the "name" of an annotation, optionally including the attributes.
    :param attributes: The attribute dictionary.
    :param category_name: Name of the object category.
    :param with_attributes: Whether to print the attributes alongside the category name.
    :return: A human readable string describing the annotation.
    """
    outstr = category_name

    if with_attributes:
        atts = [attribute['name'] for attribute in attributes]
        if len(atts) > 0:
            outstr = outstr + "--" + '.'.join(atts)

    return outstr


def mask_decode(mask: dict):
    """
    Decode the mask from base64 string to binary string, then feed it to the external pycocotools library to get a mask.
    :param mask: The mask dictionary with fields `size` and `counts`.
    :return: A numpy array representing the binary mask for this class.
    """
    # Note that it is essential to copy the mask here. If we use the same variable we will overwrite the NuImage class
    # and cause the Jupyter Notebook to crash on some systems.
    new_mask = mask.copy()
    new_mask['counts'] = base64.b64decode(mask['counts'])
    return cocomask.decode(new_mask)


def get_font(fonts_valid: List[str] = None, font_size: int = 15):
    """
    Check if there is a desired font present in the user's system. If there is, use that font; otherwise, use a default
    font.
    :param fonts_valid: A list of fonts which are desirable.
    :param font_size: The size of the font to set. Note that if the default font is used, then the font size
        cannot be set.
    :return: An ImageFont object to use as the font in a PIL image.
    """
    # If there are no desired fonts supplied, use a hardcoded list of fonts which are desirable.
    if fonts_valid is None:
        fonts_valid = ['FreeSerif.ttf', 'FreeSans.ttf', 'Century.ttf', 'Calibri.ttf', 'arial.ttf']

    # Find a list of fonts within the user's system.
    fonts_in_sys = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
    # Sort the list of fonts to ensure that the desired fonts are always found in the same order.
    fonts_in_sys = sorted(fonts_in_sys)
    # Of all the fonts found in the user's system, check if any of them are desired.
    for font_in_sys in fonts_in_sys:
        if any(os.path.basename(font_in_sys) in s for s in fonts_valid):
            return ImageFont.truetype(font_in_sys, font_size)

    # If none of the fonts in the user's system are desirable, then use the default font.
    warnings.warn('No suitable fonts were found in your system. '
                  'A default font will be used instead (the font size will not be adjustable).')
    return ImageFont.load_default()


def get_colormap():
    """
    Get the defined colormap.
    :return: A mapping from the class names to the respective RGB values.
    """

    classname_to_color = {  # RGB.
        "noise": (0, 0, 0),  # Black.
        "animal": (70, 130, 180),  # Steelblue
        "human.pedestrian.adult": (0, 0, 230),  # Blue
        "human.pedestrian.child": (135, 206, 235),  # Skyblue,
        "human.pedestrian.construction_worker": (100, 149, 237),  # Cornflowerblue
        "human.pedestrian.personal_mobility": (219, 112, 147),  # Palevioletred
        "human.pedestrian.police_officer": (0, 0, 128),  # Navy,
        "human.pedestrian.stroller": (240, 128, 128),  # Lightcoral
        "human.pedestrian.wheelchair": (138, 43, 226),  # Blueviolet
        "movable_object.barrier": (112, 128, 144),  # Slategrey
        "movable_object.debris": (210, 105, 30),  # Chocolate
        "movable_object.pushable_pullable": (105, 105, 105),  # Dimgrey
        "movable_object.trafficcone": (47, 79, 79),  # Darkslategrey
        "static_object.bicycle_rack": (188, 143, 143),  # Rosybrown
        "vehicle.bicycle": (220, 20, 60),  # Crimson
        "vehicle.bus.bendy": (255, 127, 80),  # Coral
        "vehicle.bus.rigid": (255, 69, 0),  # Orangered
        "vehicle.car": (255, 158, 0),  # Orange
        "vehicle.construction": (233, 150, 70),  # Darksalmon
        "vehicle.emergency.ambulance": (255, 83, 0),
        "vehicle.emergency.police": (255, 215, 0),  # Gold
        "vehicle.motorcycle": (255, 61, 99),  # Red
        "vehicle.trailer": (255, 140, 0),  # Darkorange
        "vehicle.truck": (255, 99, 71),  # Tomato
        "flat.driveable_surface": (0, 207, 191),  # nuTonomy green
        "flat.other": (175, 0, 75),
        "flat.sidewalk": (75, 0, 75),
        "flat.terrain": (112, 180, 60),
        "static.manmade": (222, 184, 135),  # Burlywood
        "static.other": (255, 228, 196),  # Bisque
        "static.vegetation": (0, 175, 0),  # Green
        "vehicle.ego": (255, 240, 245)
    }

    return classname_to_color

class NuImages:
    """
    Database class for nuImages to help query and retrieve information from the database.
    """

    def __init__(self,
                 version: str = 'v1.0-mini',
                 dataroot: str = '/data/sets/nuimages',
                 lazy: bool = True,
                 verbose: bool = False):
        """
        Loads database and creates reverse indexes and shortcuts.
        :param version: Version to load (e.g. "v1.0-train", "v1.0-val", "v1.0-test", "v1.0-mini").
        :param dataroot: Path to the tables and data.
        :param lazy: Whether to use lazy loading for the database tables.
        :param verbose: Whether to print status messages during load.
        """
        self.version = version
        self.dataroot = dataroot
        self.lazy = lazy
        self.verbose = verbose

        self.table_names = ['attribute', 'calibrated_sensor', 'category', 'ego_pose', 'log', 'object_ann', 'sample',
                            'sample_data', 'sensor', 'surface_ann']

        assert osp.exists(self.table_root), 'Database version not found: {}'.format(self.table_root)

        start_time = time.time()
        if verbose:
            print("======\nLoading nuImages tables for version {}...".format(self.version))

        # Init reverse indexing.
        self._token2ind: Dict[str, Optional[dict]] = dict()
        for table in self.table_names:
            self._token2ind[table] = None

        # Load tables directly if requested.
        if not self.lazy:
            # Explicitly init tables to help the IDE determine valid class members.
            self.attribute = self.__load_table__('attribute')
            self.calibrated_sensor = self.__load_table__('calibrated_sensor')
            self.category = self.__load_table__('category')
            self.ego_pose = self.__load_table__('ego_pose')
            self.log = self.__load_table__('log')
            self.object_ann = self.__load_table__('object_ann')
            self.sample = self.__load_table__('sample')
            self.sample_data = self.__load_table__('sample_data')
            self.sensor = self.__load_table__('sensor')
            self.surface_ann = self.__load_table__('surface_ann')

        self.color_map = get_colormap()

        if verbose:
            print("Done loading in {:.3f} seconds (lazy={}).\n======".format(time.time() - start_time, self.lazy))

    # ### Internal methods. ###

    def __getattr__(self, attr_name: str):
        """
        Implement lazy loading for the database tables. Otherwise throw the default error.
        :param attr_name: The name of the variable to look for.
        :return: The dictionary that represents that table.
        """
        if attr_name in self.table_names:
            return self._load_lazy(attr_name, lambda tab_name: self.__load_table__(tab_name))
        else:
            raise AttributeError("Error: %r object has no attribute %r" % (self.__class__.__name__, attr_name))

    def get(self, table_name: str, token: str):
        """
        Returns a record from table in constant runtime.
        :param table_name: Table name.
        :param token: Token of the record.
        :return: Table record. See README.md for record details for each table.
        """
        assert table_name in self.table_names, "Table {} not found".format(table_name)

        return getattr(self, table_name)[self.getind(table_name, token)]

    def getind(self, table_name: str, token: str):
        """
        This returns the index of the record in a table in constant runtime.
        :param table_name: Table name.
        :param token: Token of the record.
        :return: The index of the record in table, table is an array.
        """
        # Lazy loading: Compute reverse indices.
        if self._token2ind[table_name] is None:
            self._token2ind[table_name] = dict()
            for ind, member in enumerate(getattr(self, table_name)):
                self._token2ind[table_name][member['token']] = ind

        return self._token2ind[table_name][token]

    @property
    def table_root(self):
        """
        Returns the folder where the tables are stored for the relevant version.
        """
        return osp.join(self.dataroot, self.version)

  
    def _load_lazy(self, attr_name: str, loading_func: 'Callable'):
        """
        Load an attribute and add it to self, if it isn't already loaded.
        :param attr_name: The name of the attribute to be loaded.
        :param loading_func: The function used to load it if necessary.
        :return: The loaded attribute.
        """
        if attr_name in self.__dict__.keys():
            return self.__getattribute__(attr_name)
        else:
            attr = loading_func(attr_name)
            self.__setattr__(attr_name, attr)
            return attr

    def __load_table__(self, table_name):
        """
        Load a table and return it.
        :param table_name: The name of the table to load.
        :return: The table dictionary.
        """
        start_time = time.time()
        table_path = osp.join(self.table_root, '{}.json'.format(table_name))
        assert osp.exists(table_path), 'Error: Table %s does not exist!' % table_name
        with open(table_path) as f:
            table = json.load(f)
        end_time = time.time()

        # Print a message to stdout.
        if self.verbose:
            print("Loaded {} {}(s) in {:.3f}s,".format(len(table), table_name, end_time - start_time))

        return table

    def shortcut(self, src_table: str, tgt_table: str, src_token: str):
        """
        Convenience function to navigate between different tables that have one-to-one relations.
        E.g. we can use this function to conveniently retrieve the sensor for a sample_data.
        :param src_table: The name of the source table.
        :param tgt_table: The name of the target table.
        :param src_token: The source token.
        :return: The entry of the destination table corresponding to the source token.
        """
        if src_table == 'sample_data' and tgt_table == 'sensor':
            sample_data = self.get('sample_data', src_token)
            calibrated_sensor = self.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
            sensor = self.get('sensor', calibrated_sensor['sensor_token'])

            return sensor
        elif (src_table == 'object_ann' or src_table == 'surface_ann') and tgt_table == 'sample':
            src = self.get(src_table, src_token)
            sample_data = self.get('sample_data', src['sample_data_token'])
            sample = self.get('sample', sample_data['sample_token'])

            return sample
        else:
            raise Exception('Error: Shortcut from %s to %s not implemented!' % (src_table, tgt_table))

    def check_sweeps(self, filename: str):
        """
        Check that the sweeps folder was downloaded if required.
        :param filename: The filename of the sample_data.
        """
        assert filename.startswith('samples') or filename.startswith('sweeps'), \
            'Error: You passed an incorrect filename to check_sweeps(). Please use sample_data[''filename''].'

        if 'sweeps' in filename:
            sweeps_dir = osp.join(self.dataroot, 'sweeps')
            if not osp.isdir(sweeps_dir):
                raise Exception('Error: You are missing the "%s" directory! The devkit generally works without this '
                                'directory, but you cannot call methods that use non-keyframe sample_datas.'
                                % sweeps_dir)


    # ### Getter methods. ###

    def get_sample_content(self,
                           sample_token: str):
        """
        For a given sample, return all the sample_datas in chronological order.
        :param sample_token: Sample token.
        :return: A list of sample_data tokens sorted by their timestamp.
        """
        sample = self.get('sample', sample_token)
        key_sd = self.get('sample_data', sample['key_camera_token'])

        # Go forward.
        cur_sd = key_sd
        forward = []
        while cur_sd['next'] != '':
            cur_sd = self.get('sample_data', cur_sd['next'])
            forward.append(cur_sd['token'])

        # Go backward.
        cur_sd = key_sd
        backward = []
        while cur_sd['prev'] != '':
            cur_sd = self.get('sample_data', cur_sd['prev'])
            backward.append(cur_sd['token'])

        # Combine.
        result = backward[::-1] + [key_sd['token']] + forward

        return result

   

    # ### Rendering methods. ###

    def render_image(self,
                     sd_token: str,
                     annotation_type: str = 'all',
                     with_category: bool = False,
                     with_attributes: bool = False,
                     object_tokens: List[str] = None,
                     surface_tokens: List[str] = None,
                     render_scale: float = 1.0,
                     box_line_width: int = -1,
                     font_size: int = None,
                     out_path: str = None,
                     labels_path: str = None,
                     ) :
        """
        Renders an image (sample_data), optionally with annotations overlaid.
        :param sd_token: The token of the sample_data to be rendered.
        :param annotation_type: The types of annotations to draw on the image; there are four options:
            'all': Draw surfaces and objects, subject to any filtering done by object_tokens and surface_tokens.
            'surfaces': Draw only surfaces, subject to any filtering done by surface_tokens.
            'objects': Draw objects, subject to any filtering done by object_tokens.
            'none': Neither surfaces nor objects will be drawn.
        :param with_category: Whether to include the category name at the top of a box.
        :param with_attributes: Whether to include attributes in the label tags. Note that with_attributes=True
            will only work if with_category=True.
        :param object_tokens: List of object annotation tokens. If given, only these annotations are drawn.
        :param surface_tokens: List of surface annotation tokens. If given, only these annotations are drawn.
        :param render_scale: The scale at which the image will be rendered. Use 1.0 for the original image size.
        :param box_line_width: The box line width in pixels. The default is -1.
            If set to -1, box_line_width equals render_scale (rounded) to be larger in larger images.
        :param font_size: Size of the text in the rendered image. Use None for the default size.
        :param out_path: The path where we save the rendered image, or otherwise None.
            If a path is provided, the plot is not shown to the user.
        """
        # Validate inputs.
        sample_data = self.get('sample_data', sd_token)
        if not sample_data['is_key_frame']:
            assert annotation_type == 'none', 'Error: Cannot render annotations for non keyframes!'
            assert not with_attributes, 'Error: Cannot render attributes for non keyframes!'
        if with_attributes:
            assert with_category, 'In order to set with_attributes=True, with_category must be True.'
        assert type(box_line_width) == int, 'Error: box_line_width must be an integer!'
        if box_line_width == -1:
            box_line_width = int(round(render_scale))

        # Get image data.
        self.check_sweeps(sample_data['filename'])
        im_path = osp.join(self.dataroot, sample_data['filename'])
        print(sample_data['filename'])

        # Open labels file
        f = open(os.path.join(labels_path,"{}.txt".format(sample_data['filename'].split('/')[-1].split('.')[0])), "a")

        im = Image.open(im_path)

        # Initialize drawing.
        if with_category and font_size is not None:
            font = get_font(font_size=font_size)
        else:
            font = None
        im = im.convert('RGBA')
        draw = ImageDraw.Draw(im, 'RGBA')

        annotations_types = ['all', 'surfaces', 'objects', 'none']
        assert annotation_type in annotations_types, \
            'Error: {} is not a valid option for annotation_type. ' \
            'Only {} are allowed.'.format(annotation_type, annotations_types)
        if annotation_type is not 'none':
            if annotation_type == 'all' or annotation_type == 'surfaces':
                # Load stuff / surface regions.
                surface_anns = [o for o in self.surface_ann if o['sample_data_token'] == sd_token]
                if surface_tokens is not None:
                    sd_surface_tokens = set([s['token'] for s in surface_anns if s['token']])
                    assert set(surface_tokens).issubset(sd_surface_tokens), \
                        'Error: The provided surface_tokens do not belong to the sd_token!'
                    surface_anns = [o for o in surface_anns if o['token'] in surface_tokens]

                # Draw stuff / surface regions.
                for ann in surface_anns:
                    # Get color and mask.
                    category_token = ann['category_token']
                    category_name = self.get('category', category_token)['name']
                    color = self.color_map[category_name]
                    if ann['mask'] is None:
                        continue
                    mask = mask_decode(ann['mask'])

                    # Draw mask. The label is obvious from the color.
                    draw.bitmap((0, 0), Image.fromarray(mask * 128), fill=tuple(color + (128,)))

            if annotation_type == 'all' or annotation_type == 'objects':
                # Load object instances.
                object_anns = [o for o in self.object_ann if o['sample_data_token'] == sd_token]
                if object_tokens is not None:
                    sd_object_tokens = set([o['token'] for o in object_anns if o['token']])
                    assert set(object_tokens).issubset(sd_object_tokens), \
                        'Error: The provided object_tokens do not belong to the sd_token!'
                    object_anns = [o for o in object_anns if o['token'] in object_tokens]

                # Draw object instances.
                for ann in object_anns:
                    # Get color, box, mask and name.
                    category_token = ann['category_token']
                    category_name = self.get('category', category_token)['name']


                    # Parse category class
                    if category_name in classes_dict:
                        code = classes_dict[category_name]
                    elif custom_dict:
                    	continue
                    else:
                        classes_dict[category_name] = len(classes_dict)
                        code = classes_dict[category_name]
            

                    f.write(str(code)+ '    ')

                    # Bounding boxes to x,y center and width and height
                    color = self.color_map[category_name]
                    bbox = ann['bbox']
                    bb=bbox
                    width = bb[2]-bb[0] 
                    height = bb[3]-bb[1]
                    x_center = bb[0]+width/2
                    y_center = bb[1]+height/2

                    # Normalize
                    f.write(str(min(x_center/image_h,1)) + ' ' + str(min(y_center/image_w,1)) + ' ' + str(width/image_h) + ' ' + str(height/image_w))
                    f.write('\n')

                    
                    attr_tokens = ann['attribute_tokens']
                    attributes = [self.get('attribute', at) for at in attr_tokens]
                    name = annotation_name(attributes, category_name, with_attributes=with_attributes)
                    if ann['mask'] is not None:
                        mask = mask_decode(ann['mask'])

                        # Draw mask, rectangle and text.
                        draw.bitmap((0, 0), Image.fromarray(mask * 128), fill=tuple(color + (128,)))
                        draw.rectangle(bbox, outline=color, width=box_line_width)
                        if with_category:
                            draw.text((bbox[0], bbox[1]), name, font=font)
                    
        
        # Plot the image.
        (width, height) = im.size
        pix_to_inch = 100 / render_scale
        figsize = (height / pix_to_inch, width / pix_to_inch)
        plt.figure(figsize=figsize)
        plt.axis('off')
        plt.imshow(im)
        
        # Save to disk.
        if out_path is not None:
            plt.savefig(out_path, bbox_inches='tight', dpi=2.295 * pix_to_inch, pad_inches=0)
            plt.close()
        

    
def render_images(nuim: NuImages,
                  mode: str = 'all',
                  cam_name: str = None,
                  log_name: str = None,
                  sample_limit: int = 50,
                  filter_categories: List[str] = None,
                  out_type: str = 'image',
                  out_dir: str = '~/Downloads/nuImages',
                  cleanup: bool = True) :
    """
    Render a random selection of images and save them to disk.
    Note: The images rendered here are keyframes only.
    :param nuim: NuImages instance.
    :param mode: What to render:
      "image" for the image without annotations,
      "annotated" for the image with annotations,
      "trajectory" for a rendering of the trajectory of the vehice,
      "all" to render all of the above separately.
    :param cam_name: Only render images from a particular camera, e.g. "CAM_BACK'.
    :param log_name: Only render images from a particular log, e.g. "n013-2018-09-04-13-30-50+0800".
    :param sample_limit: Maximum number of samples (images) to render. Note that the mini split only includes 50 images.
    :param filter_categories: Specify a list of object_ann category names. Every sample that is rendered must
        contain annotations of any of those categories.
    :param out_type: The output type as one of the following:
        'image': Renders a single image for the image keyframe of each sample.
        'video': Renders a video for all images/pcls in the clip associated with each sample.
    :param out_dir: Folder to render the images to.
    :param cleanup: Whether to delete images after rendering the video. Not relevant for out_type == 'image'.
    """
    # Check and convert inputs.
    assert out_type in ['image', 'video'], ' Error: Unknown out_type %s!' % out_type
    all_modes = ['image', 'annotated', 'trajectory']
    assert mode in all_modes + ['all'], 'Error: Unknown mode %s!' % mode
    assert not (out_type == 'video' and mode == 'trajectory'), 'Error: Cannot render "trajectory" for videos!'

    if mode == 'all':
        if out_type == 'image':
            modes = all_modes
        elif out_type == 'video':
            modes = [m for m in all_modes if m not in ['annotated', 'trajectory']]
        else:
            raise Exception('Error" Unknown mode %s!' % mode)
    else:
        modes = [mode]

    if filter_categories is not None:
        category_names = [c['name'] for c in nuim.category]
        for category_name in filter_categories:
            assert category_name in category_names, 'Error: Invalid object_ann category %s!' % category_name

    # Create output folder.
    out_dir = os.path.expanduser(out_dir)
    labels_dir = out_dir + '/labels'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        os.makedirs(out_dir + '/imgs')
        os.makedirs(out_dir + '/labels')

    out_dir = out_dir + '/imgs'
    

    # Filter by camera.
    sample_tokens = [s['token'] for s in nuim.sample]
    if cam_name is not None:
        sample_tokens_cam = []
        for sample_token in sample_tokens:
            sample = nuim.get('sample', sample_token)
            key_camera_token = sample['key_camera_token']
            sensor = nuim.shortcut('sample_data', 'sensor', key_camera_token)
            if sensor['channel'] == cam_name:
                sample_tokens_cam.append(sample_token)
        sample_tokens = sample_tokens_cam

    # Filter by log.
    if log_name is not None:
        sample_tokens_cleaned = []
        for sample_token in sample_tokens:
            sample = nuim.get('sample', sample_token)
            log = nuim.get('log', sample['log_token'])
            if log['logfile'] == log_name:
                sample_tokens_cleaned.append(sample_token)
        sample_tokens = sample_tokens_cleaned

    # Filter samples by category.
    if filter_categories is not None:
        # Get categories in each sample.
        sd_to_object_cat_names = defaultdict(lambda: set())
        for object_ann in nuim.object_ann:
            category = nuim.get('category', object_ann['category_token'])
            sd_to_object_cat_names[object_ann['sample_data_token']].add(category['name'])

        # Filter samples.
        sample_tokens_cleaned = []
        for sample_token in sample_tokens:
            sample = nuim.get('sample', sample_token)
            key_camera_token = sample['key_camera_token']
            category_names = sd_to_object_cat_names[key_camera_token]
            if any([c in category_names for c in filter_categories]):
                sample_tokens_cleaned.append(sample_token)
        sample_tokens = sample_tokens_cleaned

    # Get a random selection of samples.
    random.shuffle(sample_tokens)

    # Limit number of samples.
    sample_tokens = sample_tokens[:sample_limit]

    print('Rendering %s for mode %s to folder %s...' % (out_type, mode, out_dir))
    for sample_token in tqdm.tqdm(sample_tokens):
        sample = nuim.get('sample', sample_token)
        log = nuim.get('log', sample['log_token'])
        log_name = log['logfile']
        key_camera_token = sample['key_camera_token']
        sensor = nuim.shortcut('sample_data', 'sensor', key_camera_token)
        sample_cam_name = sensor['channel']
        sd_tokens = nuim.get_sample_content(sample_token)

        # We cannot render a video if there are missing camera sample_datas.
        if len(sd_tokens) < 13 and out_type == 'video':
            print('Warning: Skipping video for sample token %s, as not all 13 frames exist!' % sample_token)
            continue

        for mode in modes:
            out_path_prefix = os.path.join(out_dir, '%s_%s_%s_%s' % (log_name, sample_token, sample_cam_name, mode))
            if out_type == 'image':
                write_image(nuim, key_camera_token, mode, '%s.jpg' % out_path_prefix, labels_dir)
            elif out_type == 'video':
                write_video(nuim, sd_tokens, mode, out_path_prefix, cleanup=cleanup)


def write_video(nuim: NuImages,
                sd_tokens: List[str],
                mode: str,
                out_path_prefix: str,
                cleanup: bool = True) :
    """
    Render a video by combining all the images of type mode for each sample_data.
    :param nuim: NuImages instance.
    :param sd_tokens: All sample_data tokens in chronological order.
    :param mode: The mode - see render_images().
    :param out_path_prefix: The file prefix used for the images and video.
    :param cleanup: Whether to delete images after rendering the video.
    """
    # Loop through each frame to create the video.
    out_paths = []
    for i, sd_token in enumerate(sd_tokens):
        out_path = '%s_%d.jpg' % (out_path_prefix, i)
        out_paths.append(out_path)
        write_image(nuim, sd_token, mode, out_path)

    # Create video.
    first_im = cv2.imread(out_paths[0])
    freq = 2  # Display frequency (Hz).
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_path = '%s.avi' % out_path_prefix
    out = cv2.VideoWriter(video_path, fourcc, freq, first_im.shape[1::-1])

    # Load each image and add to the video.
    for out_path in out_paths:
        im = cv2.imread(out_path)
        out.write(im)

        # Delete temporary image if requested.
        if cleanup:
            os.remove(out_path)

    # Finalize video.
    out.release()


def write_image(nuim: NuImages, sd_token: str, mode: str, out_path: str, labels_path: str) :
    """
    Render a single image of type mode for the given sample_data.
    :param nuim: NuImages instance.
    :param sd_token: The sample_data token.
    :param mode: The mode - see render_images().
    :param out_path: The file to write the image to.
    :param labels_path: The path to labels folder
    """
    if mode == 'annotated':
        nuim.render_image(sd_token, annotation_type='all', out_path=out_path, labels_path=labels_path)
    else:
        raise Exception('Error: Unknown mode %s!' % mode)

    # Trigger garbage collection to avoid memory overflow from the render functions.
    gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render a random selection of images and save them to disk.')
    parser.add_argument('--seed', type=int, default=42)  # Set to 0 to disable.
    parser.add_argument('--version', type=str, default='v1.0-train') #
    parser.add_argument('--dataroot', type=str, default='/mnt/md0/cfernandez/Nuscenes/nuimages-v1.0-all/')
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--mode', type=str, default='annotated')
    parser.add_argument('--cam_name', type=str, default=None)
    parser.add_argument('--log_name', type=str, default=None)
    parser.add_argument('--sample_limit', type=int, default=99999999999999999999999999999999999999999)
    parser.add_argument('--filter_categories', action='append')
    parser.add_argument('--out_type', type=str, default='image')
    parser.add_argument('--out_dir', type=str, default='/mnt/md0/cfernandez/Nuscenes/8_class')
    args = parser.parse_args()
    # Set random seed for reproducible image selection.
    if args.seed != 0:
        random.seed(args.seed)

    # Initialize NuImages class.
    nuim_ = NuImages(version=args.version, dataroot=args.dataroot, verbose=bool(args.verbose), lazy=False)


    output_dir = os.path.join(args.out_dir, args.version)
    # Render images.
    render_images(nuim_, mode=args.mode, cam_name=args.cam_name, log_name=args.log_name, sample_limit=args.sample_limit,
                  filter_categories=args.filter_categories, out_type=args.out_type, out_dir=output_dir)
    print(classes_dict)