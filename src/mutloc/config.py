import yaml
import numpy as np
import os

def arrayfromname(config, name):
    if name not in config:
        return
    data = config[name].get('data')
    shape = config[name].get('shape')
    return np.array(data, dtype=np.float32).reshape(*shape)

class CameraConfig(object):
    def __init__(self, configfile):
        self.configfile = configfile
        self.loaded = False
        self._configobj = None

    def _load(self):
        if self.loaded:
            return
        if not hasattr(self.configfile, 'readlines'):
            self.configfile = open(self.configfile)
        self._configobj = yaml.load(self.configfile)
        self.loaded = True

    def __getattr__(self, name):
        self._load()
        val = self._configobj[name]
        if hasattr(val, 'keys') and 'rows' in val and 'cols' in val:
            return np.array(val['data'], np.float32
                           ).reshape(val['rows'], val['cols'])
        else:
            return val

class Config(object):
    def __init__(self, configfile):
        self.configfile = configfile
        self.parsed = False

    def abspath(self, path_relative_to_config_file):
        configdir = os.path.dirname(self.configfile.name)
        return os.path.join(configdir, path_relative_to_config_file)

    def _parse(self):
        if self.parsed:
            return
        if not hasattr(self.configfile, 'readlines'):
            self.configfile = open(self.configfile)
        configobj = yaml.load(self.configfile)
        markers0 = arrayfromname(configobj, 'markers0')
        markers1 = arrayfromname(configobj, 'markers1')
        configdir = os.path.dirname(self.configfile.name)
        cameraconfigs = [
            CameraConfig(open(os.path.join(configdir, configobj[conf])))
            for conf in ('cameraconfig0', 'cameraconfig1')]

        self._intrinsic_camera_matrix = [camconf.camera_matrix
                                         for camconf in cameraconfigs]
        self._distortion_coefficients = [camconf.distortion_coefficients
                                         for camconf in cameraconfigs]
        self._image_width = [camconf.image_width
                             for camconf in cameraconfigs] 
        self._image_height = [camconf.image_height
                             for camconf in cameraconfigs] 

        camera_matrix0 = arrayfromname(configobj, 'camera_matrix0')
        camera_matrix1 = arrayfromname(configobj, 'camera_matrix1')
        color_reference_marker0 = arrayfromname(configobj,
                                                'color_reference_marker0')
        color_reference_marker1 = arrayfromname(configobj,
                                                'color_reference_marker1')
        self._markers = markers0, markers1
        self._color_reference_markers = (color_reference_marker0,
                                         color_reference_marker1)
        self._target_pos = arrayfromname(configobj,
                                         'target_pos')
        self._target_hue = configobj.get('target_hue')
        self._image_index = configobj.get('image_index')
        self._marked_images = configobj.get('marked_images')
        self._img_file_format = configobj.get('img_file_format0'),\
                configobj.get('img_file_format1')
        self._marker_hue = configobj.get('marker_hue0'), \
                configobj.get('marker_hue1')
        self._results_file = configobj.get('results_file')
        self._transforms_file = configobj.get('transforms_file')
        self._ground_truth = configobj.get('ground_truth')
        self.parsed = True

    def __getattr__(self, name):
        self._parse()
        attrname = '_%s' % name
        return lambda : getattr(self, attrname)

class OldStyleConf(object):
    configfile = 'test/data/two-camera/with_target/calib.yml'

    def __init__(self):
        configfile = os.path.join(os.path.dirname(__file__), self.configfile)
        config = Config(open(configfile))
        self.MARKERS = self.MARKERS0, self.MARKERS1 = config.markers()
        self.calibmat, _ = self.calibmat0, self.calibmat1 = \
                (lambda : config.intrinsic_camera_matrix()[0],
                 lambda : config.intrinsic_camera_matrix()[1])

        self.FIXED_MARKER_POS = \
                self.FIXED_MARKER_POS0, self.FIXED_MARKER_POS1 = \
                config.color_reference_markers()
        self.config = config
