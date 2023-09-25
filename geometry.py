class Geometry:
    def __init__(self, volume_shape, volume_origin, volume_spacing, detector_origin, detector_spacing):
        '''

        :param volume_shape: Image shape
        :param volume_origin: Image origin in world coordinates
        :param volume_spacing: Image spacing in mm
        :param detector_origin: Detector origin in world coordinates
        :param detector_spacing: Detector spacing in mm
        '''
        self.volume_shape = volume_shape
        self.volume_origin = volume_origin
        self.volume_spacing = volume_spacing
        self.detector_origin = detector_origin
        self.detector_spacing = detector_spacing