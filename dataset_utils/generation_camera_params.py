import numpy as np

camera_dataset_names = ['temple', 'templeRing', 'templeSparseRing', 'dino', 'dinoRing', 'dinoSparseRing']
camera_configs = {
    'horse': {
        'simpleRing': {
            'rx': (None, None, 1),
            'ry': np.pi / 2,
            'rz': (None, None, 10),
            'tz': 0.22,
            'f': 110.
        },
        'rotation': {
            'rx': (None, None, 10),
            'ry': (None, None, 5),
            'tz': 0.22,
            'f': 110.
        },
        'rotationZoom': {
            'rx': (None, None, 10),
            'ry': (None, None, 5),
            'tz': (0.18, 0.22, 3),
            'f': 110.
        },
        'rotationTranslation': {
            'rx': (None, None, 10),
            'ry': (None, None, 5),
            'tx': (-0.04, 0.04, 3),
            'ty': (-0.04, 0.04, 3),
            'tz': 0.22,
            'f': 110.
        },
        'rotationTranslationZoom': {
            'rx': (None, None, 10),
            'ry': (None, None, 5),
            'tx': (-0.04, 0.04, 3),
            'ty': (-0.04, 0.04, 3),
            'tz': (0.18, 0.22, 2),
            'f': 110.85
        },
        'rotationTranslationFocus': {
            'rx': (None, None, 10),
            'ry': (None, None, 5),
            'tx': (-0.04, 0.04, 3),
            'ty': (-0.04, 0.04, 3),
            'tz': 0.22,
            'f': [90, 110, 150]
        },
        'rotationTranslationFocusZoom': {
            'rx': (None, None, 5),
            'ry': (None, None, 5),
            'tx': (-0.04, 0.04, 3),
            'ty': (-0.04, 0.04, 3),
            'tz': (0.18, 0.22, 2),
            'f': [90, 110, 150]
        }
    },
    'big_dodge': {
        'rotation': {
            'rx': (None, None, 5),
            'ry': (None, None, 10),
            'tz': 15,
            'f': 110.
        },
        'rotationZoom': {
            'rx': (None, None, 5),
            'ry': (None, None, 10),
            'tz': (15, 45, 3),
            'f': 110.
        },
        'rotationTranslationZoom': {
            'rx': (None, None, 5),
            'ry': (None, None, 10),
            'tz': (12, 20, 3),
            'tx': (-5, 5, 3),
            'ty': (-5, 5, 3),
            'f': 110.
        }
    },
    'chopper': {
        'rotation': {
            'center': (-40, 80, -10),
            'rx': (None, None, 10),
            'ry': (None, None, 5),
            'tz': 150,
            'f': 50.
        },
        'rotationZoom': {
            'center': (-40, 80, -10),
            'rx': (None, None, 10),
            'ry': (None, None, 5),
            'tz': (130, 250, 3),
            'f': 50.
        },
        'rotationTranslationZoom':{
            'center': (-40, 80, -10),
            'rx': (None, None, 10),
            'ry': (None, None, 5),
            'tx': (-10, 10, 3),
            'ty': (-10, 10, 3),
            'tz': (140, 180, 3),
            'f': 50.
        }
    },
    'street_lamp': {
        'rotation': {
            'rx': (None, None, 5),
            'ry': (None, None, 10),
            'tz': 10,
            'f': 50.
        },
        'rotationZoom': {
            'rx': (None, None, 5),
            'ry': (None, None, 10),
            'tz': (10, 14, 3),
            'f': 50.
        }
    },
    'galleon': {
        'rotation': {
            'center': (0, -200, 0),
            'rx': (None, None, 5),
            'ry': (None, None, 10),
            'tz': 800,
            'f': 50.
        },
        'rotationZoom': {
            'center': (0, -200, 0),
            'rx': (None, None, 5),
            'ry': (None, None, 10),
            'tz': (800, 1200, 3),
            'f': 50.
        }
    },
    'stratocaster': {
        'rotation': {
            'center': (100, 20, 0),
            'rx': (None, None, 5),
            'ry': (None, None, 10),
            'tz': 300,
            'f': 50.
        }
    },
    'dolphins': {
        'rotation': {
            'center': (0, 0, 200),
            'rx': (None, None, 5),
            'ry': (None, None, 10),
            'tz': 600,
            'f': 50.
        },
        'rotationZoom': {
            'center': (0, 0, 200),
            'rx': (None, None, 5),
            'ry': (None, None, 10),
            'tz': (550, 800, 3),
            'f': 50.
        }
    }
}
mesh_files = ['horse', 'big_dodge', 'chopper', 'street_lamp', 'galleon', 'stratocaster', 'dolphins']