from enum import IntEnum
import pdb

# from spacenet_globals import *
################################################################################
## Remove these
################################################################################
# Decorator to add a function to a pre-defined class object
# src: http://bit.ly/2SN2TDG
def add_method(cls):
    def decorator(func):
        @wraps(func) 
        def wrapper(self, *args, **kwargs): 
            return func(*args, **kwargs)
        setattr(cls, func.__name__, wrapper)
        # Note we are not binding func, but wrapper which accepts self but does exactly the same as func
        return func # returning func means func can still be used normally
    return decorator



class Road(IntEnum):
    #todo remove later
    OTHERS = 0
    MOTORWAY = 1
    PRIMARY = 2
    SECONDARY = 3
    TERTIARY = 4
    RESIDENTIAL = 5
    UNCLASSIFIED = 6
    CART = 7
    

    def describe(self):
        return (self.name, self.value)


class RoadType(IntEnum):
    # TODO
    MAJOR = 1
    MINOR = 2
    OTHER = 3

    def describe(self):
        return (self.name, self.value)
    

G_WIDTHS = {Road.MOTORWAY: 3.5,
            Road.PRIMARY: 3.5,
            Road.SECONDARY: 3.,
            Road.TERTIARY: 3.,
            Road.RESIDENTIAL: 3.,
            Road.UNCLASSIFIED: 3.,
            Road.CART: 3.,
            }
################################################################################
## Spacenet Road Type
################################################################################
class SpacenetRoad(IntEnum):
    """
    Note: in the mask images, 0 refers to no-road from the road vector geojsons
    
    **todo: add 0 to 'OTHERS' and make sure it doesn't break earlier codes
    """
    OTHERS = 0
    MOTORWAY = 1
    PRIMARY = 2
    SECONDARY = 3
    TERTIARY = 4
    RESIDENTIAL = 5
    UNCLASSIFIED = 6
    CART = 7
    
    @classmethod
    def radius_mapping(cls):
        rmap = {
            cls.OTHERS: 3./2,
            cls.MOTORWAY: 3.5/2,
            cls.PRIMARY: 3.5/2,
            cls.SECONDARY: 3./2,
            cls.TERTIARY: 3./2,
            cls.RESIDENTIAL: 3./2,
            cls.UNCLASSIFIED: 3./2,
            cls.CART: 3./2,
            }
        return rmap

    def get_radius(self):
        return SpacenetRoad.radius_mapping()[self]
    
    def describe(self):
        return (self.name, self.value)
    
    
    def to_global(self):
        #todo: refer to the mapping defined in World Food Program pdf
        if self in [SpacenetRoad.MOTORWAY,SpacenetRoad.PRIMARY, SpacenetRoad.SECONDARY, 
                    SpacenetRoad.TERTIARY, SpacenetRoad.UNCLASSIFIED]: #[1,2,3,4,6]:
            return RoadType.MAJOR
        
        elif self in [SpacenetRoad.RESIDENTIAL]:
            return RoadType.MINOR
        else:
            return RoadType.OTHER


################################################################################
## OSM Road Type
################################################################################
_osm_rtypes = ['bridleway',
'cycleway',
'footway',
'living_street',
'motorway',
'motorway_link',
'path',
'pedestrian',
'primary',
'raceway',
'residential',
'road',
'secondary',
'secondary_link',
'service',
'steps',
'tertiary',
'tertiary_link',
'track',
'trunk',
'trunk_link',
'unclassified']

_osm_rtypes_str = ' '. join(list(map(lambda x: x.upper(), _osm_rtypes)))
# print(_osm_rtypes_str)

#todo: remap this based on the WFP pdf
def r_osm2spacenet(osmroad_type):

    """osm_roadtype to spacenet_roadtype mapping"""
    spacenet_type = None
    if osmroad_type.name in ['MOTORWAY', 'MOTORWAY_LINK']:
        spacenet_type = SpacenetRoad['MOTORWAY']
    elif osmroad_type.name in ['PRIMARY', 'TRUNK', 'TRUNK_LINK']:
        spacenet_type = SpacenetRoad['PRIMARY']
    elif osmroad_type.name in ['SECONDARY', 'SECONDARY_LINK']:
        spacenet_type = SpacenetRoad['SECONDARY']
    elif osmroad_type.name in ['TERTIARY', 'TERTIARY_LINK']:
        spacenet_type = SpacenetRoad['TERTIARY']
    elif osmroad_type.name in ['CYCLEWAY', 'FOOTWAY', 'LIVING_STREET', 'PEDESTRIAN',
                               'RESIDENTIAL', 'SERVICE']:
        spacenet_type = SpacenetRoad['RESIDENTIAL']
    elif osmroad_type.name in ['UNCLASSIFIED']:
        spacenet_type = SpacenetRoad['UNCLASSIFIED']
    elif osmroad_type.name in ['PATH']:
        spacenet_type = SpacenetRoad['CART']
    else:
        print("This osm_rtype is not mapped to any spacenet_rtype: {}".
              format(osmroad_type.name) )
        print(" For now we assign None")
    return spacenet_type

def r_osmnx2sp(osm_rt, others='OTHERS', verbose=False):

    """osm_roadtype to spacenet_roadtype mapping"""
    if not isinstance(osm_rt, str):
        osm_rt = osm_rt[0]
    if osm_rt.upper() in ['MOTORWAY', 'MOTORWAY_LINK']:
        sp_rt =  SpacenetRoad['MOTORWAY']
    elif osm_rt.upper() in ['PRIMARY', 'PRIMARY_LINK', 'TRUNK', 'TRUNK_LINK']:
        sp_rt =  SpacenetRoad['PRIMARY']
    elif osm_rt.upper() in ['SECONDARY', 'SECONDARY_LINK']:
        sp_rt = SpacenetRoad['SECONDARY']
    elif osm_rt.upper() in ['TERTIARY', 'TERTIARY_LINK']:
        sp_rt = SpacenetRoad['TERTIARY']
    elif osm_rt.upper() in ['CYCLEWAY', 'FOOTWAY', 'LIVING_STREET', 'PEDESTRIAN', 'RESIDENTIAL', 'SERVICE']:
        sp_rt = SpacenetRoad['RESIDENTIAL']
    elif osm_rt.upper() in ['UNCLASSIFIED']:
        sp_rt = SpacenetRoad['UNCLASSIFIED']
    elif osm_rt.upper() in ['PATH']:
        sp_rt = SpacenetRoad['CART']
    else:
        sp_rt = SpacenetRoad[others]
        if verbose:
            print(f"This osm_rtype is not mapped to any spacenet_rtype: {osm_rt}")
            print(f"For now we map it to {others}")
    return sp_rt


@staticmethod
def __osm_radius_per_lane():
    return 1.8

@staticmethod
def __osm_default_lane_nums():
    return 1.0

# def get


## todo: use add_method(cls) decorator
## https://medium.com/@mgarod/dynamically-add-a-method-to-a-class-in-python-c49204b85bd6
OSMRoad = IntEnum('OSMRoad', _osm_rtypes_str)
setattr(OSMRoad, 'describe', lambda self: (self.name, self.value) )
setattr(OSMRoad, 'to_spacenet_rtype', lambda self: r_osm2spacenet(self))
setattr(OSMRoad, 'radius_per_lane', __osm_radius_per_lane)
setattr(OSMRoad, 'default_lane_nums', __osm_default_lane_nums)


################################################################################
## Tests
################################################################################
def test_road_enum():
    moto_type = Road.MOTORWAY
    print(moto_type)
    moto_type.describe()
    
def test_osmroad_enum():
    for r in OSMRoad:
        print(r.describe())
    
    for r in OSMRoad:
        print(r.name, " --> ", r.to_spacenet_rtype().name)
        
def test_mapping():
    for osm_rtype in OSMRoad:
        print(osm_rtype, "-->", osm_rtype.to_spacenet_rtype())
    
if __name__ == '__main__':
#     test_road_enum()
    test_mapping()
