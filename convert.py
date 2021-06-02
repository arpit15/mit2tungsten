import xml.etree.ElementTree as ET
import json
from os import path as osp
from os.path import join
import numpy as np
from scipy.spatial.transform import Rotation as R

class Convertor(object):
  def __init__(self):
    pass
    
  def parseIntegrator(self, xml):
    integrator = dict()
    intType = xml.attrib["type"]
    # supported path
    integrator["type"] = "path_tracer"
    # add some properties
    integrator["enable_light_sampling"] = True
    integrator["enable_volume_light_sampling"] = True
    for child in xml:
      ele = child.find(".//*[@name='maxDepth']")
      if(ele is not None):
        integrator["max_bounces"] = int(ele.attrib["value"])
    return integrator

  def parseTransform(self, xml):
    transform = dict()
    for child in xml:
      if child.tag == "matrix":
        value = child.attrib["value"]
        value = value.split()
        assert(len(value) == 16)
        # convert to float
        value = [float(i) for i in value]
        value = np.array(value).reshape(4,4)
        position = value[:3,3]
        
        rot_mat = value[:3, :3]
        scale = [1.0]*3
        for i in range(3):
          currscale = np.linalg.norm(rot_mat[:3,i])
          scale[i] = currscale
          rot_mat[:3,i] /= currscale

        transform["scale"] = scale
        rot = R.from_matrix(rot_mat)
        transform["position"] = position.tolist()



        transform["rotation"] = rot.as_euler(seq='xyz', degrees=True).tolist()

    return transform


  def parseCamera(self, xml, scene):
    camera = dict()
    camType = xml.attrib["type"]
    # supported perspective
    camera["type"] = "pinhole"
    camera["tonemap"] = "filmic"
    camera["reconstruction_filter"] = "tent"
    for child in xml:
      if child.tag == "float":
        if child.attrib["name"] == "fov":
          camera["fov"] = float(child.attrib["value"])
        else:
          print(f'{child.attrib["name"]} not handled')
      elif child.tag == "transform":
        camera["transform"] = self.parseTransform(child)
      elif child.tag == "film":
        w_ele = child.find(".//*[@name='width']")
        h_ele = child.find(".//*[@name='height']")
        camera["resolution"] = [
                                  int(w_ele.attrib["value"]),
                                  int(h_ele.attrib["value"])
                                ]
      elif child.tag == "sampler":
        # assumed to be independent
        # only spp used for rendering
        ele = child.find(".//*[@name='sampleCount']")
        if(ele is not None):
          scene["renderer"]["spp"] = int(ele.attrib["value"])

    return camera

  def parseRGB(self, value):
    spec = [0.0]*3
    if ',' in value:
      val_list = value.split(',')
      # assert(3 == len(val_list), f"{value} should contain 3 comma separated elements")
      for i, val in enumerate(val_list):
        spec[i] = float(val)
    else:
      spec = [1.0]*float(value)

    return spec


  def parseBsdfs(self, xml, scene):
    bsdf = dict()
    bsdfType = xml.attrib["type"]
    bsdfID = xml.attrib["id"]
    bsdf["name"] = bsdfID
    if bsdfType == "diffuse":
      bsdf["type"] = "lambert"
      ele = xml.find(".//*[@name='reflectance']")
      if(ele is not None):
        bsdf["albedo"] = self.parseRGB(ele.attrib["value"])
    elif bsdfType == "roughdielectric":
      bsdf["type"] = "rough_dielectric"
      for child in xml:
        ele = child.find(".//*[@name='intIOR']")
        if(ele is not None):
          bsdf["ior"] = float(ele.attrib["value"])

        ele = child.find(".//*[@name='distribution']")
        if(ele is not None):
          bsdf["distribution"] = float(ele.attrib["value"])

        ele = child.find(".//*[@name='alpha']")
        if(ele is not None):
          bsdf["roughness"] = float(ele.attrib["value"])
    elif bsdfType == "roughconductor":
      bsdf["type"] = "rough_conductor"
      for child in xml:
        ele_eta = child.find(".//*[@name='eta']")
        if(ele_eta is not None):
          bsdf["ior"] = float(ele_eta.attrib["value"])

        ele = child.find(".//*[@name='distribution']")
        if(ele is not None):
          bsdf["distribution"] = float(ele.attrib["value"])

        ele = child.find(".//*[@name='alpha']")
        if(ele is not None):
          bsdf["roughness"] = float(ele.attrib["value"])

        ele = child.find(".//*[@name='material']")
        if(ele is not None and ele_eta is None):
          bsdf["material"] = float(ele.attrib["material"])
    elif bsdfType == "roughcoating":
      bsdf["type"] = "rough_coat"
      for child in xml:
        ele_eta = child.find(".//*[@name='eta']")
        if(ele_eta is not None):
          bsdf["ior"] = float(ele_eta.attrib["value"])

        ele = child.find(".//*[@name='thickness']")
        if(ele is not None):
          bsdf["thickness"] = float(ele.attrib["value"])

        ele = child.find(".//*[@name='distribution']")
        if(ele is not None):
          bsdf["distribution"] = float(ele.attrib["value"])

        ele = child.find(".//*[@name='alpha']")
        if(ele is not None):
          bsdf["roughness"] = float(ele.attrib["value"])

        ele = child.find(".//*[@name='sigmaA']")
        if(ele is not None):
          bsdf["sigma_a"] = float(ele.attrib["value"])

        ele = child.find(".//*[@name='bsdf']")
        if(ele is not None):
          bsdf["substrate"] = self.parseBsdfs(child, scene)

    else:
      print(f"Currently {bsdfType} is not supported")

    return bsdfID, bsdf

  def parseShapes(self, xml, scene):
    primitive = dict()
    shapeType = xml.attrib["type"]
    if shapeType == "rectangle":
      primitive["type"] = "quad"
      for child in xml:
        if child.tag == "transform":
          primitive["transform"] = self.parseTransform(child)
        elif child.tag == "ref":
          # Assumption ref is bsdf
          primitive["bsdf"] = child.attrib["id"]
        elif child.tag == "emitter":
          self.readRecursively(child, scene, primitive)
    elif(shapeType == "cube"):
      primitive["type"] = "cube"
      for child in xml:
        if child.tag == "transform":
          primitive["transform"] = self.parseTransform(child)
        elif child.tag == "ref":
          primitive["bsdf"] = child.attrib["id"]
    else:
      print(f"Currently {shapeType} is not supported")
    return primitive

  def parseEmitter(self, xml, scene): 
    bsdf = dict()
    emitterType = xml.attrib["type"]
    if emitterType == "area":
      ele = xml.find(".//*[@name='radiance']")
      if(ele is not None):
        bsdf["type"] = "null"
        bsdf["albedo"] = self.parseRGB(ele)
        bsdf["name"] = "Light001"
        print("Currently manually setting the emitter bsdf name")
        
    else:
      print(f"Currently {emitterType} is not supported")

    return bsdf

    # envmap is just primitive

  def readRecursively(self, xml, scene, primitive=None):
    for child in xml:
      if child.tag == "integrator":
        scene["integrator"] = self.parseIntegrator(child)
      elif child.tag == "sensor":
        scene["camera"] = self.parseCamera(child, scene)
      elif child.tag == "bsdf":
        bsdfID, bsdfDict = self.parseBsdfs(child, scene)
        scene["bsdfs"].append(bsdfDict)
      elif child.tag == "shape":
        scene["primitives"].append(self.parseShapes(child, scene))
      elif child.tag == "emitter":
        # only parsing area lights for now
        bsdfEntry = self.parseEmitter(child, scene)
        # add a bsdf entry
        scene["bsdfs"].append(bsdfEntry)
        # add a primitive entry
        assert(primitive is not None), "Only support area lights for now"
        primitive["bsdf"] = bsdfEntry["name"]

      else:
        print(f"Currently {child.tag} parsing is not Supported")

    return scene

  def convert(self, fname):
    # empty dict
    scene = dict()
    # add renderer entry
    scene["renderer"] = {"scene_bvh" : True}
    scene["bsdfs"] = list()
    scene["media"] = list()
    scene["primitives"] = list()

    dname = osp.dirname(fname)
    base = osp.basename(fname)
    # check ext
    base, ext = osp.splitext(base)
    
    if ext != ".xml":
      print(f"Wrong file format {ext}")
      return False
    # check version if already in correct format than do nothing
    tree = ET.parse(fname)
    root = tree.getroot()
    version = root.attrib["version"]
    MAJOR_VER = int(version[0])

    scene = self.readRecursively(root, scene)
    return json.dumps(scene, sort_keys=True, indent=4)


if __name__ == '__main__':
  fname = join("data", "mit1.xml")
  myconvertor = Convertor()
  outJson = myconvertor.convert(fname)
  with open(join("data", "converted.json"), 'w') as fp:
    fp.write(outJson)