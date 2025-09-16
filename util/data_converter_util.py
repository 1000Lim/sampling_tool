"""
The module data_converter.py is used to convert between dict, json and XML.
"""
import json
import xmltodict
from util.util_func import save_obj, load_obj


def enable_check_decorator(param_name):
    def wrapper(func):
        def decorator(self, *args, **kwargs):
            if getattr(self, param_name) is None:
                print('%s is None' % param_name)
            else:
                return func(self, *args, **kwargs)
        return decorator
    return wrapper


class DataConverter:

    def __init__(self, xml_path=None, json_path=None, dict_path=None,
                 xml_bytes=None):
        """
        Can convert between dict, json and XML.
        You can load dict from json or xml and save modified dict to json or xml.
        :param xml_path: Path of a xml file.
        :param json_path: Path of a json file.
        :param dict_path: Path of a dict file.
        """
        self.xml_path = xml_path
        self.json = None
        if json_path is not None:
            self.read_json(json_path)
        self.xml = None
        if xml_path is not None:
            self.read_xml(xml_path)
        if xml_bytes is not None:
            self.xml = xml_bytes
        self.dict = None
        if dict_path is not None:
            self.read_dict()

    def read_json(self, json_path:str):
        self.json = open(json_path, 'r').read()
        return self

    def read_xml(self, xml_path:str):
        self.xml = open(xml_path, 'r').read()
        return self

    def read_dict(self, dict_path:str):
        self.dict = load_obj(dict_path)
        return self

    @enable_check_decorator('xml')
    def xml_to_dict(self):
        dict_type = xmltodict.parse(self.xml)
        json_type = json.dumps(dict_type)
        self.dict = json.loads(json_type)
        return self

    @enable_check_decorator('json')
    def json_to_dict(self):
        self.dict = json.loads(self.json)
        return self

    @enable_check_decorator('dict')
    def dict_to_json(self):
        self.json = json.dumps(self.dict, indent=4)
        return self

    @enable_check_decorator('dict')
    def dict_to_xml(self):
        self.xml = xmltodict.unparse(self.dict, pretty=True)
        return self

    def xml_to_json(self):
        self.xml_to_dict()
        self.dict_to_json()
        return self

    def json_to_xml(self):
        self.json_to_dict()
        self.dict_to_xml()
        return self

    @enable_check_decorator('dict')
    def save_dict(self, path):
        save_obj(self.dict, path)
        return self

    @enable_check_decorator('json')
    def save_json(self, path):
        f = open(path, 'w')
        f.write(self.json)
        return self

    @enable_check_decorator('xml')
    def save_xml(self, path):
        f = open(path, 'w')
        f.write(self.xml)
        return self
