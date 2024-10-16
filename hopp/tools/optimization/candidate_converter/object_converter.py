import functools
from enum import IntEnum
from typing import (
    Callable,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    )

from ..data_logging.data_recorder import DataRecorder
from .candidate_converter import CandidateConverter


class Type(IntEnum):
    Value = 0,  # None
    Tuple = 1,  # [(element_type, element_mapping)]
    ValueTuple = 2,  # length
    List = 3,  # [(element_type, element_mapping)]
    ValueList = 4,  # length
    Dict = 5,  # [(key, element_type, element_mapping)]
    Object = 6,  # (factory, [(key, element_type, element_mapping)])


From = TypeVar('From')
ValueMapping = None
IterableMapping = List[Tuple[Type, 'Mapping']]
ValueIterableMapping = int
DictMapping = List[Tuple[any, Type, any]]
ObjectMapping = Tuple[Callable[[], object], DictMapping]
Mapping = Union[ValueMapping, IterableMapping, ValueIterableMapping, DictMapping, ObjectMapping]
ValueGenerator = Generator[any, None, None]


class ObjectConverter(CandidateConverter[From, List]):
    """
    Converts between POD objects and list of attribute's values.
    """
    
    def __init__(self, prototype: Optional[object] = None):
        self.element_type: Type = Type.Value
        self.mapping: Mapping = []
        self.length: int = 0
        
        if prototype is not None:
            self.setup(prototype)
    
    def setup(self, prototype: object, recorder: DataRecorder) -> None:
        """
        Creates a mapping of the prototype's attributes to their element types (list, tuple, dict, value)
        :param prototype: the object whose attributes are to be mapped
        :param recorder: a data recorder
        """
        self.element_type, self.mapping = build_mapping(prototype, True)
        
        # https://stackoverflow.com/questions/393053/length-of-generator-output
        self.length = len(list(convert_from_element(self.mapping, prototype, self.element_type)))
    
    def convert_from(self, candidate: From) -> List:
        """
        :param candidate: instance of the prototype's element_type
        :return: values converted from attributes of the candidate
        """
        return [value for value in convert_from_element(self.mapping, candidate, self.element_type)]
    
    def convert_to(self, candidate: Iterable) -> From:
        """
        :param candidate: list, tuple or ndarray of values of the candidate
        :return: instance of element_type with provided values
        """
        return convert_to_element(self.mapping, iter(candidate), self.element_type)


def build_iterable_map(prototype: Iterable) -> IterableMapping:
    return [build_mapping(element) for element in prototype]


def build_dict_map(prototype: dict) -> DictMapping:
    # print(sorted(prototype.keys()))
    return build_kvp_map(sorted(prototype.items()))


def build_object_map(prototype: object) -> ObjectMapping:
    """
    Entry point for building a map of the prototype object's attributes to their types
    :param prototype:
    :return:
    """
    sorted_attributes = sorted([(attr, getattr(prototype, attr))
                                  for attr in dir(prototype)
                                  if not callable(getattr(prototype, attr))
                                  and not attr.startswith("__")])
    print([e[0] for e in sorted_attributes])
    return (lambda: type(prototype)(), build_kvp_map(sorted_attributes))


def build_kvp_map(items: Iterable[Tuple[any, any]]) -> DictMapping:
    return [(key,) + build_mapping(value) for key, value in items]


def is_mapping_all_values(mapping) -> bool:
    return functools.reduce(lambda acc, e: acc and e[0] == Type.Value, mapping, True)


def build_mapping(prototype: any, on_root=False) -> ('Type', Mapping):
    if isinstance(prototype, tuple):
        mapping = build_iterable_map(prototype)
        if is_mapping_all_values(mapping):
            return Type.ValueTuple, len(mapping)
        return Type.Tuple, mapping
    
    if isinstance(prototype, list):
        mapping = build_iterable_map(prototype)
        if is_mapping_all_values(mapping):
            return Type.ValueList, len(mapping)
        return Type.List, mapping
    
    if isinstance(prototype, dict):
        return Type.Dict, build_dict_map(prototype)
    
    # if this is the root, then use an object mapping
    if on_root:
        return Type.Object, build_object_map(prototype)
    
    return Type.Value, None


def convert_from_value(_: None, source: any) -> ValueGenerator:
    yield source


def convert_from_iterable(mapping: IterableMapping, source: Iterable) -> ValueGenerator:
    for i, value in enumerate(source):
        element_type, element_mapping = mapping[i]
        yield from convert_from_element(element_mapping, value, element_type)


def convert_from_value_iterable(_: ValueIterableMapping, source: Iterable) -> ValueGenerator:
    yield from source


def convert_from_dict(mapping: DictMapping, source: {}) -> ValueGenerator:
    for key, element_type, element_mapping in mapping:
        yield from convert_from_element(element_mapping, source[key], element_type)


def convert_from_object(mapping: ObjectMapping, source: object) -> ValueGenerator:
    for key, element_type, element_mapping in mapping[1]:
        yield from convert_from_element(element_mapping, getattr(source, key), element_type)


convert_from_jump_table = {
    Type.Value:      convert_from_value,
    Type.Tuple:      convert_from_iterable,
    Type.ValueTuple: convert_from_value_iterable,
    Type.List:       convert_from_iterable,
    Type.ValueList:  convert_from_value_iterable,
    Type.Dict:       convert_from_dict,
    Type.Object:     convert_from_object,
    }


def convert_from_element(mapping: Mapping, source, element_type: Type) -> ValueGenerator:
    """
    Entry point for converting a source object's attributes to their values using the given mapping
    :param mapping: map of the source's attributes to their types
    :param source: object from which to convert
    :param element_type: type of source
    :return: a generator of the object's values
    """
    yield from convert_from_jump_table[element_type](mapping, source)


def convert_to_value(_: None, source: Iterator) -> any:
    return next(source)


def convert_to_list(mapping: IterableMapping, source: Iterator) -> []:
    return list(convert_to_generator(mapping, source))


def convert_to_value_list(mapping: ValueIterableMapping, source: Iterator) -> []:
    return list(convert_to_value_generator(mapping, source))


def convert_to_tuple(mapping: IterableMapping, source: Iterator) -> ():
    return tuple(convert_to_generator(mapping, source))


def convert_to_value_tuple(mapping: ValueIterableMapping, source: Iterator) -> []:
    return tuple(convert_to_value_generator(mapping, source))


def convert_to_generator(mapping: IterableMapping, source: Iterator) -> ValueGenerator:
    return (convert_to_element(element_mapping, source, element_type)
            for element_type, element_mapping in mapping)


def convert_to_value_generator(mapping: ValueIterableMapping, source: Iterator) -> ValueGenerator:
    return (next(source) for i in range(mapping))


def convert_to_dict(mapping: DictMapping, source: Iterator) -> {}:
    return {key: convert_to_element(element_mapping, source, element_type)
            for key, element_type, element_mapping in mapping}


def convert_to_object(mapping: ObjectMapping, source: Iterator) -> object:
    target = mapping[0]()
    for key, element_type, element_mapping in mapping[1]:
        setattr(target, key, convert_to_element(element_mapping, source, element_type))
    return target


convert_to_jump_table = {
    Type.Value:      convert_to_value,
    Type.Tuple:      convert_to_tuple,
    Type.ValueTuple: convert_to_value_tuple,
    Type.List:       convert_to_list,
    Type.ValueList:  convert_to_value_list,
    Type.Dict:       convert_to_dict,
    Type.Object:     convert_to_object,
    }


def convert_to_element(mapping, source, element_type) -> any:
    """
    Entry point for converting values into the source object's attributes using the given mapping
    :param mapping: map of the source's attributes to their types
    :param source: object from which to convert
    :param element_type: type of source
    :return: object of type element_type
    """
    return convert_to_jump_table[element_type](mapping, source)
