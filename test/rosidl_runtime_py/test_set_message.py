# Copyright 2019 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import builtins
import copy

from builtin_interfaces.msg import Time
import numpy
import pytest
import rosidl_parser.definition
from rosidl_runtime_py import set_message_fields
from std_msgs.msg import Header
from test_msgs import message_fixtures


class MockMessageStamped:

    __slots__ = [
        '_header',
    ]

    _fields_and_field_types = {
        'header': 'std_msgs/Header',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Header'),
    )

    def __init__(self):
        self.header = Header()

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def header(self):
        return self._header

    @header.setter
    def header(self, value):
        self._header = value


class MockMessageWithStampFields:

    __slots__ = [
        '_timestamp1',
        '_timestamp2',
    ]

    _fields_and_field_types = {
        'timestamp1': 'builtin_interfaces/Time',
        'timestamp2': 'builtin_interfaces/Time',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['builtin_interfaces', 'msg'], 'Time'),
        rosidl_parser.definition.NamespacedType(['builtin_interfaces', 'msg'], 'Time'),
    )

    def __init__(self):
        self.timestamp1 = Time()
        self.timestamp2 = Time()

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def timestamp1(self):
        return self._timestamp1

    @timestamp1.setter
    def timestamp1(self, value):
        self._timestamp1 = value

    @builtins.property
    def timestamp2(self):
        return self._timestamp2

    @timestamp2.setter
    def timestamp2(self, value):
        self._timestamp2 = value


def test_set_message_fields_none():
    # Smoke-test on a bunch of messages
    msgs = []
    msgs.extend(message_fixtures.get_msg_arrays())
    msgs.extend(message_fixtures.get_msg_basic_types())
    msgs.extend(message_fixtures.get_msg_bounded_sequences())
    msgs.extend(message_fixtures.get_msg_builtins())
    msgs.extend(message_fixtures.get_msg_constants())
    msgs.extend(message_fixtures.get_msg_defaults())
    msgs.extend(message_fixtures.get_msg_empty())
    msgs.extend(message_fixtures.get_msg_multi_nested())
    msgs.extend(message_fixtures.get_msg_nested())
    msgs.extend(message_fixtures.get_msg_strings())
    msgs.extend(message_fixtures.get_msg_unbounded_sequences())

    for m in msgs:
        original_m = copy.copy(m)
        set_message_fields(m, {})
        # Assert message is not modified when setting no fields
        assert original_m == m


def test_set_message_fields_partial():
    original_msg = message_fixtures.get_msg_basic_types()[0]
    original_msg.bool_value = False
    original_msg.char_value = 3
    original_msg.int32_value = 42

    modified_msg = copy.copy(original_msg)
    values = {}
    values['bool_value'] = True
    values['char_value'] = 1
    values['int32_value'] = 24
    set_message_fields(modified_msg, values)

    for attr in original_msg.get_fields_and_field_types().keys():
        # Remove underscore prefix
        if attr in values:
            assert getattr(modified_msg, attr) == values[attr]
        else:
            assert getattr(modified_msg, attr) == getattr(original_msg, attr)


def test_set_message_fields_full():
    msg_list = message_fixtures.get_msg_basic_types()
    msg0 = msg_list[0]
    msg1 = msg_list[1]

    # Set msg0 values to the values of msg1
    values = {}
    for attr in msg1.get_fields_and_field_types().keys():
        # Remove underscore prefix
        values[attr] = getattr(msg1, attr)
    set_message_fields(msg0, values)

    assert msg0 == msg1


def test_set_message_fields_invalid():
    msg = message_fixtures.get_msg_basic_types()[0]
    invalid_field = {}
    invalid_field['test_invalid_field'] = 42
    with pytest.raises(AttributeError):
        set_message_fields(msg, invalid_field)

    invalid_type = {}
    invalid_type['int32_value'] = 'this is not an integer'
    with pytest.raises(TypeError):
        set_message_fields(msg, invalid_type)

    msg = message_fixtures.get_msg_nested()[0]
    with pytest.raises(TypeError):
        set_message_fields(msg, 'not_a_dict')


def test_set_message_fields_scalar_type_mismatch():
    msg = message_fixtures.get_msg_basic_types()[0]
    string_msg = message_fixtures.get_msg_strings()[0]

    # String assigned to a bool field must not be truthy-coerced.
    with pytest.raises(TypeError):
        set_message_fields(msg, {'bool_value': 'false'})

    # Numeric values assigned to a bool field must not be truthy-coerced.
    with pytest.raises(TypeError):
        set_message_fields(msg, {'bool_value': -1})
    with pytest.raises(TypeError):
        set_message_fields(msg, {'bool_value': 0.0})

    # Float assigned to an integer field must not be silently truncated.
    with pytest.raises(TypeError):
        set_message_fields(msg, {'int8_value': 1.5})

    # String assigned to an integer field must not be parsed, even if numeric.
    with pytest.raises(TypeError):
        set_message_fields(msg, {'int8_value': '42'})

    # Bool assigned to an integer field must be rejected.
    with pytest.raises(TypeError):
        set_message_fields(msg, {'int8_value': True})

    # String assigned to a float field must not be parsed.
    with pytest.raises(TypeError):
        set_message_fields(msg, {'float64_value': '1.5'})

    # Bool assigned to a float field must be rejected.
    with pytest.raises(TypeError):
        set_message_fields(msg, {'float64_value': True})

    # Bool assigned to a string field must not be stringified.
    with pytest.raises(TypeError):
        set_message_fields(string_msg, {'string_value': True})

    # NumPy scalars must not be cross-type coerced either.
    with pytest.raises(TypeError):
        set_message_fields(msg, {'bool_value': numpy.int64(1)})
    with pytest.raises(TypeError):
        set_message_fields(msg, {'int8_value': numpy.float64(2.0)})
    with pytest.raises(TypeError):
        set_message_fields(msg, {'int8_value': numpy.bool_(True)})
    with pytest.raises(TypeError):
        set_message_fields(msg, {'float64_value': numpy.bool_(True)})

    # Exact-type assignments must continue to work.
    set_message_fields(msg, {'bool_value': True, 'int8_value': 1, 'float64_value': 1.5})
    assert msg.bool_value is True
    assert msg.int8_value == 1
    assert msg.float64_value == 1.5
    set_message_fields(string_msg, {'string_value': 'hello'})
    assert string_msg.string_value == 'hello'

    # An int assigned to a float field is a deliberately supported widening
    # (e.g. YAML/ros2 CLI commonly parse `5` as int even for a float field).
    set_message_fields(msg, {'float64_value': 5})
    assert msg.float64_value == 5.0
    assert isinstance(msg.float64_value, float)


class IntSubclass(int):
    pass


class FloatSubclass(float):
    pass


class StrSubclass(str):
    pass


def test_set_message_fields_compatible_scalar_types():
    # Python subclasses and the matching NumPy scalars are compatible with a field
    # and must be stored as the builtin field type.
    for field_name, value, expected in (
        ('bool_value', numpy.bool_(True), True),
        ('int8_value', numpy.int64(5), 5),
        ('int8_value', numpy.uint8(5), 5),
        ('int8_value', IntSubclass(5), 5),
        ('float64_value', numpy.float64(1.5), 1.5),
        ('float64_value', numpy.float32(1.5), 1.5),
        ('float64_value', FloatSubclass(1.5), 1.5),
        ('float64_value', numpy.int64(5), 5.0),
        ('float64_value', IntSubclass(5), 5.0),
    ):
        msg = message_fixtures.get_msg_basic_types()[0]
        set_message_fields(msg, {field_name: value})
        assert getattr(msg, field_name) == expected
        assert type(getattr(msg, field_name)) is type(expected)

    msg = message_fixtures.get_msg_basic_types()[0]
    msg.bool_value = True
    set_message_fields(msg, {'bool_value': numpy.bool_(False)})
    assert msg.bool_value is False

    for value in (StrSubclass('hello'), numpy.str_('hello')):
        string_msg = message_fixtures.get_msg_strings()[0]
        set_message_fields(string_msg, {'string_value': value})
        assert string_msg.string_value == 'hello'
        assert type(string_msg.string_value) is str


def test_set_nested_namespaced_fields():
    unbounded_sequence_msg = message_fixtures.get_msg_unbounded_sequences()[1]
    test_values = {
        'basic_types_values': [
            {'float64_value': 42.42, 'int8_value': 42},
            {'float64_value': 11.11, 'int8_value': 11}
        ]
    }
    set_message_fields(unbounded_sequence_msg, test_values)
    assert unbounded_sequence_msg.basic_types_values[0].float64_value == 42.42
    assert unbounded_sequence_msg.basic_types_values[0].int8_value == 42
    assert unbounded_sequence_msg.basic_types_values[0].uint8_value == 0
    assert unbounded_sequence_msg.basic_types_values[1].float64_value == 11.11
    assert unbounded_sequence_msg.basic_types_values[1].int8_value == 11
    assert unbounded_sequence_msg.basic_types_values[1].uint8_value == 0

    arrays_msg = message_fixtures.get_msg_arrays()[0]
    test_values = {
        'basic_types_values': [
            {'float64_value': 42.42, 'int8_value': 42},
            {'float64_value': 11.11, 'int8_value': 11},
            {'float64_value': 22.22, 'int8_value': 22},
        ]
    }
    set_message_fields(arrays_msg, test_values)
    assert arrays_msg.basic_types_values[0].float64_value == 42.42
    assert arrays_msg.basic_types_values[0].int8_value == 42
    assert arrays_msg.basic_types_values[0].uint8_value == 0
    assert arrays_msg.basic_types_values[1].float64_value == 11.11
    assert arrays_msg.basic_types_values[1].int8_value == 11
    assert arrays_msg.basic_types_values[1].uint8_value == 0
    assert arrays_msg.basic_types_values[2].float64_value == 22.22
    assert arrays_msg.basic_types_values[2].int8_value == 22
    assert arrays_msg.basic_types_values[2].uint8_value == 0


def test_set_nested_namespaced_fields_with_same_values():
    unbounded_sequence_msg = message_fixtures.get_msg_unbounded_sequences()[1]
    test_values = {
        'basic_types_values': [
            {'float64_value': 42.42, 'int8_value': 42},
            {'float64_value': 11.11, 'int8_value': 11}
        ]
    }
    set_message_fields(unbounded_sequence_msg, test_values)
    assert unbounded_sequence_msg.basic_types_values[0].float64_value == 42.42
    assert unbounded_sequence_msg.basic_types_values[0].int8_value == 42
    assert unbounded_sequence_msg.basic_types_values[0].uint8_value == 0
    assert unbounded_sequence_msg.basic_types_values[1].float64_value == 11.11
    assert unbounded_sequence_msg.basic_types_values[1].int8_value == 11
    assert unbounded_sequence_msg.basic_types_values[1].uint8_value == 0
    # This should still succeed because test_values should not be modified
    set_message_fields(unbounded_sequence_msg, test_values)
    assert unbounded_sequence_msg.basic_types_values[0].float64_value == 42.42
    assert unbounded_sequence_msg.basic_types_values[0].int8_value == 42
    assert unbounded_sequence_msg.basic_types_values[0].uint8_value == 0
    assert unbounded_sequence_msg.basic_types_values[1].float64_value == 11.11
    assert unbounded_sequence_msg.basic_types_values[1].int8_value == 11
    assert unbounded_sequence_msg.basic_types_values[1].uint8_value == 0


def test_set_message_fields_nested_type():
    msg_basic_types = message_fixtures.get_msg_basic_types()[0]
    msg0 = message_fixtures.get_msg_nested()[0]

    msg0.basic_types_value.bool_value = False
    msg0.basic_types_value.char_value = 3
    msg0.basic_types_value.int32_value = 42

    assert msg0.basic_types_value != msg_basic_types

    test_values = {}
    test_values['basic_types_value'] = msg_basic_types
    set_message_fields(msg0, test_values)

    assert msg0.basic_types_value == msg_basic_types


def test_set_message_fields_header_auto():
    msg = MockMessageStamped()
    values = {'header': 'auto'}
    assert msg.header.stamp.sec == 0 and msg.header.stamp.nanosec == 0
    assert msg.header.frame_id == ''
    timestamp_fields = set_message_fields(
        msg, values, expand_header_auto=True, expand_time_now=True)
    assert timestamp_fields is not None
    for field_setter in timestamp_fields:
        stamp = Time(sec=1, nanosec=2)
        field_setter(stamp)
    assert msg.header.stamp.sec == 1 and msg.header.stamp.nanosec == 2
    assert msg.header.frame_id == ''


def test_set_message_fields_header_auto_not_parsed():
    msg = MockMessageStamped()
    values = {'header': 'auto'}
    assert msg.header.stamp.sec == 0 and msg.header.stamp.nanosec == 0
    assert msg.header.frame_id == ''
    with pytest.raises(
            TypeError, match=r"^Value 'auto' is expected to be a dictionary but is a str$"):
        set_message_fields(msg, values)


def test_set_message_fields_stamp_now():
    msg = MockMessageStamped()
    values = {'header': {'stamp': 'now'}}
    assert msg.header.stamp.sec == 0 and msg.header.stamp.nanosec == 0
    assert msg.header.frame_id == ''
    timestamp_fields = set_message_fields(
        msg, values, expand_header_auto=True, expand_time_now=True)
    assert timestamp_fields is not None
    for field_setter in timestamp_fields:
        stamp = Time(sec=1, nanosec=2)
        field_setter(stamp)
    assert msg.header.stamp.sec == 1 and msg.header.stamp.nanosec == 2
    assert msg.header.frame_id == ''


def test_set_message_fields_stamp_now_not_parsed():
    msg = MockMessageStamped()
    values = {'header': {'stamp': 'now'}}
    assert msg.header.stamp.sec == 0 and msg.header.stamp.nanosec == 0
    assert msg.header.frame_id == ''
    with pytest.raises(
            TypeError, match=r"^Value 'now' is expected to be a dictionary but is a str$"):
        set_message_fields(msg, values)


def test_set_message_fields_stamp_now_with_frame_id():
    msg = MockMessageStamped()
    values = {'header': {'stamp': 'now', 'frame_id': 'hello'}}
    assert msg.header.stamp.sec == 0 and msg.header.stamp.nanosec == 0
    assert msg.header.frame_id == ''
    timestamp_fields = set_message_fields(
        msg, values, expand_header_auto=True, expand_time_now=True)
    assert timestamp_fields is not None
    for field_setter in timestamp_fields:
        stamp = Time(sec=1, nanosec=2)
        field_setter(stamp)
    assert msg.header.stamp.sec == 1 and msg.header.stamp.nanosec == 2
    assert msg.header.frame_id == 'hello'


def test_set_message_fields_stamp_now_with_timestamp_fields():
    msg = MockMessageWithStampFields()
    values = {'timestamp1': 'now', 'timestamp2': 'now'}
    assert msg.timestamp1.sec == 0 and msg.timestamp1.nanosec == 0
    assert msg.timestamp2.sec == 0 and msg.timestamp2.nanosec == 0
    timestamp_fields = set_message_fields(
        msg, values, expand_header_auto=True, expand_time_now=True)
    assert timestamp_fields is not None
    for field_setter in timestamp_fields:
        stamp = Time(sec=1, nanosec=2)
        field_setter(stamp)
    assert msg.timestamp1.sec == 1 and msg.timestamp1.nanosec == 2
    assert msg.timestamp2.sec == 1 and msg.timestamp2.nanosec == 2
