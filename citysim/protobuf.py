from typing import Any, Awaitable, TypeVar, Union, Dict
from google.protobuf.message import Message
from google.protobuf.json_format import MessageToDict

__all__ = ["parse", "async_parse"]

T = TypeVar("T", bound=Message)


def parse(res: T, dict_return: bool) -> Union[Dict[str, Any], T]:
    """
    将Protobuf返回值转换为dict或者原始值
    Convert Protobuf return value to dict or original value
    """
    if dict_return:
        return MessageToDict(
            res,
            including_default_value_fields=True,
            preserving_proto_field_name=True,
            use_integers_for_enums=True,
        )
    else:
        return res


async def async_parse(res: Awaitable[T], dict_return: bool) -> Union[Dict[str, Any], T]:
    """
    将Protobuf await返回值转换为dict或者原始值
    Convert Protobuf await return value to dict or original value
    """
    if dict_return:
        return MessageToDict(
            await res,
            including_default_value_fields=True,
            preserving_proto_field_name=True,
            use_integers_for_enums=True,
        )
    else:
        return await res
