from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterable, Sequence


MAGIC = b"GPNPUVEC"
VERSION = 1
HEADER_SIZE = 64


class BlobFlags:
    NO_RELOC = 1 << 0
    HAS_RODATA = 1 << 1
    HAS_META = 1 << 2


def _align_up(value: int, align: int) -> int:
    if align <= 0 or (align & (align - 1)) != 0:
        raise ValueError("align must be a power of two")
    return (value + (align - 1)) & ~(align - 1)


def build_meta_tlv(records: Sequence[tuple[int, bytes]]) -> bytes:
    """Build a .meta segment as TLV records.

    Layout: repeated {type_u16, size_u16, payload[size]}.
    """
    out = bytearray()
    for rec_type, payload in records:
        if not (0 <= rec_type <= 0xFFFF):
            raise ValueError("record type must fit in u16")
        if payload is None:
            payload = b""
        if len(payload) > 0xFFFF:
            raise ValueError("record payload too large for u16 size")
        out += struct.pack("<HH", rec_type, len(payload))
        out += payload
    return bytes(out)


def _pack_u16(v: int) -> bytes:
    return struct.pack("<H", v)


def _pack_u32(v: int) -> bytes:
    return struct.pack("<I", v)


def pack_blob_v1(*,
                 text: bytes,
                 rodata: bytes = b"",
                 meta: bytes = b"",
                 entry_off: int | None = None,
                 text_align: int = 16,
                 rodata_align: int = 16) -> bytes:
    """Pack a Scheme-A blob (flat bytes, no relocations) with a fixed 64B header."""

    if entry_off is None:
        entry_off = HEADER_SIZE

    if len(MAGIC) != 8:
        raise AssertionError("MAGIC must be 8 bytes")
    if entry_off < HEADER_SIZE:
        raise ValueError("entry_off must be >= HEADER_SIZE")

    flags = BlobFlags.NO_RELOC
    if rodata:
        flags |= BlobFlags.HAS_RODATA
    if meta:
        flags |= BlobFlags.HAS_META

    # Layout
    text_off = _align_up(HEADER_SIZE, text_align)
    text_size = len(text)

    rodata_off = 0
    rodata_size = 0
    meta_off = 0
    meta_size = 0

    cursor = text_off + text_size

    if rodata:
        rodata_off = _align_up(cursor, rodata_align)
        rodata_size = len(rodata)
        cursor = rodata_off + rodata_size

    if meta:
        meta_off = _align_up(cursor, 4)
        meta_size = len(meta)
        cursor = meta_off + meta_size

    total_size = cursor

    header = bytearray()
    header += MAGIC
    header += _pack_u16(VERSION)
    header += _pack_u16(flags)
    header += _pack_u32(HEADER_SIZE)
    header += _pack_u32(entry_off)
    header += _pack_u32(text_off)
    header += _pack_u32(text_size)
    header += _pack_u32(rodata_off)
    header += _pack_u32(rodata_size)
    header += _pack_u32(meta_off)
    header += _pack_u32(meta_size)

    # reserved to 64B
    if len(header) > HEADER_SIZE:
        raise AssertionError("header overflow")
    header += b"\x00" * (HEADER_SIZE - len(header))

    blob = bytearray(total_size)
    blob[0:HEADER_SIZE] = header

    blob[text_off:text_off + text_size] = text
    if rodata:
        blob[rodata_off:rodata_off + rodata_size] = rodata
    if meta:
        blob[meta_off:meta_off + meta_size] = meta

    return bytes(blob)
