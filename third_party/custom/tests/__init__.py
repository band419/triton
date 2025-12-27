"""
Custom SIMT Backend Tests

Test suite for the custom SIMT backend compilation pipeline.
"""

from .simple_matmul import (
    simple_matmul_kernel,
    vector_add_kernel,
)

__all__ = [
    'simple_matmul_kernel',
    'vector_add_kernel',
]
