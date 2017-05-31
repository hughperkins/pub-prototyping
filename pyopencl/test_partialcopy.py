"""
Not sure how to do partial buffer copy on python
"""
import pyopencl as cl
import numpy as np
import os


def enqueue_write_buffer_ext(queue, mem, hostbuf, device_offset=0, size=None,
                             wait_for=None, is_blocking=True):
    ptr_event = cl.cffi_cl._ffi.new('clobj_t*')
    c_buf, actual_size, c_ref = cl.cffi_cl._c_buffer_from_obj(hostbuf, retain=True)
    if size is None:
        size = actual_size
    c_wait_for, num_wait_for = cl.cffi_cl._clobj_list(wait_for)
    nanny_event = cl.cffi_cl.NannyEvent._handle(hostbuf, c_ref)
    cl.cffi_cl._handle_error(cl.cffi_cl._lib.enqueue_write_buffer(
            ptr_event, queue.ptr, mem.ptr, c_buf, size, device_offset, c_wait_for, num_wait_for, bool(True),
            nanny_event))
    return cl.cffi_cl.NannyEvent._create(ptr_event[0])


def enqueue_read_buffer_ext(queue, mem, hostbuf, device_offset=0, size=None,
                            wait_for=None, is_blocking=True):
    ptr_event = cl.cffi_cl._ffi.new('clobj_t*')
    c_buf, actual_size, c_ref = cl.cffi_cl._c_buffer_from_obj(hostbuf, retain=True)
    if size is None:
        size = actual_size
    c_wait_for, num_wait_for = cl.cffi_cl._clobj_list(wait_for)
    nanny_event = cl.cffi_cl.NannyEvent._handle(hostbuf, c_ref)
    cl.cffi_cl._handle_error(cl.cffi_cl._lib.enqueue_read_buffer(
            ptr_event, queue.ptr, mem.ptr, c_buf, size, device_offset, c_wait_for, num_wait_for, bool(True),
            nanny_event))
    return cl.cffi_cl.NannyEvent._create(ptr_event[0])


def test_partial_copy():
    gpu_idx = int(os.environ.get('TARGET_GPU', 0))

    platforms = cl.get_platforms()
    i = 0
    context = None
    for platform in platforms:
        gpu_devices = platform.get_devices(device_type=cl.device_type.GPU)
        if gpu_idx < i + len(gpu_devices):
            context = cl.Context(devices=[gpu_devices[gpu_idx - i]])
            break
        i += len(gpu_devices)

    if context is None:
        raise Exception('unable to find gpu at index %s' % gpu_idx)
    print('context', context)

    queue = cl.CommandQueue(context)

    N = 10

    src_host = np.random.uniform(0, 1, size=(N,)).astype(np.float32) + 1.0
    dst_host = np.zeros(N, dtype=np.float32)

    huge_buf_gpu = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=4096)

    enqueue_write_buffer_ext(queue=queue, mem=huge_buf_gpu, hostbuf=src_host, device_offset=3 * 4, size=6 * 4)
    queue.finish()
    enqueue_read_buffer_ext(queue, huge_buf_gpu, dst_host, 2 * 4, 4 * 4)
    queue.finish()
    print('src_host', src_host)
    print('dst_host', dst_host)
    assert dst_host[0] == 0
    assert dst_host[4] == 0
    assert dst_host[1] == src_host[0]
    assert dst_host[3] == src_host[2]
