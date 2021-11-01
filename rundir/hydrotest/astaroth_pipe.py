import os
import numpy as np
# no dataclasses before python 3.7 :(
# from dataclasses import dataclass 
from multiprocessing import Process
from time import sleep


data_fname = "pipe2py_data"
status_fname = "pipe2py_status"
internal_start_fname = "pipe2py_internal_start"
internal_fin_fname = "pipe2py_internal_fin"

internal_start_msg = b"start"
internal_finish_msg = b"fin"

status_ready = np.array([1],dtype=np.int32).tobytes()
data_fd = None
status_fd = None
internal_start_fd = None
internal_fin_fd = None

# data gets send in chunks of blocksize bytes
# at most presend_ready chunks will be in the pipe at any moment
blocksize = 4096
presend_ready = 0

listener = None

def test_output():
    import sys
    print("stderr visible???", file=sys.stderr)
    print("stdout visible??")
    raise ValueError("Exception visible???")
    print("stderr visible 2  ???", file=sys.stderr)
    print("stdout visible 2 ??")

def init_astaroth_pipe(pipe_folder):

    global data_fname, status_fname, internal_fin_fname, internal_start_fname
    data_fname = pipe_folder + "/" + data_fname
    status_fname = pipe_folder + "/" + status_fname 
    internal_fin_fname = pipe_folder + "/" + internal_fin_fname 
    internal_start_fname = pipe_folder + "/" + internal_start_fname 
    global data_fd, status_fd, listener, internal_start_fd, internal_fin_fd

    #os.system(f"rm -f {data_fname} {status_fname} {internal_start_fname} {internal_fin_fname}")

    #os.mkfifo(data_fname)
    #os.mkfifo(status_fname)
    #os.mkfifo(internal_start_fname)
    #os.mkfifo(internal_fin_fname)

    print("opening data pipe for reading")
    data_fd = open(data_fname, "rb", 0)
    print("opening status pipe for writing")
    status_fd = open(status_fname, "wb", 0)


    # spawn a process to match the other end of the pipe (and never do anything with it)

    print("spawning listener process")
    #listener = Process(target=lambda:print("print in process"))
    listener = Process(target=dummy_listener)
    print("starting listener process")
    try:
        print("in try")
        listener.start()
        print("in try 2")
    except Exception as e:
        print("in except")
        print(e)
    else:
        print("start did not throw")
    finally:
        print("listener start did something")



    print("opening internal pipes")
    internal_start_fd = open(internal_start_fname, "rb", 0)
    internal_fin_fd = open(internal_fin_fname, "wb", 0)


    start_msg = internal_start_fd.read(len(internal_start_msg))
    assert(start_msg == internal_start_msg)

    for i in range(presend_ready):
        _send_ready_signal()


def dummy_listener():


    print("opening datapipe for writing (in the dummy)", flush=True)
    data_fd_write_end = open(data_fname, "wb", 0)
    print("opening statuspipe for reading (in the dummy)")
    status_fd_read_end = open(status_fname, "rb", 0)

    internal_start_fd_dummy_side = open(internal_start_fname, "wb", 0)
    internal_fin_fd_dummy_side = open(internal_fin_fname, "rb", 0)

    internal_start_fd_dummy_side.write(internal_start_msg)
    
    # hang here the entire time, just keep the fd open...
    

    fin_msg = internal_fin_fd_dummy_side.read(len(internal_finish_msg))
    assert(fin_msg == internal_finish_msg)

    data_fd_write_end.close()
    status_fd_read_end.close()
    internal_start_fd_dummy_side.close()
    internal_fin_fd_dummy_side.close()


def close_astaroth_pipe():

    internal_fin_fd.write(internal_finish_msg)

    data_fd.close()
    status_fd.close()
    internal_start_fd.close()
    internal_fin_fd.close()

    listener.join()


def _send_ready_signal(num_sent=[0]):

    assert(status_fd)
    num_sent[0] += 1
    status_fd.write(status_ready)
    #print(f"sent {num_sent} ready-signals")

#@dataclass
class Header():
    floatsize : int
    nx : int
    ny : int
    nz : int
    timestep : int
    phystime : float
    name : str

    ####
    #unnecessary if this was a dataclass
    ####

    def __init__(self,floatsize, nx, ny, nz, timestep, phystime, name):
        self.floatsize =floatsize 
        self.nx =nx 
        self.ny =ny 
        self.nz =nz 
        self.timestep =timestep 
        self.phystime =phystime 
        self.name =name 
    
    def __str__(self):
        return str((
            self.floatsize, 
            self.nx ,
            self.ny ,
            self.nz ,
            self.timestep ,
            self.phystime ,
            self.name
        ))



#@dataclass
class VelocityFieldInfo():
    fullname : str
    dim : str
    phystime : float
    timestep : int

    def __init__(self, header):

        h = header
        prefix = "VTXBUF_UU"
        if h.name.startswith(prefix):
            self.dim = h.name[len(prefix)]
            assert(self.dim) in "XYZ"
        else:
            raise ValueError(f"header {header} does not seem to represent a velocity field")

        self.fullname = h.name
        self.phystime = h.phystime
        self.timestep = h.timestep
    
    ######
    # unnecessary if this is a dataclass 
    ######

    def __str__(self):
        return str((self.dim, self.phystime, self.timestep, self.fullname))


def _get_header():
    
    
    header_data = data_fd.read(128)
    floatsize, nx, ny, nz, timestep = list(np.frombuffer(header_data[:20],dtype="int32"))
    phystime = np.frombuffer(header_data[20:28],dtype="<f8")[0]
    name = header_data[28:].split(sep=b"\0", maxsplit=1)[0].decode("UTF-8")
    
    return Header(floatsize, nx, ny, nz, timestep, phystime, name)



def get_array_blocking():


    _send_ready_signal()
    h = _get_header()

    num_blocks = h.nx*h.ny*h.nz*h.floatsize // blocksize
    assert(h.nx*h.ny*h.nz*h.floatsize % blocksize == 0)
    assert(blocksize%h.floatsize==0)

    msg = []
    for i in range(num_blocks):
        _send_ready_signal()
        msg.append(data_fd.read(blocksize))
        # arr_head = np.frombuffer(msg[-1], dtype="<f4")
        # print("printing arrayhead and leaving")
        # with open("read_by_python.txt", "w") as f:
        #     for i in range(10):
        #         print(arr_head[i], file=f)
        # exit(1)
    msg = b"".join(msg)


    # this actually assumes an endianness...
    if h.floatsize == 4:
        dtype = "<f4"
    elif h.floatsize == 8:
        dtype = "<f8"
    else:
        raise ValueError(f"floatsize should be 4 or 8, received {h.floatsize} from header")

    arr = np.frombuffer(msg, dtype=dtype).reshape(h.nx, h.ny, h.nz, order="F")
    info = VelocityFieldInfo(h)

    return arr, info
