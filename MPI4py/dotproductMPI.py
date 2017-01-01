#"to run" syntax example: mpiexec -n 4 python26 dotProductMPI.py
from mpi4py import MPI
import numpy

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n = 10000000    #length of vectors

#arbitrary example vectors, generated to be evenly divided by the number of
#processes for convenience

x = numpy.linspace(0,10,n) if comm.rank == 0 else None
y = numpy.linspace(10,40,n) if comm.rank == 0 else None

#initialize as numpy arrays
dot = numpy.array([0.])
local_n = numpy.array([0])

#test for conformability
if rank == 0:
        if (n != y.size):
                print "vector length mismatch"
                comm.Abort()

        #currently, our program cannot handle sizes that are not evenly divided by
        #the number of processors
        if(n % size != 0):
                print "the number of processors must evenly divide n."
                comm.Abort()

        #length of each process's portion of the original vector
        local_n = numpy.array([n/size])

#communicate local array size to all processes
comm.Bcast(local_n, root=0)

#initialize as numpy arrays
local_x = numpy.zeros(local_n[0])
local_y = numpy.zeros(local_n[0])

#divide up vectors
comm.Scatter(x, local_x, root=0)
comm.Scatter(y, local_y, root=0)

#local computation of dot product
local_dot = numpy.array([numpy.dot(local_x, local_y)])

#sum the results of each
comm.Reduce(local_dot, dot, op = MPI.SUM)

if (rank == 0):
        print "The dot product is", dot[0], "computed in parallel"
        print "and", numpy.dot(x,y), "computed serially"