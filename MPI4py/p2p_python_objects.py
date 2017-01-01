#Point-to-Point Communication
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print('my rank is: ', rank)
if rank == 0:
	data1 = {"a" : 7, "b": 3.1415}
	data2 = {'c': 9, "d": 2.718}
	comm.send(data1, dest=1, tag=11)
	comm.send(data2, dest=2, tag=11)

elif rank != 0:
	data = comm.recv(source=0, tag=11)
	print("data in process {} is {}".format(rank, data))

