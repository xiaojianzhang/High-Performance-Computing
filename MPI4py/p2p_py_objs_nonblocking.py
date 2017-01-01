#Python objects with non-blocking communication
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
	data_send = {'a': 7, 'b': 3.14}
	req = comm.isend(data_send, dest=1, tag=11)
	req.wait()
	req = comm.irecv(source=1, tag=11)
	data_rec = req.wait()
	print('process {} received data {}'.format(rank, data_rec))

elif rank == 1:
	data_send = {'c':98, 'd': 2.718}
	req = comm.irecv(source=0, tag=11)
	data_rec = req.wait()
	req = comm.isend(data_send, dest=0, tag=11)
	req.wait()
	print('process {} received data {}'.format(rank, data_rec))