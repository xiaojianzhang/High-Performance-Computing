#scalar advection equation: q_t + u * q_x = 0
#where u is a constant.

from mpi4py import MPI
import numpy

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
u = 1.0
num_cells = 1000
delta_x = 1.0 / float(num_cells)
cell_centers = numpy.arange(-delta_x / 2.0, 1.0 + delta_x, delta_x)

t = 0.0
T = 0.5
CFL = 0.8
delta_t = CFL * delta_x / u


def cell_centers_decomposition(cell_centers, size):
	if size == 1:
		return cell_centers.reshape(1,cell_centers.shape[0])

	else:
		q = len(cell_centers) // size
		cell_centers=[]
		for i in range(size):
			if i < size - 1 and i > 0
				cell_centers.append(cell_centers[i*q-1 : (i+1)q+1])
			elif i == 0:
				cell_centers.append(cell_centers[:q+1])
			elif i == size - 1:
				cell_centers.append(cell_centers[i*q-1:])
		cell_centers = numpy.asarray(cell_centers)
		return cell_centers