#!~/anaconda2/bin/python
#example to run: mpiexec -n 4 python MPI_2d_advection_virtual_topology.py 2 2 or mpirun -n 4 python MPI_2d_advection_virtual_topology.py 2 2
#parallel 2d advection solver using upwind method
import numpy
import sys
from mpi4py import MPI

#takes in command-line arguments [rows,cols] to create Cartesian topologies
grid_rows = int(sys.argv[1])
grid_columns = int(sys.argv[2])

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
if rank == 0:
	start = MPI.Wtime()

up=0
down=1
left=2
right=3
neighbor_processes = [0, 0, 0, 0]

a=1.0
b=1.0
u=1.0
v=1.0
mx=2000
my=2000
CFL=0.8
T=0.5
init_cond_func=lambda x,y: numpy.exp(-200.0*((x-0.25)**2.0+(y-0.25)**2.0))

def plot_function(x_grids, y_grids, dt, T, results):
	from matplotlib import animation
	import matplotlib.pyplot as plt
	
	t = numpy.arange(0.0,T+1e-7,dt)
	k = numpy.arange(len(results))
	fig = plt.figure()
	axes = fig.add_subplot(1,1,1)
	xx, yy = numpy.meshgrid(x_grids, y_grids)
	z = results[0]
	surf = plt.pcolormesh(xx, yy, z, cmap='RdBu', vmin=0.0, vmax=1.0)
	def plot_q(i,z,surf):
		axes.clear()
		z = results[int(i)]
		surf = plt.pcolormesh(xx, yy, z, cmap='RdBu', vmin=0.0, vmax=1.0)
		return surf,
	# Animate the solution
	ani = animation.FuncAnimation(fig, plot_q, frames=k, fargs=(z,surf), blit=False)
	plt.colorbar()
	ani.save('MPI_2d_advection_upwind.mp4', writer="mencoder", fps=15)
	plt.show()

if rank == 0:
	print('mx = {}, my = {}'.format(mx,my))
	print('Building a {} x {} grid topology:'.format(grid_rows, grid_columns))

cartesian_communicator = comm.Create_cart((grid_rows,grid_columns), periods=(False,False), reorder=True)

local_row, local_column = cartesian_communicator.Get_coords(cartesian_communicator.rank)

neighbor_processes[up], neighbor_processes[down] = cartesian_communicator.Shift(0,1)
neighbor_processes[left], neighbor_processes[right] = cartesian_communicator.Shift(1,1)

dx = a/float(mx)
dy = b/float(my)

dt = min(CFL*dx/u, CFL*dy/v)
t=dt

#x direction decomposition
x_grids = numpy.arange(dx/2.0, a, dx)
local_x_grids = numpy.array_split(x_grids, grid_columns)[local_column]
first_center = local_x_grids[0]
last_center = local_x_grids[-1]
local_x_grids = numpy.concatenate((numpy.array([first_center-dx]), local_x_grids, numpy.array([last_center+dx])), axis=0)

#y direction decomposition
y_grids = numpy.arange(dy/2.0, b, dy)
local_y_grids = numpy.array_split(y_grids, grid_rows)[local_row]
first_center = local_y_grids[0]
last_center = local_y_grids[-1]
local_y_grids = numpy.concatenate((numpy.array([first_center-dy]), local_y_grids, numpy.array([last_center+dy])), axis=0)

if rank == 0:
	xx,yy = numpy.meshgrid(x_grids, y_grids)
	results=[init_cond_func(xx,yy)]


local_xx, local_yy = numpy.meshgrid(local_x_grids, local_y_grids)
local_Q_old = init_cond_func(local_xx,local_yy) #initial data

#set ghost cells using zero-order extrapolation
local_Q_old[:,0] = local_Q_old[:,1]
local_Q_old[:,-1] = local_Q_old[:,-2]
local_Q_old[0,:] = local_Q_old[1,:]
local_Q_old[-1,:] = local_Q_old[-2,:]

#neighbor data exchange
comm.Barrier()
if rank == 0:
	end_init = MPI.Wtime() #time inilization
	print('Parallel Initialization runs for {} seconds'.format(end_init - start))
	collect = 0.0
	data_exchange_start = MPI.Wtime() #time data exchange 
	data_exchange = 0.0

if neighbor_processes[up] >= 0:
	send_buffer = local_Q_old[1,:]
	#recv_buffer = numpy.zeros(local_x_grids.shape[0])
	recv_buffer = comm.sendrecv(send_buffer, dest=neighbor_processes[up], source=neighbor_processes[up])
	#print('neighbor communication: process {} sended data to process {} and received data from process {}'.format(rank, neighbor_processes[up],neighbor_processes[up]))
	local_Q_old[0,:] = recv_buffer

if neighbor_processes[down] >= 0:
	send_buffer = local_Q_old[-2,:]
	#recv_buffer = numpy.zeros(local_x_grids.shape[0])
	recv_buffer = comm.sendrecv(send_buffer, dest=neighbor_processes[down], source=neighbor_processes[down])
	#print('neighbor communication: process {} sended data to process {} and received data from process {}'.format(rank, neighbor_processes[down],neighbor_processes[down]))
	local_Q_old[-1,:] = recv_buffer

if neighbor_processes[left] >= 0:
	send_buffer = local_Q_old[:,1]
	#recv_buffer = numpy.zeros(local_y_grids.shape[0])
	recv_buffer = comm.sendrecv(send_buffer, dest=neighbor_processes[left], source=neighbor_processes[left])
	#print('neighbor communication: process {} sended data to process {} and received data from process {}'.format(rank, neighbor_processes[left],neighbor_processes[left]))
	local_Q_old[:,0] = recv_buffer

if neighbor_processes[right] >= 0:
	send_buffer = local_Q_old[:,-2]
	#recv_buffer = numpy.zeros(local_y_grids.shape[0])
	recv_buffer = comm.sendrecv(send_buffer, dest=neighbor_processes[right], source=neighbor_processes[right])
	#print('neighbor communication: process {} sended data to process {} and received data from process {}'.format(rank, neighbor_processes[right],neighbor_processes[right]))
	local_Q_old[:,-1] = recv_buffer

#diagonal data exchange
comm.Barrier()
if neighbor_processes[up] >= 0 and neighbor_processes[left] >= 0:
	dest_rank = neighbor_processes[up] - 1
	send_buffer = numpy.array([local_Q_old[1,1]])
	#recv_buffer = numpy.zeros(1)
	recv_buffer = comm.sendrecv(send_buffer, dest=dest_rank, source=dest_rank)
	#print('diagonal communication: process {} sended data to process {} and received data from process {}'.format(rank, dest_rank, dest_rank))
	local_Q_old[0,0] = recv_buffer[0]

if neighbor_processes[down] >= 0 and neighbor_processes[right] >= 0:
	dest_rank = neighbor_processes[down] + 1
	send_buffer = numpy.array([local_Q_old[-2,-2]])
	#recv_buffer = numpy.zeros(1)
	recv_buffer = comm.sendrecv(send_buffer, dest=dest_rank, source=dest_rank)
	#print('diagonal communication: process {} sended data to process {} and received data from process {}'.format(rank, dest_rank, dest_rank))
	local_Q_old[-1,-1] = recv_buffer[0]

if neighbor_processes[up] >= 0 and neighbor_processes[right] >= 0:
	dest_rank = neighbor_processes[up] + 1
	send_buffer = numpy.array([local_Q_old[1,-2]])
	#recv_buffer = numpy.zeros(1)
	recv_buffer = comm.sendrecv(send_buffer, dest=dest_rank, source=dest_rank)
	#print('diagonal communication: process {} sended data to process {} and received data from process {}'.format(rank, dest_rank, dest_rank))
	local_Q_old[0,-1] = recv_buffer[0]

if neighbor_processes[down] >= 0 and neighbor_processes[left] >= 0:
	dest_rank = neighbor_processes[down] - 1
	send_buffer = numpy.array([local_Q_old[-2,1]])
	#recv_buffer = numpy.zeros(1)
	recv_buffer = comm.sendrecv(send_buffer, dest=dest_rank, source=dest_rank)
	#print('diagonal communication: process {} sended data to process {} and received data from process {}'.format(rank, dest_rank, dest_rank))
	local_Q_old[-1,0] = recv_buffer[0]

comm.Barrier()
if rank == 0:
	data_exchange += MPI.Wtime() - data_exchange_start #time data exchange
	concurrent_start = MPI.Wtime() #time concurrent computations
	concurrent = 0.0 #time concurrent computations

#update cell average:
local_Q_new = local_Q_old.copy()
while t <= T+1e-6:
	#x-sweeps:
	for j in range(1,local_y_grids.shape[0]-1):
		local_Q_new[j,1:-1] = local_Q_old[j,1:-1] - u * dt * (local_Q_old[j,1:-1] - local_Q_old[j,0:-2]) / dx

	#y-sweeps:
	for i in range(1,local_x_grids.shape[0]-1):
		local_Q_new[1:-1,i] = local_Q_new[1:-1,i] - v * dt * (local_Q_new[1:-1,i] - local_Q_new[0:-2,i]) / dy

	#collect data from other processes
	comm.Barrier()
	if rank == 0:
		concurrent += MPI.Wtime() - concurrent_start #time concurent computations
		collect_start = MPI.Wtime()                  #time data collection
		Q_new = local_Q_new[1:-1,1:-1]
		data={'0':Q_new}
		for pid in range(1,size):
			recv_buffer = comm.recv(source=pid)
			data[str(pid)] = recv_buffer

		Q_new=[]
		rank_count=0
		for j in range(grid_rows):
			Q_new_sub=[]
			for i in range(grid_columns):
				Q_new_sub.append(data[str(rank_count)])
				rank_count+=1
			Q_new.append(numpy.concatenate(list(Q_new_sub),axis=1))
		Q_new = numpy.concatenate(list(Q_new), axis=0)
		results.append(Q_new)

		collect += MPI.Wtime() - collect_start #time data collection

	else:
		send_buffer = local_Q_new[1:-1,1:-1]
		comm.send(send_buffer, dest=0)

	local_Q_old = local_Q_new.copy()
	#set ghost cells using zero-order extrapolation
	local_Q_old[:,0] = local_Q_old[:,1]
	local_Q_old[:,-1] = local_Q_old[:,-2]
	local_Q_old[0,:] = local_Q_old[1,:]
	local_Q_old[-1,:] = local_Q_old[-2,:]

	#neighbor data exchange
	comm.Barrier()
	if rank == 0:
		data_exchange_start = MPI.Wtime()
	if neighbor_processes[up] >= 0:
		send_buffer = local_Q_old[1,:]
		#recv_buffer = numpy.zeros(local_x_grids.shape[0])
		recv_buffer = comm.sendrecv(send_buffer, dest=neighbor_processes[up], source=neighbor_processes[up])
		#print('neighbor communication: process {} sended data to process {} and received data from process {}'.format(rank, neighbor_processes[up],neighbor_processes[up]))
		local_Q_old[0,:] = recv_buffer

	if neighbor_processes[down] >= 0:
		send_buffer = local_Q_old[-2,:]
		#recv_buffer = numpy.zeros(local_x_grids.shape[0])
		recv_buffer = comm.sendrecv(send_buffer, dest=neighbor_processes[down], source=neighbor_processes[down])
		#print('neighbor communication: process {} sended data to process {} and received data from process {}'.format(rank, neighbor_processes[down],neighbor_processes[down]))
		local_Q_old[-1,:] = recv_buffer

	if neighbor_processes[left] >= 0:
		send_buffer = local_Q_old[:,1]
		#recv_buffer = numpy.zeros(local_y_grids.shape[0])
		recv_buffer = comm.sendrecv(send_buffer, dest=neighbor_processes[left], source=neighbor_processes[left])
		#print('neighbor communication: process {} sended data to process {} and received data from process {}'.format(rank, neighbor_processes[left],neighbor_processes[left]))
		local_Q_old[:,0] = recv_buffer

	if neighbor_processes[right] >= 0:
		send_buffer = local_Q_old[:,-2]
		#recv_buffer = numpy.zeros(local_y_grids.shape[0])
		recv_buffer = comm.sendrecv(send_buffer, dest=neighbor_processes[right], source=neighbor_processes[right])
		#print('neighbor communication: process {} sended data to process {} and received data from process {}'.format(rank, neighbor_processes[right],neighbor_processes[right]))
		local_Q_old[:,-1] = recv_buffer

	#diagonal data exchange
	comm.Barrier()
	if neighbor_processes[up] >= 0 and neighbor_processes[left] >= 0:
		dest_rank = neighbor_processes[up] - 1
		send_buffer = numpy.array([local_Q_old[1,1]])
		#recv_buffer = numpy.zeros(1)
		recv_buffer = comm.sendrecv(send_buffer, dest=dest_rank, source=dest_rank)
		#print('diagonal communication: process {} sended data to process {} and received data from process {}'.format(rank, dest_rank, dest_rank))
		local_Q_old[0,0] = recv_buffer[0]

	if neighbor_processes[down] >= 0 and neighbor_processes[right] >= 0:
		dest_rank = neighbor_processes[down] + 1
		send_buffer = numpy.array([local_Q_old[-2,-2]])
		#recv_buffer = numpy.zeros(1)
		recv_buffer = comm.sendrecv(send_buffer, dest=dest_rank, source=dest_rank)
		#print('diagonal communication: process {} sended data to process {} and received data from process {}'.format(rank, dest_rank, dest_rank))
		local_Q_old[-1,-1] = recv_buffer[0]

	if neighbor_processes[up] >= 0 and neighbor_processes[right] >= 0:
		dest_rank = neighbor_processes[up] + 1
		send_buffer = numpy.array([local_Q_old[1,-2]])
		#recv_buffer = numpy.zeros(1)
		recv_buffer = comm.sendrecv(send_buffer, dest=dest_rank, source=dest_rank)
		#print('diagonal communication: process {} sended data to process {} and received data from process {}'.format(rank, dest_rank, dest_rank))
		local_Q_old[0,-1] = recv_buffer[0]

	if neighbor_processes[down] >= 0 and neighbor_processes[left] >= 0:
		dest_rank = neighbor_processes[down] - 1
		send_buffer = numpy.array([local_Q_old[-2,1]])
		#recv_buffer = numpy.zeros(1)
		recv_buffer = comm.sendrecv(send_buffer, dest=dest_rank, source=dest_rank)
		#print('diagonal communication: process {} sended data to process {} and received data from process {}'.format(rank, dest_rank, dest_rank))
		local_Q_old[-1,0] = recv_buffer[0]

	local_Q_new = local_Q_old.copy()
	t += dt
	comm.Barrier()
	if rank == 0:
		data_exchange += MPI.Wtime() - data_exchange_start #time data exchange
		concurrent_start = MPI.Wtime() #time concurrent computations


if rank == 0:
	end = MPI.Wtime()
	print('Concurrent part runs for {} seconds'.format(concurrent))
	print('Data exchange runs for {} seconds'.format(data_exchange))
	print('Data collection from local runs for {} seconds'.format(collect))
	print('Parallel Program runs for {} seconds'.format(end-start))
	#plot_function(x_grids, y_grids, dt, T, results)
