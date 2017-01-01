import numpy
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

up=0
down=1
left=2
right=3
neighbor_processes = [0, 0, 0, 0]

if rank == 0:
	start = MPI.Wtime()
a=1.0
b=1.0
u=1.0
v=1.0
mx=800
my=800
CFL=0.8
T=0.5
init_cond_func=lambda x,y: numpy.exp(-200.0*((x-0.25)**2.0+(y-0.25)**2.0))

def plot_function(x_grids, y_grids, dt, T, results, method):
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
	ani.save('2d_advection_{}.mp4'.format(method), writer="mencoder", fps=15)
	plt.show()

grid_rows = int(numpy.floor(numpy.sqrt(comm.size)))
grid_columns = size // grid_rows
if grid_rows*grid_columns < size:
	grid_rows -= 1
	grid_columns = size // grid_rows
if rank == 0:
	print('Building a {} x {} grid topology:'.format(grid_rows, grid_columns))

cartesian_communicator = comm.Create_cart((grid_rows,grid_columns), periods=(False,False), reorder=True)

local_row, local_column = cartesian_communicator.Get_coords(cartesian_communicator.rank)

neighbor_processes[up], neighbor_processes[down] = cartesian_communicator.Shift(0,1)
neighbor_processes[left], neighbor_processes[right] = cartesian_communicator.Shift(1,1)
'''
print('process {}: row {} column {} neighbor_porcess[up]={}  neighbor_porcess[down]={}  neighbor_porcess[left]={}  neighbor_porcess[right]={}'.format(\
	   rank, local_row, local_column, neighbor_processes[up], neighbor_processes[down], neighbor_processes[left], neighbor_processes[right]))
'''
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

'''
print('process {} \
	   local_x_grids: {} \
	   local_y_grids: {}'.format(rank, local_x_grids, local_y_grids))
'''

local_xx, local_yy = numpy.meshgrid(local_x_grids, local_y_grids)
local_Q_old = init_cond_func(local_xx,local_yy) #initial data

#set ghost cells using zero-order extrapolation
local_Q_old[:,0] = local_Q_old[:,1]
local_Q_old[:,-1] = local_Q_old[:,-2]
local_Q_old[0,:] = local_Q_old[1,:]
local_Q_old[-1,:] = local_Q_old[-2,:]

#neighbor data exchange
comm.Barrier()
if neighbor_processes[up] >= 0:
	send_buffer = local_Q_old[1,:]
	recv_buffer = numpy.zeros(local_x_grids.shape[0])
	recv_buffer = comm.sendrecv(send_buffer, dest=neighbor_processes[up], source=neighbor_processes[up])
	#print('neighbor communication: process {} sended data to process {} and received data from process {}'.format(rank, neighbor_processes[up],neighbor_processes[up]))
	local_Q_old[0,:] = recv_buffer

if neighbor_processes[down] >= 0:
	send_buffer = local_Q_old[-2,:]
	recv_buffer = numpy.zeros(local_x_grids.shape[0])
	recv_buffer = comm.sendrecv(send_buffer, dest=neighbor_processes[down], source=neighbor_processes[down])
	#print('neighbor communication: process {} sended data to process {} and received data from process {}'.format(rank, neighbor_processes[down],neighbor_processes[down]))
	local_Q_old[-1,:] = recv_buffer

if neighbor_processes[left] >= 0:
	send_buffer = local_Q_old[:,1]
	recv_buffer = numpy.zeros(local_y_grids.shape[0])
	recv_buffer = comm.sendrecv(send_buffer, dest=neighbor_processes[left], source=neighbor_processes[left])
	#print('neighbor communication: process {} sended data to process {} and received data from process {}'.format(rank, neighbor_processes[left],neighbor_processes[left]))
	local_Q_old[:,0] = recv_buffer

if neighbor_processes[right] >= 0:
	send_buffer = local_Q_old[:,-2]
	recv_buffer = numpy.zeros(local_y_grids.shape[0])
	recv_buffer = comm.sendrecv(send_buffer, dest=neighbor_processes[right], source=neighbor_processes[right])
	#print('neighbor communication: process {} sended data to process {} and received data from process {}'.format(rank, neighbor_processes[right],neighbor_processes[right]))
	local_Q_old[:,-1] = recv_buffer

#diagonal data exchange
comm.Barrier()
if neighbor_processes[up] >= 0 and neighbor_processes[left] >= 0:
	dest_rank = neighbor_processes[up] - 1
	send_buffer = numpy.array([local_Q_old[1,1]])
	recv_buffer = numpy.zeros(1)
	recv_buffer = comm.sendrecv(send_buffer, dest=dest_rank, source=dest_rank)
	#print('diagonal communication: process {} sended data to process {} and received data from process {}'.format(rank, dest_rank, dest_rank))
	local_Q_old[0,0] = recv_buffer[0]

if neighbor_processes[down] >= 0 and neighbor_processes[right] >= 0:
	dest_rank = neighbor_processes[down] + 1
	send_buffer = numpy.array([local_Q_old[-2,-2]])
	recv_buffer = numpy.zeros(1)
	recv_buffer = comm.sendrecv(send_buffer, dest=dest_rank, source=dest_rank)
	#print('diagonal communication: process {} sended data to process {} and received data from process {}'.format(rank, dest_rank, dest_rank))
	local_Q_old[-1,-1] = recv_buffer[0]

if neighbor_processes[up] >= 0 and neighbor_processes[right] >= 0:
	dest_rank = neighbor_processes[up] + 1
	send_buffer = numpy.array([local_Q_old[1,-2]])
	recv_buffer = numpy.zeros(1)
	recv_buffer = comm.sendrecv(send_buffer, dest=dest_rank, source=dest_rank)
	#print('diagonal communication: process {} sended data to process {} and received data from process {}'.format(rank, dest_rank, dest_rank))
	local_Q_old[0,-1] = recv_buffer[0]

if neighbor_processes[down] >= 0 and neighbor_processes[left] >= 0:
	dest_rank = neighbor_processes[down] - 1
	send_buffer = numpy.array([local_Q_old[-2,1]])
	recv_buffer = numpy.zeros(1)
	recv_buffer = comm.sendrecv(send_buffer, dest=dest_rank, source=dest_rank)
	#print('diagonal communication: process {} sended data to process {} and received data from process {}'.format(rank, dest_rank, dest_rank))
	local_Q_old[-1,0] = recv_buffer[0]

comm.Barrier()
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
	if rank == 0:
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
	if neighbor_processes[up] >= 0:
		send_buffer = local_Q_old[1,:]
		recv_buffer = numpy.zeros(local_x_grids.shape[0])
		recv_buffer = comm.sendrecv(send_buffer, dest=neighbor_processes[up], source=neighbor_processes[up])
		#print('neighbor communication: process {} sended data to process {} and received data from process {}'.format(rank, neighbor_processes[up],neighbor_processes[up]))
		local_Q_old[0,:] = recv_buffer

	if neighbor_processes[down] >= 0:
		send_buffer = local_Q_old[-2,:]
		recv_buffer = numpy.zeros(local_x_grids.shape[0])
		recv_buffer = comm.sendrecv(send_buffer, dest=neighbor_processes[down], source=neighbor_processes[down])
		#print('neighbor communication: process {} sended data to process {} and received data from process {}'.format(rank, neighbor_processes[down],neighbor_processes[down]))
		local_Q_old[-1,:] = recv_buffer

	if neighbor_processes[left] >= 0:
		send_buffer = local_Q_old[:,1]
		recv_buffer = numpy.zeros(local_y_grids.shape[0])
		recv_buffer = comm.sendrecv(send_buffer, dest=neighbor_processes[left], source=neighbor_processes[left])
		#print('neighbor communication: process {} sended data to process {} and received data from process {}'.format(rank, neighbor_processes[left],neighbor_processes[left]))
		local_Q_old[:,0] = recv_buffer

	if neighbor_processes[right] >= 0:
		send_buffer = local_Q_old[:,-2]
		recv_buffer = numpy.zeros(local_y_grids.shape[0])
		recv_buffer = comm.sendrecv(send_buffer, dest=neighbor_processes[right], source=neighbor_processes[right])
		#print('neighbor communication: process {} sended data to process {} and received data from process {}'.format(rank, neighbor_processes[right],neighbor_processes[right]))
		local_Q_old[:,-1] = recv_buffer

	#diagonal data exchange
	comm.Barrier()
	if neighbor_processes[up] >= 0 and neighbor_processes[left] >= 0:
		dest_rank = neighbor_processes[up] - 1
		send_buffer = numpy.array([local_Q_old[1,1]])
		recv_buffer = numpy.zeros(1)
		recv_buffer = comm.sendrecv(send_buffer, dest=dest_rank, source=dest_rank)
		#print('diagonal communication: process {} sended data to process {} and received data from process {}'.format(rank, dest_rank, dest_rank))
		local_Q_old[0,0] = recv_buffer[0]

	if neighbor_processes[down] >= 0 and neighbor_processes[right] >= 0:
		dest_rank = neighbor_processes[down] + 1
		send_buffer = numpy.array([local_Q_old[-2,-2]])
		recv_buffer = numpy.zeros(1)
		recv_buffer = comm.sendrecv(send_buffer, dest=dest_rank, source=dest_rank)
		#print('diagonal communication: process {} sended data to process {} and received data from process {}'.format(rank, dest_rank, dest_rank))
		local_Q_old[-1,-1] = recv_buffer[0]

	if neighbor_processes[up] >= 0 and neighbor_processes[right] >= 0:
		dest_rank = neighbor_processes[up] + 1
		send_buffer = numpy.array([local_Q_old[1,-2]])
		recv_buffer = numpy.zeros(1)
		recv_buffer = comm.sendrecv(send_buffer, dest=dest_rank, source=dest_rank)
		#print('diagonal communication: process {} sended data to process {} and received data from process {}'.format(rank, dest_rank, dest_rank))
		local_Q_old[0,-1] = recv_buffer[0]

	if neighbor_processes[down] >= 0 and neighbor_processes[left] >= 0:
		dest_rank = neighbor_processes[down] - 1
		send_buffer = numpy.array([local_Q_old[-2,1]])
		recv_buffer = numpy.zeros(1)
		recv_buffer = comm.sendrecv(send_buffer, dest=dest_rank, source=dest_rank)
		#print('diagonal communication: process {} sended data to process {} and received data from process {}'.format(rank, dest_rank, dest_rank))
		local_Q_old[-1,0] = recv_buffer[0]

	local_Q_new = local_Q_old.copy()
	t += dt
	comm.Barrier()

if rank == 0:
	end = MPI.Wtime()
	print(end-start)
	#plot_function(x_grids, y_grids, dt, T, results, 'upwind_MPI_vitual_topology')