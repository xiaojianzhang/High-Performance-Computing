import numpy
import scipy
import pdb

def Gaussian_wave(x,y,t):
#this function generate Gaussian wave
	return numpy.exp(-200.0*((x-0.25-t)**2.0+(y-0.25-t)**2.0))

def Square_wave(x,y,t):
#this function generate the square wave
	z = numpy.empty(x.shape)
	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			z[i,j] = 1.0 if x[i,j] >= 0.20+t and x[i,j] <= 0.30+t and y[i,j] >= 0.10+t and y[i,j] <= 0.30+t else 0.0

	return z

def upwind(a=1.0, b=1.0, u=1.0, v=1.0, mx=200, my=200, CFL=0.8, T=0.5, init_cond_func='Gaussian'):
#2d solver using upwind method

	dx = a/float(mx)
	dy = b/float(my)

	#spatial discretization, including ghost cells
	x_grids = numpy.arange(-dx/2.0, a+dx, dx)
	y_grids = numpy.arange(-dy/2.0, b+dy, dy)

	dt = min(CFL*dx/u, CFL*dy/v)

	xx, yy = numpy.meshgrid(x_grids, y_grids)

	if init_cond_func is 'Gaussian':
		Q_old = Gaussian_wave(xx,yy,0.0) #initial data
	elif init_cond_func is 'Square':
		Q_old = Square_wave(xx,yy,0.0)   #initial data

	#set ghost cells using zero-order extrapolation
	Q_old[:,0] = Q_old[:,1]
	Q_old[:,-1] = Q_old[:,-2]
	Q_old[0,:] = Q_old[1,:]
	Q_old[-1,:] = Q_old[-2,:]

	t=dt
	results=[]
	results.append(Q_old[1:-1, 1:-1])
	Q_new = Q_old.copy()
	while t <= T+1e-6:

		#x-sweeps:
		for j in range(1,len(y_grids)-1):
			Q_new[1:-1,j] = Q_old[1:-1,j] - u * dt * (Q_old[1:-1,j] - Q_old[0:-2,j]) / dx

		#y-sweeps:
		for i in range(1,len(x_grids)-1):
			Q_new[i,1:-1] = Q_new[i,1:-1] - v * dt * (Q_new[i,1:-1] - Q_new[i,0:-2]) / dy

		Q_new[:,0] = Q_new[:,1]
		Q_new[:,-1] = Q_new[:,-2]
		Q_new[0,:] = Q_new[1,:]
		Q_new[-1,:] = Q_new[-2,:]
		Q_old = Q_new.copy()
		results.append(Q_old[1:-1, 1:-1])
		t += dt

	return results, x_grids[1:-1], y_grids[1:-1], dt, T

def Lax_Wendroff(a=1.0, b=1.0, u=1.0, v=1.0, mx=200, my=200, CFL=0.8, T=0.5, init_cond_func='Gaussian'):
#2d solver using Lax_Wendroff method
	dx = a/float(mx)
	dy = b/float(my)

	#spatial discretization, including ghost cells
	x_grids = numpy.arange(-dx/2.0, a+dx, dx)
	y_grids = numpy.arange(-dy/2.0, b+dy, dy)

	dt = min(CFL*dx/u, CFL*dy/v)

	xx, yy = numpy.meshgrid(x_grids, y_grids)

	if init_cond_func is 'Gaussian':
		Q_old = Gaussian_wave(xx,yy,0.0) #initial data
	elif init_cond_func is 'Square':
		Q_old = Square_wave(xx,yy,0.0)   #initial data

	#set ghost cells using zero-order extrapolation
	Q_old[:,0] = Q_old[:,1]
	Q_old[:,-1] = Q_old[:,-2]
	Q_old[0,:] = Q_old[1,:]
	Q_old[-1,:] = Q_old[-2,:]

	t=dt
	results=[]
	results.append(Q_old[1:-1, 1:-1])
	Q_new = Q_old.copy()
	while t <= T+1e-6:

		#x-sweeps:
		for j in range(1,len(y_grids)-1):
			Q_new[1:-1,j] = Q_old[1:-1,j] - u * dt * (Q_old[2:,j] - Q_old[0:-2,j]) / (2.0*dx) + 0.5 * (u*dt/dx)**2.0 * (Q_old[0:-2,j] - 2.0*Q_old[1:-1,j] + Q_old[2:,j])

		#y-sweeps:
		for i in range(1,len(x_grids)-1):
			Q_new[i,1:-1] = Q_new[i,1:-1] - v * dt * (Q_new[i,2:] - Q_new[i,0:-2]) / (2.0*dy) + 0.5 * (v*dt/dy)**2.0 * (Q_new[i,0:-2] - 2*Q_new[i,1:-1] + Q_new[i,2:])

		Q_new[:,0] = Q_new[:,1]
		Q_new[:,-1] = Q_new[:,-2]
		Q_new[0,:] = Q_new[1,:]
		Q_new[-1,:] = Q_new[-2,:]
		Q_old = Q_new.copy()
		results.append(Q_old[1:-1, 1:-1])
		t += dt

	return results, x_grids[1:-1], y_grids[1:-1], dt, T

def Beam_Warming(a=1.0, b=1.0, u=1.0, v=1.0, mx=200, my=200, CFL=0.8, T=0.5, init_cond_func='Gaussian'):
#2d solver using Beam_Warming method
	dx = a/float(mx)
	dy = b/float(my)

	#spatial discretization, including ghost cells
	x_grids = numpy.arange(-dx/2.0-dx, a+2*dx, dx)
	y_grids = numpy.arange(-dy/2.0-dx, b+2*dy, dy)

	dt = min(CFL*dx/u, CFL*dy/v)

	xx, yy = numpy.meshgrid(x_grids, y_grids)

	if init_cond_func is 'Gaussian':
		Q_old = Gaussian_wave(xx,yy,0.0) #initial data
	elif init_cond_func is 'Square':
		Q_old = Square_wave(xx,yy,0.0)   #initial data

	#set ghost cells using zero-order extrapolation
	Q_old[:,0] = Q_old[:,2]
	Q_old[:,1] = Q_old[:,2]
	Q_old[:,-1] = Q_old[:,-3]
	Q_old[:,-2] = Q_old[:,-3]
	Q_old[0,:] = Q_old[2,:]
	Q_old[1,:] = Q_old[2,:]
	Q_old[-1,:] = Q_old[-3,:]
	Q_old[-2,:] = Q_old[-3,:]

	t=dt
	results=[]
	results.append(Q_old[2:-2, 2:-2])
	Q_new = Q_old.copy()
	while t <= T+1e-6:

		#x-sweeps:
		for j in range(2,len(y_grids)-2):
			Q_new[2:-2,j] = Q_old[2:-2,j] - u * dt * (3*Q_old[2:-2,j] - 4*Q_old[1:-3,j] + Q_old[0:-4,j]) / (2.0*dx) + 0.5 * (u*dt/dx)**2.0 * (Q_old[2:-2,j] - 2*Q_old[1:-3,j] + Q_old[0:-4,j])

		#y-sweeps:
		for i in range(2,len(x_grids)-2):
			Q_new[i,2:-2] = Q_new[i,2:-2] - v * dt * (3*Q_new[i,2:-2] - 4*Q_new[i,1:-3] + Q_new[i,0:-4]) / (2.0*dy) + 0.5 * (v*dt/dy)**2.0 * (Q_new[i,2:-2] - 2*Q_new[i,1:-3] + Q_new[i,0:-4])

		Q_new[:,0] = Q_new[:,2]
		Q_new[:,1] = Q_new[:,2]
		Q_new[:,-1] = Q_new[:,-3]
		Q_new[:,-2] = Q_new[:,-3]
		Q_new[0,:] = Q_new[2,:]
		Q_new[1,:] = Q_new[2,:]
		Q_new[-1,:] = Q_new[-3,:]
		Q_new[-2,:] = Q_new[-3,:]
		Q_old = Q_new.copy()
		results.append(Q_old[2:-2, 2:-2])
		t += dt

	return results, x_grids[2:-2], y_grids[2:-2], dt, T

def analytical_solver(x_grids, y_grids, dt, T, init_cond_func='Gaussian'):
#true solution
	if init_cond_func is 'Gaussian':
		solution = Gaussian_wave
	elif init_cond_func is 'Square':
		solution = Square_wave

	xx,yy = numpy.meshgrid(x_grids, y_grids)

	results =[]
	results.append(solution(xx,yy,0.0))
	t = dt
	while t<=T+1e-6:

		results.append(solution(xx,yy,t))
		t+=dt

	return results
		

def plot_animation(x_grids, y_grids, dt, T, results, method):
	#generate animation of solutions
	from matplotlib import animation
	import matplotlib.pyplot as plt
	
	t = numpy.arange(0.0,T+1e-7,dt)
	k = numpy.arange(len(results))
	fig = plt.figure()
	axes = fig.add_subplot(1,1,1)
	xx, yy = numpy.meshgrid(x_grids, y_grids)
	z = results[0]
	surf = plt.pcolormesh(xx, yy, z, cmap='RdBu_r', vmin=0.0, vmax=1.0)
	def plot_q(i,z,surf):
		axes.clear()
		z = results[int(i)]
		surf = plt.pcolormesh(xx, yy, z, cmap='RdBu_r', vmin=0.0, vmax=1.0)
		return surf,
	# Animate the solution
	ani = animation.FuncAnimation(fig, plot_q, frames=k, fargs=(z,surf), blit=False)
	plt.colorbar()
	ani.save('serial_2d_advection_{}.mp4'.format(method), writer="mencoder", fps=15)
	#plt.show()

def plot_image(x_grids, y_grids, results, method):
	#plot image of the solution at one time step
	import matplotlib
	import matplotlib.pyplot as plt
	fig = plt.figure()
	axes = fig.add_subplot(1,1,1)
	xx, yy = numpy.meshgrid(x_grids, y_grids)
	z = results[-1]	
	surf = plt.pcolormesh(xx, yy, z, cmap='RdBu_r', vmin=-0.1, vmax=1.0)
	plt.colorbar()
	plt.xlabel('x')
	plt.ylabel("y")
	plt.title('{} at t = 0.5'.format(method))
	plt.savefig('{} at t = 0.5.png'.format(method))

def two_norm_error(result1, result2):
	#calciulate 2-norm error
	result = (result1 - result2)**2.0
	error = numpy.sqrt((numpy.sum(result)/result.size))

	return error

def inf_norm_error(result1, result2):
	#calciulate inf-norm error
	result = numpy.fabs(result1 - result2)
	error = numpy.max(result)

	return error

if __name__ == '__main__':

	pdb.set_trace()
	results_upwind, x_grids, y_grids, dt, T = upwind(a=1.0, b=1.0, u=1.0, v=1.0, mx=200, my=200, CFL=0.8, T=0.5, init_cond_func='Square')
	plot_image(x_grids, y_grids, results_upwind, 'Upwind')
	results_exact = analytical_solver(x_grids, y_grids, dt, T, init_cond_func='Square')
	plot_image(x_grids, y_grids, results, 'Exact solution')
	results_LW, x_grids, y_grids, dt, T = Lax_Wendroff(a=1.0, b=1.0, u=1.0, v=1.0, mx=200, my=200, CFL=0.8, T=0.5, init_cond_func='Square')
	plot_image(x_grids, y_grids, results, 'Lax_Wendroff')
	results_BW, x_grids, y_grids, dt, T = Beam_Warming(a=1.0, b=1.0, u=1.0, v=1.0, mx=200, my=200, CFL=0.8, T=0.5, init_cond_func='Square')
	two_norm_error = [two_norm_error(results_upwind[-1], results_exact[-1]), two_norm_error(results_LW[-1], results_exact[-1]), two_norm_error(results_BW[-1], results_exact[-1])]
	inf_norm_error = [inf_norm_error(results_upwind[-1], results_exact[-1]), inf_norm_error(results_LW[-1], results_exact[-1]), inf_norm_error(results_BW[-1], results_exact[-1])] 