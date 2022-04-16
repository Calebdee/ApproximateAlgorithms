from gmq_example import gass_hermite_quad
import numpy as np
import matplotlib.pyplot as plt

def inner(x=[], m=10, b=3):
	x = np.array([x])
	x = m*x + b
	out = np.zeros(x.shape)
	pos = (x >= 0)
	out[pos] = 1 / (1 + np.exp(-x[pos]))
	out[-1*pos] = np.divide(np.exp(x[-1*pos]), (1 + np.exp(x[-1*pos])))
	return out

norm = gass_hermite_quad(inner, degree=80)
print("Normalizing Constant: {:.2f}".format(norm))


delta= 0.01
z = np.arange(-5, 5+delta, delta)
pz = np.multiply(np.exp(-np.multiply(z, z)), inner(z))/norm

print(z)
print(pz[0])
plt.plot(z, pz[0], label= 'Gauss-Hermite')



# Get MAP estimation
theta_0 = z[pz.argmax()]

# Now we need to expand the log joint probabilitity at this posterior mode
# We will need to calculate the second derivative inverse for this.
m=10
b=3

#d1_lp = (-2 - inner(theta_0)) * m
#d2_lp = -(d1_lp * (1 - inner(theta_0))) * m
d2_lp = -(-2 - inner(theta_0) * (1- inner(theta_0)) * m * m)
print("Mean: {:.2f}".format(theta_0))
print("Variance: {:.2f}".format(d2_lp[0]))
laplace_z =  pz[0][pz.argmax()] * np.exp(- 0.5 * np.power(z-theta_0, 2) * d2_lp[0])
plt.plot(z, laplace_z, label= 'Laplace')
plt.legend(loc= 'upper right')

lva = np.zeros((z.shape))
# Initial value of xi
xi  = 0
while(True):
    first = np.exp(-np.multiply(z, z)) * inner(xi)
    second= np.exp(5*(z-xi) + -1/(2*inner(xi, m, b)) * (inner(xi,m ,b) - 0.5) *np.multiply(10*(z-xi), 10*(z + xi) + 6))  
    lva = np.multiply(first, second)/norm

    xi_new  = z[lva.argmax()]
    diff    = np.abs(xi_new - xi)
    if diff < 1e-4:
        break
    else:
        xi = xi_new

# Draw the curve now
plt.plot(z, lva, label= 'LVA')
plt.legend(loc= 'upper right')


plt.title("p(z)")
plt.show()