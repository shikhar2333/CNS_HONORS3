from lib import *
import matplotlib
matplotlib.use('pdf')
from matplotlib import pyplot as plt 
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from mc_integrator import MultiCanIntegrator
kB = BOLTZMANN_CONSTANT_kB * AVOGADRO_CONSTANT_NA
def fit_function(x, A, mu, sigma):
    return A*np.exp( -1.0 * (x - mu)**2 / (2 * sigma**2) )

def MCIntegrator(temperature=298.0*kelvin, collision_rate=91.0/picoseconds, timestep=1.0*femtoseconds):
    # Compute constants.
    kT = kB * temperature
    gamma = collision_rate
    
    # Create a new custom integrator.
    integrator = CustomIntegrator(timestep)
    #
    # integrator initialization.
    #
    integrator.addGlobalVariable("kT", kT) # thermal energy
    integrator.addGlobalVariable("b", np.exp(-gamma*timestep)) # velocity mixing parameter
    integrator.addPerDofVariable("sigma", 0) 
    integrator.addPerDofVariable("x1", 0) # position before application of constraints 

    #
    # Allow context updating here.
    #
    integrator.addUpdateContextState()
    #
    # Pre-computation.
    # This only needs to be done once, but it needs to be done for each degree of freedom.
    # Could move this to initialization?
    #
    integrator.addComputePerDof("sigma", "sqrt(kT/m)")

    # 
    # Velocity perturbation.
    #
    integrator.addComputePerDof("v", "sqrt(b)*v + sqrt(1-b)*sigma*gaussian")
    integrator.addConstrainVelocities()
    
    #
    # Metropolized symplectic step.
    #
    #integrator.addComputePerDof("f", "f")
    integrator.addComputePerDof("v", "v + 0.5*dt*f/m")
    integrator.addComputePerDof("x", "x + v*dt")
    integrator.addComputePerDof("x1", "x")
    integrator.addConstrainPositions()
    integrator.addComputePerDof("v", "v + 0.5*dt*f/m + (x-x1)/dt")
    integrator.addConstrainVelocities()

    #
    # Velocity randomization
    #
    integrator.addComputePerDof("v", "sqrt(b)*v + sqrt(1-b)*sigma*gaussian")
    integrator.addConstrainVelocities()

    return integrator

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

pdb = PDBFile('input.pdb')
forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME,
        nonbondedCutoff=1*nanometer, constraints=HBonds) 
integrator = MCIntegrator(700*kelvin, 91.0/picoseconds, 0.5*femtoseconds) 
platform = Platform.getPlatformByName('CUDA')
print(platform)
simulation = Simulation(pdb.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)
simulation.minimizeEnergy()
positions = simulation.context.getState(getPositions=True).getPositions()
PDBFile.writeFile(simulation.topology, positions, open('outs.pdb', 'w'))

def simulate_constT(T):
    integrator.setGlobalVariableByName("kT", kB*T)
    simulation.reporters.append(StateDataReporter("file-"+str(T)+".txt", 10, step=False,
            potentialEnergy=True, temperature=False))
    simulation.step(2000000)
    no_lines = file_len("file-"+str(T)+".txt")
    PE_file = open("file-"+str(T)+".txt", "r")
    PE_file.readline()
    print(simulation.context.getState(getEnergy=True).getPotentialEnergy())
    PE = np.empty(no_lines-1)
    counter = 0
    for E in PE_file.readlines():
        PE[counter] = E
        counter = counter + 1
    PE_file.close()
    #plt.plot(PE, np.arange(500))
    #plt.show()
    return (PE, np.amax(PE))

T_range = np.array([700, 600, 500, 400, 300])
no_temp = len(T_range)
E_max = np.empty(no_temp)
counter = 0
PE = np.empty((no_temp, 200000))
for T in T_range:
    PE[counter], E_max[counter] = simulate_constT(T)
    counter = counter + 1

maxPE = np.amax(PE[0])
print("Canonical at 700K:", np.amax(PE[0]), np.amin(PE[0]))
minPE = np.amin(PE[no_temp-1])
fig = plt.figure()
plt.plot(E_max, T_range)
# plt.show()
fig.savefig('Emax_VS_T.png')
# PE_entries, bins = np.histogram(PE[2], bins=np.arange(40))
# popt, pcov = curve_fit(fit_function, xdata=bins[0:39], ydata=PE_entries)
# Plot the histogram and the fitted function.
# xspace = np.linspace(0, 6, 100000)
# plt.bar(bins[0:39], PE_entries, color='navy', label=r'Histogram entries')
# plt.plot(xspace, fit_function(xspace, *popt), color='darkorange', linewidth=2.5, label=r'Fitted function')
# plt.show()

#plot of dEmc(E)/dE vs E; dEmc(E)/dE = T0/E^-1max(E)
x = np.linspace(E_max[4]-10000, E_max[4], 100)
y = np.repeat(T_range[0]/T_range[4], 100)
# plt.plot(x, y, 'co')
E = x
DEmc = y
for i in range(4,0,-1):
    alpha_i = ( T_range[i-1] - T_range[i] )/ ( E_max[i-1] - E_max[i] )
    beta_i = ( T_range[i]*E_max[i-1] - T_range[i-1]*E_max[i] )/ ( E_max[i-1] - E_max[i] )
    x = np.linspace(E_max[i]+1,E_max[i-1],100)
    E = np.append(E, x)
    #print(E_max[i], E_max[i-1])
    y = alpha_i*x + beta_i
    DEmc = np.append(DEmc, T_range[0]/y)
    #plt.plot(x, T_range[0]/y, 'co')

x = np.linspace(E_max[0]+1, E_max[0]+10000, 100)
y = np.repeat(1, 100)
E = np.append(E, x)
DEmc = np.append(DEmc, y)

# plt.plot(E, DEmc, 'co')
# plt.show()

cs = CubicSpline(E, DEmc)
fig = plt.figure()
plt.plot(E, cs(E))
# plt.show()
fig.savefig('DEmc(E)_vs_E')
pdb = PDBFile('input.pdb')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME,
        nonbondedCutoff=1*nanometer, constraints=HBonds)
func = Continuous1DFunction(DEmc, np.amin(E), np.amax(E))
MC_Integrator = MultiCanIntegrator(700*kelvin, 91.0/picoseconds, 0.5*femtoseconds, cs)

simulation = Simulation(pdb.topology, system, MC_Integrator, platform)
simulation.context.setPositions(pdb.positions)
#simulation.reporters.append(PDBReporter('outs.pdb', 1000))
# simulation.minimizeEnergy()
simulation.reporters.append(StateDataReporter("fileMC-700"+".txt", 10, step=False, potentialEnergy=True, temperature=False))
simulation.step(500000) 
no_lines = file_len("fileMC-700"+".txt")
PE_file = open("fileMC-700"+".txt", "r")
PE_file.readline()
PE = np.empty(no_lines-1)
counter = 0
for E in PE_file.readlines():
    PE[counter] = E
    counter = counter + 1
PE_file.close()
print("Max PE = ", np.amax(PE), maxPE)
print("Min PE = ", np.amin(PE), minPE)
# plt.plot(PE)
binwidth = 1.5
fig = plt.figure()
plt.hist(PE, bins=np.arange(minPE, maxPE + binwidth, binwidth) )
# plt.show()
fig.savefig('Energy_Dist.png')
