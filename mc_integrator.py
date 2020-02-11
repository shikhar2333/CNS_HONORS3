from lib import *
kB = BOLTZMANN_CONSTANT_kB * AVOGADRO_CONSTANT_NA
def MultiCanIntegrator(temperature=298.0*kelvin, collision_rate=91.0/picoseconds, timestep=1.0*femtoseconds, DEmc=0):
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
    integrator.addGlobalVariable('pe', 0)
    integrator.addGlobalVariable('pe1', 0)
    # integrator.addGlobalVariable("a0", nE_coeff[0] )
    # integrator.addGlobalVariable("a1", nE_coeff[1] )
    # integrator.addGlobalVariable("a2", nE_coeff[2] )
    # integrator.addGlobalVariable("a3", nE_coeff[3] )
    #integrator.addTabulatedFunction("DEmc_function", DEmc)
    #
    # Allow context updating here.
    #
    integrator.addUpdateContextState()

    #
    # Pre-computafunc = Continuous1DFunction(functional_values, Emin, Emax)
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
    #Continuous1DFunction
    integrator.addComputePerDof('pe','energy')
    pe = integrator.getGlobalVariableByName('pe')
    #integrator.addComputePerDof("f", "f*"+str(DEmc(pe)))
    integrator.addComputePerDof("v", "v + 0.5*dt*fprime/m; fprime=f*" + str(DEmc(pe)))
    integrator.addComputePerDof("x", "x + v*dt")
    integrator.addComputePerDof("x1", "x")
    integrator.addConstrainPositions()
    integrator.addComputePerDof('pe1','energy')
    pe = integrator.getGlobalVariableByName('pe1')
    #integrator.addComputePerDof("f", "f*"+str(DEmc(pe)))
    integrator.addComputePerDof("v", "v + 0.5*dt*fprime/m + (x-x1)/dt; fprime=f*" + str(DEmc(pe)))
    integrator.addConstrainVelocities()

    #
    # Velocity randomization
    #
    integrator.addComputePerDof("v", "sqrt(b)*v + sqrt(1-b)*sigma*gaussian")
    integrator.addConstrainVelocities()

    return integrator