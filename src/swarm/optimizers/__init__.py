import swarm.optimizers.gd as gd
import swarm.optimizers.pso_adam as pso_adam
import swarm.optimizers.pso as pso

get = {
    "gd": gd.gd,
    "pso": pso.pso,
    "pso_adam": pso_adam.pso,
}
