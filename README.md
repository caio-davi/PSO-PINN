# Physics-Informed Neural Networks Trained with Particle Swarm Optimization

_Physics-informed neural networks (PINNs) have recently emerged as a promising application of deep learning in a wide range of engineering and scientific problems based on partial differential equation models. However, evidence shows that PINN training by gradient descent displays pathologies and stiffness in gradient flow dynamics. In this paper, we propose the use of a hybrid particle swarm optimization and gradient descent approach to train PINNs. The resulting PSO-PINN algorithm not only mitigates the undesired behaviors of PINNs trained with standard gradient descent, but also presents an ensemble approach to PINN that affords the possibility of robust predictions with quantified uncertainty. Experimental results using the Poisson, advection, and Burgers equations show that PSO-PINN consistently outperforms a baseline PINN trained with Adam gradient descent._

This Full paper available [here](https://arxiv.org/pdf/2202.01943.pdf).

### Setup virtual environment

Create environment:

```bash
python3 -m venv --system-site-packages ./venv
source ./venv/bin/activate
```

Install requirements:

```
pip install -r requirements
```

Install the project module (on the root folder):

```bash
pip install -e ./src
```

### Citation

```
 @article{davi2022pso,
  title={PSO-PINN: Physics-Informed Neural Networks Trained with Particle Swarm Optimization},
  author={Davi, Caio and Braga-Neto, Ulisses},
  journal={arXiv preprint arXiv:2202.01943},
  year={2022}
}
```
