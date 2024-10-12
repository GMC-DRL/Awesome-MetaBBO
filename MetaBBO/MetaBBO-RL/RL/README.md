# Introduction to Reinforcement Learning

Reinforcement learning (RL) is learning what to do - how to map situations to actions - so as to maximize a numerical reward signal [1]. The concept of RL is inspired by the trial-and-error learning seen in humans and animals. In MetaBBO, two major approaches to RL are the value-based method and the policy gradient method. Below is a brief introduction to these approaches, aimed at helping you gain a clearer understanding of the MetaBBO-RL learning paradigm.

## Specific RL algorithms

### Basic RL theory
A Markov Decision Process (MDP) can be defined as $\mathcal{M}:=\langle\mathcal{S}, \mathcal{A}, \Gamma, R\rangle$. At each time step $t$, the agent is provided with a state $s_t$, which contains all relevant information about the current situation. Based on $s_t$, the RL agent selects an action $a_t$ and interacts with the environment through the dynamics $\Gamma(s_{t+1} \mid s_{t}, a_{t})$, which determines the next state $s_{t+1}$. The reward function $R: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$ evaluates the performance improvement during this transition. The sequence of state-action pairs forms a trajectory $\tau:=(s_{0}, a_{0}, s_{1}, \dots, s_{T})$.

The goal of MDPs is to find a policy $\pi_{\theta^{* }}$ that maximizes the cumulative rewards $J(\theta)$ over the trajectory $\mathcal{T}$:

$$
    \pi_{\theta^{*}}= \underset{\theta}{\arg \max } J(\theta) = \underset{\theta}{\arg \max } \sum_{t=0}^{T} \gamma^{t} R\left(s_{t}, a_{t}\right)
$$

where $\gamma$ is the discount factor and $T$ is the trajectory length. RL, as an unsupervised learning paradigm, aims to learn the policy $\pi_{\theta}$ through the interaction between an agent and its environment, which is a specific approach to solving MDPs. As you can see, the mapping $\left(s_{t}, a_{t}\right)$ is central to finding the optimal policy $\pi_{\theta^{*}}$, which is the ultimate goal of RL.



One straightforward approach to learning the mapping $\left(s_{t}, a_{t}\right)$ is to collect all the state-action pairs along with their corresponding rewards. This approach is the foundation of value-based methods. However, value-based methods face a significant challenge: if the state and action spaces are continuous, it becomes impractical to learn all possible pairs. An alternative approach is to learn the policy directly, rather than focusing on the explicit mapping. By learning a policy that selects the optimal action in each state to maximize the overall reward, we effectively achieve the same goal, but through a more scalable and flexible means. This approach is the foundation of policy gradient method.
### Value-based method 

### Policy gradient method
In policy-based methods, the policy for selecting actions is no longer determined by a value function. Instead, the method directly learns the policy itself by parameterizing it with a set of parameters, $\theta$. The objective of the policy is to maximize the cumulative rewards $J(\theta)$. More specifically, the goal is to identify actions that yield higher rewards in each state and to increase the probability of selecting those actions, so that the policy becomes more likely to choose them. 

We can define $J(\theta)$ more clearly as follows:

$$
    \max_{\theta} J(\theta)=\max_{\theta} E_{\tau \sim \pi_{\theta}} R(\tau)=\max_{\theta} \sum_{\tau} P(\tau ; \theta) R(\tau)
$$

where $P(\tau ; \theta)$ is defined as:

$$
    P(\tau ; \theta)=\left[ \prod_{t=0}^{T} P\left(s_{t+1} \mid s_{t}, a_{t}\right) \cdot \pi_{\theta}\left(a_{t} \mid s_{t}\right)\right ]
$$

This expression represents the product of the state transition probability and the action selection probability, as the trajectory $\tau$ is determined by both the states and the actions.

The optimization of $J(\theta)$ is similar to that of a neural network, as it can be optimized using gradient.

$$
\nabla_\theta J(\theta) = \sum_{\tau} \nabla_\theta P(\tau ; \theta) R(\tau) = \sum_{\tau} P(\tau ; \theta) \frac{\nabla_\theta P(\tau ; \theta)}{P(\tau ; \theta)} R(\tau) = \sum_{\tau} P(\tau ; \theta) \nabla_\theta \log P(\tau ; \theta) R(\tau) = E_{\tau \sim \pi_\theta} [\nabla_\theta \log P(\tau ; \theta) R(\tau)].
$$

*The logarithmic derivative technique* is applied to express $P(\tau ; \theta)$ in logarithmic form, allowing it to be conveniently rewritten as:

$$
\nabla_{\theta} \log P(\tau ; \theta) =\nabla_{\theta}\left[\sum_{t=0}^{T} \log P\left(s_{t+1} \mid s_{t}, a_{t}\right)+\sum_{t=0}^{T} \log \pi_{\theta}\left(a_{t} \mid s_{t}\right)\right] = \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}\left(a_{t} \mid s_{t}\right)
$$

The first term, which represents the state transition probability, depends only on the environment and is independent of $\theta$, which simplifies the optimization process significantly.

So what exactly is the policy $\pi_{\theta}\left(a_{t} \mid s_{t}\right)$? Here are two common policy examples and their corresponding gradients.

- For discrete action spaces, the Softmax policy is a commonly used approach:
  
$$
    \pi_{\theta}(s, a)=\frac{e^{\phi(s, a)^{\top} \theta}}{\sum_{a^{\prime} \in A} e^{\phi\left(s, a^{\prime}\right)^{\top}} \theta}
$$

Here, $\phi(s, a)$ is a feature vector that represents the relationship between state $s$ and action $a$, allowing a complex state-action relationship to be mapped into a numerical vector.

Then it's corresponing policy gradient is:


