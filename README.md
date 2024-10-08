# Awesome-MetaBBO
This is a collection of MetaBBO papers and their corresponding code sources. We first classify them by learning paradigm, followed by different autonomous algorithm designs. All papers are sorted by year.
## What is MetaBBO?
Meta-Black-Box-Optimization (MetaBBO) is an emerging research topic, leveraging the generalization power of Meta Learning to enhance the optimization performance of existing BBO methods. By utilizing Meta Learning, the reliance on expert-level knowledge decreases, highlighting the trend toward autonomous algorithm design in BBO.

<!--使用quickchart来生成动态图表，方便随时调整-->

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://quickchart.io/chart?c=%7Btype%3A%27bar%27%2Cdata%3A%7Blabels%3A%5B%272014%27%2C%272015%27%2C%272016%27%2C%272017%27%2C%272018%27%2C%272019%27%2C%272020%27%2C%272021%27%2C%272022%27%2C%272023%27%2C%272024%27%5D%2Cdatasets%3A%5B%7Btype%3A%27bar%27%2Clabel%3A%27paper%20counts%20per%20year%27%2CbackgroundColor%3A%27rgba(255%2C%2099%2C%20132%2C%200.2)%27%2CborderColor%3A%27rgba(255%2C%2099%2C%20132%2C%201)%27%2CborderWidth%3A1%2Cdata%3A%5B2%2C3%2C1%2C5%2C3%2C4%2C9%2C12%2C15%2C26%2C46%5D%7D%2C%7Btype%3A%27line%27%2Clabel%3A%27development%20trend%27%2Cfill%3Afalse%2CborderColor%3A%27black%27%2CborderWidth%3A2%2CpointStyle%3A%27circle%27%2CpointRadius%3A5%2CpointBackgroundColor%3A%27black%27%2Cdata%3A%5B2%2C3%2C1%2C5%2C3%2C4%2C9%2C12%2C15%2C26%2C46%5D%7D%5D%7D%2Coptions%3A%7Bscales%3A%7By%3A%7BbeginAtZero%3Atrue%7D%7D%7D%7D" width="500"/>










We warmly invite you to read our survey on MetaBBO, "[Toward Autonomous Algorithm Design: A Survey and Empirical Evaluation of Meta-Black-Box-Optimization]()," for more detailed insights into MetaBBO!


## Content

- 📝 [1. Survey Papers & Benchmarks](#1-survey-papers--benchmarks)
  - 📚[1.1. Survey Papers](#11-survey-papers)
  - 🔍[1.2. Benchmarks](#12-benchmarks)

- 🎯[2. MetaBBO](#2-metabbo)
  <details>
    <summary>🤖 2.1 MetaBBO-RL</summary>
    
    - [2.1.1. Algorithm Selection](#211-algorithm-selection)
    - [2.1.2. Algorithm Configuration](#212-algorithm-configuration)
    - [2.1.3. Algorithm Generation](#213-algorithm-generation)
    - [2.1.4. Algorithm Imitation](#214-algorithm-imitation)
  </details>

  <details>
    <summary>💻 2.2 MetaBBO-SL</summary>
    
    - [2.2.1. Algorithm Selection](#221-algorithm-selection)
    - [2.2.2. Algorithm Configuration](#222-algorithm-configuration)
    - [2.2.3. Algorithm Generation](#223-algorithm-generation)
    - [2.2.4. Algorithm Imitation](#224-algorithm-imitation)

  </details>

  <details>
    <summary>🧬 2.3 MetaBBO-NE</summary>
    
    - [2.3.1. Algorithm Selection](#231-algorithm-selection)
    - [2.3.2. Algorithm Configuration](#232-algorithm-configuration)
    - [2.3.3. Algorithm Generation](#233-algorithm-generation)
    - [2.3.4. Algorithm Imitation](#234-algorithm-imitation)

  </details>

  <details>
    <summary>🧠 2.4 MetaBBO-ICL</summary>
    
    - [2.4.1. Algorithm Selection](#241-algorithm-selection)
    - [2.4.2. Algorithm Configuration](#242-algorithm-configuration)
    - [2.4.3. Algorithm Generation](#243-algorithm-generation)
    - [2.4.4. Algorithm Imitation](#244-algorithm-imitation)

  </details>

</details>

  - 🔧[3. Others](#3-others)
    - 📈[3.1 Evaluation Indicator](#31-evaluation-indicator)
    - 📊[3.2 Landscape Feature](#32-landscape-feature)



## 1. 📝Survey Papers \& Benchmarks

### 1.1. 📚Survey Papers

|Paper|
|:-:|
|Li P, Hao J, Tang H, et al. "[**Bridging Evolutionary Algorithms and Reinforcement Learning: A Comprehensive Survey on Hybrid Algorithms**](https://ieeexplore.ieee.org/abstract/document/10637292). IEEE Transactions on Evolutionary Computation. (2024).
|Song Y, Wu Y, Guo Y, et al. "[**Reinforcement learning-assisted evolutionary algorithm: A survey and research opportunities**](https://www.sciencedirect.com/science/article/pii/S2210650224000506). Swarm and Evolutionary Computation. (2024).
|Nikolikj, Ana, et al. "[**Quantifying Individual and Joint Module Impact in Modular Optimization Frameworks**](https://arxiv.org/abs/2405.11964)." 2024 IEEE Congress on Evolutionary Computation (CEC). (2024).
|Qian, Chao, Ke Xue, and Ren-Jian Wang. "[**Quality-Diversity Algorithms Can Provably Be Helpful for Optimization**](https://arxiv.org/abs/2401.10539)." arXiv preprint arXiv:2401.10539. (2024).
|Huang, Beichen, et al. "[**Exploring the True Potential: Evaluating the Black-box Optimization Capability of Large Language Models**](https://arxiv.org/abs/2404.06290)." arXiv preprint arXiv:2404.06290. (2024).
|Chernigovskaya, Maria, Andrey Kharitonov, and Klaus Turowski. "[**A Recent Publications Survey on Reinforcement Learning for Selecting Parameters of Meta-Heuristic and Machine Learning Algorithms**](https://www.scitepress.org/Papers/2023/119543/119543.pdf)." CLOSER. (2023).
|Drugan, Madalina M. "[**Reinforcement learning versus evolutionary computation: A survey on hybrid algorithms**](https://www.sciencedirect.com/science/article/abs/pii/S2210650217302766)." Swarm and Evolutionary Computation. (2019).

### 1.2. 🔍Benchmarks

|Benchmark|Paper|Code Source|Optimization Type|
|:-:|:-:|:-:|:-:|
|GP-based|He Y, Aranha C. "[**Evolving Benchmark Functions to Compare Evolutionary Algorithms via Genetic Programming**](https://arxiv.org/abs/2403.14146)". arXiv preprint arXiv:2403.14146 (2024).|[GP-based](https://github.com/Y1fanHE/cec2024)||
|SELECTOR|Benjamins, Carolin, et al. "[**Instance Selection for Dynamic Algorithm Configuration with Reinforcement Learning: Improving Generalization**](https://arxiv.org/abs/2407.13513)." arXiv preprint arXiv:2407.13513 (2024).|[automl/instance-dac]( https://github.com/automl/instance-dac)||
|MetaBox|Ma, Zeyuan, et al. "[**MetaBox: A Benchmark Platform for Meta-Black-Box Optimization with Reinforcement Learning**](https://neurips.cc/virtual/2023/oral/73737)." Advances in Neural Information Processing Systems 36 (2023).|[GMC-DRL/MetaBox]( https://github.com/GMC-DRL/MetaBox)||
|NN-based|Prager R P, Dietrich K, Schneider L, et al. "[**Neural Networks as Black-Box Benchmark Functions Optimized for Exploratory Landscape Features**](https://dl.acm.org/doi/abs/10.1145/3594805.3607136)" Proceedings of the 17th ACM/SIGEVO Conference on Foundations of Genetic Algorithms (2023).| - | |
|NeuroEvoBench|Lange, Robert, Yujin Tang, and Yingtao Tian. "[**Neuroevobench: Benchmarking evolutionary optimizers for deep learning applications**](https://neurips.cc/virtual/2023/oral/73737)." Advances in Neural Information Processing Systems 36 (2023)|[neuroevobench/neuroevobench](https://github.com/neuroevobench/neuroevobench)||
|MA-BBOB|Vermetten, Diederick, et al. "[**MA-BBOB: A Problem Generator for Black-Box Optimization Using Affine Combinations and Shifts**](https://arxiv.org/abs/2312.11083)." arXiv preprint arXiv:2312.11083 (2023).|[Dvermetten/Many-affine-BBOB](https://github.com/Dvermetten/Many-affine-BBOB)||
|IEEE CEC 2022|Abhishek Kumar, Kenneth V. Price, Ali Wagdy Mohamed, Anas A. Hadi, P. N. Suganthan, "[**Problem definitions and evaluation criteria for the cec 2022 Special Session and Competition on Single Objective Bound Constrained Numerical Optimization**](https://www3.ntu.edu.sg/home/epnsugan/index_files/CEC2022/CEC2022.htm)." Technical Report 2022|[P-N-Suganthan/2022-SO-BO](https://github.com/P-N-Suganthan/2022-SO-BO)||
|Affine Recombination|Dietrich K, Mersmann O. "[**Increasing the diversity of benchmark function sets through affine recombination**](https://link.springer.com/chapter/10.1007/978-3-031-14714-2_41)" International Conference on Parallel Problem Solving from Nature. (2022).| - | |
|IEEE CEC 2021|Ali Wagdy, Anas A Hadi, Ali K. Mohamed, Prachi Agrawal, Abhishek Kumar and P. N. Suganthan, "[**Problem definitions and evaluation criteria for the cec 2021 Special Session and Competition on Single Objective Bound Constrained Numerical Optimization**](https://www3.ntu.edu.sg/home/epnsugan/index_files/CEC2021/CEC2021-2.htm)." Technical Report 2021|[P-N-Suganthan/2021-SO-BCO](https://github.com/P-N-Suganthan/2021-SO-BCO)||
|Zigzag BBO|Kudela, Jakub. "[**Novel zigzag-based benchmark functions for bound constrained single objective optimization**](https://ieeexplore.ieee.org/abstract/document/9504720/)." 2021 IEEE Congress on Evolutionary Computation (CEC). IEEE, (2021).<br>Kudela, Jakub, and Radomil Matousek. "[**New benchmark functions for single-objective optimization based on a zigzag pattern**](https://ieeexplore.ieee.org/abstract/document/9684455/)." IEEE Access 10 (2022).|[JakubKudela89/Zigzag](https://github.com/JakubKudela89/Zigzag)||
|HPOBench|Eggensperger, Katharina, et al. "[**HPOBench: A collection of reproducible multi-fidelity benchmark problems for HPO**](https://arxiv.org/abs/2109.06716)." arXiv preprint arXiv:2109.06716 (2021).|[automl/HPOBench](https://github.com/automl/HPOBench)||
|DACBench|Eimer, Theresa, et al. "[**DACBench: A benchmark library for dynamic algorithm configuration**](https://arxiv.org/abs/2105.08541)." arXiv preprint arXiv:2105.08541 (2021).|[automl/DACBench](https://github.com/automl/DACBench)||
|Olympus|Häse, Florian, et al. "[**Olympus: a benchmarking framework for noisy optimization and experiment planning**](https://iopscience.iop.org/article/10.1088/2632-2153/abedc8/meta)." Machine Learning: Science and Technology (2021).|[aspuru-guzik-group/olympus](https://github.com/aspuru-guzik-group/olympus)||
|NeurIPS BBO challenge|Turner R, Eriksson D, McCourt M, et al. "[**Bayesian optimization is superior to random search for machine learning hyperparameter tuning: Analysis of the black-box optimization challenge 2020**](https://proceedings.mlr.press/v133/turner21a.html)" NeurIPS 2020 Competition and Demonstration Track. (2021)|[NeurIPS BBO challenge](https://github.com/rdturnermtl/bbo_challenge_starter_kit/) | |
|Random function generator|Tian Y, Peng S, Zhang X, et al. "[**A recommender system for metaheuristic algorithms for continuous optimization based on deep recurrent neural networks**](https://ieeexplore.ieee.org/abstract/document/9187549)". IEEE transactions on artificial intelligence (2020).|[Random function generator](https://github.com/BIMK/Algorithm-Recommendation) | |
|CEC 2020 competition on real-world optimization problem|Kumar A, Wu G, Ali M Z, et al. "[**A test-suite of non-convex constrained optimization problems from the real-world and some baseline results**](https://www.sciencedirect.com/science/article/pii/S2210650219308946). Swarm and Evolutionary Computation (2020).|[CEC 2020 real-world](https://github.com/P-N-Suganthan/2020-RW-Constrained-Optimisation)||
|COCO|Hansen, Nikolaus, et al. "[**COCO: A platform for comparing continuous optimizers in a black-box setting**](https://www.tandfonline.com/doi/abs/10.1080/10556788.2020.1808977)." Optimization Methods and Software (2021).|[numbbo/coco](https://github.com/numbbo/coco)||
|EVOBBO|Muñoz, Mario A., and Kate Smith-Miles. "[**Generating new space-filling test instances for continuous black-box optimization**](https://direct.mit.edu/evco/article-abstract/28/3/379/94997)." Evolutionary computation (2020).|[andremun/EVOBBO_Instances](https://github.com/andremun/EVOBBO_Instances)||
|Bayesmark|Turner R, Eriksson D. "[**Bayesmark: Benchmark framework to easily compare bayesian optimization methods on real machine learning tasks**](https://bayesmark.readthedocs.io/en/latest/)." (2019). |[Bayesmark](https://github.com/uber/bayesmark)| |
|IOHprofiler (IOHexperimenter)|Doerr, Carola, et al. "[**IOHprofiler: A benchmarking and profiling tool for iterative optimization heuristics**](https://arxiv.org/abs/1810.05281)." arXiv preprint arXiv:1810.05281 (2018).<br>de Nobel, Jacob, et al. "[**Iohexperimenter: Benchmarking platform for iterative optimization heuristics**](https://direct.mit.edu/evco/article/doi/10.1162/evco_a_00342/116949)." Evolutionary Computation (2023): 1-6.|[IOHprofiler/<br>IOHexperimenter](https://github.com/IOHprofiler/IOHexperimenter)||
|MTMOOP|Yuan Y, Ong Y S, Feng L, et al. "[**Evolutionary multitasking for multiobjective continuous optimization: Benchmark problems, performance metrics and baseline results**](https://arxiv.org/abs/1706.02766)." arXiv preprint arXiv:1706.02766 (2017).|- | |
|MTSOP|Da B, Ong Y S, Feng L, et al. "[**Evolutionary multitasking for single-objective continuous optimization: Benchmark problems, performance metric, and baseline results**](https://arxiv.org/abs/1706.03470)". arXiv preprint arXiv:1706.03470 (2017).|- | |
|IEEE CEC 2017|N. H. Awad, M. Z. Ali, J. J. Liang, B. Y. Qu and P. N. Suganthan, "[**Problem definitions and evaluation criteria for the CEC 2017 competition on constrained real-parameter optimization**](https://www3.ntu.edu.sg/home/epnsugan/index_files/CEC2017/CEC2017.htm)." Technical Report (2017)|[P-N-Suganthan/CEC2017-BoundContrained](https://github.com/P-N-Suganthan/CEC2017-BoundContrained)||
|IEEE CEC 2015|J. J. Liang, B. Y. Qu, P. N. Suganthan, Q. Chen, "[**Problem Definitions and Evaluation Criteria for the CEC 2015 Competition on Learning-based Real-Parameter Single Objective Optimization**](https://www3.ntu.edu.sg/home/epnsugan/index_files/CEC2015/CEC2015.htm)", Technical Report, Computational Intelligence Laboratory (2015).|[P-N-Suganthan/CEC2015-Learning-Based](https://github.com/P-N-Suganthan/CEC2015-Learning-Based)||
|AClib|Hutter, Frank, et al. "[**AClib: A benchmark library for algorithm configuration**](https://link.springer.com/chapter/10.1007/978-3-319-09584-4_4)." Learning and Intelligent Optimization: 8th International Conference (2014).|[aclib.net](https://www.aclib.net/)||
|IEEE CEC 2013|J. J. Liang, B-Y. Qu, P. N. Suganthan, Alfredo G. Hernández-Díaz, "[**Problem Definitions and Evaluation Criteria for the CEC 2013 Special Session and Competition on Real-Parameter Optimization**](https://www3.ntu.edu.sg/home/epnsugan/index_files/CEC2013/CEC2013.htm)", Technical Report, Computational Intelligence Laboratory (2013).|[P-N-Suganthan/CEC2013](https://github.com/P-N-Suganthan/CEC2013)||
|Protein–Docking|Hwang, Howook, et al. "[**Protein–protein docking benchmark version 4.0**](https://onlinelibrary.wiley.com/doi/abs/10.1002/prot.22830)." Proteins: Structure, Function, and Bioinformatics (2010).|[Protein–Docking](http://zlab.umassmed.edu/benchmark/)||
|BBOB 2009|Hansen N, Finck S, Ros R, et al. "[**Real-parameter black-box optimization benchmarking 2009: Noiseless functions definitions**](https://inria.hal.science/inria-00362633/)". INRIA. (2009). |[BBOB 2009](https://web.archive.org/web/20200811021008/https://coco.gforge.inria.fr/doku.php?id=bbob-2009-results) | |
|WFG|Huband S, Hingston P, Barone L, et al. "[**A review of multiobjective test problems and a scalable test problem toolkit**](https://ieeexplore.ieee.org/abstract/document/1705400)." IEEE Transactions on Evolutionary Computation. (2006).|[WFG](https://github.com/White-Chen/MOEA-Benchmark) ||
|DTLZ|Deb K, Thiele L, Laumanns M, et al. "[**Scalable multi-objective optimization test problems**](https://ieeexplore.ieee.org/abstract/document/1007032)." Proceedings of the 2002 Congress on Evolutionary Computation (2002).|[DTLZ](https://github.com/msu-coinlab/pymop/tree/master?tab=readme-ov-file) ||
|ZDT|Zitzler, E., Deb, K., and Thiele, L. "[**Comparison of Multiobjective Evolutionary Algorithms: Empirical Results**]( https://dl.acm.org/doi/10.1162/106365600568202)." Evolutionary Computation (2000). |[ZDT](https://github.com/White-Chen/MOEA-Benchmark)| |

**The complete list of IEEE CEC series can be access at [ntu.edu.sg](https://www3.ntu.edu.sg/home/epnsugan/index_files/).*

**The complete list of BBOB series can be access at [numbbo](https://numbbo.github.io/workshops/bbob2023.html).*

<p align="right">
<a href="https://github.com/GMC-DRL/Awesome-MetaBBO/tree/main#awesome-metabbo">Back to Top</a>
</p>


## 2. 🎯MetaBBO

### 2.1 🤖MetaBBO-RL
#### 2.1.1 Algorithm Selection
|Algorithm|Paper|Optimization Type|Low-Level Optimizer|RL|Code Source|
|:-:|:-:|:-:|:-:|:-:|:-:|
|HHRL-MAR|Zhu N, Zhao F, Cao J. "[**A Hyperheuristic and Reinforcement Learning Guided Meta-heuristic Algorithm Recommendation**](https://ieeexplore.ieee.org/abstract/document/10580058/)" 2024 27th International Conference on Computer Supported Cooperative Work in Design (CSCWD) (2024)|SOP|SI| | |
|R2-RLMOEA|Tahernezhad-Javazm F, Rankin D, Bois N D, et al. "[**R2 Indicator and Deep Reinforcement Learning Enhanced Adaptive Multi-Objective Evolutionary Algorithm**](https://arxiv.org/abs/2404.08161)". arXiv preprint arXiv:2404.08161 (2024).|MOOP|EAs|DDQN| |
|RL-DAS|Guo, Hongshu, et al. "[**Deep Reinforcement Learning for Dynamic Algorithm Selection: A Proof-of-Principle Study on Differential Evolution**](https://ieeexplore.ieee.org/abstract/document/10496708/)." IEEE Transactions on Systems, Man, and Cybernetics: Systems (2024).|SOP|DE|PPO| |

#### 2.1.2 Algorithm Configuration

|Algorithm|Paper|Optimization Type|Low-Level Optimizer|RL|Code Source|
|:-:|:-:|:-:|:-:|:-:|:-:|
|UES-CMAES-RL|Bolufé-Röhler A, Xu B. "[**Deep Reinforcement Learning for Smart Restarts in Exploration-Only Exploitation-Only Hybrid Metaheuristics Metaheuristics International Conference**](https://link.springer.com/chapter/10.1007/978-3-031-62922-8_2)" 2024.|SOP|UES-CMAES|DQN|   |
|MSoRL|Wang X, Wang F, He Q, et al. "[**A multi-swarm optimizer with a reinforcement learning mechanism for large-scale optimization**](https://www.sciencedirect.com/science/article/pii/S2210650224000191)". Swarm and Evolutionary Computation (2024).|LSOP|PSO|Tabular Q-learning|    |
|MRL-MOEA|Wang, Jing, et al. "[**A Novel Multi-State Reinforcement Learning-Based Multi-Objective Evolutionary Algorithm**](https://www.sciencedirect.com/science/article/pii/S0020025524013112)." Information Sciences (2024).|MOOP|MOEA|Tabular Q-learning| |
|RLEMMO|Lian, Hongqiao, et al. "[**RLEMMO: Evolutionary Multimodal Optimization Assisted By Deep Reinforcement Learning**](https://arxiv.org/abs/2404.08242)." arXiv preprint arXiv:2404.08242 (2024).|MMOP|DE|PPO|  |
|SA-DQN-DE|Liao, Zuowen, Qishuo Pang, and Qiong Gu. "[**Differential evolution based on strategy adaptation and deep reinforcement learning for multimodal optimization problems**](https://www.sciencedirect.com/science/article/pii/S2210650224001068)." Swarm and Evolutionary Computation 87 (2024): 101568.|MMOP|DE|DQN|  |
|PG-DE \& PG-MPEDE|Zhang, Haotian, et al. "[**Learning to select the recombination operator for derivative-free optimization**](https://link.springer.com/article/10.1007/s11425-023-2252-9)." Science China Mathematics (2024).|SOP|DE|REINFORCE|  |
|RLNS|Hong, Jiale, Bo Shen, and Anqi Pan. "[**A reinforcement learning-based neighborhood search operator for multi-modal optimization and its applications**](https://www.sciencedirect.com/science/article/pii/S0957417424000150)." Expert Systems with Applications (2024).|MMOP|SSA,PSO,EO|Tabular Q-learning|   |
|RLMODE|Yu, Xiaobing, et al. "[**Reinforcement learning-based differential evolution algorithm for constrained multi-objective optimization problems**](https://www.sciencedirect.com/science/article/pii/S0952197623020018)." Engineering Applications of Artificial Intelligence (2024).|MOOP|DE|Tabular Q-learning|   |
|GLEET|Ma, Zeyuan, et al. "[**Auto-configuring Exploration-Exploitation Tradeoff in Evolutionary Computation via Deep Reinforcement Learning**](https://arxiv.org/abs/2404.08239)." arXiv preprint arXiv:2404.08239 (2024).|SOP|DE,PSO|PPO| |
|RLDMDE|Yang, Qingyong, et al. "[**Dynamic multi-strategy integrated differential evolution algorithm based on reinforcement learning for optimization problems**](https://link.springer.com/article/10.1007/s40747-023-01243-9)." Complex & Intelligent Systems (2023).|SOP|DE|Tabular Q-learning|   |
|MPSORL|Meng, Xiaoding, Hecheng Li, and Anshan Chen. "[**Multi-strategy self-learning particle swarm optimization algorithm based on reinforcement learning**](http://www.aimspress.com/aimspress-data/mbe/2023/5/PDF/mbe-20-05-373.pdf)." Mathematical Biosciences and Engineering (2023).|SOP|PSO|Tabular Q-learning|  |
|IRLMFO|Zhao F, Wang Q, Wang L. "[**An inverse reinforcement learning framework with the Q-learning mechanism for the metaheuristic algorithm**](https://www.sciencedirect.com/science/article/pii/S0950705123001181)". Knowledge-Based Systems (2023).|SOP|MFO|IRL+Tabual Q-learning|   |
|RLAM|Yin, Shiyuan, et al. "[**Reinforcement-learning-based parameter adaptation method for particle swarm optimization**](https://link.springer.com/article/10.1007/s40747-023-01012-8)." Complex & Intelligent Systems (2023).|SOP|PSO|DDPG|    |
|LADE|Liu X, Sun J, Zhang Q, et al. "[**Learning to learn evolutionary algorithm: A learnable differential evolution**](https://ieeexplore.ieee.org/abstract/document/10068274/)". IEEE Transactions on Emerging Topics in Computational Intelligence (2023).|SOP|DE|REINFORCE|   |
|MOEADRL|Gao, Mengqi, et al. "[**An efficient evolutionary algorithm based on deep reinforcement learning for large-scale sparse multiobjective optimization**](https://link.springer.com/article/10.1007/s10489-023-04574-9)." Applied Intelligence (2023).|LS-MOOP|SpareEAs|A2C|  |
|Q-LSHADE|Zhang H, Sun J, Bäck T, et al. "[**Controlling Sequential Hybrid Evolutionary Algorithm by Q-Learning**](https://ieeexplore.ieee.org/abstract/document/10035716/)". IEEE Computational Intelligence Magazine (2023).|SOP|LSHADE|Tabular Q-learning|   |
|NRLPSO|Li, Wei, et al. "[**Reinforcement learning-based particle swarm optimization with neighborhood differential mutation strategy**](https://www.sciencedirect.com/science/article/pii/S2210650223000482)." Swarm and Evolutionary Computation (2023).|SOP|PSO|Tabular Q-learning|  |
|RL-SHADE|Fister I, Fister D, Fister Jr I. "[**Reinforcement learning-based differential evolution for global optimization Differential Evolution: From Theory to Practice**](https://link.springer.com/chapter/10.1007/978-981-16-8082-3_3)" (2022).|SOP|SHADE|Tabular Q-learning|  |
|MOEA/D-DQN|Tian, Ye, et al. "[**Deep reinforcement learning based adaptive operator selection for evolutionary multi-objective optimization**](https://ieeexplore.ieee.org/abstract/document/9712324/)." IEEE Transactions on Emerging Topics in Computational Intelligence (2022).|MOOP|MOEA|DDQN|   |
|RL-CORCO|Hu Z, Gong W. "[**Constrained evolutionary optimization based on reinforcement learning using the objective function and constraints**](https://www.sciencedirect.com/science/article/pii/S0950705121009709)". Knowledge-Based Systems (2022).|COP|DE|Tabular Q-learning|   |
|MADAC|Xue, Ke, et al. "[**Multi-agent dynamic algorithm configuration**](https://proceedings.neurips.cc/paper_files/paper/2022/hash/7f02b39c0424cc4a422994289ca03e46-Abstract-Conference.html)." Advances in Neural Information Processing Systems (2022).|MOOP|MOEA/D|VDN|  |
|RLLPSO|Wang F, Wang X, Sun S. "[**A reinforcement learning level-based particle swarm optimization algorithm for large-scale optimization**](https://www.sciencedirect.com/science/article/pii/S0020025522004054)." Information Sciences (2022).|LSOP|PSO|Tabular Q-learning|  |
|RL-PSO|Wu, Di, and G. Gary Wang. "[**Employing reinforcement learning to enhance particle swarm optimization methods**](https://www.tandfonline.com/doi/abs/10.1080/0305215X.2020.1867120)." Engineering Optimization (2022).|SOP|PSO|REINFORCE|  |
|RLEA-SSC|Xia H, Li C, Zeng S, et al. "[**A reinforcement-learning-based evolutionary algorithm using solution space clustering for multimodal optimization problems**](https://ieeexplore.ieee.org/abstract/document/9504896) 2021 IEEE Congress on Evolutionary Computation (CEC). (2021).|MMOP|DE|Tabular Q-learning|   |
|DE-DQN|Tan, Zhiping, and Kangshun Li. "[**Differential evolution with mixed mutation strategy based on deep reinforcement learning**](https://www.sciencedirect.com/science/article/abs/pii/S1568494621005998)." Applied Soft Computing 111 (2021).|SOP|DE|Tabular Q-learning|  |
|RLEPSO|Yin, Shiyuan, et al. "[**RLEPSO: Reinforcement learning based Ensemble particle swarm optimizer**](https://dl.acm.org/doi/abs/10.1145/3508546.3508599)." Proceedings of the 2021 4th International Conference on Algorithms, Computing and Artificial Intelligence. (2021).|SOP|PSO|DDPG|   |
|RLDE|Hu Z, Gong W, Li S. "[**Reinforcement learning-based differential evolution for parameters extraction of photovoltaic models**](https://www.sciencedirect.com/science/article/pii/S2352484721000974)." Energy Reports (2021).|SOP|DE|Tabular Q-learning| |
|LRMODE|Huang Y, Li W, Tian F, et al. "[**A fitness landscape ruggedness multiobjective differential evolution algorithm with a reinforcement learning strategy**](https://www.sciencedirect.com/science/article/pii/S1568494620306311)". Applied Soft Computing 2020.|MOOP|DE|Tabular Q-learning|  |
|MARLwCMA|Sallam, Karam M., et al. "[**Evolutionary framework with reinforcement learning-based mutation adaptation**](https://ieeexplore.ieee.org/abstract/document/9239320/)." IEEE Access 8 (2020).|SOP|DE|Tabular Q-learning| |
|QLPSO|Xu Y, Pi D. "[**A reinforcement learning-based communication topology in particle swarm optimization**](https://link.springer.com/article/10.1007/s00521-019-04527-9)." Neural Computing and Applications 2020.|SOP|PSO|Tabular Q-learning|  |
|LTO|Shala G, Biedenkapp A, Awad N, et al. "[**Learning step-size adaptation in CMA-ES**](https://link.springer.com/chapter/10.1007/978-3-030-58112-1_48)." Parallel Problem Solving from Nature–PPSN XVI: 16th International Conference (2020).|SOP|CMA-ES|GPS|  |
|NSGA-RL|Kaur A, Kumar K. "[**A reinforcement learning based evolutionary multi-objective optimization algorithm for spectrum allocation in cognitive radio networks**](https://www.sciencedirect.com/science/article/pii/S1874490720302731)". Physical Communication (2020).|MOOP|NSGA-II|Tabular Q-learning| |
|DE-DDQN|Sharma, Mudita, et al. "[**Deep reinforcement learning based parameter control in differential evolution**](https://dl.acm.org/doi/abs/10.1145/3321707.3321813)." Proceedings of the Genetic and Evolutionary Computation Conference (2019).|SOP|DE|Tabular Q-learning|  |
|QLMOPSO|Liu Y, Lu H, Cheng S, et al. "[**An adaptive online parameter control algorithm for particle swarm optimization based on reinforcement learning**](https://ieeexplore.ieee.org/abstract/document/8790035)" 2019 IEEE congress on evolutionary computation (CEC) (2019).|SOP|PSO|Tabular Q-learning|  |
|RL-MOEA/D|Ning W, Guo B, Guo X, et al. "[**Reinforcement learning aided parameter control in multi-objective evolutionary algorithm based on decomposition**](https://link.springer.com/article/10.1007/s13748-018-0155-7)". Progress in Artificial Intelligence 2018.|MOOP|MOEA/D|SARSA| |
|QFA|Sadhu A K, Konar A, Bhattacharjee T, et al. "[**Synergism of firefly algorithm and Q-learning for robot arm path planning**](https://www.sciencedirect.com/science/article/pii/S2210650217306776)". Swarm and Evolutionary Computation 2018.|SOP|FA|Tabular Q-learning| |

#### 2.1.3 Algorithm Generation
|Algorithm|Paper|Optimization Type|Low-Level Optimizer|RL|Code Source|
|:-:|:-:|:-:|:-:|:-:|:-:|
|ALDes|Zhao, Qi, et al. "[**Automated Metaheuristic Algorithm Design with Autoregressive Learning**](https://arxiv.org/abs/2405.03419)." arXiv preprint arXiv:2405.03419 (2024).|SOP|-|-|  |
|SYMBOL|Chen, Jiacheng, et al. "[**Symbol: Generating Flexible Black-Box Optimizers through Symbolic Equation Learning**](https://arxiv.org/abs/2402.02355)." The Twelfth International Conference on Learning Representations. 2024.|SOP|-|PPO|[SYMBOL](https://github.com/GMC-DRL/Symbol)|


#### 2.1.4 Algorithm Imitation
|Algorithm|Paper|Optimization Type|Low-Level Optimizer|RL|Code Source|
|:-:|:-:|:-:|:-:|:-:|:-:|
|MELBA|Chaybouti, Sofian, et al. "[**Meta-learning of Black-box Solvers Using Deep Reinforcement Learning**](https://hal.science/hal-03930140/)." NeurIPS 2022, MetaLearn Workshop. 2022.|SOP|-|PPO| |

<p align="right">
<a href="https://github.com/GMC-DRL/Awesome-MetaBBO/tree/main#awesome-metabbo">Back to Top</a>
</p>

### 2.2 💻MetaBBO-SL
#### 2.2.1 Algorithm Selection
|Algorithm|Paper|Optimization Type|Low-Level Optimizer|Code Source|
|:-:|:-:|:-:|:-:|:-:|
|ASF-ALLFV|Li Y, Liang J, Yu K, et al. "[**Adaptive local landscape feature vector for problem classification and algorithm selection**](https://www.sciencedirect.com/science/article/pii/S1568494622008006)". Applied Soft Computing, 2022, 131: 109751.|SOP|EAs,SI|  |
|AR-BB|Tian Y, Peng S, Zhang X, et al. "[**A recommender system for metaheuristic algorithms for continuous optimization based on deep recurrent neural networks**](https://ieeexplore.ieee.org/abstract/document/9187549)". IEEE transactions on artificial intelligence (2020).|SOP|EAs,SI|   |
|Meta-VRP|Gutierrez-Rodríguez A E, Conant-Pablos S E, Ortiz-Bayliss J C, et al. "[**Selecting meta-heuristics for solving vehicle routing problems with time windows via meta-learning**]". Expert Systems with Applications (2019).|CO|MOEA|    |
|Meta-MOP|Tian Y, Peng S, Rodemann T, et al. "[**Automated selection of evolutionary multi-objective optimization algorithms**](https://ieeexplore.ieee.org/abstract/document/9003018)" 2019 IEEE Symposium Series on Computational Intelligence (SSCI). (2019).|MOOP|MOEA|   |
|Meta-TSP|Kanda J Y, de Carvalho A C, Hruschka E R, et al. "[**Using meta-learning to recommend meta-heuristics for the traveling salesman problem**](https://ieeexplore.ieee.org/abstract/document/6146996)" 2011 10th international conference on machine learning and applications and workshops. (2011).|CO|GA|   |
|Meta-QAP|Smith-Miles K A. "[**Towards insightful algorithm selection for optimisation using meta-learning concepts**](https://ieeexplore.ieee.org/abstract/document/4634391)" 2008 IEEE international joint conference on neural networks (IEEE world congress on computational intelligence). (2008).|CO|MMAS|   |

#### 2.2.2 Algorithm Configuration

#### 2.2.3 Algorithm Generation

#### 2.2.4 Algorithm Imitation
|Algorithm|Paper|Optimization Type|Low-Level Optimizer|Code Source|
|:-:|:-:|:-:|:-:|:-:|
|B2Opt|Li X, Wu K, Zhang X, et al. "[**B2Opt: Learning to Optimize Black-box Optimization with Little Budget**](https://arxiv.org/abs/2304.11787)". arXiv preprint arXiv:2304.11787, (2023).|SOP|GA|  |
|GLHF|Li, Xiaobin, et al. "[**GLHF: General Learned Evolutionary Algorithm Via Hyper Functions**](https://arxiv.org/abs/2405.03728)." arXiv preprint arXiv:2405.03728 (2024).|SOP|DE| |
|EvoTF|Lange, Robert Tjarko, Yingtao Tian, and Yujin Tang. "[**Evolution Transformer: In-Context Evolutionary Optimization**](https://arxiv.org/abs/2403.02985)." arXiv preprint arXiv:2403.02985 (2024).|SOP|-|[RobertTLange/evosax](https://github.com/RobertTLange/evosax)|
|LEO-SL|Yu, Peiyu, et al. "[**Latent Energy-Based Odyssey: Black-Box Optimization via Expanded Exploration in the Energy-Based Latent Space**](https://arxiv.org/abs/2405.16730)." arXiv preprint arXiv:2405.16730 (2024).|SOP|-| |
|RIBBO|Song, Lei, et al. "[**Reinforced In-Context Black-Box Optimization**](https://arxiv.org/abs/2402.17423)." arXiv preprint arXiv:2402.17423 (2024).|SOP|-|[RIBBO](https://github.com/songlei00/RIBBO)|
|NAP|Maraval, Alexandre, et al. "[**End-to-end meta-Bayesian optimisation with transformer neural processes**](https://proceedings.neurips.cc/paper_files/paper/2023/hash/2561721d0ca69bab22b749cfc4f48f6c-Abstract-Conference.html)." Advances in Neural Information Processing Systems 36 (2024).|SOP|-| |
|OptFormer|Chen, Yutian, et al. "[**Towards learning universal hyperparameter optimizers with transformers**](https://proceedings.neurips.cc/paper_files/paper/2022/hash/cf6501108fced72ee5c47e2151c4e153-Abstract-Conference.html)." Advances in Neural Information Processing Systems 35 (2022).|SOP|-|[Optformer](https://github.com/google-research/optformer)|
|RNN-Opt|TV, Vishnu, et al. "[**Meta-learning for black-box optimization**](http://proceedings.mlr.press/v70/chen17e.html)." Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Cham: Springer International Publishing (2019).|SOP|-|   |
|RNN-OI|Chen, Yutian, et al. "[**Learning to learn without gradient descent by gradient descent**](http://proceedings.mlr.press/v70/chen17e.html)." International Conference on Machine Learning. PMLR (2017).|SOP|-|   |

<p align="right">
<a href="https://github.com/GMC-DRL/Awesome-MetaBBO/tree/main#awesome-metabbo">Back to Top</a>
</p>

### 2.3 🧬MetaBBO-NE
#### 2.3.1 Algorithm Selection
#### 2.3.2 Algorithm Configuration
|Algorithm|Paper|Optimization Type|Low-Level Optimizer|Code Source|
|:-:|:-:|:-:|:-:|:-:|
|LES|Lange, Robert, et al. "[**Discovering evolution strategies via meta-black-box optimization**](https://iclr.cc/virtual/2023/poster/11005)." The Eleventh International Conference on Learning Representations. (2023).|SOP|CMA-ES|  |
#### 2.3.3 Algorithm Generation
#### 2.3.4 Algorithm Imitation
|Algorithm|Paper|Optimization Type|Low-Level Optimizer|Code Source|
|:-:|:-:|:-:|:-:|:-:|
|LGA|Lange, Robert, et al. "[**Discovering attention-based genetic algorithms via meta-black-box optimization**](https://dl.acm.org/doi/abs/10.1145/3583131.3590496)." Proceedings of the Genetic and Evolutionary Computation Conference. (2023).|SOP|GA|  |
|LTO-POMDP|Gomes H S, Léger B, Gagné C. "[**Meta learning black-box population-based optimizers**](https://arxiv.org/abs/2103.03526)". arXiv preprint arXiv:2103.03526 (2021).|SOP|-| |

<p align="right">
<a href="https://github.com/GMC-DRL/Awesome-MetaBBO/tree/main#awesome-metabbo">Back to Top</a>
</p>

### 2.4 🧠MetaBBO-ICL
#### 2.4.1 Algorithm Selection
|Algorithm|Paper|Optimization Type|Low-Level Optimizer|Code Source|
|:-:|:-:|:-:|:-:|:-:|
|AS-LLM|Wu, Xingyu, et al. "[**Large language model-enhanced algorithm selection: towards comprehensive algorithm representation**](https://ira.lib.polyu.edu.hk/handle/10397/108348)." International Joint Conference on Artificial Intelligence (2024).|SOP|-| |
#### 2.4.2 Algorithm Configuration
#### 2.4.3 Algorithm Generation
|Algorithm|Paper|Optimization Type|Low-Level Optimizer|Code Source|
|:-:|:-:|:-:|:-:|:-:|
|LLaMoCo|Ma, Zeyuan, et al. "[**LLaMoCo: Instruction Tuning of Large Language Models for Optimization Code Generation**](https://arxiv.org/abs/2403.01131)." arXiv preprint arXiv:2403.01131 (2024).|SOP|-|[LLaMoCo-722A](https://anonymous.4open.science/r/LLaMoCo-722A)|
|LLaMEA|van Stein, Niki, and Thomas Bäck. "[**LLaMEA: A Large Language Model Evolutionary Algorithm for Automatically Generating Metaheuristics**](https://arxiv.org/abs/2405.20132)." arXiv preprint arXiv:2405.20132 (2024).|SOP|-| |
|LLMOPT|Huang, Yuxiao, et al. "[**Towards Next Era of Multi-objective Optimization: Large Language Models as Architects of Evolutionary Operators**](https://arxiv.org/abs/2406.08987)." arXiv preprint arXiv:2406.08987 (2024).|MOOP|-|  |
|OptiMUS|AhmadiTeshnizi, Ali, Wenzhi Gao, and Madeleine Udell. "[**OptiMUS: Optimization Modeling Using mip Solvers and large language models**](https://arxiv.org/abs/2310.06116)." arXiv preprint arXiv:2310.06116 (2023).|MILP|-|[teshnizi/OptiMUS](https://github.com/teshnizi/OptiMUS)|
|AEL|Liu, Fei, et al. "[**Algorithm evolution using large language model**](https://arxiv.org/abs/2311.15249)." arXiv preprint arXiv:2311.15249 (2023).|CO|-| |
|EoH|Liu, Fei, et al. "[**Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model**](https://www.researchgate.net/publication/380399749_Evolution_of_Heuristics_Towards_Efficient_Automatic_Algorithm_Design_Using_Large_Language_Model)." arXiv preprint arXiv:2309.03409 (2023).|CO|-| |
#### 2.4.4 Algorithm Imitation
|Algorithm|Paper|Optimization Type|Low-Level Optimizer|Code Source|
|:-:|:-:|:-:|:-:|:-:|
|CMOEA-LLM|Wang, Zeyi, et al. "[**Large Language Model-Aided Evolutionary Search for Constrained Multiobjective Optimization**](https://arxiv.org/abs/2405.05767)." arXiv preprint arXiv:2405.05767 (2024).|CMOP|   |   |
|LEO|Brahmachary, Shuvayan, et al. "[**Large Language Model-Based Evolutionary Optimizer: Reasoning with elitism**](https://arxiv.org/abs/2403.02054)." arXiv preprint arXiv:2403.02054 (2024).|SOP|   |   |
|EvoLLM|Lange, Robert Tjarko, Yingtao Tian, and Yujin Tang. "[**Large Language Models As Evolution Strategies**](https://arxiv.org/abs/2402.18381)." arXiv preprint arXiv:2402.18381 (2024).|SOP|  |   |
|MOEA/D-LLM|Liu, Fei, et al. "[**Large language model for multi-objective evolutionary optimization**](https://arxiv.org/abs/2310.12541)." arXiv preprint arXiv:2310.12541 (2023).|MOOP|MOEA/D|   |
|OPRO|Yang, Chengrun, et al. "[**Large language models as optimizers**](https://arxiv.org/abs/2309.03409)." arXiv preprint arXiv:2309.03409 (2023).|SOP|-|  |
|LMEA|Liu, Shengcai, et al. "[**Large language models as evolutionary optimizers**](https://arxiv.org/abs/2310.19046)." arXiv preprint arXiv:2310.19046 (2023).|SOP|-|  |

<p align="right">
<a href="https://github.com/GMC-DRL/Awesome-MetaBBO/tree/main#awesome-metabbo">Back to Top</a>
</p>

## 🔧3. Others
### 📈3.1 Evaluation Indicator
|Indicator|Paper|
|:-:|:-:|
|ECDF|López-Ibáñez M, Vermetten D, Dreo J, et al. "[**Using the empirical attainment function for analyzing single-objective black-box optimization algorithms**](https://arxiv.org/abs/2404.02031)". arXiv preprint arXiv:2404.02031 (2024).|
|EAF|da Fonseca V G, Fonseca C M. "[**A link between the multivariate cumulative distribution function and the hitting function for random closed sets**](https://www.sciencedirect.com/science/article/pii/S0167715202000469)". Statistics & probability letters (2002).|
### 📊3.2 Landscape Feature
|Feature|Paper|
|:-:|:-:|
|DoE2Vec|van Stein B, Long F X, Frenzel M, et al. "[**Doe2vec: Deep-learning based features for exploratory landscape analysis**](https://dl.acm.org/doi/abs/10.1145/3583133.3590609)" Proceedings of the Companion Conference on Genetic and Evolutionary Computation. (2023).|
|TransOpt|Cenikj G, Petelin G, Eftimov T. "[**TransOptAS: Transformer-Based Algorithm Selection for Single-Objective Optimization**](https://dl.acm.org/doi/abs/10.1145/3638530.3654191)" Proceedings of the Genetic and Evolutionary Computation Conference Companion. (2024).|
|Deep ELA|Seiler M V, Kerschke P, Trautmann H. "[**Deep-ELA: Deep Exploratory Landscape Analysis with Self-Supervised Pretrained Transformers for Single-and Multi-Objective Continuous Optimization Problems**](https://arxiv.org/abs/2401.01192)". arXiv preprint arXiv:2401.01192 (2024).|
|LvsC ELA|Seiler M, Škvorc U, Cenikj G, et al. "[**Learned Features vs. Classical ELA on Affine BBOB Functions**](https://link.springer.com/chapter/10.1007/978-3-031-70068-2_9)" International Conference on Parallel Problem Solving from Nature. Cham: Springer Nature Switzerland (2024).|
|Comparable Feature|Long F X, Vermetten D, van Stein B, et al. "[**BBOB instance analysis: Landscape properties and algorithm performance across problem instances**](https://link.springer.com/chapter/10.1007/978-3-031-30229-9_25)" International Conference on the Applications of Evolutionary Computation (Part of EvoStar). Cham: Springer Nature Switzerland (2023).|
|ISA|Smith-Miles K, Muñoz M A. "[**Instance space analysis for algorithm testing: Methodology and software tools**](https://dl.acm.org/doi/abs/10.1145/3572895)". ACM Computing Surveys 2023.|
|ELA|Mersmann O, Bischl B, Trautmann H, et al. "[**Exploratory landscape analysis**](https://dl.acm.org/doi/abs/10.1145/2001576.2001690)" Proceedings of the 13th annual conference on Genetic and evolutionary computation. (2011).|

<p align="right">
<a href="https://github.com/GMC-DRL/Awesome-MetaBBO/tree/main#awesome-metabbo">Back to Top</a>
</p>




