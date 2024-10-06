# Awesome-MetaBBO

- [1. Survey Papers \& Benchmarks](#1-survey-papers--benchmarks)
  - [1.1. Survey Papers](#11-survey-papers)
  - [1.2. Benchmarks](#12-benchmarks)
- [2. MetaBBO](#2-metabbo)
  - [2.1. MetaBBO-RL](#21-metabbo-rl)
    - [2.1.1. Algorithm Selection](#211-algorithm-selection)
    - [2.1.2. Algorithm Configuration](#212-parameter-contorl)
    - [2.1.3. Algorithm Generation \& Parameter](#213-operator--parameter)
    - [2.1.4. Algorithm Imitation](#214-symbolic)
    - [2.1.5. Others](#215-others)
  - [2.2. MetaBBO-SL](#22-metabbo-with-supervised-learning-metabbo-sl)
    - [2.2.1. Algorithm Selection](#221-operator-selection)
    - [2.2.2. Algorithm Configuration](#222-parameter-contorl)
    - [2.2.3. Algorithm Generation \& Parameter](#223-operator--parameter)
    - [2.2.4. Algorithm Imitation](#224-symbolic)
    - [2.2.5. Others](#225-others)
  - [2.3. MetaBBO-NE](#23-metabbo-with-neuroevolution-metabbo-ne)
    - [2.3.1. Algorithm Selection](#231-operator-selection)
    - [2.3.2. Algorithm Configuration](#232-parameter-contorl)
    - [2.3.3. Algorithm Generation \& Parameter](#233-operator--parameter)
    - [2.3.4. Algorithm Imitation](#234-symbolic)
    - [2.3.5. Others](#235-others)
  - [2.4. MetaBBO-ICL](#24-metabbo-with-in-context-learning )
    - [2.4.1. Algorithm Selection](#241-operator-selection)
    - [2.4.2. Algorithm Configuration](#242-parameter-contorl)
    - [2.4.3. Algorithm Generation \& Parameter](#243-operator--parameter)
    - [2.4.4. Algorithm Imitation](#244-symbolic)
    - [2.4.5. Others](#245-others)
  - [2.5. Others](#25-others)
    - [2.5.1 Evaluation Indicator](#251-evaluation-indicator)
    - [2.5.2 Landscape Feature](#252-landscape-feature)


## 1. Survey Papers \& Benchmarks

### 1.1. Survey Papers

|Paper|
|:-:|
|Li P, Hao J, Tang H, et al. "[**Bridging Evolutionary Algorithms and Reinforcement Learning: A Comprehensive Survey on Hybrid Algorithms**](https://ieeexplore.ieee.org/abstract/document/10637292). IEEE Transactions on Evolutionary Computation. (2024).
|Song Y, Wu Y, Guo Y, et al. "[**Reinforcement learning-assisted evolutionary algorithm: A survey and research opportunities**](https://www.sciencedirect.com/science/article/pii/S2210650224000506). Swarm and Evolutionary Computation. (2024).
|Nikolikj, Ana, et al. "[**Quantifying Individual and Joint Module Impact in Modular Optimization Frameworks**](https://arxiv.org/abs/2405.11964)." 2024 IEEE Congress on Evolutionary Computation (CEC). (2024).
|Qian, Chao, Ke Xue, and Ren-Jian Wang. "[**Quality-Diversity Algorithms Can Provably Be Helpful for Optimization**](https://arxiv.org/abs/2401.10539)." arXiv preprint arXiv:2401.10539. (2024).
|Huang, Beichen, et al. "[**Exploring the True Potential: Evaluating the Black-box Optimization Capability of Large Language Models**](https://arxiv.org/abs/2404.06290)." arXiv preprint arXiv:2404.06290. (2024).
|Chernigovskaya, Maria, Andrey Kharitonov, and Klaus Turowski. "[**A Recent Publications Survey on Reinforcement Learning for Selecting Parameters of Meta-Heuristic and Machine Learning Algorithms**](https://www.scitepress.org/Papers/2023/119543/119543.pdf)." CLOSER. (2023).
|Drugan, Madalina M. "[**Reinforcement learning versus evolutionary computation: A survey on hybrid algorithms**](https://www.sciencedirect.com/science/article/abs/pii/S2210650217302766)." Swarm and Evolutionary Computation. (2019).

### 1.2. Benchmarks

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
<a href="https://github.com/GMC-DRL/psc4MetaBBO/tree/main#useful-papers-and-source-codes-for-meta-black-box-optimization-metabbo">Back to Top</a>
</p>


## 2. MetaBBO

### 2.1 MetaBBO-RL
#### 2.1.1 Algorithm Selection
|Algorithm|Paper|Optimization Type|Low-Level Optimizer|RL|Code Source|
|:-:|:-:|:-:|:-:|:-:|:-:|
|HHRL-MAR|Zhu N, Zhao F, Cao J. "[**A Hyperheuristic and Reinforcement Learning Guided Meta-heuristic Algorithm Recommendation**](https://ieeexplore.ieee.org/abstract/document/10580058/)" 2024 27th International Conference on Computer Supported Cooperative Work in Design (CSCWD) (2024)|SOP|SI| | |
|R2-RLMOEA|Tahernezhad-Javazm F, Rankin D, Bois N D, et al. "[**R2 Indicator and Deep Reinforcement Learning Enhanced Adaptive Multi-Objective Evolutionary Algorithm**](https://arxiv.org/abs/2404.08161)". arXiv preprint arXiv:2404.08161 (2024).|MOOP|EAs|DDQN| |
|RL-DAS|Guo, Hongshu, et al. "[**Deep Reinforcement Learning for Dynamic Algorithm Selection: A Proof-of-Principle Study on Differential Evolution**](https://ieeexplore.ieee.org/abstract/document/10496708/)." IEEE Transactions on Systems, Man, and Cybernetics: Systems (2024).|SOP|DE|PPO| |

#### 2.1.2 Algorithm Configuration

|Algorithm|Paper|Optimization Type|Low-Level Optimizer|RL|Code Source|
|:-:|:-:|:-:|:-:|:-:|:-:|
|DE-DDQN|Sharma, Mudita, et al. "[**Deep reinforcement learning based parameter control in differential evolution**](https://dl.acm.org/doi/abs/10.1145/3321707.3321813)." Proceedings of the Genetic and Evolutionary Computation Conference (2019).|SOP|DE|Tabular Q-learning|  |
|QLMOPSO|Liu Y, Lu H, Cheng S, et al. "[**An adaptive online parameter control algorithm for particle swarm optimization based on reinforcement learning**](https://ieeexplore.ieee.org/abstract/document/8790035)" 2019 IEEE congress on evolutionary computation (CEC) (2019).|SOP|PSO|Tabular Q-learning|  |
|RL-MOEA/D|Ning W, Guo B, Guo X, et al. "[**Reinforcement learning aided parameter control in multi-objective evolutionary algorithm based on decomposition**](https://link.springer.com/article/10.1007/s13748-018-0155-7)". Progress in Artificial Intelligence 2018.|MOOP|MOEA/D|SARSA| |
|QFA|Sadhu A K, Konar A, Bhattacharjee T, et al. "[**Synergism of firefly algorithm and Q-learning for robot arm path planning**](https://www.sciencedirect.com/science/article/pii/S2210650217306776)". Swarm and Evolutionary Computation 2018.|SOP|FA|Tabular Q-learning| |




