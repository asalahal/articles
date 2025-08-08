11218 IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 35, NO. 8, AUGUST 2024

## Meta Learning With Graph Attention Networks for Low-Data Drug Discovery


[Qiujie Lv, Guanxing Chen, Ziduo Yang, Weihe Zhong, and Calvin Yu-Chian Chen](https://orcid.org/0000-0003-4979-7906 )



_**Abstract**_ **— Finding candidate molecules with favorable phar-**
**macological activity, low toxicity, and proper pharmacokinetic**
**properties is an important task in drug discovery. Deep neural**
**networks have made impressive progress in accelerating and**
**improving drug discovery. However, these techniques rely on**
**a large amount of label data to form accurate predictions**
**of molecular properties. At each stage of the drug discovery**
**pipeline, usually, only a few biological data of candidate molecules**
**and derivatives are available, indicating that the application**
**of deep neural networks for low-data drug discovery is still**
**a formidable challenge. Here, we propose a meta learning**
**architecture with graph attention network, Meta-GAT, to predict**
**molecular properties in low-data drug discovery. The GAT**
**captures the local effects of atomic groups at the atom level**
**through the triple attentional mechanism and implicitly captures**
**the interactions between different atomic groups at the molecular**
**level. GAT is used to perceive molecular chemical environment**
**and connectivity, thereby effectively reducing sample complexity.**
**Meta-GAT further develops a meta learning strategy based on**
**bilevel optimization, which transfers meta knowledge from other**
**attribute prediction tasks to low-data target tasks. In summary,**
**our work demonstrates how meta learning can reduce the amount**
**of data required to make meaningful predictions of molecules in**
**low-data scenarios. Meta learning is likely to become the new**
**learning paradigm in low-data drug discovery. The source code**
**is publicly available at: https://github.com/lol88/Meta-GAT.**


_**Index Terms**_ **— Drug discovery, few examples, graph attention**
**network, meta learning, molecular property.**


I. I NTRODUCTION

RUG discovery is a high-investment, long-period, and
# D high-risk systems engineering [1]. When molecular biol
ogy studies have identified an effective target associated with
a disease, the subsequent path of drug discovery becomes
relatively clear [2]. With the help of various computer-aided
virtual screening technologies and high-throughput omics


Manuscript received 24 November 2021; revised 21 May 2022, 16 October
2022, and 26 December 2022; accepted 24 February 2023. Date of publication
6 March 2023; date of current version 6 August 2024. This work was
supported in part by the National Natural Science Foundation of China under
Grant 62176272 and in part by the China Medical University Hospital under
Grant DMR-112-085. _(Corresponding author: Calvin Yu-Chian Chen.)_
Qiujie Lv, Guanxing Chen, Ziduo Yang, and Weihe Zhong are with
the Artificial Intelligence Medical Research Center, School of Intelligent
Systems Engineering, Shenzhen Campus of Sun Yat-sen University, Shenzhen,
Guangdong 518107, China.
Calvin Yu-Chian Chen is with the Artificial Intelligence Medical Research
Center, School of Intelligent Systems Engineering, Shenzhen Campus of
Sun Yat-sen University, Shenzhen, Guangdong 518107, China, also with
the Department of Medical Research, China Medical University Hospital,
Taichung 40447, Taiwan, and also with the Department of Bioinformatics
and Medical Engineering, Asia University, Taichung 41354, Taiwan (e-mail:
chenyuchian@mail.sysu.edu.cn).
Digital Object Identifier 10.1109/TNNLS.2023.3250324



technologies, researchers can integrate the relevant knowledge
of computational chemistry, physics, and structural biology to
effectively screen and design molecular compounds [3], [4],

[5], [6], [7]. The key issue of drug discovery is the screening
and optimization of candidate molecules, which must meet a
series of criteria: the compound needs to have suitable potential for biological targets, and exhibit good physicochemical
properties; absorption, distribution, metabolism, excretion, and
toxicity (ADMET); water solubility; and mutagenicity [8], [9].
However, there are usually only a few validated leads and
derivatives that can be used for lead optimization [10], [11].
Also, due to the possible toxicity, low activity, and low
solubility, there are often only a few real biological data
on candidate molecules and analog molecules. The accuracy
of the physical chemical properties of candidate molecules
directly affects the results of the drug development process.
Therefore, researchers have paid more and more attention to
accurately predict the physicochemical properties of candidate
molecules with low data.

In the past few years, deep learning technology has been
implemented to accelerate and improve the drug discovery
process [12], [13], [14], [15], and some key advances have
been made in molecular property prediction [16], [17], [18],

[19], [20], side effect prediction [21], [22], [23], and virtual
screening [24], [25]. In particular, the graph neural network
(GNN), which can learn the information contained in the nodes
and edges directly from the chemical graph structure, has
aroused the strong interest of bioinformatics scientists [26],

[27], [28], [29]. The performance of deep learning algorithm
depends largely on the size of the training data, and a larger
sample size usually produces a more accurate model. Given
a large amount of labeled data, deep neural networks have
enough ability to learn complex representations of inputs [30].
However, this is obviously in contradiction with insufficient
data in the initial stage of drug discovery. Due to the scarcity
of labeled data, achieving satisfactory results for low-data
drug discovery remains a challenge. The paradigm of artificial
intelligence for drug discovery has changed: from large-scale
sample learning to small sample learning [31], [32], [33].
The human brain’s understanding of objective things
does not necessarily require large sample training, and it
can be learned in many cases based on simple analogies

[34], [35], [36]. DeepMind explores how the brain learns
with few experience, that is, “meta learning” or “learning
to learn” [37]. The understanding of meta learning mode is
one of the important ways to achieve general intelligence.
Biswas et al. [38] developed UniRep for protein engineering to



2162-237X © 2023 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.
See https://www.ieee.org/publications/rights/index.html for more information.


Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:27:22 UTC from IEEE Xplore. Restrictions apply.


LV et al.: META LEARNING WITH GRAPH ATTENTION NETWORKS FOR LOW-DATA DRUG DISCOVERY 11219



efficiently use resource-intensive high-fidelity assays without
sacrificing throughput, and subsequent low- _N_ supervision then
identifies improvements to the activity of interest. Liu et al.

[39] from the Chinese Academy of Sciences have established
a complete and effective screening method for disease target markers based on few examples (or even one sample).
Lin et al. [40] proposed a prototypical graph contrastive learning (PGCL) method for learning graph representation, which
improved the results of molecular property prediction. Yu and
Tran [25] proposed an XGBoost-based fitted _Q_ iteration
algorithm with fewer training data for finding the optimal
structured treatment interruption (STI) strategies for HIV
patients. They have made certain explorations and attempts
in the field of drug virtual screening and combination drug
prediction based on the few examples learning method [41],

[42]. The abovementioned work is a useful attempt by meta
learning for few samples learning problems, indicating that the
meta learning method has the potential to be a useful tool in
drug discovery and other bioinformatics research fields.
Meta learning uses meta knowledge to reduce requirement
for sample complexity, thus solving the core problem of
minimizing the risk of unreliable experience. However, the
molecular structure is usually composed of the interaction
between atoms and complex electronic configurations. Even
small changes in the molecular structure may lead to completely opposite molecular properties. The model learns the
complexity of molecular structure, which requires that the
model should perfectly extract the local environmental influence of neighboring atoms on the central atom and the rich
nonlocal information contained between pairs of atoms that
are topologically far apart. Therefore, meta learning for lowdata drug discovery is highly dependent on the structure of
the network and needs to be redesigned for widely varying
tasks.

Meta learning has made some representative attempts to
predict molecular properties. Altae-Tran et al. [43] introduced
an architecture of iteratively refined long short-term memory
(IterRefLSTM) that uses IterRefLSTM to generate dually
evolved embeddings for one-shot learning. Adler et al. [44]
proposed cross-domain Hebbian ensemble few-shot learning
(CHEF), which achieves representation fusion by an ensemble
of Hebbian learners acting on different layers of a deep neural
network. The Meta-molecular graph neural network (MGNN)
leverages a pretrained GNN and introduces additional selfsupervised tasks, such as bond reconstruction and atomtype prediction to be jointly optimized with the molecular
property prediction tasks [45]. Meta-MGNN, CHEF, obtains
meta knowledge through pretraining on a large-scale molecular corpus and additional self-supervised model parameters.
IterRefLSTM trains the memory-augmented model, which
restricts the model structure and can only be used in specific domain scenarios. How to represent molecular features
effectively and how to capture common knowledge between
different tasks are great challenges that exist in meta learning.
In this work, we propose a meta learning architecture
based on graph attention network, Meta-GAT, to predict
the biochemical properties of molecules in low-data drug



discovery. The graph attention network captures the local
effects of atomic groups at the atomic level through the
triple attentional mechanism, so that the GAT can learn the
influence of the atom group on the properties of the compound.
At the molecular level, GAT treats the entire molecule as a
supervirtual node that connects every atom in a molecule,
implicitly capturing the interactions between different atomic
groups. The gated recurrent unit (GRU) hierarchical model
mainly focuses on abstracting or transferring limited molecular
information into higher-level feature vectors or meta knowledge, improving the ability of the GAT to perceive chemical
environment and connectivity in molecules, thereby efficiently
reducing sample complexity. This is very important for lowdata drug discovery. Meta-GAT benefits from meta knowledge
and further develops a meta learning strategy based on bilevel
optimization, which transfers meta knowledge from other
attribute prediction tasks to low-data target tasks, allowing
the model to quickly adapt to molecular attribute predictions
with few examples. Meta-GAT achieved accurate prediction of
few examples’s molecular new properties on multiple public
benchmark datasets. These advantages indicate that Meta-GAT
is likely to become a viable option for low-data drug discovery.
In addition, the Meta-GAT code and data are open source at
https://github.com/lol88/Meta-GAT, so that the results can be
easily replicated.
Our contributions can be summarized as follows.

1) We create a chemical tool to predict multiple physiological properties of new molecules that are invisible to the
model. This tool could push the boundaries of molecular
representation for low-data drug discovery.
2) The proposed Meta-GAT captures the local effects of
atomic groups at the atomic level through the triplet
attentional mechanism and can also model global effects
of molecules at the molecular level.

3) We propose a meta learning strategy to selectively
update parameters within each task through a bilevel
optimization, which is particularly helpful to capture the
generic knowledge shared across different tasks.
4) Meta-GAT demonstrates how meta learning can reduce
the amount of data required to make meaningful predictions of molecules in low-data drug discovery.


II. M ETHODS


In this section, we first briefly introduce the mathematical
formalism of Meta-GAT and then introduce the meta learning
strategy and graph attention network structure. Finally, the
parameters and details of the model training are shown. Fig. 1
shows the overall architecture of Meta-GAT for low-data drug
discovery.


_A. Problem Formulations_


Consider several common drug discovery tasks _T_, such as
predicting the toxicity and side effects of new molecules, _x_ is
the compound molecule to be measured, and the label _y_ is the
binary experimental label (positive/negative) of the molecular
properties. Suppose that all some potential laws considered



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:27:22 UTC from IEEE Xplore. Restrictions apply.


11220 IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 35, NO. 8, AUGUST 2024


Fig. 1. Meta learning framework for few examples molecular property prediction. The blue box and the orange box represent the data flow in the training
phase and the test phase, respectively.



by the model are called hypothesis space _H_ . _h_ is the optimal
hypothesis from _x_ to _y_ . The expected risk _R(h)_ represents
the prediction ability of the decision model for all samples.
The empirical risk _R(h_ _I_ _)_ represents the predictive ability of
the model for samples in the training set by calculating the
average value of the loss function, and _I_ represents the number
of samples in the training set. The empirical risk _R(h_ _I_ _)_ is used
to estimate expected risk _R(h)_ . In real-world applications, only
a few examples are available for a property prediction task of
a new molecule, that is, _I_ → few. According to the empirical
risk minimization theory, if only a few training samples can be
provided, which makes the empirical risk _R(h_ _I_ _)_ far from the
approximation of the expected risk _R(h)_, the obtained empirical risk minimizer is unreliable [46]. The learning challenge
is to obtain a reliable empirical risk minimization from a few
examples. This minimizer results in _R(h_ _I_ _)_ approaching the
optimal _R(h)_, as shown in the following equation:


E[ _R(h_ _I_ →few _)_ − _R(h)_ ] = 0 _._ (1)


The empirical risk minimization is closely related to sample
complexity. Sample complexity refers to the number of training samples required to minimize the loss of empirical risk
_R(h_ _I_ _)_ . According to Vapnikâ A¸S Chervonenkis (VC), when [˘]
samples are insufficient, _H_ needs less complexity, so that the
few examples provided are sufficient for compensation. We use
meta knowledge _w_ to reduce the complexity of learning
samples, thus solving the core problem of minimizing the risk
of unreliable experience.


_B. Meta Learning_


Meta learning, also known as learning to learn, means
learning a learning experience by systematically observing
how the model performs in a wide range of learning tasks.
This learning experience is called meta knowledge _w_ . The
goal of meta learning is to find the _w_ shared across different
tasks, so that the model can quickly generalize to new tasks
that contain only a few examples with supervised information.



The difference between meta learning and transfer learning is
that transfer learning is usually fitting the distribution of one
data, while meta learning is fitting the distribution of multiple
similar tasks. Therefore, the training samples of meta learning
are a series of tasks.

Model-agnostic meta-learning (MAML) [47] is used as a
base meta learning algorithm for the Meta-GAT framework.
Meta-GAT selectively updates parameters within each task
through a bilevel optimization and transfers meta knowledge
to new tasks with few label samples, as shown in Fig. 1.
Bilevel optimization means that one optimization contains the
another optimization as a constraint. In inner-level optimization, we hope to learn a general meta knowledge _w_ from
the support set of training tasks, so that the loss of different
tasks can be as small as possible. The inner level optimization
phase can be formalized, as shown in (3). In outer-level
optimization, Meta-GAT calculates the gradient relative to the
optimal parameter in the query set of each task and calculates
the minimum total loss value of all training tasks to optimize
the _w_ parameter, thereby reducing the expected loss of the
training task, as shown in (2). Algorithm 1 shows the specific
algorithm details



_w_ [∗] = argmin

_w_



_M_
� _L_ [meta] _f_ _θ_ � _θ_ [∗] _[(][i][)]_ _(w), D_ train _[q]_ � (2)


_i_ =1



_θ_ [∗] _[(][i][)]_ _(w)_ = argmin _θ_ _L_ [task] _f_ _θ_ � _θ, w, D_ train _[s][(][i][)]_ � (3)


where _L_ meta and _L_ task refer to the outer and inner objectives,
respectively. _i_ represents the _i_ th training task.
Specifically, first, the train tasks _T_ train and test tasks _T_ test are
extracted from a set of multitask _T_ for drug discovery, where
each task has a support set _D_ _[s]_ and a query set _D_ _[q]_ . MetaGAT uses a large number of training tasks _T_ train to fitting
the distribution of multiple similar tasks _T_ . Second, MetaGAT sequentially iterates a batch of training tasks, learns
task-specific parameters, and tries to minimize the loss using



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:27:22 UTC from IEEE Xplore. Restrictions apply.


LV et al.: META LEARNING WITH GRAPH ATTENTION NETWORKS FOR LOW-DATA DRUG DISCOVERY 11221



**Algorithm 1** Pseudocode of Meta-GAT for Low-Data Drug
Discovery
**Require:** A set of tasks for predicting molecular properties
_T_ ;
**Ensure:** GAT parameters _θ_, step sizes _α, β_ ;

1: Randomly initialize _θ_ ;

2: **while** not done **do**


3: Sample batch of tasks _T_ _train_ ∼ _T_ ;

4: **for all** _T_ _train_ **do**

5: Sample _m_ examples _D_ _train_ _[s]_ [= {] _[D]_ [1] _[,][ D]_ [2] _[, . . .,][ D]_ _[m]_ [} ∈]
_D_ _train_ ;

6: Evaluate ∇ _θ_ _L_ _T_ _train_ _( f_ _θ_ _)_ by _y_ _train_ _[s]_ =
_G AT (D_ _train_ _[s]_ _[, θ)]_ [;]

7: Compute adapted parameters with gradient
descent:
_θ_ _i_ [′] [=] _[ θ]_ [ −] _[α]_ [∇] _[θ]_ _[L]_ _[T]_ _train_ _[(][ f]_ _[θ]_ _[)]_ [;]

8: Sample _n_ examples _D_ _train_ _[q]_ [= {] _[D]_ [1] _[,][ D]_ [2] _[, . . .,][ D]_ _[n]_ [} ∈]
_D_ _train_ − _D_ _train_ _[s]_ [;]

9: Evaluate _L_ [′] _T_ _train_ [by] _[ y]_ _train_ _[q]_ [=] _[ G AT][ (][D]_ _train_ _[q]_ _[, θ]_ [′] _[)]_ [;]
10: **end for**
11: Updat _θ_ ← _θ_ − _β_ ∇ _θ_ � _T_ _train_ ∼ _p(T )_ _[L]_ [′] _T_ _train_
12: **end while**

13: Sample batch of tasks _T_ _test_ ∼ _T_ − _T_ _train_
14: **for all** _T_ _test_ **do**

15: Sample k examples _D_ _test_ _[s]_ [= {] _[D]_ [1] _[,][ D]_ [2] _[, . . .,][ D]_ _[k]_ [} ∈] _[D]_ _[test]_
16: // Similar to the training phase
17: Evaluate and Compute adapted parameters with gradient descent

18: Updat _θ_

19: Sample j examples _D_ _test_ _[q]_ = { _D_ 1 _, D_ 2 _, . . ., D_ _j_ } ∈
_D_ _test_ − _D_ _test_ _[s]_
20: _y_ _test_ _[q]_ [=] _[ G AT][ (][D]_ _test_ _[q]_ _[, θ)]_
21: **end for**


gradient descent. The corresponding optimal parameters _θ_ are
obtained from each task’s support set, as shown in (4). This
parameters are not assigned to _θ_ directly, but are cached


_θ_ _i_ [′] [=] _[ θ]_ [ −] _[α]_ [∇] _[θ]_ _[L]_ _[T]_ train _[(][ f]_ _[θ]_ _[)]_ (4)

_θ_ ← _θ_ − _β_ ∇ _θ_ � _L_ [′] _T_ train � _f_ _θ_ _i_ [′] � _._ (5)

_T_ train ∼ _p(T )_


Then, the outer-level optimization learns _w_, such that it
produces models _f_ _θ_ [see (5)]. Each task’s query set is used to
obtain a gradient value on each task-specific parameter _θ_ . The
vector sum of the gradient values, obtained from the above
batch task query set, is used to update the parameters of the
meta learner. The model continues iterating up to a preset
number of times, and the best meta model is selected based
on the query set


_θ_ [∗] = argmin E � _θ, w, D_ test _[s]_ � _._ (6)
_θ_ _T_ test ∈ _T_ _[L]_


Finally, in the testing phase, Meta-GAT, which has learned
meta knowledge _w_, learns the specificity of the new test task
through a few inner optimizations on the support set of the
new task, as shown in (6). Note that the model parameters _θ_
exist separately or within meta knowledge _w_ . We evaluate the



TABLE I


C ODED I NFORMATION FOR A TOMIC AND B OND F EATURES


performance of the model by the accuracy of _θ_ on the query
set of the new task. In the process of learning new tasks, the
model benefits from meta knowledge to reduce the requirement
of sample complexity, so as to realize the optimization strategy,
which is faster to search the parameterized _θ_ of the hypothesis
_h_ ∈ _H_ in the hypothesis space _H_ .
Meta-GAT essentially searches for a hypothesis that is better
for all tasks of predicting the properties of drug molecules.
Therefore, when updating parameters, it combines the loss of
all tasks on the query set to specify the gradient update. The
parameter _θ_ obtained in this way is already an approximate
optimal hypothesis on the new task, and the optimal hypothesis
can be reached with few inner iterations.

Our Meta-GAT uses meta knowledge _w_ to guide the model
to search for the parameter _θ_ that approximates the optimal
hypothesis _h_ in the hypothesis space _H_, leading to the
minimization of empirical risk. The meta knowledge _w_ is
obtained through limited analysis of new molecules and prior
knowledge analysis of many similar molecules. The meta
knowledge _w_ changes the search strategy by providing a better
parameter initialization or providing a search direction. MetaGAT was rapidly migrated from a better hypothesis site to
the new task through several inner optimizations on fewer
new molecular instances, and then, the percentage of correctly
assigned molecules with/without toxicity was increased.


_C. Molecular Graph Representation_


Molecules are coded into graphs with node features, edge
features, and adjacency matrices for input into the graph
network. We use a total of nine atomic features and four

bond features to characterize the molecular graph’s structure
(see Table I). Atom features include hybridization, aromaticity, chirality, and so on, and bond features include type,
conjugation, ring, and so on. Molecular structures usually
involve atomic interactions and complex electronic structures,
and bond features contain rich information about molecular

scaffolds and conformational isomers. The encoded molecular

graph can implicitly capture the local environment of the
molecule and the key interactions between atoms and electrons



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:27:22 UTC from IEEE Xplore. Restrictions apply.


11222 IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 35, NO. 8, AUGUST 2024


Fig. 2. Schematic of graph attention network architecture for meta learning.



and provide insight into the edge characteristics of molecular
bonds.


_D. Graph Attention Network_


GNN has made substantial progress in the field of chemical
informatics. It has the extraordinary capacity to learn the
intricate relationships between structures and properties [16],

[48], [49], [50]. The attention mechanism has proved its
outstanding performance in predicting molecular properties.
Molecular structure involves the spatial position of atoms
and the types of chemical bonds. Topologically adjacent
nodes in molecules have a greater chance of interacting with
each other. In some cases, they can also form functional
groups that determine the chemical properties of the molecule.
In addition, pairs of atoms that are topologically far apart
may also have significant interactions, such as intramolecular
hydrogen bonds. Our graph attention network extracts insights
on molecular structure and features from both local and global
perspectives, as shown in Fig. 2. GAT captures the local effects
of atomic groups at the atomic level through the attentional
mechanism and can also model global effects of molecules at
the molecular level.

The molecule _G_ = _(v, e)_ can be defined as a graph composed of a set of atoms (nodes) _v_ and a set of bonds (edges) _e_ .
Similar to the previous study, we encode chemical information
including nine atomic features and four bond features into the
molecular graph as the input of graph attention network. For
the local environment within the molecule, previous graph networks only aggregate the neighbor nodes’ information, which
may lead to insufficient edge (bond) information extraction.
Our GAT gradually aggregates the triplet embedding of target
node _v_ _i_, neighbor node _v_ _j_, and edge _e_ _i j_ through the triple
attention mechanism.



Specifically, GAT first performs linear transformation and
nonlinear activation on the neighbor nodes’ state vectors _v_ _i_, _v_ _j_
and their edge hidden states _e_ _i j_ to align these vectors to the
same dimension, and concatenate them into triplet embedding
vectors. Then, _h_ _i j_ is normalized by the softmax function over
all neighbor nodes to get attention weights _a_ _i j_ . Finally, the
node hidden state and edge hidden state elementwise multiplied by neighbor node representation, and the information of
neighbors (including neighbor nodes and edges) is aggregated
according to the attention weight to obtain the context state _c_ _i_
of the atom _i_ . The formula is shown below


_h_ _i j_ = LeakyReLU� _W_     - � _v_ _i_ _, e_ _i j_ _, v_ _j_ �� (7)

exp� _h_ _i j_ �
_a_ _i j_ = softmax� _h_ _i j_ � = (8)

~~�~~ _j_ ∈ _N_ _(i)_ [exp] ~~�~~ _h_ _i j_ ~~�~~

_c_ _i_ = � _a_ _i j_      - _W_      - � _e_ _i j_ _, v_ _j_ � (9)

_j_ ∈ _N_ _(i)_


where _N_ _(i)_ is the set of neighbor nodes for node _i_ . _W_ is the
trainable weight matrix. Then, the GRU is used as a message
transfer function to fuse messages with a farther radius to
generate a new context state, as shown in Fig. 2 (bottom left).
As the time step _t_ increases, messages of nodes and edges in
the range centered on node _I_ and whose radius increases with
_t_ are collected successively to generate new states _h_ _i_ _[t]_ [, which]
is computed by


_h_ _i_ _[t]_ [=][ GRU] � _h_ _i_ _[t]_ [−][1] _, c_ _i_ _[t]_ [−][1] � _._ (10)


In order to include more global information from the
molecule, GAT aggregates the atomic level representation
through the readout function, which treats the entire molecule
as a supervirtual node that connects every atom in a molecule.
We use the bidirectional GRU (BiGRU) with attention to



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:27:22 UTC from IEEE Xplore. Restrictions apply.


LV et al.: META LEARNING WITH GRAPH ATTENTION NETWORKS FOR LOW-DATA DRUG DISCOVERY 11223



TABLE II



TABLE II Meta-GAT is implemented using the pytorch framework

D ETAILED D ESCRIPTION OF THE B ENCHMARK D ATASETS and uses the Adam optimizer [52] with a 0.001 learning rate

for gradient descent optimization. The learning rate for inner
iterations is 0.1. Generates information around atoms with a

radius of 2. The output unit of the full connection layer is 200.
Both GRU and BiGRU also have 200 hidden units. Gradient

descent is performed five times in each iteration of the training
and testing phase, _α, β_ = 5. During training, 10 000 episodes

connect node features with historical information from two are generated _K_ = _N_ pos + _N_ neg _(N_ pos _, N_ neg ∈[1 _,_ 5 _,_ 10] _)_ in _N_ directions, so as to obtain a graph-level (molecular) represen- way _K_ -shot. _N_ pos and _N_ neg, respectively, represent the number
_M_ . The update gate in the GRU recurring network cell of positive and negative examples in the support set. We use
ensures that information is effectively transmitted to distant CrossEntropyLoss as the loss function of the classification
nodes, while the reset gate helps to filter out information that is task. When training Meta-GAT on the task of molecular
not relevant to the learning task. Moreover, different attention biochemical property prediction, multiple tasks are divided
weights can focus on the implicit interactions between distant into two disjoint task sets, training task and test task. The
atoms and extract more information related to learning tasks. training/testing division method for each dataset is the same
The formula of readout function is shown below as the comparison experiment. During the prediction phase,

_M_ = ←−−GRU att ←− _h_ _[t]_ [ ��][−−→] GRU _,_ att −→ _h_ _[t]_ [ ���] (11) a batch of support sets with size _N_ pos + _N_ neg and a batch of
� � � � � query sets with size _K_ = 128 are randomly sampled from a

where att uses the same attention mechanism as before. The task’s dataset. For each test task, 20 independent runs were
final molecular representation _M_ is used by the classifier for performed based on different random seeds, and the average
molecular property prediction. value of area under the receiver operating characteristic curve
GAT learns the contextual representation of each atom (ROC-AUC) was calculated in the report of experimental
by aggregating the triple information from the atom feature, section.
the neighboring atoms feature information, and the feature In addition, we also analyze the total training time,
information of the connecting bond through the message pass- meta training time, meta testing time, number of
ing mechanism and attention mechanism. Then, the context multiply–accumulate operations (MACs), and model size
representations of atoms are gradually aggregated by BiGRU to evaluate the computational complexity of the proposed
based on the attention mechanism to generate a global state method. Meta-GAT consists of two steps, the meta training
vector for the entire molecule. The final vector representation phase and the meta testing phase. Total training time refers
is a high-quality descriptor of molecular structure information, to the cost of stabilizing the performance of Meta-GAT on
which reduces the difficulty of learning the unsupervised new tasks. Meta training time is the cost of one iteration
information in molecular graph by the meta learning model. in the meta training phase. Meta testing time refers to the

cost of Meta-GAT learning the prediction task of molecular
new property with few samples in the meta testing stage.
Within an iteration, both the support set and the query set

We report the experimental results of the Meta-GAT model

participate in the model forward calculation and perform

on multiple public benchmark datasets. Table II shows the

one or more iterations of gradient descent. The cost of

detailed information of the benchmark dataset, including cat
one iteration in meta training stage, namely, meta training

egories, tasks, and the number of molecules. All datasets are time, is 2 _N_ ∗ _α_ times of the model’s forward calculation
available for download at the public project MoleculNet [51]. time, while meta testing time is 2 ∗ _β_ times longer than the

model’s forward computation time. GeForce RTX 2060 is

_F. Model Implementation and Evaluation Protocols_ used in this experiment, and _N_ is 8, and _α_ and _β_ are 5.

Meta-GAT performs linear transformation and nonlinear The average forward computation time of Meta-GAT on
activation from both atomic features and neighbor features the Tox21 and Side Effect Resource (SIDER) datasets is
to unify vector lengths [see (7)]. Then, the triplet embedding 14.84 and 23.08 ms, respectively. Therefore, the meta training
vectors of atoms are aligned using a fully connected layer, and time is about 1187.2 and 1846.4 ms, the meta testing time
the attention weights are calculated using softmax [see (8)]. is about 148.4 and 230.8 ms, and the total training time is
The weighted sum of the atoms current state vectors is taken about 7.3 and 6 h, respectively. The MACs of Meta-GAT are
as the attention context vector of a single atom [see (9)], 3.17E9, and the model size is 4.8 M. The training time of
which is fed into the GRU along with the current state meta-learning-based GAT is longer than that of GAT, but the
10)]. This process is repeated twice to generate size of obtained prediction model is the same as GAT.
a new state vector for each atom. Finally, we assume that We compare Meta-GAT with multiple baseline modthe molecular embedding is a virtual node embedding, so that els, including random forest (RF) [53], Graph Conv [54],
the whole molecule can be embedded as if it was a single Siamese [55], MAML [47], attention LSTM (attnLSTM) [43],
atom. Similar to the above process, we combine the context IterRefLSTM [43], Meta-MGNN [45], edge-labeling GNN
state vectors of each atom from both directions into a BiGRU (EGNN) [56], PreGNN [57], prototypical networks (PN)
11)]. This process is also repeated twice to obtain a graph [58], CHEF [44], attentive fingerprint (Attentive FP) [16],
representation at the molecular level. communicative message passing neural network (CMPNN)

Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:27:22 UTC from IEEE Xplore. Restrictions apply.



D ETAILED D ESCRIPTION OF THE B ENCHMARK D ATASETS



connect node features with historical information from two

directions, so as to obtain a graph-level (molecular) representation _M_ . The update gate in the GRU recurring network cell
ensures that information is effectively transmitted to distant
nodes, while the reset gate helps to filter out information that is
not relevant to the learning task. Moreover, different attention
weights can focus on the implicit interactions between distant
atoms and extract more information related to learning tasks.
The formula of readout function is shown below



←−− ←− −→
_M_ = GRU att _h_ _[t]_ [ ��][−−→] GRU _,_ att _h_ _[t]_ [ ���] (11)
� � � � �



where att uses the same attention mechanism as before. The

final molecular representation _M_ is used by the classifier for
molecular property prediction.
GAT learns the contextual representation of each atom
by aggregating the triple information from the atom feature,
the neighboring atoms feature information, and the feature
information of the connecting bond through the message passing mechanism and attention mechanism. Then, the context
representations of atoms are gradually aggregated by BiGRU
based on the attention mechanism to generate a global state
vector for the entire molecule. The final vector representation
is a high-quality descriptor of molecular structure information,
which reduces the difficulty of learning the unsupervised
information in molecular graph by the meta learning model.



_E. Datasets_



We report the experimental results of the Meta-GAT model
on multiple public benchmark datasets. Table II shows the
detailed information of the benchmark dataset, including categories, tasks, and the number of molecules. All datasets are
available for download at the public project MoleculNet [51].



_F. Model Implementation and Evaluation Protocols_



Meta-GAT performs linear transformation and nonlinear
activation from both atomic features and neighbor features
to unify vector lengths [see (7)]. Then, the triplet embedding
vectors of atoms are aligned using a fully connected layer, and
the attention weights are calculated using softmax [see (8)].
The weighted sum of the atoms current state vectors is taken
as the attention context vector of a single atom [see (9)],
which is fed into the GRU along with the current state
vector [see (10)]. This process is repeated twice to generate
a new state vector for each atom. Finally, we assume that
the molecular embedding is a virtual node embedding, so that
the whole molecule can be embedded as if it was a single
atom. Similar to the above process, we combine the context
state vectors of each atom from both directions into a BiGRU

[see (11)]. This process is also repeated twice to obtain a graph
representation at the molecular level.


11224 IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 35, NO. 8, AUGUST 2024


TABLE III


S CORES FOR C ONSISTENCY C HECKS ON THE T OX 21 D ATASET U SING

K APPA AND P AIRED W ILCOXON T ESTS


Fig. 3. ROC-AUC scores of Meta-GAT and previous models in the Tox21
few examples prediction task. 1+/5− represent the number of positive and
negative examples is 1 and 5, respectively.




[59], Weave [60], continuous and data-driven descriptors
(CDDD) [49], DeepTox [61], molecule attention transformer (MAT) [62], molecular prediction model fine-tuning
(MolPMoFiT) [63], N-Gram [64], molecular to context vector (Mol2Context-vec) [17], and triplet message network
(TrimNet) [29]. In the reproducibility settings, Siamese,
MAML, AttnLSTM, IterRefLSTM, Meta-MGNN, EGNN,
PN, and CHEF are based on meta learning methods, using
the same training settings as Meta-GAT. RF and Graph Conv
are single-task models. DeepTox, Weave, Attentive FP, and
CMPNN are multitask models. For each assay prediction task,
randomly select _N_ pos + _N_ neg samples as the support set and
128 samples as the query set. Repeat this process 20 times,
and calculate the final average value as the model result. MAT,
PreGNN, CDDD, MolPMoFiT, and N-Gram, is a pretrained
GNN model that uses self-supervised learning on a large-scale
molecular corpus, resulting in better parameter initialization.
Similarly, 128 samples were randomly collected for testing and
repeated 20 times to avoid the randomness of model testing.


III. R ESULTS AND D ISCUSSION


_A. Tox21_


The “Toxicology in the 21st Century” (Tox21), collected
by the 2014 Tox21 Data Challenge, is a public database
containing 12 assays measures the toxicity of biological target.
We treat each assay as a single task. The first nine assays
were used for training, including NR-AR, NR-AR-LBD,
NR-AHR, NR-Aromatase, NR-ER, NR-ER-LBD, NR-PPARGamma, SR-ARE, and SR-ATAD5. The last three assays were
used for testing, including SR-HSE, SR-MMP, and SR-P53.
Meta-GAT is compared with other 12 models, and the
experimental results are shown in Fig. 3. The numbers of
positive and negative samples in the support set are both
increased from 1 to 10, and the improvement of model
performance is not obvious. However, the change in the ratio
of positive and negative samples has great influence on the
model performance. Interestingly, when there are only a few
examples, the balanced ratio of positive and negative samples
in the support sets may be more important than the increase in
the number. To some extent, the ratio of positive and negative
samples in the support set represents the distribution of task.
A balanced ratio of positive and negative samples may better
guide the model to search for the parameters of the optimal



Fig. 4. Performance comparison of Meta-GAT with other representative
molecular models.


hypothesis, making the Meta-GAT model easier to learn the
meta knowledge of binary classification. Meta-GAT has shown
impressive performance in toxicity assay tasks with few data.
We used Kappa and paired Wilcoxon Test to conduct consistency checks on the three test tasks of Tox21, and the results
are shown in Table III. Kappa analysis is used to evaluate
the consistency degree between predicted results of MetaGAT and actual measured results. The paired Wilcoxon Test
in nonparametric test is used to test whether the distribution
of the predictions (independent samples) produced by the two
models is equal. It is not limited by the data distribution, the
test conditions are relatively loose, and it can be applied to the
overall unknown samples. Wilcoxon _p-value <_ 0 _._ 05 indicated
that the distribution of Meta-GAT predicted results was different from that of other models. The results of Kappa analysis
show that SR-HSE and real measurement results are highly
consistent within the allowable error range, and SR-MMP
and SR-p53 are extremely consistent. These statistical tests
indicate that the prediction results of Meta-GAT can replace
real measurements within the allowable error range.
In addition, Fig. 4 (left) also shows the performance comparison of Meta-GAT with other representative molecular
models. We observe that Meta-GAT still achieves state-of-the
art (SOTA) performance compared with fully supervised models. Self-supervised models (CDDD, N-Gram, MolPMoFiT,
and Mol2Context-vec) pretrain models from unlabeled large
datasets and then fine-tune models on specific target datasets.
Due to its powerful feature transfer capability, its model
outperforms multitask models that only use the target dataset.



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:27:22 UTC from IEEE Xplore. Restrictions apply.


LV et al.: META LEARNING WITH GRAPH ATTENTION NETWORKS FOR LOW-DATA DRUG DISCOVERY 11225


TABLE IV


S CORES FOR C ONSISTENCY C HECKS ON THE SIDER D ATASET U SING

K APPA AND P AIRED W ILCOXON T ESTS


Fig. 5. ROC-AUC scores of Meta-GAT and previous models in the SIDER
few examples prediction task. 1+/5− represent the number of positive and
negative examples is 1 and 5, respectively.



The graph attention network introduces an attention mechanism by assigning different weights to different nodes, and
the corresponding graph convolution operation aggregates the
weighted sum of the atoms local information together, which
can force the model to learn the most meaningful neighbors
and local environments part. Compared with other common
GCN architectures, graph attention networks (GAT, Attentive
FP, and TrimNet) tend to achieve better performance. Overall,
Meta-GAT shows a powerful improvement over the existing
baseline model, indicating that meta learning method may be
a better solution for low-data drug discovery.


_B. SIDER_


The SIDER is a public database containing 1427 marketed
drugs and their adverse drug reactions [65]. According to
the MedDRA classifications, drug side effects are grouped
into 27 systemic organ classes. Among them, “renal and
urinary disorders” (RUD), “pregnancy, puerperium and perinatal conditions” (PPPC), “ear and labyrinth disorders” (ELD),
“cardiac disorders” (CD), “nervous system disorders” (NSD),
and “injury, poisoning and procedural complications” (IPPC),
six indications were used for testing, and the remaining
21 indications were used for training.
The performance comparison of Meta-GAT with other few
examples methods is shown in Fig. 5. The meta learning model
still shows a strong improvement in this set, demonstrating
the potential of meta learning in few examples molecular
property prediction tasks. As is shown before, a balanced ratio
of positive and negative samples helps to further improve
performance. The graph attention mechanism introduced in
Meta-GAT can focus on task-related information from the

neighborhood, which helps to achieve accurate iteration of
few examples. It can be observed that the GNN based on
meta learning (EGNN, Meta-MGNN, and Meta-GAT) obtains
more advanced model performance than other meta learning
methods (Siamese, MAML, AttnLSTM, and IterRefLSTM).
Fig. 4 (right) shows that Meta-GAT achieves SOTA performance compared with fully supervised representative models.
In addition, the distribution of Meta-GAT predicted results was



Fig. 6. ROC-AUC Scores of Meta-GAT and previous models in the MUV
few examples prediction task. 1+/5− represent the number of positive and
negative examples is 1 and 5, respectively.


different from that of other models (Wilcoxon _p-value <_ 0 _._ 05)
for the six indications in the SIDER dataset (see Table IV).
PPPC, ELD, NSD, IPPC, and real measurement results are
highly consistent within the allowable error range, while RUD
and CD are extremely consistent.


_C. MUV_


The maximum unbiased validation (MUV) dataset contains 17 binary classification tasks for more than 90 000
molecules and is specifically designed to be challenging for
standard virtual screening [51], [66]. The positives examples
are selected to be structurally distinct from one another. MUV
is a best-case scenario for baseline machine learning (since
each data point is maximally informative) and a worst case
test for the low-data methods, since structural similarity cannot
be effectively exploited to predict behavior of new active
molecules [43].
The first 12 assays were used for training. The five assays,
MUV-832, MUV-846, MUV-852, MUV-858, and MUV-859,
were used for model test. Fig. 6 reports the overall performance of all methods on the MUV dataset. Experimental
results show that Meta-GAT outperforms other baseline models. In terms of average improvement, for one-shot learning,
the average improvements are +0.72%. The value equals
0.39% for five-shot learning. Both Meta-MGNN and PreGNN
provide considerable performance, with an average ROCAUC of 0.6451 and 0.6554 on the test task set, respectively, which are slightly worse than that of Meta-GAT
(ROC-AUC = 0.6626). Note that Meta-MGNN and PreGNN



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:27:22 UTC from IEEE Xplore. Restrictions apply.


11226 IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 35, NO. 8, AUGUST 2024


TABLE V


S CORES FOR C ONSISTENCY C HECKS ON THE MUV D ATASET U SING

K APPA AND P AIRED W ILCOXON T ESTS


Fig. 7. High agreement between the predicted results of the Meta-GAT
(Our) and GT assessed by the Bland–Altman analysis. The _x_ -axis and _y_ -axis
represent the average values and the bias between the GT and the predicted
values by the Meta-GAT, respectively. The blue dashed line indicates the mean
bias. The red dashed lines indicate the 95% confidence intervals of the bias.


TABLE VI


C OMPARISON OF P REDICTIVE P ERFORMANCES (MAE) ON THE QM9
D ATASET Q UANTUM P ROPERTIES . N OTE T HAT FOR MAE, L OWER
V ALUE I NDICATES B ETTER P ERFORMANCE


Fig. 8. Comparison of distribution differences between Tox21 and SIDER
using kernel density estimation.



require a large-scale molecular corpus and additional selfsupervised model parameters. Furthermore, we observe that
IterRefLSTM and MAML baseline methods do not have stable

performance on different tasks. In other words, they may
perform well on Tox21 or SIDER, but perform poorly on the
MUV task. In contrast, the performance of Meta-GAT on all
three classification datasets is SOTA and stable. In addition,
Wilcoxon _p-value <_ 0 _._ 05 in Table V indicated that the
distribution of Meta-GAT predicted results is different from
that of other models on the five assays of the MUV dataset.
Kappa analysis results in the last row of Table V show that the
predicted results and real measurement results are moderately
consistent within the allowable error range.


_D. QM9_


Due to the huge computational cost of density functional
theories approaches, there has been considerable interest in
applying machine learning models to task of molecular quantum property prediction. QM9 is a comprehensive dataset
that provides quantum mechanical properties, which include
12 calculated quantum properties for 134k stable small organic
molecules composed of up to nine heavy atoms.
The three quantum properties of LUMO, G, and Cv were
used as test tasks, and the other nine quantum properties were
used for training tasks. QM9 is a regression dataset, and mean
absolute error (MAE) is used to evaluate the performance
of regression models. As shown in Table VI, Meta-GAT
outcompetes other models on two out of three testing tasks
in the QM9 datasets. Two pretrained-based models (N-Gram



and Mol2Context-vec) provided more promising predictions
than other meta learning models. Meta-GAT shows noticeable
improvement in low-data drug discovery and is a promising
meta learning method.
Moreover, Fig. 7 illustrates that the predicted results of the
Meta-GAT are highly agreed with the ground truth (GT). Each
subplot shows the results of Bland–Altman analysis for the
three test task sets in QM9. The _x_ -axis and _y_ -axis represent the
average values and the bias between the GT and the predicted
values by the Meta-GAT, respectively. The blue dashed line
indicates the mean bias. The red dashed lines indicate the

95% confidence intervals of the bias. The results show that
the mean bias of the three test task sets is −0.0516, 0.1817,
and 0.1384, respectively. The percentages of the scatter points
falling within the 95% confidence interval are greater than
98%. Therefore, the results show the high agreement between
the GT and the predicted results by the Meta-GAT. In other
words, the prediction results of Meta-GAT can replace the GT
measured by experiment within the allowable error range.


_E. Transfer Learning to SIDER From Tox21_


The experiments, thus far, have demonstrated that MetaGAT is able to learn an efficient learning process to transfer
meta knowledge from a range of training tasks, allowing the
model to rapidly adapt to closely related molecular property predictions with few examples. Transferability and taskrelatedness issues need to be carefully evaluated in real-world
use cases for drug discovery to determine whether transfer



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:27:22 UTC from IEEE Xplore. Restrictions apply.


LV et al.: META LEARNING WITH GRAPH ATTENTION NETWORKS FOR LOW-DATA DRUG DISCOVERY 11227


TABLE VII


ROC-AUC S CORES OF M ODELS T RAINED ON T OX 21 T ESTED ON SIDER


Fig. 9. Visualizations of molecular embeddings generated by Meta-GAT on
each internal iteration during the test phase. (a)–(f) Number of iterations 0–5
using the support set.



learning can be used. We train the model on the Tox21
dataset and fine-tune on ten samples taken from the test task
on the SIDER dataset, and then evaluate on the remaining
samples. There is a large domain transfer in two datasets. The
Tox21 measures the results of nuclear receptor assays, and
the SIDER measures adverse effects from real patients. This
problem becomes so challenging that even domain experts may
not be able to accurately judge it.
We only use ten labeled data to transfer on one or more
SIDER test tasks. The evaluation results in Table VII show

that neither meta learning nor multitask models achieve generalization between unrelated tasks. The performance of these
methods using knowledge transferred from Tox21 to SIDER
is inferior to that of molecular models trained only on SIDER.
Clearly, how to quantify the correlation between different tasks
is important for transfer learning in drug discovery.
Kernel density estimation is used to estimate unknown
density functions in probability theory. It does not attach
any assumption to the data distribution and is a method to
study the characteristics of the data distribution from the
data sample itself. Fig. 8 shows the distribution differences
of NR-AR-LBD, NR-AHR, SR-HSE, SR-MMP in Tox21,
and eye disorders and product issues in SIDER using kernel
density estimation. There was a strong correlation between
the four Tox21 assays, so the training task NR-AR-LBD and
NR-AHR could be transferred to the test tasks SR-HSE and

SR-MMP. Due to the large distribution difference between
Tox21 and SIDER, it may lead to negative transfer, overfitting problems in the case of data scarcity, thus failing to
obtain meaningful molecular models. Identifying and addressing these possible limitations are research directions for our
future work. Furthermore, kernel density estimation may be
a method for assessing transferability in the field of drug
discovery, which can measure the distributional differences
between source and target tasks, thus revealing task relatedness. We hope our work can promote low-data drug discovery
tasks.


_F. Feature Visualization and Interpretation for Meta-GAT_


The interpretability of the model is crucial, and reducing
the gap between the visualization of the model and the



chemical intuition of human understanding is conducive to
the application of meta learning in drug discovery. When
predicting new task of molecular properties, our model uses a
few examples in the new task support set to perform several
internal iterations, and then evaluates the performance of the
query set in the new task. This raises an obvious question: can
learning just a few molecules build a competitive classifier?
Taking the test dataset SR-HSE in the Tox21 toxicity
prediction task as an example, we explored the performance
of the molecular embedding representation generated by the
Meta-GAT mode. Specifically, the high-dimensional feature
generated by Meta-GAT is a 200-D embedding, similar to
the fingerprint vector representation of a molecule. We reduce
the high-dimensional vector to 2-D embedding space by
t-distributed neighbor embedding (T-SNE) [67] to observe
the distribution of molecular representations of different categories. Fig. 9 shows the distribution visualization of molecular

–
embeddings in the SR-HSE dataset, and Fig. 9(a) (f) represents the number of iterations (0–5) using the support set. The
blue dots and red dots represent the numerators of positive and
negative examples, respectively.
During model training, an initialization parameter that is
approximately optimal for multiple toxicity prediction tasks
has been searched. Before using the support set iteration in a
new task of SR-HSE toxicity prediction [see Fig. 9(a)], the
molecular representation had some degree of separation in
2-D mapping space. But, the model could not clearly classify
positive and negative samples, and the blue dots and red
dots were still mixed together. After one iteration using the
support set [see Fig. 9(b)], the mixing degree of blue dots
and red dots weakened and showed aggregation phenomenon
to some extent. It shows that Meta-GAT iterates well in the

right direction under the guidance of meta knowledge through
feature analysis of limited data in the new task. Continue 1–2
iterates, it has been clearly observed that the model can better
distinguish between blue and red dots in Fig. 9(c) and (d).
The blue dots are mostly gathered in the bottom left corner
of the space, and the red dots are mostly in the top-right corner.
The model has reached the best performance on the new task.
After 4–5 iterations using the same support set, the model
may have overfitting. Therefore, we have to set an early stop
to make the iterative process terminate early.



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:27:22 UTC from IEEE Xplore. Restrictions apply.


11228 IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 35, NO. 8, AUGUST 2024


Fig. 11 shows that the attention weights learned from the
Meta-GAT model are used to highlight each atom in nine
molecules on Tox21 datasets. Meta-GAT model may pay
more attention to atomic groups that may cause toxicity, such
as sulfonic acid or aniline. The sulfonic acid has potential
hazards, including eye burns, skin burns, digestive tract burns
if swallowed, and respiratory tract burns if inhaled. Aniline
leakage may cause combustion, and explosion hazards, and it
is very toxic to the blood and nerves, can be poisoned by the
skin or the respiratory tract absorption. These observations
suggest that Meta-GAT has indeed successfully extracted
relevant information by learning from a specific task, and
the attention weight at the atom level indeed has chemical
implications. For more intricate problems, attention weight
may also be taken as hints for discovering new knowledge.


IV. C ONCLUSION



Fig. 10. Heatmap of atomic similarity matrices for six molecules.


Fig. 11. Attention weights learned from the Meta-GAT are used to highlight
each atom in nine molecules in the toxicity prediction task on Tox21 datasets.


In addition, we conducted two visualization experiments on
the atom similarity matrix and attention weights to rationalize
Meta-GAT. We obtained the similarity coefficient between
atom pairs by calculating the Pearson correlation coefficient
for those feature vectors and plotted the heatmap of the atomic
similarity matrices for the six molecules, as shown in Fig. 10.
Taking the molecule structure of Dipyrone as an example, the
atoms in Dipyrone are clearly separated into three clusters,
as follows: a benzene (atoms 0–5), an aminomethanesulfonic
acid (atoms 6–13), and a pyrazolidone (atoms 14–20). The
first impression of the visual pattern in the heat map for the
compound iodoantipyrine may show some degree of chaos,
which is caused by the disorder of the atom numbers in
SMILES. Combining atoms 0–6, atom N13, and atom C14
of iodoantipyrine, the atoms in iodoantipyrine are clearly
divided into two clusters. The visual pattern of these heat
maps strongly agrees with our chemical intuition regarding
these molecular structure.



Drug discovery is the process of discovering new molecules
properties and identifying the useful molecules as new drugs
after optimization. In the initial stage of optimization of
candidate molecules, due to low solubility or possible toxicity,
new molecules or analog molecules do not have many records
of real physicochemical properties and biological activities.
Therefore, the key problem of AI-assisted drug discovery is
few examples learning. Here, we propose a meta learning
method based on graph attention network, Meta-GAT, which
uses graph attention network to extract the interaction of
atom pairs and the edge features of bonds in molecules.
Also, the meta learning algorithm trains a well-initialized
parameter through multiple prediction tasks and, on this basis,
performs one or more steps of gradient adjustment to achieve
the purpose of quickly adapting to a new task with only
few data. Meta-GAT achieves SOTA performance on multiple
public benchmark datasets, indicating that it can adapt to new
tasks faster than other models. This algorithm is expected
to fundamentally solve the problem of few samples in drug
discovery. We have proved that Meta-GAT can provide a powerful impetus for low-data drug discovery. The development
of meta learning is an important direction of AI-assisted drug
discovery. It is believed that the new learning paradigm can
be applied in the field of drug discovery in the future.


A CKNOWLEDGMENT


The authors would like to thank the anonymous reviewers
for their valuable suggestions.


R EFERENCES


[1] H. Dowden and J. Munro, “Trends in clinical success rates and therapeutic focus,” _Nature Rev. Drug Discovery_, vol. 18, no. 7, pp. 495–497,
2019.

[2] L. Wang et al., “Accurate and reliable prediction of relative ligand
binding potency in prospective drug discovery by way of a modern
free-energy calculation protocol and force field,” _J. Amer. Chem. Soc._,
vol. 137, no. 7, pp. 2695–2703, Feb. 2015.

[3] G. Sliwoski, S. Kothiwale, J. Meiler, and E. W. Lowe, “Computational
methods in drug discovery,” _Pharmacological Rev._, vol. 66, no. 1,
pp. 334–395, 2014.

[4] Z. Yang, W. Zhong, L. Zhao, and C. Y.-C. Chen, “ML-DTI: Mutual
learning mechanism for interpretable drug–target interaction prediction,”
_J. Phys. Chem. Lett._, vol. 12, no. 17, pp. 4247–4261, 2021.



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:27:22 UTC from IEEE Xplore. Restrictions apply.


LV et al.: META LEARNING WITH GRAPH ATTENTION NETWORKS FOR LOW-DATA DRUG DISCOVERY 11229




[5] J.-Q. Chen, H.-Y. Chen, W.-J. Dai, Q.-J. Lv, and C. Y.-C. Chen, “Artificial intelligence approach to find lead compounds for treating tumors,”
_J. Phys. Chem. Lett._, vol. 10, no. 15, pp. 4382–4400, Aug. 2019.

[6] J.-Y. Li, H.-Y. Chen, W.-J. Dai, Q.-J. Lv, and C. Y.-C. Chen, “Artificial
intelligence approach to investigate the longevity drug,” _J. Phys. Chem._
_Lett._, vol. 10, no. 17, pp. 4947–4961, Sep. 2019.

[7] C. Y. Lee and Y.-P.-P. Chen, “New insights into drug repurposing for
COVID-19 using deep learning,” _IEEE Trans. Neural Netw. Learn. Syst._,
vol. 32, no. 11, pp. 4770–4780, Nov. 2021.

[8] M. J. Waring et al., “An analysis of the attrition of drug candidates from
four major pharmaceutical companies,” _Nature Rev. Drug Discovery_,
vol. 14, no. 7, pp. 475–486, Jul. 2015.

[9] J. Wenzel, H. Matter, and F. Schmidt, “Predictive multitask deep neural
network models for ADME-Tox properties: Learning from large data
sets,” _J. Chem. Inf. Model._, vol. 59, no. 3, pp. 1253–1268, Mar. 2019.

[10] J. Ma, R. P. Sheridan, A. Liaw, G. E. Dahl, and V. Svetnik, “Deep
neural nets as a method for quantitative structure–activity relationships,”
_J. Chem. Inf. Model._, vol. 55, no. 2, pp. 263–274, 2015.

[11] R. S. Simões, V. G. Maltarollo, P. R. Oliveira, and K. M. Honorio,
“Transfer and multi-task learning in QSAR modeling: Advances and
challenges,” _Frontiers Pharmacol._, vol. 9, p. 74, Feb. 2018.

[12] C. Li et al., “Geometry-based molecular generation with deep constrained variational autoencoder,” _IEEE Trans. Neural Netw. Learn. Syst._,
[early access, 2022, doi: 10.1109/TNNLS.2022.3147790.](http://dx.doi.org/10.1109/TNNLS.2022.3147790)

[13] C. Ji, Y. Zheng, R. Wang, Y. Cai, and H. Wu, “Graph polish: A
novel graph generation paradigm for molecular optimization,” _IEEE_
_Trans. Neural Netw. Learn. Syst._, early access, Sep. 14, 2021, doi:
[10.1109/TNNLS.2021.3106392.](http://dx.doi.org/10.1109/TNNLS.2021.3106392)

[14] P. Schneider et al., “Rethinking drug design in the artificial intelligence era,” _Nature Rev. Drug Discovery_, vol. 19, no. 5, pp. 353–364,
May 2020.

[15] X. Jing and J. Xu, “Fast and effective protein model refinement using
deep graph neural networks,” _Nature Comput. Sci._, vol. 1, no. 7,
pp. 462–469, Jul. 2021.

[16] Z. Xiong et al., “Pushing the boundaries of molecular representation
for drug discovery with the graph attention mechanism,” _J. Medicinal_
_Chem._, vol. 63, no. 16, pp. 8749–8760, Aug. 2019.

[17] Q. Lv, G. Chen, L. Zhao, W. Zhong, and C. Yu-Chian Chen,
“Mol2Context-vec: Learning molecular representation from context
awareness for drug discovery,” _Briefings Bioinf._, vol. 22, no. 6,
Nov. 2021, Art. no. bbab317.

[18] L. A. Bugnon, C. Yones, D. H. Milone, and G. Stegmayer, “Deep
neural architectures for highly imbalanced data in bioinformatics,”
_IEEE Trans. Neural Netw. Learn. Syst._, vol. 31, no. 8, pp. 2857–2867,
Aug. 2020.

[19] J. Song et al., “Local–global memory neural network for medication
prediction,” _IEEE Trans. Neural Netw. Learn. Syst._, vol. 32, no. 4,
pp. 1723–1736, Apr. 2021.

[20] R. Huang, X. Tan, and Q. Xu, “Learning to learn variational quantum algorithm,” _IEEE Trans. Neural Netw. Learn. Syst._, early access,
[Feb. 28, 2022, doi: 10.1109/TNNLS.2022.3151127.](http://dx.doi.org/10.1109/TNNLS.2022.3151127)

[21] Y. Yamanishi, E. Pauwels, and M. Kotera, “Drug side-effect prediction
based on the integration of chemical and biological spaces,” _J. Chem._
_Inf. Model._, vol. 52, no. 12, pp. 3284–3292, Dec. 2012.

[22] Á. Duffy et al., “Tissue-specific genetic features inform prediction of
drug side effects in clinical trials,” _Sci. Adv._, vol. 6, no. 37, Sep. 2020,
Art. no. eabb6242.

[23] G. Yu, Y. Xing, J. Wang, C. Domeniconi, and X. Zhang, “Multiview
multi-instance multilabel active learning,” _IEEE Trans. Neural Netw._
_Learn. Syst._, vol. 33, no. 9, pp. 4311–4321, Sep. 2022.

[24] A. Morro et al., “A stochastic spiking neural network for virtual
screening,” _IEEE Trans. Neural Netw. Learn. Syst._, vol. 29, no. 4,
pp. 1371–1375, Apr. 2018.

[25] Y. Yu and H. Tran, “An XGBoost-based fitted Q iteration for
finding the optimal STI strategies for HIV patients,” _IEEE Trans._
_Neural_ _Netw._ _Learn._ _Syst._, early access, Jun. 2, 2022, doi:
[10.1109/TNNLS.2022.3176204.](http://dx.doi.org/10.1109/TNNLS.2022.3176204)

[26] K. V. Chuang, L. M. Gunsalus, and M. J. Keiser, “Learning molecular
representations for medicinal chemistry: Miniperspective,” _J. Medicinal_
_Chem._, vol. 63, no. 16, pp. 8705–8722, Aug. 2020.

[27] M. Sun, S. Zhao, C. Gilvary, O. Elemento, J. Zhou, and F. Wang,
“Graph convolutional networks for computational drug development and discovery,” _Briefings Bioinf._, vol. 21, no. 3, pp. 919–935,
May 2020.




[28] D. Duvenaud et al., “Convolutional networks on graphs for learning
molecular fingerprints,” in _Proc. Adv. Neural Inf. Process. Syst., Annu._
_Conf. Neural Inf. Process. Syst._ Montreal, QC, Canada: Curran Associates, Inc., Dec. 2015, pp. 2224–2232.

[29] P. Li et al., “TrimNet: Learning molecular representation from triplet
messages for biomedicine,” _Briefings Bioinf._, vol. 22, no. 4, Jul. 2021,
Art. no. bbaa266.

[30] Q.-J. Lv et al., “A multi-task group bi-LSTM networks application on
electrocardiogram classification,” _IEEE J. Transl. Eng. Health Med._,
vol. 8, pp. 1–11, 2020.

[31] C. Cai et al., “Transfer learning for drug discovery,” _J. Medicinal Chem._,
vol. 63, no. 16, pp. 8683–8694, 2020.

[32] S. Guo, L. Xu, C. Feng, H. Xiong, Z. Gao, and H. Zhang, “Multilevel semantic adaptation for few-shot segmentation on cardiac image
sequences,” _Med. Image Anal._, vol. 73, Oct. 2021, Art. no. 102170.

[33] M. Huisman, J. N. Van Rijn, and A. Plaat, “A survey of deep metalearning,” _Artif. Intell. Rev._, vol. 54, pp. 1–59, Aug. 2021.

[34] A. Banino et al., “Vector-based navigation using grid-like representations
in artificial agents,” _Nature_, vol. 557, no. 7705, pp. 429–433, May 2018.

[35] T. Hospedales, A. Antoniou, P. Micaelli, and A. Storkey, “Meta-learning
in neural networks: A survey,” 2020, _arXiv:2004.05439_ .

[36] J. Vanschoren, “Meta-learning: A survey,” 2018, _arXiv:1810.03548_ .

[37] J. X. Wang et al., “Prefrontal cortex as a meta-reinforcement learning
system,” _Nature Neurosci._, vol. 21, no. 6, pp. 860–868, May 2018.

[38] S. Biswas, G. Khimulya, E. C. Alley, K. M. Esvelt, and G. M. Church,
“Low- _N_ protein engineering with data-efficient deep learning,” _Nature_
_Methods_, vol. 18, no. 4, pp. 389–396, Apr. 2021.

[39] R. Liu, X. Yu, X. Liu, D. Xu, K. Aihara, and L. Chen, “Identifying
critical transitions of complex diseases based on a single sample,”
_Bioinformatics_, vol. 30, no. 11, pp. 1579–1586, Jun. 2014.

[40] S. Lin et al., “Prototypical graph contrastive learning,” _IEEE_
_Trans. Neural Netw. Learn. Syst._, early access, Jul. 27, 2022, doi:
[10.1109/TNNLS.2022.3191086.](http://dx.doi.org/10.1109/TNNLS.2022.3191086)

[41] Y. Sun et al., “Combining genomic and network characteristics for
extended capability in predicting synergistic drugs for cancer,” _Nature_
_Commun._, vol. 6, no. 1, pp. 1–10, Sep. 2015.

[42] Q. Liu, H. Zhou, L. Liu, X. Chen, R. Zhu, and Z. Cao, “Multi-target
QSAR modelling in the analysis and design of HIV-HCV co-inhibitors:
An in-silico study,” _BMC Bioinf._, vol. 12, no. 1, pp. 1–20, Dec. 2011.

[43] H. Altae-Tran, B. Ramsundar, A. S. Pappu, and V. Pande, “Low data
drug discovery with one-shot learning,” _ACS Central Sci._, vol. 3, no. 4,
pp. 283–293, 2017.

[44] T. Adler et al., “Cross-domain few-shot learning by representation
fusion,” 2020, _arXiv:2010.06498_ .

[45] Z. Guo et al., “Few-shot graph learning for molecular property prediction,” in _Proc. Web Conf._, J. Leskovec, M. Grobelnik, M. Najork, J. Tang,
and L. Zia, Eds., Ljubljana, Slovenia, Apr. 2021, pp. 2559–2567.

[46] Y. Wang, Q. Yao, J. T. Kwok, and L. M. Ni, “Generalizing from a few
examples: A survey on few-shot learning,” _ACM Comput. Surv._, vol. 53,
no. 3, pp. 1–34, 2020.

[47] C. Finn, P. Abbeel, and S. Levine, “Model-agnostic meta-learning for
fast adaptation of deep networks,” in _Proc. 34th Int. Conf. Mach. Learn._,
vol. 70, Sydney, NSW, Australia, Aug. 2017, pp. 1126–1135.

[48] D. Jiang et al., “Could graph neural networks learn better molecular
representation for drug discovery? A comparison study of descriptorbased and graph-based models,” _J. Cheminformatics_, vol. 13, no. 1,
pp. 1–23, Feb. 2021.

[49] R. Winter, F. Montanari, F. Noé, and D.-A. Clevert, “Learning continuous and data-driven molecular descriptors by translating equivalent
chemical representations,” _Chem. Sci._, vol. 10, no. 6, pp. 1692–1701,
Jul. 2019.

[50] J. Cui, B. Yang, B. Sun, X. Hu, and J. Liu, “Scalable and parallel deep
Bayesian optimization on attributed graphs,” _IEEE Trans. Neural Netw._
_Learn. Syst._, vol. 33, no. 1, pp. 103–116, Jan. 2020.

[51] Z. Wu et al., “MoleculeNet: A benchmark for molecular machine
learning,” _Chem. Sci._, vol. 9, no. 2, pp. 513–530, 2018.

[52] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,”
2014, _arXiv:1412.6980_ .

[53] F. Fabris, A. Doherty, D. Palmer, J. P. De Magalhães, and A. A. Freitas,
“A new approach for interpreting random forest models and its application to the biology of ageing,” _Bioinformatics_, vol. 34, no. 14,
pp. 2449–2456, Jul. 2018.

[54] T. N. Kipf and M. Welling, “Semi-supervised classification with graph
convolutional networks,” in _Proc. 5th Int. Conf. Learn. Represent._
_(ICLR)_, Toulon, France, Apr. 2017, pp. 1–14.



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:27:22 UTC from IEEE Xplore. Restrictions apply.


11230 IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, VOL. 35, NO. 8, AUGUST 2024




[55] G. Koch et al., “Siamese neural networks for one-shot image recognition,” in _Proc. ICML Deep Learn. Workshop_, vol. 2, Lille, France, 2015,
pp. 1–30.

[56] J. Kim, T. Kim, S. Kim, and C. D. Yoo, “Edge-labeling graph neural
network for few-shot learning,” in _Proc. IEEE/CVF Conf. Comput. Vis._
_Pattern Recognit. (CVPR)_ . Long Beach, CA, USA: Computer Vision
Foundation, Jun. 2019, pp. 11–20.

[57] W. Hu et al., “Strategies for pre-training graph neural networks,” in
_Proc. 8th Int. Conf. Learn. Represent. (ICLR)_, Addis Ababa, Ethiopia,
Apr. 2020, pp. 1–22.

[58] J. Snell, K. Swersky, and R. Zemel, “Prototypical networks for fewshot learning,” in _Proc. Adv. Neural Inf. Process. Syst._, vol. 30, 2017,
pp. 1–11.

[59] Y. Song, S. Zheng, Z. Niu, Z.-H. Fu, Y. Lu, and Y. Yang, “Communicative representation learning on attributed molecular graphs,” in
_Proc. 29th Int. Joint Conf. Artif. Intell._, C. Bessiere, Ed., Jul. 2020,
[pp. 2831–2838, doi: 10.24963/IJCAI.2020/392.](http://dx.doi.org/10.24963/IJCAI.2020/392)

[60] S. Kearnes, K. McCloskey, M. Berndl, V. Pande, and P. Riley, “Molecular graph convolutions: Moving beyond fingerprints,” _J. Comput.-Aided_
_Mol. Des._, vol. 30, no. 8, pp. 595–608, Aug. 2016.

[61] A. Mayr, G. Klambauer, T. Unterthiner, and S. Hochreiter, “DeepTox:
Toxicity prediction using deep learning,” _Frontiers Environ. Sci._, vol. 3,
p. 80, Feb. 2016.

[62] L. Maziarka, T. Danel, S. Mucha, K. Rataj, J. Tabor, and S. Jastrzebski,
“Molecule attention transformer,” 2020, _arXiv:2002.08264_ .

[63] X. Li and D. Fourches, “Inductive transfer learning for molecular activity
prediction: Next-gen QSAR models with MolPMoFiT,” _J. Cheminfor-_
_matics_, vol. 12, no. 1, pp. 1–15, Dec. 2020.

[64] S. Liu, M. F. Demirel, and Y. Liang, “N-gram graph: Simple unsupervised representation for graphs, with applications to molecules,” in _Proc._
_Adv. Neural Inf. Process. Syst._, vol. 32, 2019, pp. 8464–8476.

[65] M. Kuhn, I. Letunic, L. J. Jensen, and P. Bork, “The SIDER database
of drugs and side effects,” _Nucleic Acids Res._, vol. 44, no. D1,
pp. D1075–D1079, Jan. 2016.

[66] S. G. Rohrer and K. Baumann, “Maximum unbiased validation (MUV)
data sets for virtual screening based on PubChem bioactivity data,”
_J. Chem. Inf. Model._, vol. 49, no. 2, pp. 169–184, Feb. 2009.

[67] L. Van Der Maaten and G. Hinton, “Visualizing data using t-SNE,”
_J. Mach. Learn. Res._, vol. 9, pp. 2579–2605, Nov. 2008.


**Qiujie Lv** is currently pursuing the Ph.D. degree
with the Artificial Intelligence Medical Research
Center, School of Intelligent Systems Engineering, Shenzhen Campus of Sun Yat-sen University,
Shenzhen, Guangdong, China.
His research interests include graph neural network, drug discovery, artificial intelligence, and
bioinformatics.



**Guanxing Chen** is currently pursuing the Ph.D.
degree with the Artificial Intelligence Medical
Research Center, School of Intelligent Systems Engineering, Shenzhen Campus of Sun Yat-sen University, Shenzhen, Guangdong, China.
His research interests include explainable artificial
intelligence, drug discovery, deep learning, biosynthesis, and vaccine design.


**Ziduo Yang** is currently pursuing the Ph.D. degree
with the Artificial Intelligence Medical Research
Center, School of Intelligent Systems Engineering, Shenzhen Campus of Sun Yat-sen University,
Shenzhen, Guangdong, China.
His main research interests include explainable
graph neural network, computer vision, reinforcement learning, and chemoinformatics.


**Weihe Zhong** is currently pursuing the Ph.D. degree
with the Artificial Intelligence Medical Research
Center, School of Intelligent Systems Engineering, Shenzhen Campus of Sun Yat-sen University,
Shenzhen, Guangdong, China.
His main research interests include graph neural
network, chemoinformatics, and drug discovery.


**Calvin Yu-Chian Chen** is currently the Director of the Artificial Intelligent Medical Center
and a Professor with the School of Intelligent
Systems Engineering, Shenzhen Campus of Sun
Yat-sen University, Shenzhen, Guangdong, China.
He also serves as an Advisor at China Medical
University Hospital, Taichung, China, and Asia University, Taichung, and a Guest Professor at the
Massachusetts Institute of Technology (MIT), Cambridge, MA, USA, and the University of Pittsburgh,
Pittsburgh, PA, USA. He has published more
than 300 SCI articles and with H-index more than 47. In 2020–2023, he is
the highly cited candidate (in the field of computer science and technology).
In 2021–2023, he was also selected as the world’s top 100 000 scientists.
In 2018–2023, he was also selected as the world’s top 2% scientists. He had
built several artificial intelligence medical systems for hospital, including
various pathological image processing, MRI image processing, and big data
modeling. He also built the world’s largest traditional Chinese medicine
database (http://TCMBank.cn/). His laboratory general research interests
include developing structured machine learning techniques for computer
vision tasks to investigate how to exploit the human commonsense and
incorporate them to develop the advanced artificial intelligence system.



Authorized licensed use limited to: Necmettin Erbakan Universitesi. Downloaded on August 06,2025 at 09:27:22 UTC from IEEE Xplore. Restrictions apply.


