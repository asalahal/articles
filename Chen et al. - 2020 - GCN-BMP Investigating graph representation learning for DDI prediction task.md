[Methods 179 (2020) 47–54](https://doi.org/10.1016/j.ymeth.2020.05.014)


[Contents lists available at ScienceDirect](http://www.sciencedirect.com/science/journal/10462023)

## Methods


[journal homepage: www.elsevier.com/locate/ymeth](https://www.elsevier.com/locate/ymeth)

### GCN-BMP: Investigating graph representation learning for DDI prediction task


Xin Chen [a], Xien Liu [a], Ji Wu [a][,][b][,][⁎]


a Department of Electronic Engineering, Tsinghua University, Beijing 100084, China
b Institute for Precision Medicine, Tsinghua University, Beijing 100084, China





A R T I C L E I N F O


Keywords:

DDI

Graph representation learning
Scalability

Robustness

Interpretability


1. Introduction



A B S T R A C T


One drug's pharmacological activity may be changed unexpectedly, owing to the concurrent administration of
another drug. It is likely to cause unexpected drug-drug interactions (DDIs). Several machine learning approaches have been proposed to predict the occurrence of DDIs. However, existing approaches are almost dependent heavily on various drug-related features, which may incur noisy inductive bias. To alleviate this problem, we investigate the utilization of the end-to-end graph representation learning for the DDI prediction task.
We establish a novel DDI prediction method named GCN-BMP (Graph Convolutional Network with Bond-aware
Message Propagation) to conduct an accurate prediction for DDIs. Our experiments on two real-world datasets
demonstrate that GCN-BMP can achieve higher performance compared to various baseline approaches.
Moreover, in the light of the self-contained attention mechanism in our GCN-BMP, we could find the most vital
local atoms that conform to domain knowledge with certain interpretability.



Drugs may interact with each other when concurrently administrated, which would lead to patients’ death or drug withdrawal. What’s
worse, recent findings unveil that most human diseases are caused by
complex biological processes which take on the resistance towards the
activity of any single drug [1]. A promising solution to cure diseases is
the co-prescription of multiple drugs, also termed as combinatorial
therapy. Although the application of combinatorial therapy can combat
many diseases, it would also increase the chance of drug-drug interactions (DDIs) [2]. It seems that DDI is inevitable even though the
patient is hugely self-discipline during the treatment. Hence, there is an
urgent need to identify DDIs effectively.
The machine learning methods can help scientists identify DDIs
effectively, without lengthy periods or high expense. To our best
knowledge, there exist many machine learning methods proposed for
this purpose. Though the existing approaches have achieved considerable performances, they heavily depend on a broad range of drug-related features which may be unavailable for most drugs. The dependence stems from the input molecule representation that existing
approaches have exploited – chemical fingerprint(s).
The chemical fingerprint is one of the conventional representation
methods for molecule data. The other two representations are SMILES
(Simplified Molecular-Input Line-Entry System) string [3] and



molecular graph, respectively. A specific kind of chemical fingerprint is
a pre-defined feature vector characterizing particular property of a
molecule, such as substructures, targets, signaling pathways, and so on.
It can be viewed as an abstract feature representation for molecules and
can be fed into machine learning models easily. Therefore, most conventional DDI prediction approaches used fingerprint(s) to represent
drug molecules [4–9]. However, a single kind of chemical fingerprint
can only depict the specific property of drug objects. It’s hard to determine whether the information contained in one particular fingerprint is relevant to the occurrence of DDIs. The incorporation of irrelevant information would incur noisy inductive bias. To overcome the
shortcomings, recent machine learning methods developed for DDI
prediction usually integrated a broad range of fingerprints to fulfill the
prediction [6–9]. However, the integration is not very ideal because
some important fingerprints are only available for a small subset of
drugs [8,10]. The unavailability of fingerprints lays a constraint on the
scale of datasets since the more drugs a dataset contains, the higher the
probability of missing fingerprints is. In other words, the models based
on the joint use of various fingerprints are lacking in scalability. Furthermore, the models mentioned above were almost based on the empirical assumption that chemically, biologically, or topologically similar drug molecules have higher chances of interacting with each
other. Therefore, the models built on top of this assumption are sensitive to pairwise similarity information. Incorrect pairwise similarity



⁎ Corresponding author at: Department of Electronic Engineering, Tsinghua University, Beijing 100084, China.
[E-mail addresses: chenx17@mails.tsinghua.edu.cn (X. Chen), xeliu@mail.tsinghua.edu.cn (X. Liu), wuji_ee@mail.tsinghua.edu.cn (J. Wu).](mailto:chenx17@mails.tsinghua.edu.cn)


[https://doi.org/10.1016/j.ymeth.2020.05.014](https://doi.org/10.1016/j.ymeth.2020.05.014)
Received 22 February 2020; Received in revised form 29 April 2020; Accepted 14 May 2020

Available online 03 July 2020
1046-2023/ © 2020 Published by Elsevier Inc.


X. Chen, et al. _Methods 179 (2020) 47–54_


Fig. 1. The overview of our proposed approach. The model adheres to the thought of end-to-end representation learning, and thus exploits only raw inputs of drug
pairs (i.e. pairwise SMILES representations), or rather, molecular graphs extracted from pairwise SMILES representation. The Siamese GCN functions as a generic
encoder transforming irregular-structured molecular data into real-valued embedding vectors. The following interaction predictor based on the HOLE-style neural
network predicts whether there will be an interaction between input drugs. The running example composed of Morphine and Naloxone are very similar structurally,
and our proposed model labels them as an interacting pair with a high confidence score 0.9831.



incurred by unrelated fingerprints would make the model give wrong
predictions. In other words, traditional models based on various fingerprints are at a disadvantage in terms of robustness. We will demonstrate the deficiency in scalability and robustness of conventional
DDI prediction approaches in the later experimental section.
Since the problem originates from the improper molecule representation, we consider changing the input molecule representation
for the DDI prediction model. As mentioned earlier, another two molecule representations are SMILES string [3] and molecular graph.
First, we analyze the feasibility of taking SMILES string [3] as input.
The SMILES string is a line notation to describe the structure of compounds using texts and a set of human-defined grammar rules. SMILES
can be analogous to text which is sequential and composite. By viewing
SMILES strings representing molecule objects as sequence data, typical
NLP-style approaches can be transferred directly to the field of molecular machine learning tasks. For example, there is an analogous algorithm to Word2vec [11] whose name is Mol2vec [12]. Similar to
Word2vec, Mol2vec can provide the pretrained representation for an
individual compound. It can be used as a fundamental component in the
molecular property prediction system [13]. Apart from Mol2vec, there
are other research works in cheminformatics inspired by NLP algorithms such as seq2seq [14] and cddd fingerprint [15] inspired by
seq2seq machine translation model [16]. Though taking SMILES string
as input can provide effective pretrained representation, it only views
the molecule data with sophisticated interior connectivity as sequences,
ignoring much valuable information. Also, it is hard to interpret.
Therefore, we claim that it is improper to represent molecules as
SMILES strings. We will verify our viewpoints in detail in the later
experimental section.
Then we analyze the possibility and rationality of representing
molecules as graph-structured objects. Firstly, this representation is
natural and rational because molecules are principally graph-structured
with atoms as nodes and bonds as edges. Secondly, graph objects such
as molecule graphs can be directly handled by graph neural networks
(GNN) [17–21]. Thirdly, GNNs can be trained in an end-to-end manner,
automatically obtaining task-related data-driven representations and
avoiding the incorporation of noisy prior bias. Last but not least, GNNs
can guarantee the smoothness of molecules’ hidden representations.
Namely, two molecules with similar structures or properties will be
embedded into near points in the latent space. Hence, representing
molecule data as graphs is an applicable option. The problem we need
to solve is how we design the GNN that is responsible for converting
molecular graphs into embeddings.
In this paper, we will propose the Graph Convolutional Network
with Bond-aware Message Propagation (GCN-BMP) model to conduct



the encoding of molecule graphs. Extensive experiments show that our
GCN-BMP model can achieve more excellent performance than the
various baseline approaches in real-world DDI datasets. Moreover, in
the light of self-contained attention mechanism in GCN-BMP, it could
find the most vital local atoms which are consistent with domain
knowledge with certain interpretability.


2. Materials and methods


2.1. Datasets


We want to prove the effectiveness of our proposed model by demonstrating that it can achieve better performance state-of-the-art approaches on two real-world DDI datasets. The task of the first DDI dataset is to predict the occurrence of DDIs. We refer to the dataset as
BinaryDDI. The task of another dataset is to predict the specific types
of DDIs. We named it as MultiDDI because the task is a multi-label
classification task.

# • [BinaryDDI][. BinaryDDI is released by][ [9]][. BinaryDDI dataset in-]

cludes 548 drugs, 48,548 pairwise DDIs as well as multiple types of
pairwise similarity information about these drug pairs. At the stage
of data preprocessing, we remove the molecules and associated similarity information if the SMILES strings of those molecules cannot
be converted into graph objects successfully by the RDKit tool.
# • [MultiDDI][. MultiDDI is contributed by][ [10]][. The dataset contains 86]

distinct interaction labels, and each drug is represented as a canonical SMILES string. Our data preprocessing procedure also removes
the data items that cannot be converted into graph objects by the
RDKit tool. The final dataset contains 1704 molecules and 191,400
interacting pairs.


2.2. GCN-BMP model


Our proposed DDI prediction method is an end-to-end graph representation learning model, composed of two sequential parts: the
former is a Siamese GNN which serves as a general molecule encoder
converting graph-structured molecule data to embeddings in the lowdimensional latent space, whereas the latter is an interaction predictor
responsible for predicting the occurrence or specific type of DDIs.
The overview of our model is illustrated in Fig. 1. The proposed
model takes canonical SMILES strings as input. The canonical SMILES
string is a typical line notation of molecules. We transform the pairwise
SMILES strings into a pair of molecular graphs via RDKit. With the help
of RDKit, we can extract two types of structural information from



48


X. Chen, et al. _Methods 179 (2020) 47–54_


̂


̃



molecular graphs. The two kinds of structural information are atom list
and multi-channel adjacency matrix, which will be fed into the following Siamese GNN. Setting the data-driven features of pairwise drugs
as inputs, we build a HOLE-based neural network to compute the interaction probability of the input drugs. In this section, we will describe
the detailed architecture of GNN, the interaction classifier, and the
training method.
Before that, we give the instructions on the symbols that we adopt in
this paper. Throughout this paper, we denote vectors by lowercase
boldface letters (e.g. **g** ∈ � _d_ ), matrices by uppercase boldface letters
(e.g. **A** ∈ � _m n_ × ), and scalars as non-boldface letters (e.g. _d_ _h_ for the dimension of node level hidden states, d for the dimension of graph level
hidden representations). Furthermore, a graph G is denoted as
G = (V E, where ) V is the set of nodes, and E is the set of edges. More ̂
specifically, we represent the i-th atom as _v_ _i_ ∈ V and the chemical
bond connecting the i-th atom and j-th atom as _e_ _ij_ ∈ E .


2.2.1. GNN to encode molecular graphs
We will expand on the use of the GNN for the encoding of drugs’
molecular graphs. In this part, we describe the GNN encoder, which is
to convert complex graph-structured molecule objects into low-dimensional dense real-valued embeddings. Formally speaking, the GNN
can learn an embedding _g_ ∈ � _d_ for individual molecules via two sequential processes: iterative update process and output process. In
our work, we incorporate simple chemistry knowledge into the design
of the weight-sharing mechanism contained in the GNN encoder. Apart
from that, we also adopt an attention mechanism inspired by [22] to
aggregate atom-level embeddings to generate a graph-level embedding.
We will elaborate on the GNN from three aspects: input, iterative update process, and output process. The architecture of the GNN encoder
is demonstrated in Fig. 2.
Input. A drug molecule can be regarded as a graph where every atom
is represented as a node, while every bond as an edge if we neglect the ̃
three-dimensional spatial structural information. Based on the abstract,
we can utilize the devised GNN to extract structural information and

compress them in a low-dimensional dense embedding. For each atom
in a molecule, we assign a randomly-initialized embedding vector
sampled from the standard normal distribution in consideration of its
nuclear charge number. Stacking these randomly-initialized embeddings forms the embedding matrix, which is one of the two inputs for



GNN. Another input for GNN is the multi-channel adjacency matrix,
whose channel dimension denotes the specific type of chemical bond. In
our work, we take the following four kinds of chemical bonds into
consideration: single, double, triple, and aromatic. We represent the
specific bond type by one-hot encoding.
Iteratiave Update Process. The iterative update process is implemented by stacking multiple graph convolution layers. The stacking
mechanism can consider a more chemical environment. We propose
two different mechanisms to inject simple chemistry knowledge into
the iterative update process. The two novel mechanisms are Bond-aware
Message Propagation and Highway-based Update, respectively.
Bond-aware Message Propagation. For each node, we aggregate
the message sent by its direct neighboring nodes, and sum the messages

( ) ̂ _l_
equally to serve as the candidate hidden state **h** _i_ at the l-th layer for
node _v_ _i_ . This process can be described by the formula (1):


̃



̂


where N( ) _i_ denotes the first-order neighborhood nodes for the centering node _v_ _i_ **A**, ( ) _el_ _ij_ ∈ � _d_ _h_ × _d_ _h_ denotes the trainable weight parameters

shared by the same type of bond _e_ _ij_ in the l-th layer, and **h** ( _jl_ −1) ∈ � _d_ _h_
denotes the hidden state for the neighboring node _v_ _j_ in the previous
layer. According to formula (1), the message sent by the neighboring
node _v_ _j_ to center one _v_ _i_ is modeled as a linear transformation whose
trainable weight parameters are shared by the edges with the same type
of chemical bond. The modeling of node-level interactions has an intuitive interpretation in chemistry: the impact from the neighboring
atoms with the identical chemical bond would be similar, whereas the
effect from those with the different kinds of chemical bonds would be
very different.
Highway-based Update. Though formula (1) has describe the impact on the focused centering node exerted by neighboring ones, we
should not ignore the impact of the centering node in the previous

( ) ̃ _l_
layer. In order to fuse the candidate hidden state **h** _i_ and the previous
hidden state **h** _i_ ( _l_ −1), we devise the fusion gate layer inspired by the
highway network [23]:


**z** _i_ ( ) _l_ = tanh( **W h** _z_ [ _i_ ( _l_ −1) ; **h** _i_ ( ) _l_ ] + **b** _z_ ) (2)


**r** _i_ ( ) _l_ = _σ_ ( **W h** _r_ [ _i_ ( _l_ −1) ; **h** _i_ ( ) _l_ ] + **b** _r_ ) (3)


Fig. 2. The architecture of the Graph
Convolution Neural Network for the encoding of
molecular data. The input layer assigns the
randomly initialized embedding vector for each
vertex/atom in consideration of the nuclear

charge number. The consecutive graph convolution layer (layer 1 to layer L) completes the
iterative update process, refining the atomic representations given by the input layer. The
bottom part shows the output process comprising a simple attention mechanism.



̂


_i_ ( ) _l_ = ∑ **A h** _e_ ( ) _l_ _ij_ ( _jl_

_j_ ∈N( ) _i_


̃



̂


**h** _i_ ( ) _l_ = ∑ **A h** _e_ ( ) _l_ ( _l_ −


̃



̂


=
∑


̃



̂


∈


̃



̂


∑ **A h** _e_ ( ) _l_ _ij_ ( _jl_ −1)

N( ) _i_ (1)


̃



̂


( ) _i_


̃



̂


̃



̂


̃


49


X. Chen, et al. _Methods 179 (2020) 47–54_



**f** _i_ ( ) _l_ = _σ_ ( **W h** _f_ [ _i_ ( _l_ −1) ; **h** _i_ ( ) _l_ ] + **b** _f_ )



(4)



to minimizing the cross-entropy loss:


_N_



**h** _i_ ( ) _l_ = **r** _i_ ( ) _l_ ⊙ **h** _i_ ( _l_ −1) + **f** _i_ ( ) _l_ ⊙ **z** _i_ ( ) _l_ (5)


where [;] denotes the concatenation operation along the feature di
mension.

Output process. We stack multiple graph convolutional layers (L
layers in total) to learn the embedding for each node v in a graph G . At
the final graph convolutional layer, we obtain the set of final hidden
states for each node in a graph
_S_ _g_ = { **h** _i_ ( ) _L_ ∈ � _d_ _h_ | _v_ _i_ ∈ V, _i_ = 1, 2, …,|V|} . With the set _S_ _g_ at hand, we
need to generate an embedding **g** ∈ � _d_ to represent the whole graph.
We conjecture that the shallow layers in GNNs would extract more
concrete features, while the deeper layers would learn more abstract
features. This conjecture is based on the analogy with the observation
on CNN (Convolutional Neural Network) [24]. In light of this thinking,
the graph-level embedding must contain both concrete and abstract
features. Therefore, we introduce the attention-based graph pooling
layer motivated by [22] as follows:


**a** _i_ = _σ_ ( **W h** _a_ [ _i_ (0) ; **h** _i_ ( ) _L_ ] + **b** _a_ ). (6)



= − ∑ **y** _i_ log( ) **p** _i_ + (1 − **y** _i_ )log(1 −



_l_ (Θ) = − ∑ **y** _i_ log( ) **p** _i_ + (1 − **y** _i_ )log(1 − **p** _i_ )


_i_ =1



_i_



_i_ _i_ _i_ _i_
1 (11)



= ∑ **a** _i_ ⊙ ( **Wh** _i_ ( ) _L_ +



∑ **a** _i_ ⊙ ( **Wh** _i_ ( ) _L_


_v_ ∈V



**g** = ∑ **a** _i_ ⊙ ( **Wh** _i_ + **b** )


_v_ ∈V



_v_ _i_ ∈V (7)



where [;] denotes concatenation operation along the feature dimension,
**h** _i_ (0) is the randomly initialized embedding vector for each node _v_ _i_, **h** _i_ ( ) _L_

is the hidden state in the L-th graph convlutional layer (the final one),
attention score **a** _i_ ∈ � _d_ denotes the importance score of the node _v_ _i_, ⊙
denotes Hadamard multiplication, **g** ∈ � _d_ is the required embedding
for the embedding of the whole graph. It should be noted that the
parameters **W** _a_ ∈ � _d d_ × _h_ and **b** _a_ ∈ � _d_ are shared across all the nodes, and
so do the weight parameters **W** ∈ � _d d_ × _h_ and bias parameter **b** ∈ � _d_ .


2.2.2. Drug-drug interaction prediction
Interaction Predictor. We first compress two embeddings representing a pair of molecule objects by the circular correlation operation inspired by [25].


**g** = **g g** 1 ∘ 2 (8)



_d_



−1



1 ∘ 2 _i_ = ∑



**g g** 1 ∘ 2 _i_ = ∑ [ **g** 1 ] ·[ _k_ **g** 2 ] [(] _k_ + _i mod d_ ) . .
_k_ =0



1 _k_ **g** 2 [(] _k_ + _i mod d_ ) .



(9)



1 2 _i_ 1 _k_ 2 [(] _k_ + _i mod d_ ) . .


_k_ =0



where Θ is the set of all weight and bias parameters in our model, N is
the total number of drug-drug pairs in the self-augmented training set.
With the cross-entropy as loss function, we adopt the back-propagation
algorithm to find the best solution for the trainable parameters Θ .


3. Results and discussion


In this section, we demonstrate the experimental results to verify the
advantages of our GCN-BMP model in the DDI prediction task.
Additionally, we also conduct more experiments to probe the reason
why our GCN-BMP model can perform better than traditional fingerprint-based DDI prediction models. Finally, we compare the robustness
of our GCN-BMP model and the sub-optimal conventional model to
verify our claim on the robustness.


3.1. Baselines


To illustrate the superiority of our model compared to the state-ofthe-art approaches for DDI prediction, we implement the following
baseline approaches to compare their performances:

# • [Nearest Neighobr][ [4]][. This model used the combination of known]

pairwise interactions and similarity derived from substructure [4] to
conduct DDI prediction. We named the model as NN for short.
# • [Label Propagation][ [7]][. This model resorted to the label propaga-]

tion(LP) algorithm to build three similarity-based predictive models.
The pairwise similarity information is computed based on substructures, side effects and off-label side effects, respectively. We
named the model as LP-Sub, LP-SE, and LP-OSE, respectively.
# • [Multi-Feature Ensemble][ [9]][. Multi-Feature Ensenmble is built on]

top of the combination of three different algorithms: neighbor recommendation(NR), label propagation(LP), and matrix disturb(MD)
algorithms. We refer to the model as Ens.
# • [SSP-MLP][. Ref.][ [10]][ use the sequential combination of structural]

similarity profile(SSP) and multi-layer perceptron to conduct the
classification. We refer to the model as SSP-MLP.
# • [Mol2vec][. This model uses the pre-trained reprsentations provided]

by Mol2vec [12] as inputs. The vector representations of a drug pair
are fed into a feed-forward neural network.

# • [Graph Autoencoder][. Ref.][ [26]][ developed an attention mechanism]

to integrate multiple types of drug features, which will be passed
into a GCN-style graph autoencoder to learn the embedding for individual drug node. We refer to the model as GA.
# • [NFP][. Neural][ fi][ngerprint developed by][ [18]][ is the][ fi][rst graph neural]

network tailored for molecular property prediction. We substitute
our siamese graph neural network with neural fingerprint. We name
the model as NFP for short.

# • [GIN][ [20]][. Graph Isomorphism Network(GIN) is the state-of-the-art]

graph neural network. We change our siamese GNN encoder with
the graph isomorphism network and name the model as GIN.


3.2. Evaluatioin setups


Setups for BinaryDDI Dataset. We divide the whole dataset into
the training set, validation set, and test set with the ratio 8:1:1. Note
that we have only reliable positive drug pairs in the dataset, we regard
the same number of randomly sampled negative drug pairs as the negative training samples for simplicity. As for the valid set and test set,
we keep it the same as the original situation. We implement our model
with Chainer [27]. We use the Adam optimizer, set the initial learning
rate as 0.001. We also exploit the exponential shift strategy with a ratio



=



where [∘] denotes the circular correlation operation, **g** is the desired
graph-level embedding. Additionally, the calculation of circular correlation is described by formula (9). Note that the circular correlation
operation does not satisfy the symmetry property, the model performance may be perturbed by the order of two drug molecules in an input
pair. To deal with this problem, we doubled the size of the training set
by replicating each pair and reversing the order of two molecules. We
train the model in the newly-generated training set. We conduct the
inference process in the following manner: use the original pairs and
pairs with the reversed order and then average the two estimated interaction probability to obtain the final prediction. With the final
graph-level embedding encoding the input pair, we feed it into a
shallow-layered neural network whose non-linear activation function is
sigmoid to predict the probability that an interaction occurs between
the pair of drugs or the specific DDI type (as shown in formula (10)).


**p** = _σ_ ( **W g** _o_ + **b** _o_ ) (10)


where **p** ∈ � _k_ . The parameter k will be determined by the task setting.
If the purpose is to predict the occurrence of DDIs, k is set as 2, while k
is set as the total number of DDI types if our aim is to predict the
specific DDI types. Training Our model is typical of end-to-end graph
representation learning. Based on a set of drug-drug pairs as well as the
corresponding labels, our model is trained by maximizing the likelihood
of observing the training data. The optimization criterion is equivalent



50


X. Chen, et al. _Methods 179 (2020) 47–54_



Table 1

Model comparison on the small-scale dataset.


Model Performance


Name AUROC AUPRC F1


NN 67.81 ± 0.25 52.61 ± 0.27 49.84 ± 0.43

LP-Sub 93.70 ± 0.13 90.36 ± 0.18 76.41 ± 0.28

LP-SE 93.79 ± 0.28 90.53 ± 0.39 78.48 ± 0.50

LP-OSE 93.88 ± 0.14 90.63 ± 0.36 79.44 ± 0.43

Ens 95.54 ± 0.13 92.75 ± 0.33 83.81 ± 0.39

SSP-MLP 93.09 ± 0.34 88.58 ± 0.51 78.38 ± 0.57

GA 93.84 ± 0.61 90.27 ± 0.66 54.84 ± 0.47

Mol2vec 93.63 ± 0.14 88.74 ± 0.32 81.03 ± 0.26

NFP 81.82 ± 0.13 68.89 ± 0.21 60.93 ± 0.24

GIN 61.63 ± 0.26 48.28 ± 0.31 56.00 ± 0.56

GCN-BMP 96.66 ± 0.09 94.02 ± 0.12 85.00 ± 0.17


of 0.5 every 10 epochs. The batch size is 32. For the hyper-parameters
of the GNN encoder, we set the dimension of node-level hidden states as
32, that of the whole graph as 16. The total number of graph convolutional layers is 8, deeper than the usual graph neural networks such
as GCN [17]. Since the tasks conducted on BinaryDDI dataset is binary
classification, we select three metrics: area under ROC curve(AUC), area
under PRC curve(PRC) and F1 to evaluate the model performances. For
the sake of reliability, we report the mean and standard deviation of the
four metrics over 20 repetitions in Table I. The experimental results are
obtained on the test set.

Setups for MultiDDI Dataset. We use the same dataset as DeepDDI

[10]. The hyperparameter configuration is the same as the binary
classification task performed on BinaryDDI dataset. Since the task
conducted on the MultiDDI dataset is in essence multi-label classification, we select macro AUC, macro PRC and macro F1 to measure the
model performance.


3.3. Comparison results


Table 1 shows the performance comparison among our proposed
model and baseline approaches. It is evident that our model can achieve
more exceptional performance. The model Ens obtain the sub-optimal
performance, following our model closely. Ens exploit eight kinds of
drug feature similarities as well as six kinds of topological ones and use
three different algorithms to predict the occurrence of DDIs. Ens is
more brilliant compared with LP, SSP-MLP, and NN, all of the three
models only incorporate only one kind of drug feature. From the performance comparison among Ens, LP, and NN, it appears that the integration of multi-view drug features is essential to the model performance for the DDI prediction task. However, this stereotype has been
broken by the better performance shown by our model, suggesting that
the end-to-end graph representation learning method is a better option.
The model named GA views individual drug entity as a node, pairwise
interaction or similarity information as an edge. Then it learns the
embedding for each drug node through Graph Autoencoder [17]. From
the perspective of GA, the embedding of each drug molecule is a nodelevel representation. Our model is different in that the embedding
vector for one drug molecule is graph-level representation. The performance of GA is worse than our model, indicating that viewing a drug
molecule as a node in an interaction graph is improper. According to
the experimental results shown in Table 1, our model outperforms other
methods that also incorporate graph-level representation acquired by
other graph neural networks such as NFP and GIN. Finally, the SMILESbased model named Mol2vec is not suitable for the DDI prediction task
because of its model performance.

Table 2 demonstrates the model performance comparison results
obtained on the MultiDDI dataset. The model named SSP-MLP is the
first catering for the prediction of specific DDI type. The original version of NN is developed for the prediction of the occurrence of DDIs. We



Table 2

Model comparison on the large-scale dataset


Model Performance


Name AUROC AUPRC F1


NN 98.52 ± 0.17 63.04 ± 0.29 13.74 ± 0.36

NR-Sub 99.01 ± 0.09 66.18 ± 0.24 47.89 ± 0.92

SSP-MLP 74.30 ± 0.21 61.20 ± 0.32 53.56 ± 0.54

NFP 98.95 ± 0.16 68.32 ± 0.17 62.56 ± 0.33

GIN 87.67 ± 0.18 14.58 ± 0.13 10.42 ± 0.24

GCN-BMP 99.01 ± 0.09 80.18 ± 0.30 67.31 ± 0.23


adapt it for the multi-label classification task. NR-Sub is a component
of Ens. Because we cannot get access to other types of drug features and
other components(Label Propagation and Matrix Disturb) cannot converge, we can only utilize the substructure similarity information as
well as Neighbor Recommendation(NR) method to fulfill the multi-label
classification task. The experimental results shown in Table 2 indicate
that our model can achieve higher performance in terms of different

metrics.


3.4. Simple abalation study


In order to probe the influence of the three components in our
model, namely Bond-aware Message Propagation (BMP), Highwaybased Update (HU) and attention-based graph pooling layer (ATT), we
conduct ablation experiments. The associated experimental results are
shown in Table 3. The results suggest that these three components can
boost model performance. Furthermore, Bond-aware Message Propagation contributes more significantly than the other two components.


3.5. Visualization analysis


In this section, we endeavor to illustrate that our proposed model
can capture meaningful and interpretable data-driven molecular representations for the DDI prediction task. We qualitatively visualize
molecular embedding vectors related to one certain drug DB00250
(Dapsone) in Fig. 3. Fig. 3 exhibits the t-SNE visualization of ECFP4
features, embedding vectors learned by mol2vec [12], SSP(Structural
Similarity Profile) [10] and the molecular representations generated by
our model respectively. We observe that only the embedding vectors
learned by our proposed model can separate the interacting drug molecules and the non-interacting ones easily, which means that our
model can learn more discriminative data-driven molecular re
presentations for the DDI prediction task.
To further probe the reason why our model works so well, we
choose Methotrexate (DB00563) as a representative drug molecule in
input pairs. We visualize the molecular graph of this drug molecule, and
label atoms with different colors according to the attention scores in the
output process of our GNN encoder. Fig. 4 suggests that the rightmost
carboxyl functional group is one of the local structures which are crucial to determining whether there will be an interaction with other drug
molecules.

We list the molecular graphs of interacting drug molecules in the


Table 3

Results of Abalation Study on Small-scale DDI Dataset.


Model Performance


Name AUROC AUPRC F1


-BMP 94.57 ± 0.13 90.99 ± 0.17 81.13 ± 0.39

-HU 95.31 ± 0.10 92.22 ± 0.16 82.69 ± 0.25

-ATT 95.13 ± 0.08 91.99 ± 0.09 82.14 ± 0.13

Our 96.66 ± 0.09 94.02 ± 0.12 85.00 ± 0.17



51


X. Chen, et al. _Methods 179 (2020) 47–54_


Fig. 3. The t-SNE visualization of drug molecule embedding vectors with respect to DB00250 (Dapsone) in the test set. Green circle points denote the interacting drug
molecules while red ones the non-interacting drug entities.


Fig. 4. The molecular graph of Methotrexate (DB00563). The atoms in red
denote the most important ones in DDI prediction while those in orange represent the second most crucial atoms for this task.



test set in Figs. 5–7. We also label the crucial atoms according to the
attention score in the output process. We can see several common
substructures in the 3 figures mentioned above: (1) Organic weak
acids represented by Acetylsalicylic acid and Salicylic acid The
drugs shown in Fig. 5 are all organic substances with weak acidity
whose pKa metric is greater than 0. Their labeled hydroxides are connected to the benzene ring, carbonyl or P]O, leading to weak acidity. It
is the weak acidity that makes these drugs bind to plasma proteins,
contributing to the dramatically rising plasma concentration of Methotrexate to cause toxicity. The interpretation is accurate according to
the professional pharmacology knowledge, indicating that the emphasis
on hydroxides conducted by our model conforms to domain knowledge.
In other words, our proposed model is blessed with certain interpretability. (2) Statins symbolized by Pravastatin and Simvastatin
which have common local substructures. The molecular graphs of
Pravastatin and Simvastatin are shown in Fig. 6. For Pravastatin and
Simvastatin, our model considers the chiral carbon/oxygen atoms
connecting to the Siamese rings are most significant for the DDI prediction task. (3) Dipping drugs such as Nifedipine and Nisoldipine



Fig. 6. The molecular graphs of interacting drugs involving with Methotrexate
(DB00563). The drugs are Pravastatin (DB00175) and Simvastatin (DB00641)
which share the common local substructures.


which have common local substructures. The molecular graphs of Nifedipine and Nisoldipine are demonstrated in Fig. 7. For Nifedipine and
Nisoldipine, the distribution of importance is almost the same as each
other.

From Fig. 5–7, we can observe an interesting phenomenon that our
proposed model will assign similar attention weights for the matters
with shared local substructures or chemical properties. For the substances in Fig. 5, their hydroxides are connected to the benzene ring,
carbonyl or P]O, leading to weak acidity. As for Nifedipine and Nisoldipine, the story is almost the same. They are highly similar structurally, and our model assigns almost the same attention weights to the
two drugs. Therefore, our model can pay more attention to the local
substructures which are more related to the occurrence of DDI, neglecting the insignificant part. Obviously, this is the reason why the
proposed end-to-end representation learning method can work well
than other predictive models especially for the DDI prediction task.


Fig. 5. The molecular graphs of interacting drugs
involving with Methotrexate (DB00563). The atoms
in red denote the most important one in DDI prediction while those in orange represent the second
most crucial atoms for this task (sometimes green
color for the third important atoms). The drugs are
all organic matters with weak acidity.



52


X. Chen, et al. _Methods 179 (2020) 47–54_


Fig. 9. Example of wrongly-predicted interacting drug pairs whose size is
varied dramatically and share no common structures. The top molecule is
DB01327 (Cefazolin) while the bottom is DB00435 (Nitric Oxide).



Fig. 7. The molecular graphs of interacting drugs involving with Methotrexate
(DB00563). The drugs are Nifedipine (DB01115) and Nisoldipine (DB00401)
which share the common local substructures.


Table 4

Model comparison on the low similarity dataset.


Model Performance


Name AUROC AUPRC Precision Recall


Ens 0.9587 0.8817 0.8276 0.7956

Our 0.9724 0.9223 0.8037 0.9148


Fig. 8. Example of wrongly-predicted interacting drug pairs which share
common structures (O]S]O structure) though look different from the global
perspective. The left molecule is DB00250 (Dapsone), while the right is
DB01008 (Busulfan).


3.6. Robustness analysis


In the Introduction section, we claim that the models based on
chemical fingerprints tend to be perturbed by the extremely low pairwise similarity. The foundation of these models is the approximation of
DDI occurrence probability with the combination of distinct drug-associated features. In line with this assumption, the drug pairs with
higher pairwise similarity tend to interact, and vice versa. However, the
interaction probability cannot be precisely approximated by the combination of various drug features. In fact, there are still interacting drug
pairs whose pairwise similarity is immensely low. In order to verify our
conjecture, we choose the drug pairs whose pairwise similarity is less
than 0.2 except that derived from side effect fingerprint to produce the
desired low similarity dataset. The pairwise similarity mentioned above
is computed as the cosine similarity of specific fingerprint vectors. The
low similarity dataset contains 521 drug pairs. We compare the predictive performance of our model and the sub-optimal model Ens on the
low similarity dataset. The detailed experimental results are demonstrated in Table 4. It is evident that our model is superior to the suboptimal Ens on the low similarity dataset in terms of its significantly
better AUROC, AUPRC and recall score and comparable precision score.
What we concentrate more on is the recall score, the metric indicating
that our model can still retrieve more interacting drug pairs even when
the pairwise similarity is extremely low.
With the purpose of finding out the reason why the prediction error
occurs in the Ens model, we choose some interacting drug pairs



randomly from the low similarity dataset. We present two typical interacting drug pairs in Figs. 8 and 9. As shown in Fig. 8, Dapsone
(DB00250) and Bulsulfan(DB01008) are dissimilar globally, but there
still exist shared chemical substructures locally which are deemed as
the key indicator to determine the occurrence of DDIs. Or rather,
Dapsone and Busulfan share the common “O]S]O”-style chemical
substructure, but immensely distinct globally. However, the models
based on chemical fingerprints such as Ens can only model the global
pairwise similarities. In contrast, our model can fulfill the prediction
task in a different manner. Our model can capture the local similarity
between pairwise drugs and use the related similarity to conduct the
correct prediction. The story a bit different when it comes to Cefazolin
(DB01327) and Nitric Oxide (DB00345). Cefazolin is much bigger than
Nitric Oxide in terms of size. To make matters worse, we cannot find
any common chemical substructures shared by the two drugs(shown in
Fig. 9). The decision made by Ens is wrong, whereas our model gives
the correct answer that these two drugs will interact even though they
are different from all aspects.


4. Conclusion


In this paper, we have proposed a novel DDI prediction model which
exploits the end-to-end representation learning. The extensive experiments illustrate that our model can achieve the state-of-the-art predictive performance, suggesting that the power of graph representation
learning in the DDI prediction task. The visualization of molecule graph
attention distribution indicates that our model can capture the meaningful data-driven molecular representations which are beneficial for
the DDI prediction task, which can provide insights conforming to domain knowledge. We believe that there is still room for further improvements. The graph convolution operator in prevailing GNNs can
only model 2D molecular structure, which may ignore some critical 3D
structural information. It would be a promising direction to develop
graph convolution operator on 3D structures.


Funding


The research presented in the paper is supported by the National
Key Research and Development Program of China (No.
2018YFC0116800).


CRediT authorship contribution statement


Xin Chen: Conceptualization, Methodology, Software, Data curation, Writing - original draft, Validation, Investigation. Xien Liu:
Writing - review & editing, Project administration. Ji Wu: Supervision.


References


[[1] K. Han, E.E. Jeng, G.T. Hess, D.W. Morgens, A. Li, M.C. Bassik, Synergistic drug](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0005)
[combinations for cancer identified in a crispr screen for pairwise genetic interac-](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0005)
[tions, Nat. Biotechnol. 35 (5) (2017) 463.](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0005)

[2] N.P. Tatonetti, P.Y. Patrick, R. Daneshjou, R.B. Altman, Data-driven prediction of
drug effects and interactions, Sci. Transl. Med. 4(125) (2012) 125ra31–125ra31.

[[3] D. Weininger, Smiles a chemical language and information system. 1. Introduction](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0015)



53


X. Chen, et al. _Methods 179 (2020) 47–54_



[to methodology and encoding rules, J. Chem. Inf. Comput. Sci. 28 (1) (1988) 31–36.](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0015)

[[4] S. Vilar, R. Harpaz, E. Uriarte, L. Santana, R. Rabadan, C. Friedman, Drug-drug](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0020)
[interaction through molecular structure similarity analysis, J. Am. Med. Inform.](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0020)
[Assoc. 19 (6) (2012) 1066–1074.](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0020)

[[5] S. Vilar, E. Uriarte, L. Santana, N.P. Tatonetti, C. Friedman, Detection of drug-drug](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0025)
[interactions by modeling interaction profile fingerprints, PloS One 8 (3) (2013)](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0025)
[e58321.](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0025)

[[6] S. Vilar, E. Uriarte, L. Santana, T. Lorberbaum, G. Hripcsak, C. Friedman,](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0030)
[N.P. Tatonetti, Similarity-based modeling in large-scale prediction of drug-drug](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0030)
[interactions, Nat. Protoc. 9 (9) (2014) 2147.](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0030)

[[7] P. Li, C. Huang, Y. Fu, J. Wang, Z. Wu, J. Ru, C. Zheng, Z. Guo, X. Chen, W. Zhou,](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0035)
[et al., Large-scale exploration and analysis of drug combinations, Bioinformatics 31](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0035)
[(12) (2015) 2007–2016.](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0035)

[[8] P. Zhang, F. Wang, J. Hu, R. Sorrentino, Label propagation prediction of drug-drug](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0040)
[interactions based on clinical side effects, Scientific Rep. 5 (2015) 12339.](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0040)

[[9] W. Zhang, Y. Chen, F. Liu, F. Luo, G. Tian, X. Li, Predicting potential drug-drug](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0045)
[interactions by integrating chemical, biological, phenotypic and network data, BMC](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0045)
[Bioinf. 18 (1) (2017) 18.](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0045)

[[10] J.Y. Ryu, H.U. Kim, S.Y. Lee, Deep learning improves prediction of drug–drug and](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0050)
[drug–food interactions, Proc. Natl. Acad. Sci. 115 (18) (2018) E4304–E4311.](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0050)

[11] T. Mikolov, K. Chen, G.S. Corrado, J. Dean, Efficient estimation of word representations in vector space (2013).

[[12] S. Jaeger, S. Fulle, S. Turk, Mol2vec: unsupervised machine learning approach with](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0060)
[chemical intuition, J. Chem. Inf. Model. 58 (1) (2018) 27–35.](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0060)

[[13] S. Zheng, X. Yan, Y. Yang, J. Xu, Identifying structure-property relationships](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0065)
[through smiles syntax analysis with self-attention mechanism, J. Chem. Inf. Model.](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0065)
[59 (2) (2019) 914–923.](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0065)

[14] Z. Xu, S. Wang, F. Zhu, J. Huang, Seq2seq fingerprint: an unsupervised deep molecular embedding for drug discovery (2017) 285–294.

[[15] R. Winter, F. Montanari, F. Noe, D. Clevert, Learning continuous and data-driven](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0075)



[molecular descriptors by translating equivalent chemical representations, Chem.](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0075)
[Sci. 10 (6) (2019) 1692–1701.](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0075)

[[16] I. Sutskever, O. Vinyals, Q.V. Le, Sequence to sequence learning with neural net-](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0080)
[works, Comput. Lang. (2014).](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0080)

[17] T.N. Kipf, M. Welling, Semi-supervised classification with graph convolutional
networks, arXiv preprint arXiv:1609.02907 (2016).

[[18] D.K. Duvenaud, D. Maclaurin, J. Iparraguirre, R. Bombarell, T. Hirzel, A. Aspuru-](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0090)
[Guzik, R.P. Adams, Convolutional networks on graphs for learning molecular fin-](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0090)
[gerprints, Adv. Neural Inf. Process. Syst. (2015) 2224–2232.](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0090)

[[19] S. Kearnes, K. McCloskey, M. Berndl, V. Pande, P. Riley, Molecular graph con-](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0095)
[volutions: moving beyond fingerprints, J. Comput.-Aided Mol. Design 30 (8) (2016)](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0095)
[595–608.](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0095)

[20] K. Xu, W. Hu, J. Leskovec, S. Jegelka, How powerful are graph neural networks?,
arXiv preprint arXiv:1810.00826 (2018).

[[21] J. Gilmer, S.S. Schoenholz, P.F. Riley, O. Vinyals, G.E. Dahl, Neural message passing](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0105)
[for quantum chemistry, Proceedings of the 34th International Conference on](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0105)
[Machine Learning-Volume 70, JMLR. org, 2017, pp. 1263–1272.](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0105)

[22] Y. Li, D. Tarlow, M. Brockschmidt, R. Zemel, Gated graph sequence neural networks, arXiv preprint arXiv:1511.05493 (2015).

[23] R.K. Srivastava, K. Greff, J. Schmidhuber, Highway networks, arXiv preprint
arXiv:1505.00387 (2015).

[[24] A. Mahendran, A. Vedaldi, Visualizing deep convolutional neural networks using](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0120)
[natural pre-images, Int. J. Comput. Vision 120 (3) (2016) 233–255.](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0120)

[[25] M. Nickel, L. Rosasco, T. Poggio, Holographic embeddings of knowledge graphs,](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0125)
[Thirtieth Aaai Conference on Artificial Intelligence, 2016.](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0125)

[26] T. Ma, C. Xiao, J. Zhou, F. Wang, Drug similarity integration through attentive
multi-view graph auto-encoders, arXiv preprint arXiv:1804.10850 (2018).

[[27] S. Tokui, K. Oono, S. Hido, J. Clayton, the twenty-ninth annual conference on](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0135)
[neural information processing systems (NIPS), Vol, 2015, pp. 1–6.](http://refhub.elsevier.com/S1046-2023(20)30060-8/h0135)



54


