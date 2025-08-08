### Highlights

**Hierarchical Graph Representation Learning for the Prediction of Drug-Target Binding Affin-**
**ity**


Zhaoyang Chu,Shichao Liu,Wen Zhang


  - We propose a novel hierarchical graph representation learning model for drug-target binding affinity prediction, named HGRL-DTA. The model can capture global-level affinity relationships and local-level chemical
structures involving drug/target molecules synergistically, and integrate such hierarchical information with a
message broadcasting strategy.


  - To solve the cold start problem of inferring representations for unseen drugs and targets, we design a similaritybased embedding map from known to unknown drugs/targets, which can infer the representation of the unknown
drug (target) by aggregating the representations of its most similar known drugs (targets).


  - Extensive experiments under four experimental scenarios are conducted to evaluate the performance of the proposed model. Compared with four state-of-the-art methods, HGRL-DTA achieves significantly better model
generalization among all four scenarios.


## Hierarchical Graph Representation Learning for the Prediction of Drug-Target Binding Affinity

Zhaoyang Chu _[a]_, Shichao Liu _[a]_ [,][∗] and Wen Zhang _[a]_ [,][∗]


_a_ _College of Informatics, Huazhong Agricultural University, Wuhan, 430070, China_



A R T I C L E I N F O


_Keywords_ :
Graph Representation Learning
Binding Affinity Prediction
Hierarchical Graph
Message Broadcasting
Drug Discovery


**1. Introduction**



A B S T R A C T


The identification of drug-target binding affinity (DTA) has attracted increasing attention in the
drug discovery process due to the more specific interpretation than binary interaction prediction.
Recently, numerous deep learning-based computational methods have been proposed to predict
the binding affinities between drugs and targets benefiting from their satisfactory performance.
However, the previous works mainly focus on encoding biological features and chemical structures of drugs and targets, with a lack of exploiting the essential topological information from the
drug-target affinity network. In this paper, we propose a novel hierarchical graph representation
learning model for the drug-target binding affinity prediction, namely HGRL-DTA. The main
contribution of our model is to establish a hierarchical graph learning architecture to incorporate the intrinsic properties of drug/target molecules and the topological affinities of drug-target
pairs. In this architecture, we adopt a message broadcasting mechanism to integrate the hierarchical representations learned from the global-level affinity graph and the local-level molecular
graph. Besides, we design a similarity-based embedding map to solve the cold start problem
of inferring representations for unseen drugs and targets. Comprehensive experimental results
under different scenarios indicate that HGRL-DTA significantly outperforms the state-of-the-art
models and shows better model generalization among all the scenarios.



Molecular drugs exert their therapeutic effects by working as ligands to interact with target proteins and activate
or inhibit the biological process of targets. Investigation of drug-target interactions (DTI) plays a crucial role in drug
discovery, which helps in understanding the specific interactions between drug compounds and target proteins. However, due to the numerous possible protein-compound combinations with more than 5,000 potential proteins [18] and
over 100 million drug candidate compounds [26], computational approaches have become increasingly necessary to
predict drug-target interactions. The existing approaches for drug-target interaction prediction mainly fall into two
categories. The first category of methods formulates the interaction prediction as a binary classification task (interacts
or not) [8, 16, 21, 31, 54]. However, these methods suffer from two primary defects: (1) the inability to differentiate
between true-negative and unknown interactions, and (2) the binary relationship is unable to indicate the continuous
binding affinity that quantifies how tightly a drug binds to a target [39]. To tackle the above limitations, the other kind
of approaches considers the interaction prediction as a regression problem to predict continuous binding affinities,
which has attracted more and more attention in recent studies [4, 22, 32, 38, 48].
Most computational approaches for binding affinity prediction focus on the usage of 3D structures of proteincompound complexes [4, 14, 23, 32, 43]. Although these 3D structure-based methods have achieved relatively high
predictive performance, they remain limitations because the exploitation of 3D structural data is costly and timeconsuming. In practice, the co-crystallized 3D structures of protein-compound complexes are usually difficult to obtain,
and predicting them by docking individual structures of proteins and compounds together also remains a challenging
task [24, 25]. For this reason, structure-free prediction of drug-target binding affinities has emerged to overcome the
limitations of 3D structure-based methods, without the analysis of 3D structural data and the molecular docking process

[53, 58]. The early structure-free approaches utilize statistic machine learning to predict binding affinities of drug-target
pairs with hand-crafted features [20], similarity information [39] of drugs and targets. However, these approaches
heavily rely on expert knowledge and feature engineering, which may lead to limited accuracy and generalization of
models.


∗ Corresponding author: zhangwen@mail.hzau.edu.cn, scliu@mail.hzau.edu.cn
ORCID (s): `0000-0003-4333-8063` (Z. Chu)


Zhaoyang Chu et al.: _Preprint submitted to Elsevier_ Page 1 of 15


HGRL-DTA


Recently, deep learning [30] has achieved remarkable success in various machine learning tasks and demonstrated
its expansion capability for diverse application scenarios. Deep learning-based affinity prediction has become a popular
research [38], which can automatically learn feature representations of drugs and targets in an end-to-end way and
show its superior performance. Most of the existing works represent drug compounds as simplified molecular input
line entry system (SMILES) strings and target proteins as amino acid sequences, respectively, and learn hidden patterns
from the sequence data using various neural networks, such as convolutional neural networks (CNNs) [1, 24, 38, 64],
recurrent neural networks (RNNs) [1, 24] and generative adversarial networks (GANs) [63]. Moreover, owing to the
significant advance of deep learning on graph-structured data [15, 51], some works have been designed to model 2D
graph structure information of drug compounds [22, 24, 33, 36] and target proteins [22, 37].
Despite the success of representation learning for drugs and targets, the affinity information cannot be properly
exploited by the previous deep learning-based methods, which only use affinities as the true labels for model optimization. Recent studies aim to construct heterogeneous networks based on binary drug-target affinities [10, 34, 41, 50, 56],
which tend to lose more realistic information hidden in continuous affinities. To take advantage of the essential topological information deriving from drug-target affinity relationships, the widely used graph representation learning
method, i.e., graph neural networks (GNNs) [62], can be applied to facilitate the predictive performance of models.
However, the graph representation learning method works based on the graph connectivity of the affinity data, which
cannot learn representations for unseen drugs/targets. In addition, previous graph-based binding affinity prediction
works only concentrate on utilizing the chemical structures to learn representations for drugs and targets [22, 36]. The
combined analysis of affinity relationships and chemical structures is often overlooked by these methods, which is
essentially necessary and proven effective in some relative researches [3, 57, 59].
To overcome the limitations mentioned above, we propose a novel hierarchical graph representation learning
method for the drug-target binding affinity prediction, namely HGRL-DTA, which can capture global-level affinity
relationships and local-level chemical structures involving drug/target molecules synergistically. HGRL-DTA establishes a hierarchical graph architecture, where drug-target affinity relationships are encoded as a global-level affinity
graph and each node inside it, i.e., drug or target, is encoded as a local-level molecular graph. We firstly utilize a popular graph neural network, i.e., graph convolutional network (GCN) [28], to encode affinity relationships and chemical
structures of drugs and targets. Then, we design a message broadcasting mechanism to integrate the hierarchical information learned from the global-level affinity graph and local-level molecular graph. After that, we reuse GCNs to
conduct a refinement process for molecular representations and readout the final embeddings of drugs and targets to
make drug-target binding affinity predictions. Moreover, to solve the cold start problem of inferring representations for
unseen drugs and targets, we build a similarity-based embedding map from known to unknown drugs/targets, which
can infer the unknown drug (target) by aggregating the representations of its most similar known drugs (targets).


**2. Preliminaries**


This section introduces some definitions used in our proposed method and describes the problem formulation of
drug-target binding affinity prediction.


_Definition 2.1._ **Affinity Graph.** Given a set of drugs  _푑_ = { _푑_ 1 _,_ … _, 푑_ _푛_ _푑_ } and a set of targets  _푡_ = { _푡_ 1 _,_ … _, 푡_ _푛_ _푡_ },
we define the affinity graph as a weighted bipartite graph  = { _푑_ _,_  _푡_ _,_  _푎_ }, where  _푎_ _⊆_  _푑_ ×  _푡_ denotes the set of
drug-target pairs with known affinities. Note that a subset of  _푎_ is selected as the training set , and the rest of samples
as the test set are masked during training.


Specifically, the drug-target pair set  _푎_ can be formulated as a drug-target co-occurrence matrix **퐘** ∈ ℝ _[푛]_ _[푑]_ [×] _[푛]_ _[푡]_, whose
entries represents the continuous affinity values of the respective drug-target pairs. Further, we formulate the affinity
graph  as a normalized adjacency matrix **퐀** ∈[0 _,_ 1] [(] _[푛]_ _[푑]_ [+] _[푛]_ _[푡]_ [)×(] _[푛]_ _[푑]_ [+] _[푛]_ _[푡]_ [)], where **퐀** _푖,푗_ + _푛_ _푑_ = _푓_ _푛표푟푚_ ( **퐘** _푖,푗_ ) if there exists a
known affinity from the _푖_ -th drug to the _푗_ -th target (i.e., ( _푑_ _푖_ _, 푡_ _푗_ ) ∈  _푎_ ) and otherwise **퐀** _푖,푗_ = 0, _푓_ _푛표푟푚_ (⋅) denotes the
normalization function. Note that **퐀** is symmetric because  is an undirected graph. The initial signals of nodes (i.e.,
drugs and targets) are summarized in a matrix **퐗** ∈ ℝ [(] _[푛]_ _[푑]_ [+] _[푛]_ _[푡]_ [)×] _[푑]_ _[푎푔]_, where _푑_ _푎푔_ denotes the signal dimension.


_Definition 2.2._ **Drug Molecular Graph.** Given a drug _푑_ _푖_ and a set of atoms  _푎_ = { _푎_ 1 _,_ … _, 푎_ _푛_ _푎_ } inside it, the drug
molecular graph is defined as  _푑_ _푖_ = { _푎_ _,_  _푏_ }, where  _푏_ _⊆_  _푎_ ×  _푎_ denotes the set of atom-atom covalent bonds. An
initial feature vector **퐱** _푎_ _푢_ ∈ ℝ _[푑]_ _[푑푔]_ is assigned for each atom _푎_ _푢_, where _푑_ _푑푔_ denotes the feature dimension.


_Definition 2.3._ **Target Molecular Graph.** Given a target _푡_ _푗_ and a set of residues  _푟_ = { _푟_ 1 _,_ … _, 푟_ _푛_ _푟_ } inside it, the


Zhaoyang Chu et al.: _Preprint submitted to Elsevier_ Page 2 of 15


HGRL-DTA


target molecular graph is defined as  _푡_ _푗_ = { _푟_ _,_  _푐_ }, where  _푐_ _⊆_  _푟_ ×  _푟_ denotes the set of residue-residue contacts.

Each residue _푟_ _푝_ is encoded as an attributed vector **퐱** _푟_ _푝_ ∈ ℝ _[푑]_ _[푡푔]_, where _푑_ _푡푔_ is the attribution dimension.


_Definition 2.4._ **Drug-Target Binding Affinity Prediction.** Given the training set , the affinity graph , the drug
molecular graphs { _푑_ _푖_ } _[푛]_ _푖_ =1 _[푑]_ [and the target molecular graphs][ {][] _[푡]_ _푗_ [}] _[푛]_ _푗_ =1 _[푡]_ [, our goal of drug-target binding affinity prediction]
is to learn a mapping function _훩_ ( _휔_ ) ∶( _,_  _푑_ _푖_ _,_  _푡_ _푗_ ) → _푦_ _푖,푗_ to precisely predict the binding affinity _푦_, where _휔_ is the
trainable parameter and ( _푑_ _푖_ _, 푡_ _푗_ ) ∈  .


**3. Model framework**


In this section, we introduce the proposed hierarchical graph representation learning model for the drug-target
binding affinity prediction, namely HGRL-DTA. The model builds a hierarchical graph setting, where the affinity data
is formulated as an affinity graph and each node inside it, i.e., drug or target, is formulated as a molecular graph. We
firstly utilize GCNs to learn the global-level affinity relationships on the affinity graph and the local-level chemical
structures on the molecular graph, respectively. Then, we integrate the learned hierarchical graph information using
a message broadcasting mechanism and reuse GCNs to refine the molecular representations. Finally, we readout the
representations of drugs and targets and combine them to make binding affinity predictions. Figure 1 illustrates the
framework of our proposed HGRL-DTA model.









































**Figure 1:** Overview of HGRL-DTA.


**3.1. Global-level graph representation learning on affinity graph**
In this paper, we build a graph-based encoder with a two-layer GCN to capture the global-level affinity relationships.
This encoder learns global-level representations for drugs and targets through neighbourhood aggregation and feature
transformation.
Given the training set  and the affinity graph  = { _푑_ _,_  _푡_ _,_  _푎_ }, we recompose the drug-target pair set  _푎_ by
removing test pairs, i.e., masking test entries in the drug-target matrix **퐘** . Then, we preprocess the adjacency matrix
**퐀** deriving from **퐘** :



**̂퐀** = **퐃** [−] [1] 2




[1]

2 **퐀퐃** [−] 2 [1]



2 (1)



where **퐃** ∈ ℝ [(] _[푛]_ _[푑]_ [+] _[푛]_ _[푡]_ [)×(] _[푛]_ _[푑]_ [+] _[푛]_ _[푡]_ [)] denotes a diagonal matrix whose diagonal elements **퐃** _푖,푖_ = [∑] _푗_ **[퐀]** _푖,푗_ [are the degrees of]
corresponding nodes.


Zhaoyang Chu et al.: _Preprint submitted to Elsevier_ Page 3 of 15


HGRL-DTA


Given the preprocessed adjacency matrix **퐀** **[̂]** and the initial graph signal **퐗** as inputs, we apply the GCN propagation
process as follows:


**퐇** = _푅푒퐿푈_ ( **퐀** **[̂]** _푅푒퐿푈_ ( **퐀퐗퐖** **[̂]** [(1)] _푎_ [)] **[퐖]** [(2)] _푎_ [)] (2)


where _푅푒퐿푈_ (⋅) denotes the ReLU activation function, **퐖** [(1)] _푎_ and **퐖** [(2)] _푎_ are two weight parameters at the 1-th and 2-th
layer of the GCN encoder, respectively. Specifically, this encoder directly captures affinity relationships between drugs
and targets at the first GCN propagation iteration, which can be regarded as a spatial-based graph filter aggregating
directly connected neighbours (i.e., 1-hop neighbours) to update node representations. The second GCN propagation
aggregates the information of 2-hop neighbour nodes to recognize the hidden similarity patterns, whose chemical
interpretability is based on the empirical assumption that chemically, biologically, or topologically similar drugs are
more likely to interact with the same target and vice versa. After two iterations, the GCN encoder generates the globallevel embedding matrix **퐇** ∈ ℝ [(] _[푛]_ _[푑]_ [+] _[푛]_ _[푡]_ [)×] _[푑]_ _푎푔_ [′], whose row vectors correspond to drugs  _푑_ = { _푑_ 1 _,_ … _, 푑_ _푛_ _푑_ } and targets
 _푡_ = { _푡_ 1 _,_ … _, 푡_ _푛_ _푡_ } in the affinity graph , _푑_ _푎푔_ [′] [is the embedding dimension.]
Assumed that a drug-target pair ( _푑_ _푖_ _, 푡_ _푗_ ) is selected from the training set , we extract the global-level representations
**퐡** _푑_ _푖_ ∈ ℝ _[푑]_ _[푑푔]_ for drug _푑_ _푖_ and **퐡** _푡_ _푗_ ∈ ℝ _[푑]_ _[푡푔]_ for target _푡_ _푗_, respectively, using two non-linear Multi Layer Perceptron (MLP)
operators:


**퐡** _푑_ _푖_ = _푀퐿푃_ ( **퐖** [(1)] _푏_ _[,]_ **[ 퐖]** [(2)] _푏_ _[,]_ **[ 퐇]** [[] _[푖,]_ [ ∶])] _[,]_ **[ 퐡]** _[푡]_ _푗_ [=] _[ 푀퐿푃]_ [(] **[퐖]** _푐_ [(1)] _[,]_ **[ 퐖]** [(2)] _푐_ _[,]_ **[ 퐇]** [[] _[푗]_ [+] _[ 푛]_ _[푑]_ _[,]_ [ ∶])] (3)


where **퐇** [ _푖,_ ∶] and **퐇** [ _푗_ + _푛_ _푑_ _,_ ∶] denote the row vectors in the embedding matrix **퐇** corresponding to _푑_ _푖_ and _푡_ _푗_ respectively,

**퐖** [(1)] _푏_ [,] **[ 퐖]** [(2)] _푏_ [,] **[ 퐖]** _푐_ [(1)] and **퐖** [(2)] _푐_ are the weight parameters of the MLP operators.


**3.2. Local-level graph representation learning on molecular graph**
For each molecular graph, we equip our model with self-loop GCNs to learn local-level molecular representations.
In the GCN propagation process, we smooth the biological features of each node (i.e., atom or residue) over the
molecular graph to map the intrinsic spatial structures into the low-dimensional embedding space.
Mathematically, given a molecular graph  = { _,_ }, where  is the set of nodes and  is the set of edges, we

⋅
introduce the self-loop GCN propagation operator _푓_ _푝푟표푝_ ( ) over the molecular graph:



**퐡** [(] _푣_ _[푙]_ _푖_ [)] [=] _[ 푓]_ _[푝푟표푝]_ [({] **[퐡]** _푣_ [(] _[푙]_ _푗_ [−1)] ∣ _푣_ _푗_ ∈ _퐶_ ( _푣_ _푖_ ) ∪{ _푣_ _푖_ }}) = _푅푒퐿푈_ (∑

_푣_ _푗_



1
~~√~~ _̂푑_ _푣_ _푖_ _̂푑_ _푣_ _푗_



**퐡** [(] _푣_ _[푙]_ _푗_ [−1)] **퐖** [(] _푑_ _[푙]_ [)] [)] (4)



where _푣_ _푖_ represents the _푖_ -th node in the graph , _퐶_ ( _푣_ _푖_ ) is the neighbours of the node _푣_ _푖_, **퐡** [(] _푣_ _[푙]_ _푖_ [)] [denotes the hidden state at]

the _푙_ -th graph convolutional layer for node _푣_ _푖_, **퐡** [(0)] _푣_ _푖_ [denotes the initial feature of node] _[ 푣]_ _푖_ [,] _[ ̂푑]_ _푣_ _푖_ [= 1+ ∣] _[퐶]_ [(] _[푣]_ _푖_ [) ∣] [denotes the]

degree of node _푣_ _푖_, **퐖** [(] _푑_ _[푙]_ [)] [is the weight parameter at the] _[ 푙]_ [-th layer. It should be noted that the self-loop is added into the]
calculation of the degree _푑_ _[̂]_ _푣_ _푖_, where the node _푣_ _푖_ itself is viewed as its 1-hop neighbour to be taken into consideration.
Through the self-loop operation, we can take full advantage of the biological features of nodes (i.e., atoms or residues)
during the GCN propagation.
In order to learn local-level molecular representations of drugs/targets, we apply the above self-loop GCN propagation framework over the drug molecular graph and the target molecular graph, respectively. To be specific, given
the drug _푑_ _푖_ and its corresponding drug molecular graph  _푑_ _푖_ = { _푎_ _,_  _푏_ }, we apply the self-loop convolutional operator
over the graph’s topology with the initial attributions { **퐱** _푎_ _푢_ } of atoms  _푎_ = { _푎_ 1 _,_ … _, 푎_ _푛_ _푎_ } as input. On this basis, we
encode multiple atomic features and chemical structure information of drug molecules into the local-level atom representations { **퐡** _푎_ _푢_ ∈ ℝ _[푑]_ _[푑푔]_ } through Equation (4). Similarly, for the target molecular graph  _푡_ _푗_ = { _푟_ _,_  _푐_ } corresponding
to the target _푡_ _푗_, we take the initial features { **퐱** _푟_ _푝_ } of residue  _푟_ = { _푟_ 1 _,_ … _, 푟_ _푛_ _푟_ } as input to obtain the local-level residue

representations { **퐡** _푟_ _푝_ ∈ ℝ _[푑]_ _[푡푔]_ } through Equation (4). Note that the parameters of GCNs on the molecular graph are
shared across all the atoms or residues.


**3.3. Integrating hierarchical graph information via message broadcasting**
After the hierarchical graph representation learning introduced above, for drug/target molecules, we obtain their
global-level representations deriving from the affinity graph and local-level representations learned from the molecular


Zhaoyang Chu et al.: _Preprint submitted to Elsevier_ Page 4 of 15


HGRL-DTA


graph, respectively. In order to integrate the hierarchical graph information, we make use of a message broadcasting
mechanism to encode the global-level representations of drugs/targets into their corresponding local-level molecular
representations, and reuse GCN propagation to refine the molecular representations. It is worth noticing that the
global-level affinity information is used to guide the representation learning of the local-level molecular graph via the
refinement process.
In the message broadcasting strategy, a sender can deliver information to plenty of recipients simultaneously. In our
model, the global-level drug/target embedding and its corresponding local-level atom/residue embeddings are regarded
a sender and recipients. Through message broadcasting, the global-level representation of a drug (or target) is shared
by all of the atoms (or residues) to update their local-level representations. Concretely, given drug embedding **퐡** _푑_ _푖_,
target embedding **퐡** _푡_ _푗_, atom representations { **퐡** _푎_ _푢_ } and residue representations { **퐡** _푟_ _푝_ }, the message broadcasting process
can be formulated as follows:


{ **퐳** _푎_ _푢_ } = { **퐡** _푎_ _푢_ _⊕_ **퐡** _푑_ _푖_ ∣∣ **퐡** _푎_ _푢_ _⊖_ **퐡** _푑_ _푖_ } _,_ { **퐳** _푟_ _푝_ } = { **퐡** _푟_ _푝_ _⊕_ **퐡** _푡_ _푗_ ∣∣ **퐡** _푟_ _푝_ _⊖_ **퐡** _푡_ _푗_ } (5)


where { **퐳** _푎_ _푢_ ∈ ℝ [2][⋅] _[푑]_ _[푑푔]_ } and { **퐳** _푟_ _푝_ ∈ ℝ [2][⋅] _[푑]_ _[푡푔]_ } denote the updated atom embeddings and residue embeddings, _⊕_ and _⊖_
denote element-wise addition and subtraction respectively, ∣∣ denotes the concatenation operation. In particular, the
element-wise addition represents the fusion of global-level and local-level information and the element-wise subtraction explores the difference between them.
Through the message broadcasting process, we can integrate the global-level affinity information into the locallevel molecular graphs to update atom and residue embeddings. Whereas, this update process may cause the chemical
structure information blurred in the molecular representations. Thus, it is necessary to reload the molecular chemical
structures in our model. For this reason, we reuse GCNs over the molecular graph’s topology to refine the updated
atom and residue embeddings. To be specific, given the updated atom embeddings { **퐳** _푎_ _푢_ } and residue embeddings

{ **퐳** _푟_ _푝_ } as input, we generate the refined representations { **̂퐳** _푎_ _푢_ ∈ ℝ _[푑]_ _푑푔_ [′] } for atoms and { **̂퐳** _푟_ _푝_ ∈ ℝ _[푑]_ _푡푔_ [′] } for residues through

Equation (4), where ℝ _[푑]_ _푑푔_ [′] and ℝ _[푑]_ _푡푔_ [′] are the representation dimensions.
We readout the final drug embedding **퐝** _푖_ from the drug molecular graph _퐺_ _푑_ _푖_ through an average pooling operation
along the atom dimension. Similarly, we readout the final target embedding **퐭** _푗_ through combining representation
vectors across all residues in the target molecular graph _퐺_ _푡_ _푗_ . The readout layer is formulated as follows:


**퐝** _푖_ = _푀퐿푃_ ( **퐖** [(1)] _푒_ _[,]_ **[ 퐖]** [(2)] _푒_ _[, 푓]_ _[푎푣푔]_ [({] **[̂퐳]** _[푎]_ _푢_ [}))] _[,]_ **[ 퐭]** _[푗]_ [=] _[ 푀퐿푃]_ [(] **[퐖]** [(1)] _푓_ _[,]_ **[ 퐖]** [(2)] _푓_ _[, 푓]_ _[푎푣푔]_ [({] **[̂퐳]** _[푟]_ _푝_ [}))] (6)


where _푓_ _푎푣푔_ denotes the average pooling operation, **퐖** [(1)] _푒_ [,] **[ 퐖]** [(2)] _푒_ [,] **[ 퐖]** [(1)] _푓_ [and] **[ 퐖]** [(2)] _푓_ [are the weight parameters of the MLP]
operators.


**3.4. Drug-target binding affinity prediction**
In this study, the binding affinity prediction problem is treated as a regression task. For each drug-target pair
( _푑_ _푖_ _, 푡_ _푗_ ) ∈ , we combine the drug embedding **퐝** _푖_ and the target embedding **퐭** _푗_ into an affinity embedding vector and
then make the final affinity prediction using a three-layer MLP:


_푦_ _푖,푗_ = _푀퐿푃_ ( **퐖** [(1)] _푔_ _[,]_ **[ 퐖]** [(2)] _푔_ _[,]_ **[ 퐖]** [(3)] _푔_ _[,]_ **[ 퐝]** _[푖]_ [∣∣] **[퐭]** _[푗]_ [)] (7)


where _푦_ _푖,푗_ is the predicted affinity value of the drug-target pair ( _푑_ _푖_ _, 푡_ _푗_ ), ∣∣ denotes the concatenation operation, **퐖** [(1)] _푔_ [,]

**퐖** [(2)] _푔_ and **퐖** [(3)] _푔_ are the weight parameters of the MLP operator.
Given a batch of drug-target pairs in the training set  and their corresponding ground-truth affinity values, we
train the weight parameters with the following MSE loss function:



 = [1]


_푚_



−
∑( _푦_ _푖,푗_ _̂푦_ _푖,푗_ ) [2] (8)



where _푚_ represents the number of drug-target pairs in a training batch, _̂푦_ _푖,푗_ denotes the ground-truth affinity value of the
drug-target pair ( _푑_ _푖_ _, 푡_ _푗_ ). Based on the loss function, we optimize the mapping function _훩_ ( _휔_ ) using the back-propagation
algorithm and the Adam [27] optimizer, to find the best solution for the trainable parameter _휔_ .


Zhaoyang Chu et al.: _Preprint submitted to Elsevier_ Page 5 of 15


HGRL-DTA


**Algorithm 1** The Hierarchical Graph Representation Learning Algorithm

**Input:** Training set  ; affinity graph  ; drug molecular graphs { _푑_ _푖_ } ~~_[푛]_~~ _푖_ =1 _[푑]_ [; target molecular graphs][ {][] _[푡]_ _푗_ [}] ~~_[푛]_~~ _푗_ =1 _[푡]_
**Output:** Mapping function _훩_ ( _휔_ )

1: **while** HGRL-DTA not converge **do**

2: **̂퐀** ← preprocess the adjacency matrix via Eq.(1)
3: **퐇** ← calculate the global-level embedding matrix via Eq.(2)
4: **for** ( _푑_ _푖_ _, 푡_ _푗_ ) ∈  **do**
5: **퐡** _푑_ _푖_ _,_ **퐡** _푡_ _푗_ ← calculate the global-level representations of the drug and the target via Eq.(3)

6: **for** _푎_ _푢_ ∈  _푎_ **do**
7: **퐡** _푎_ _푢_ ← calculate the local-level atom representation via Eq.(4)
8: **end for**

9: **for** _푟_ _푝_ ∈  _푟_ **do**
10: **퐡** _푟_ _푝_ ← calculate the local-level residue representation via Eq.(4)

11: **end for**
12: { **퐳** _푎_ _푢_ } _,_ { **퐳** _푟_ _푝_ } ← update the representations of atoms and residues via Eq.(5)

13:14: {{ **̂퐳̂퐳** _푎푟_ _푝푢_ }} ← ← refine the residue representations likerefine the atom representations like **line 6-8 line 9-11**

15: **퐝** _푖_ _,_ **퐭** _푗_ ← readout the final drug embedding and target embedding via Eq.(6)
16: _푦_ _푖,푗_ ← predict the binding affinity via Eq.(7)
17: **end for**


18:  ← calculate the loss function via Eq.(8)
19: ∇ _휔_ 
20: **end while**

21: **return** _훩_ ( _휔_ )


**4. Experiment**


In this section, we firstly present the datasets, experimental settings, and evaluation metrics used in our experiments. Then, we describe several important implementation details of our proposed model. Afterwards, we compare
the HGRL-DTA model with other state-of-the-art methods under different experimental scenarios. Finally, we make
further and deeper analyses on HGRL-DTA with ablation study, parameter analysis and visualization analysis.


**4.1. Datasets**
To evaluate the performance of our proposed model on the binding affinity prediction task, two classic benchmark
datasets, the Davis dataset [12] and the KIBA dataset [52], are chosen to conduct experiments.
**Davis.** The Davis dataset contains 68 unique drugs and 442 unique targets, with 30,056 kinase dissociation constant
_퐾_ _푑_ values as drug-target affinities. He et al. [20] transform _퐾_ _푑_ values in the Davis dataset into log space as: _푝퐾_ _푑_ =
−log 10 ( _퐾_ _푑_ ∕10 [9] ). The preprocessed Davis dataset is filled with affinities ranging from 5.0 to 10.8, where the boundary
value 5.0 is regarded as the true negative drug-target pair that either has very weak binding affinities or is not detected
in the wet lab experiment. The Davis dataset collected drugs’ SMILES strings from the PubChem compound database

[6] based on PubChem CIDs, and targets’ protein sequences from the UniProt protein database [2] according to gene
names/RefSeq accession numbers.
**KIBA.** The KIBA dataset introduces KIBA scores as drug-target affinities, based on the integration of kinase inhibitor bioactivities from various sources such as _퐾_ _푖_, _퐾_ _푑_, and _퐼퐶_ 50 [52]. The dataset originally consists of 52,498
drugs and 467 targets with 246,088 affinities. He et al. [20] filtered it to comprise 118,254 affinities between 2,111
unique drugs and 229 unique targets with 10 affinities at least of each drug and target. The preprocessed KIBA dataset
contains affinity values ranging from 0.0 to 17.2, and nan values indicating there are no experimental values for corresponding drug-target pairs. The KIBA dataset converted drugs’ CHEMBL IDs [17] into their corresponding PubChem
CIDs to extract SMILES strings based on PubChem CIDs and collected protein sequences for targets through UniProt
IDs.


Zhaoyang Chu et al.: _Preprint submitted to Elsevier_ Page 6 of 15


HGRL-DTA


**4.2. Experimental settings and evaluation metrics**
Following previous works [22, 38], we evaluate the performance of various models with the division ratio for
training and testing as 5:1. When conducting experiments, we train the proposed model and compared methods on the
training set and evaluate them on the test set. We conduct 5-fold cross validation (CV) on the training set to select the
best hyper-parameters from fixed ranges for our proposed model. For all the compared methods, their hyper-parameters
are set as the optimal values provided in the corresponding studies [22, 36, 38, 64].
In this study, to verify the generalization and robustness of the models comprehensively, we consider the following
four experimental scenarios [39]:


∙ **S1:** Entries in the drug-target matrix **퐘** are randomly selected for testing.


∙ **S2:** Row vectors in the drug-target matrix **퐘** are randomly selected for testing.


∙ **S3:** Column vectors in the drug-target matrix **퐘** are randomly selected for testing.


∙ **S4:** The intersection set of the row vectors in the setting **S2** and the column vectors in the setting **S3** is selected
for testing, and their non-intersection parts are used for neither training nor testing.


In setting **S1**, both the drug and the target of the test drug-target pairs can be observed in the training phase. As
the most widely used experimental setting in previous studies, the setting **S1** assumes that a part of known drug-target
affinities are randomly masked and our aim is to infer these masked affinities using other known ones. Compared to
setting **S1**, the settings **S2** and **S3** have attracted more attention in real-world applications recently, where only part
of drug/target information is available during training and the models need to predict affinities for new drugs/targets
which do not bind any known affinities. The setting **S4** corresponds to the most challenging case in computational
works, which aims to predict affinities between unknown drugs and targets.
We choose four classic metrics to evaluate the performance of the models: Mean Squared Error (MSE), Concordance Index (CI) [19], _푟_ [2] _푚_ [[][42][,][ 47][] and Pearson correlation coefficient (Pearson) [][5][]. For each model, we report the]
average and the standard deviation (std) of these indicators across ten random runs.


**4.3. Implementation details**
In this subsection, we present the important implementation details of the proposed model, which correspond to
GNNs’ practical issues including constructing graphs, inferring representations for unseen nodes, and preventing overfitting and over-smoothing. Then we describe the runtime environment for modeling and conducting experiments.


_**4.3.1. Constructing graphs**_
In the following, we introduce the construction process of the input graphs, i.e., the affinity graph, the drug molecular graph, and the target molecular graph, defined in Section 2.
**Affinity graph.** For the affinity graph, we normalize raw affinity values into the range [0, 1] using min-max
normalization and formulate normalized affinities as its corresponding adjacency matrix. Each node (i.e., drug or
target) in the affinity graph is encoded as a multi-dimensional binary feature vector, which consists of two pieces of
information: one-hot encoding of the node type (i.e., either drug-type or target-type) and one-hot encoding of the
neighbour nodes (i.e., row vector in the connectivity matrix corresponding to the affinity graph).
**Drug molecular graph.** We transform drugs’ SMILES (Simplified Molecular Input Line Entry System) strings

[60], which were invented to represent molecules to be readable by computers, into their corresponding graphs with
the open-source chemical informatics software RDKit [29]. A group of atomic features adopted from DeepChem [44]
is used as the initial drug molecular graph signals.
**Target molecular graph.** Targets’ complex folded structure are composed of a variety of spatial characteristics,
which contain residues, peptide bonds, and non-bonded interactions such as hydrogen bonds and van der Waals forces.
However, obtaining the tertiary structures of protein by crystallization in the laboratory is time-consuming and costly,
which results in that there exist a large number of protein structures unavailable. To handle this issue, Pconsc4 [35],
an open-source and highly efficient protein structure prediction approach, is adopted in our work to generate target
molecular graphs through mining useful topological information hidden in protein sequences. The Pconsc4 method
transforms targets’ protein sequences into their corresponding contact maps, i.e., residue-residue association matrixes,
whose entries are the Euclidean distance-based contacts. In this contact map, there exists a contact between two atoms
if the Euclidean distance between them is less than a specified threshold [61]. We set the threshold as 0.5 according
to the previous study DGraphDTA [22]. A set of residue features extracted by DGraphDTA [22] is used as the initial
attributions of residues in the target molecular graph.


Zhaoyang Chu et al.: _Preprint submitted to Elsevier_ Page 7 of 15


HGRL-DTA


_**4.3.2. Inferring representations for unseen nodes**_
In view of the affinity graph, the GCN encoder works based on the graph connectivity of the affinity data. However,
there exists a challenge that this encoder cannot learn representations for unseen nodes (i.e., unknown drugs/targets).
Such a cold start problem occurs when conducting experiments in the settings **S2**, **S3**, and **S4**, where a part of drugs or
targets are not observed in the training phase. This problem limits the generalization ability of our model for unseen
drugs and targets.
To solve this problem, we introduce drug-drug similarities and target-target similarities to construct a **similarity-**
**based embedding map** from known to unknown drugs/targets, which can infer global-level representations for unknown drugs/targets during testing. In detail, we firstly calculate structural fingerprint similarities between known
and unknown drugs using PubChem Score Matrix Service [6] and sequence similarities between known and unknown
targets using Smith-Waterman (SW) algorithm [49], respectively. Then, take the drug as an example (target is similar),
we consider a bipartite similarity graph, in which each unknown drug is connected to its _푠푖푚퐾_ most similar known
drugs. We implement the embedding map to infer an unknown drug by aggregating the learned global-level representations of its neighbours with a weighted summation operation, which utilizes normalized similarities as weights.
Through this map, we can infer the representations of unseen drugs and targets to generalize our model for the cold
start situation. In this paper, we set the _푠푖푚퐾_ of drug _푠푖푚퐾_ _푑_ = 2 and target _푠푖푚퐾_ _푡_ = 7, respectively.
Compared to the learned representations of known drugs and targets, the inferred representations of unseen drugs
and targets contain less topological affinity information. Moreover, the model may lose some global-level affinity
information when integrating the hierarchical graph representations. Therefore, in order to allow sufficient globallevel affinity information to be retained, we add a skip connection from the global-level drug/target embedding to the
readout layer. It should be noted that we only conduct the above operations under settings **S2**, **S3**, and **S4** .


_**4.3.3. Preventing over-fitting and over-smoothing**_
Over-fitting and over-smoothing frequently occur during training GNN-based models. To alleviate these issues, we
introduce a regularization technique DropEdge [45], which randomly removes a certain number of edges from the input
graph at each training epoch, into the graph convolutional process over the affinity graph. Also, it has been theoretically
demonstrated that DropEdge either reduces the convergence speed of over-smoothing or relieves the information loss
caused by it [45]. In this paper, we set the DropEdge rate _훼_ = 0 _._ 2.
In addition, it has been theoretically and empirically proved that nodes with high degrees are more likely to suffer
from over-smoothing in multi-layer GNN-based models [9]. In the KIBA dataset, we observe that each target connects
with too many (on average 518 and up to 1,452) drugs, which causes the over-smoothing problem occurring. To handle
this issue, we selectively dropout edges of the affinity graph in the data preprocessing phase, which only preserves the
_푡표푝퐾_ highest affinity edges related to each target and removes other connected ones. In this paper, we set the _푡표푝퐾_ of
target _푡표푝퐾_ _푡_ = 150 under settings **S1**, **S3** and _푡표푝퐾_ _푡_ = 90 under settings **S2**, **S4** . Without loss of generality, we perform
the same operation for each drug and set the _푡표푝퐾_ of drug _푡표푝퐾_ _푑_ = 40. Note that we selectively remove affinities only
when conducting experiments on the KIBA dataset.


We implement our proposed model with Pytorch 1.4.0 [40] and Pytorch-geometric 1.7.0 [13]. We run HGRLDTA on our workstation with 2 Intel(R) Xeon(R) Gold 6146 3.20GHZ CPUs, 128GB RAM, and 2 NVIDIA 1080 Ti
GPUs. For more detailed parameter settings of HGRL-DTA, please refer to the source code: `[https://github.com/](https://github.com/Zhaoyang-Chu/HGRL-DTA)`
`[Zhaoyang-Chu/HGRL-DTA](https://github.com/Zhaoyang-Chu/HGRL-DTA)` .


**4.4. Comparison with state-of-the-art methods**
To demonstrate the superiority of the proposed model, we conduct experiments to compare our approach with the
following state-of-the-art methods:


∙ **DeepDTA** [38] employs CNNs to learn 1D drug and target representations from drug SMILES strings and
target protein sequences.


∙ **AttentionDTA** [64] utilizes 1D CNNs to learn sequence representations of drugs and targets and an attention
mechanism to find the weight relationships between drug subsequences and protein subsequences.


∙ **GraphDTA** [36] models drugs as molecular graphs to capture the bonds among atoms with GNNs and leverages CNNs to learn 1D representations of target proteins.


Zhaoyang Chu et al.: _Preprint submitted to Elsevier_ Page 8 of 15


HGRL-DTA


**Table 1**

Performances of HGRL-DTA and compared methods.


Davis KIBA
Architecture

MSE↓ (std) CI↑ (std) _푟_ [2] _푚_ [↑] [(std)] Pearson↑ (std) MSE↓ (std) CI↑ (std) _푟_ [2] _푚_ [↑] [(std)] Pearson↑ (std)



**S1**


**S2**


**S3**


**S4**



DeepDTA 0.245 (0.014) 0.888 (0.004) 0.665 (0.015) 0.842 (0.004) 0.181 (0.007) 0.868 (0.004) 0.711 (0.021) 0.864 (0.003)
AttentionDTA 0.233 (0.006) 0.889 (0.002) 0.676 (0.020) 0.845 (0.004) 0.150 (0.002) 0.883 (0.001) 0.760 (0.018) 0.888 (0.001)
GraphDTA 0.243 (0.005) 0.887 (0.002) 0.685 (0.016) 0.839 (0.003) 0.148 (0.006) 0.891 (0.001) 0.730 (0.015) 0.895 (0.001)
DGraphDTA 0.216 (0.003) 0.900 (0.001) 0.686 (0.015) 0.857 (0.002) 0.132 (0.002) 0.902 (0.001) **0.800 (0.011)** 0.903 (0.001)
HGRL-DTA **0.166 (0.002)** **0.911 (0.002)** **0.751 (0.006)** **0.892 (0.001)** **0.125 (0.001)** **0.906 (0.001)** 0.789 (0.017) **0.907 (0.001)**


DeepDTA 0.985 (0.114) 0.548 (0.045) 0.027 (0.022) 0.126 (0.109) 0.494 (0.070) 0.747 (0.012) 0.337 (0.026) 0.623 (0.023)
AttentionDTA 0.869 (0.053) 0.642 (0.028) 0.079 (0.024) 0.289 (0.048) 0.506 (0.018) 0.744 (0.005) 0.298 (0.015) 0.618 (0.006)
GraphDTA 0.801 (0.038) 0.659 (0.015) 0.160 (0.019) 0.416 (0.022) 0.475 (0.047) 0.753 (0.002) **0.382 (0.007)** 0.652 (0.002)
DGraphDTA 0.818 (0.012) 0.646 (0.006) 0.114 (0.005) 0.356 (0.010) 0.458 (0.008) 0.754 (0.002) 0.362 (0.012) 0.622 (0.004)
HGRL-DTA **0.776 (0.019)** **0.684 (0.007)** **0.163 (0.015)** **0.422 (0.018)** **0.434 (0.007)** **0.757 (0.003)** 0.370 (0.010) **0.653 (0.003)**


DeepDTA 0.552 (0.086) 0.729 (0.017) 0.258 (0.029) 0.523 (0.028) 0.732 (0.197) 0.676 (0.016) 0.273 (0.026) 0.587 (0.033)
AttentionDTA 0.436 (0.017) 0.787 (0.018) 0.304 (0.022) 0.588 (0.027) 0.529 (0.039) 0.693 (0.008) 0.254 (0.024) 0.592 (0.022)
GraphDTA 0.860 (0.083) 0.666 (0.012) 0.134 (0.014) 0.379 (0.018) 0.469 (0.089) 0.710 (0.005) 0.388 (0.013) 0.627 (0.009)
DGraphDTA 0.445 (0.019) 0.788 (0.009) 0.289 (0.016) 0.558 (0.017) 0.364 (0.010) 0.718 (0.007) 0.429 (0.022) 0.671 (0.009)
HGRL-DTA **0.383 (0.010)** **0.816 (0.008)** **0.375 (0.018)** **0.621 (0.012)** **0.322 (0.014)** **0.741 (0.004)** **0.502 (0.016)** **0.729 (0.007)**


DeepDTA 0.767 (0.091) 0.508 (0.057) 0.009 (0.012) 0.015 (0.098) 0.700 (0.075) 0.627 (0.009) 0.140 (0.017) 0.401 (0.025)
AttentionDTA 0.679 (0.021) 0.554 (0.030) 0.005 (0.008) 0.036 (0.062) 0.609 (0.021) 0.629 (0.007) 0.143 (0.015) 0.407 (0.022)
GraphDTA 0.988 (0.096) 0.569 (0.017) 0.020 (0.006) 0.141 (0.020) 0.676 (0.113) 0.641 (0.003) 0.149 (0.007) 0.404 (0.009)
DGraphDTA 0.658 (0.026) 0.569 (0.008) 0.031 (0.005) 0.180 (0.015) 0.594 (0.022) 0.632 (0.009) 0.148 (0.013) 0.403 (0.019)
HGRL-DTA **0.642 (0.016)** **0.602 (0.009)** **0.044 (0.005)** **0.215 (0.013)** **0.532 (0.008)** **0.642 (0.004)** **0.207 (0.009)** **0.491 (0.010)**


∙ **DGraphDTA** [22] constructs target molecular graphs from the corresponding protein sequences via the protein
structure prediction method and applies GNNs to mine structural information hidden in drug molecular graphs
and target molecular graphs.



Table 1 compares our HGRL-DTA model’s performance against the state-of-the-art methods on the two benchmark
datasets under four experimental scenarios. We highlight the best results in boldface. According to the experimental
results, we can observe that the proposed HGRL-DTA model achieves the best performance compared to the stateof-the-art methods under all the scenarios, which demonstrates the generalization and robustness of our model. In
four experimental settings, over the best baseline models, we achieve 23.1%, 3.1%, 12.2% and 2.4% improvement of
MSE on the Davis dataset, 5.3%, 5.2%, 11.5% and 10.4% improvement of MSE on the KIBA dataset. In most cases,
the standard deviations of the HGRL-DTA model are lower than other compared methods, which demonstrates the
stability of the predictive model.
Among all baselines, the performance of the sequence-based methods (i.e., DeepDTA, AttentionDTA) is relatively
poor due to the inadequate exploitation of the molecular chemical structures. It indicates that simply modeling drugs
as SMILES strings and targets as protein sequences is not a natural way to capture the intrinsic properties of molecules.
By contrast, the graph-based models (i.e., GraphDTA and DGraphDTA) represent molecules as molecular graphs to
take advantage of their chemical structural information, which produces better predictive performance. However, these
graph-based models mainly focus on encoding molecular structures but ignore the abundant topological information
deriving from drug-target affinity relationships, which causes them inferior to our proposed model. Compared with
the state-of-the-art models, HGRL-DTA can capture the local-level intrinsic molecular properties and the global-level
topological affinity relationships simultaneously and incorporate such hierarchical information into the representations
of drugs and targets, which significantly facilitates the performance of predicting drug-target binding affinities.
In addition, we observe that all the models perform best in setting **S1**, have relatively poor performance in settings
**S2** and **S3**, and perform worst in setting **S4** . With more unknown drugs or targets in the four experimental settings, the
predictive performance of the models significantly declines. Different from setting **S1**, the settings **S2**, **S3** and **S4** test
the generalization and robustness of the models for unseen drugs or targets, which is another necessary measurement
of performance evaluation. Through the similarity-based embedding map, HGRL-DTA infers unseen drugs/targets
using the learned embeddings of known drugs/targets, which can make full use of the known affinity and similarity
information to improve the generalization and robustness of the model. As illustrate in Table 1, the HGRL-DTA model
obtains the best performance in the settings **S2**, **S3** and **S4**, which indicates that HGRL-DTA is more generalizable and
more robust compared to baseline methods when only part of drug/target information is known.


Zhaoyang Chu et al.: _Preprint submitted to Elsevier_ Page 9 of 15


HGRL-DTA


**4.5. Ablation study**
To investigate the important factors that impact the predictive capacity of our model, we conduct the ablation study
with the following variants of HGRL-DTA in setting **S1** :


∙
**HGRL-DTA without global-level affinity graph** (w/o GAG) only learns local-level representations on the
molecular graph without the GCN propagation on the global-level affinity graph. Note that this variant keeps
the same number of GCN iterations on the molecular graph as HGRL-DTA, where GCN iterations in the
refinement procedure of HGRL-DTA are also considered.


∙
**HGRL-DTA without local-level molecular graph** (w/o LMG) only applies the GCN propagation on the
global-level affinity graph without considering the local-level molecular graphs. The MLP-based predictor is
directly applied with the input of the global-level representations of drugs and targets for the binding affinity
prediction task.


∙
**HGRL-DTA without weighted affinities** (w/o WA) addresses the affinity graph as an unweighted graph,
which only considers binary relationships instead of continuous affinities.


∙
**HGRL-DTA without message broadcasting** (w/o MB) extracts the global-level representations from the
affinity graph, readouts the local-level representations from the molecular graph, and then integrates them
using the combination of element-wise addition, element-wise subtraction, and concatenation. Different from
HGRL-DTA, this variant learns representations from the affinity graph and the molecular graph separately,
without using the global-level affinity information to guide learning local-level molecular properties. Note
that this variant keeps the same number of GCN iterations on the molecular graph as HGRL-DTA.













Figure 2 compares HGRL-DTA with its four variants on the two benchmark datasets. Overall, the proposed HGRLDTA outperforms other variants, which demonstrates the effectiveness of the hierarchical graph learning architecture.
In detail, HGRL-DTA (w/o GAG) and HGRL-DTA (w/o LMG) have the most significant performance gaps with
HGRL-DTA. These results suggest that global-level and local-level components contribute the most to our model, and
removing either component will undermine its predictive performance. Besides, HGRL-DTA (w/o WA) performances
worse than HGRL-DTA since it only constructs the affinity graph using binary relationships, which loses more realistic
information hidden in continuous affinities. The deprecation of the message broadcasting mechanism in HGRL-DTA

|0.908|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0.890<br>0.893<br>0.896<br>0.899<br>0.902<br>0.905<br><br>0.850<br>0.859<br>0.868<br>0.877<br>0.886<br>0.895<br>0.904<br> the tw<br>ﬀectiv<br> LMG<br>al-lev<br>rform<br>graph<br>n of t||||||||||||||
|0.890<br>0.893<br>0.896<br>0.899<br>0.902<br>0.905<br><br>0.850<br>0.859<br>0.868<br>0.877<br>0.886<br>0.895<br>0.904<br> the tw<br>ﬀectiv<br> LMG<br>al-lev<br>rform<br>graph<br>n of t||||||||||||||
|0.890<br>0.893<br>0.896<br>0.899<br>0.902<br>0.905<br><br>0.850<br>0.859<br>0.868<br>0.877<br>0.886<br>0.895<br>0.904<br> the tw<br>ﬀectiv<br> LMG<br>al-lev<br>rform<br>graph<br>n of t||||||||||||||
|0.890<br>0.893<br>0.896<br>0.899<br>0.902<br>0.905<br><br>0.850<br>0.859<br>0.868<br>0.877<br>0.886<br>0.895<br>0.904<br> the tw<br>ﬀectiv<br> LMG<br>al-lev<br>rform<br>graph<br>n of t||||Dav|is|Pea|rson||K|IBA||||
|0.890<br>0.893<br>0.896<br>0.899<br>0.902<br>0.905<br><br>0.850<br>0.859<br>0.868<br>0.877<br>0.886<br>0.895<br>0.904<br> the tw<br>ﬀectiv<br> LMG<br>al-lev<br>rform<br>graph<br>n of t||||||||||||||
|0.890<br>0.893<br>0.896<br>0.899<br>0.902<br>0.905<br><br>0.850<br>0.859<br>0.868<br>0.877<br>0.886<br>0.895<br>0.904<br> the tw<br>ﬀectiv<br> LMG<br>al-lev<br>rform<br>graph<br>n of t||||||||||||||
|0.890<br>0.893<br>0.896<br>0.899<br>0.902<br>0.905<br><br>0.850<br>0.859<br>0.868<br>0.877<br>0.886<br>0.895<br>0.904<br> the tw<br>ﬀectiv<br> LMG<br>al-lev<br>rform<br>graph<br>n of t||o b<br>en<br>)<br>el<br>an<br>usi<br>he|en<br>ess<br>hav<br>co<br>ce.<br>ng<br>me|Dav<br>ch<br> o<br>e<br>mp<br> B<br>bi<br>ss|is<br>ma<br>f th<br>the<br>one<br>esid<br>nar<br>age|rk d<br>e hi<br> mo<br>nts<br>es,<br>y re<br> bro|ata<br>era<br>st<br> co<br> H<br>lat<br>ad|set<br>rch<br>sig<br>ntr<br>GR<br>ion<br>cas|K<br>s.<br>ica<br>niﬁ<br>ibu<br>L-<br>shi<br>tin|IBA<br>Ov<br>l g<br>ca<br>te<br>DT<br>ps<br>g|era<br>rap<br>nt<br>the<br>A<br>, w<br>me|ll,<br>h<br>pe<br> m<br>(w/<br>hic<br>cha|th<br>le<br>rf<br>o<br>o<br>h<br>n|



Zhaoyang Chu et al.: _Preprint submitted to Elsevier_ Page 10 of 15


|0.185 0.202|Col2|Col3|Col4|Col5|Col6|Col7|HGRL-DTA HGRL-DTA (w/o GAG) HGRL-DTA (w/o LMG) HGRL-DTA (w/o WA)|Col9|Col10|Col11|Col12|Col13|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0.100<br>0.117<br>0.134<br>0.151<br>0.168<br>0.185<br>0.500<br>0.545<br>0.590<br>0.635<br>0.680<br>0.725<br>0.770<br>f ablation<br>res HGR<br>other var<br>DTA (w/<br>e results<br>mponent<br>DTA sin<br> in conti|||||||HGRL~~-~~DTA (w/o WA)<br>HGRL~~-~~DTA (w/o MB)|HGRL~~-~~DTA (w/o WA)<br>HGRL~~-~~DTA (w/o MB)|HGRL~~-~~DTA (w/o WA)<br>HGRL~~-~~DTA (w/o MB)|HGRL~~-~~DTA (w/o WA)<br>HGRL~~-~~DTA (w/o MB)|HGRL~~-~~DTA (w/o WA)<br>HGRL~~-~~DTA (w/o MB)|HGRL~~-~~DTA (w/o WA)<br>HGRL~~-~~DTA (w/o MB)|
|0.100<br>0.117<br>0.134<br>0.151<br>0.168<br>0.185<br>0.500<br>0.545<br>0.590<br>0.635<br>0.680<br>0.725<br>0.770<br>f ablation<br>res HGR<br>other var<br>DTA (w/<br>e results<br>mponent<br>DTA sin<br> in conti|||||||HGRL~~-~~DTA (w/o WA)<br>HGRL~~-~~DTA (w/o MB)||||||
|0.100<br>0.117<br>0.134<br>0.151<br>0.168<br>0.185<br>0.500<br>0.545<br>0.590<br>0.635<br>0.680<br>0.725<br>0.770<br>f ablation<br>res HGR<br>other var<br>DTA (w/<br>e results<br>mponent<br>DTA sin<br> in conti||||Dav|is|r2<br>m||K|IBA||||
|0.100<br>0.117<br>0.134<br>0.151<br>0.168<br>0.185<br>0.500<br>0.545<br>0.590<br>0.635<br>0.680<br>0.725<br>0.770<br>f ablation<br>res HGR<br>other var<br>DTA (w/<br>e results<br>mponent<br>DTA sin<br> in conti|||||||||||||
|0.100<br>0.117<br>0.134<br>0.151<br>0.168<br>0.185<br>0.500<br>0.545<br>0.590<br>0.635<br>0.680<br>0.725<br>0.770<br>f ablation<br>res HGR<br>other var<br>DTA (w/<br>e results<br>mponent<br>DTA sin<br> in conti|||||||||||||
|0.100<br>0.117<br>0.134<br>0.151<br>0.168<br>0.185<br>0.500<br>0.545<br>0.590<br>0.635<br>0.680<br>0.725<br>0.770<br>f ablation<br>res HGR<br>other var<br>DTA (w/<br>e results<br>mponent<br>DTA sin<br> in conti||ion<br>GR<br>var<br>w/<br>lts<br>ent<br>sin<br>nti|ex<br>L-<br>ian<br>o G<br>su<br> wi<br>ce<br>nu|Dav<br>p<br>D<br>ts<br>A<br>gg<br>ll<br>it<br>ou|is<br>erim<br>TA<br>, wh<br>G)<br>est<br>und<br>only<br>s a|ents<br>with<br>ich<br> and<br>that<br>erm<br> con<br>ﬃniti|.<br> its fo<br>demo<br> HGR<br>globa<br>ine it<br>struc<br>es. T|K<br>ur<br>nst<br>L-<br>l-le<br>s pr<br>ts t<br>he|IBA<br>var<br>rat<br>D<br>ve<br>ed<br>he<br> de|ia<br>es<br>TA<br>l a<br>ict<br>a<br>pr|nts<br> th<br> (<br>nd<br>ive<br>ﬃni<br>eca|o<br>e<br>w<br> l<br> <br>t<br>t|
|l.:_ Pre_|l.:_ Pre_|_pri_|_nt s_|_ub_|_mit_|_ted to_|_ Else_|_vier_|||||


HGRL-DTA


(w/o MB) also leads to performance reduction. It indicates that this mechanism is more effective to integrate the
global-level affinity information and the local-level molecular properties.


**4.6. Parameter analysis**
To further validate the effectiveness of the similarity-based embedding map for inferring unseen nodes, we analyze
the impacts of two major hyper-parameters _푠푖푚퐾_ _푑_ and _푠푖푚퐾_ _푡_ used in this map.
We conduct the parameter study experiment on the Davis dataset by changing the hyper-parameters _푠푖푚퐾_ _푑_ and
_푠푖푚퐾_ _푡_ from 1 to 8 while keeping other hyper-parameters fixed as default settings. We test _푠푖푚퐾_ _푑_ under setting **S2**
where drug is unseen and _푠푖푚퐾_ _푡_ under setting **S3** where target is unknown, respectively. To more directly analyze
the effect of inferred representations of nodes absent in the affinity graph, we observe the performance variation of
HGRL-DTA w/o LMG, which is a variant of HGRL-DTA only learning global-level representations on the affinity
graph for binding affinity predictions.
As shown in Figure 3, the similarity-based embedding map influences the predictive performance of HGRL-DTA
w/o LMG by changing _푠푖푚퐾_ _푑_ and _푠푖푚퐾_ _푡_ . We can see that the model performs best when _푠푖푚퐾_ _푑_ = 2 and _푠푖푚퐾_ _푡_ = 7.
With the increase of _푠푖푚퐾_ _푑_ or _푠푖푚퐾_ _푡_, aggregating more known drug/target embeddings to infer unseen drugs/targets
can encode more useful information, which leads to dramatic performance improvements. When _푠푖푚퐾_ _푑_ or _푠푖푚퐾_ _푡_
reaches its optimal value, the performance begins to decline because the aggregation of too many embeddings may
introduce redundant and noisy information that can harm the predictive capacity. Furthermore, the non-zero choices of
_푠푖푚퐾_ _푑_ and _푠푖푚퐾_ _푡_ demonstrate the importance of utilizing the similarity-based embedding map to infer unseen nodes
in our method.















**Figure 3:** Parameter study of _푠푖푚퐾_ _푑_ and _푠푖푚퐾_ _푡_ for inferring unseen nodes.


In this subsection, we design an additional experiment to explore the representation power of the proposed model

To simplify the discussion, we divide affinities into two clusters through predefined thresholds provided in the
previous studies [20, 52], where the _푝퐾_ _푑_ value 7 and the KIBA score 12.1 are selected as thresholds for the Davis
dataset and the KIBA dataset, respectively. Affinities below the predefined threshold are classified as weak ones and
above as strong ones. It should be noted that such division is conducted on the test sets of the two benchmark datasets in

|0.625<br>0.611<br>0.597<br>0.583<br>0.569<br>0.555<br>0<br>0.446<br>0.440<br>0.434<br>0.428<br>0.422<br>0.416<br>0.410<br>0<br>igure 3<br>.7. V<br>In t<br>rom th<br>To<br>reviou<br>ataset<br>bove a|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|~~0~~<br><br>0.555<br>0.569<br>0.583<br>0.597<br>0.611<br>0.625<br>~~0~~<br><br>0.410<br>0.416<br>0.422<br>0.428<br>0.434<br>0.440<br>0.446<br>**igure 3**<br>**.7. V**<br>In t<br>rom th<br>To<br>reviou<br>ataset<br>bove a||||||||||
|~~0~~<br><br>0.555<br>0.569<br>0.583<br>0.597<br>0.611<br>0.625<br>~~0~~<br><br>0.410<br>0.416<br>0.422<br>0.428<br>0.434<br>0.440<br>0.446<br>**igure 3**<br>**.7. V**<br>In t<br>rom th<br>To<br>reviou<br>ataset<br>bove a||~~1~~|~~2~~|~~3~~<br>~~4~~<br>sim<br>MS|~~5~~<br>Kd<br>E|~~6~~|~~7~~|~~8~~||
|~~0~~<br><br>0.555<br>0.569<br>0.583<br>0.597<br>0.611<br>0.625<br>~~0~~<br><br>0.410<br>0.416<br>0.422<br>0.428<br>0.434<br>0.440<br>0.446<br>**igure 3**<br>**.7. V**<br>In t<br>rom th<br>To<br>reviou<br>ataset<br>bove a||||||||||
|~~0~~<br><br>0.555<br>0.569<br>0.583<br>0.597<br>0.611<br>0.625<br>~~0~~<br><br>0.410<br>0.416<br>0.422<br>0.428<br>0.434<br>0.440<br>0.446<br>**igure 3**<br>**.7. V**<br>In t<br>rom th<br>To<br>reviou<br>ataset<br>bove a||||||||||
|~~0~~<br><br>0.555<br>0.569<br>0.583<br>0.597<br>0.611<br>0.625<br>~~0~~<br><br>0.410<br>0.416<br>0.422<br>0.428<br>0.434<br>0.440<br>0.446<br>**igure 3**<br>**.7. V**<br>In t<br>rom th<br>To<br>reviou<br>ataset<br>bove a||~~1~~<br>**:** <br>**is**<br>his<br>e v<br>sim<br>s s<br>an<br>s s|~~2~~<br>Par<br>**ual**<br> su<br>ie<br>pl<br>tu<br>d t<br>tro|~~3~~<br>~~4~~<br>si<br>ame<br>**izat**<br>bsec<br>w of<br>ify t<br>dies<br>he K<br>ng o|~~5~~<br>mKt<br>ter<br>**ion**<br>tion<br> aﬃ<br>he<br>[20,<br>IBA<br>nes.|~~6~~<br>stu<br>** a**<br>, w<br>nit<br>dis<br> 5<br> d<br> It|~~7~~<br>dy<br>**nal**<br>e<br>y r<br>cus<br>2],<br>ata<br>sh|~~8~~<br>of<br>**y**<br>de<br>ep<br>si<br>w<br>s<br>ou|<br>**s**<br>s<br>r<br>o<br><br>et<br>l|


|0.740<br>0.735<br>0.730<br>0.725<br>0.720<br>0.715<br>0 1<br>0.803<br>0.799<br>0.795<br>0.791<br>0.787<br>0.783<br>0.779<br>0.775<br>0 1<br>푖푚퐾 and<br>푑<br>s<br>gn an ad<br>sentation<br>n, we div<br>ere the 푝<br>respectiv<br>be noted|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|~~0~~<br>~~1~~<br>0.715<br>0.720<br>0.725<br>0.730<br>0.735<br>0.740<br><br>~~0~~<br>~~1~~<br>0.775<br>0.779<br>0.783<br>0.787<br>0.791<br>0.795<br>0.799<br>0.803<br>_푖푚퐾푑_and<br>**s**<br>gn an ad<br>sentation<br>n, we div<br>ere the_ 푝_<br> respectiv<br> be noted||~~1~~|~~2~~|~~3~~<br>~~4~~<br>sim<br>CI|~~5~~<br>Kd<br>|~~6~~|~~7~~|~~8~~||
|~~0~~<br>~~1~~<br>0.715<br>0.720<br>0.725<br>0.730<br>0.735<br>0.740<br><br>~~0~~<br>~~1~~<br>0.775<br>0.779<br>0.783<br>0.787<br>0.791<br>0.795<br>0.799<br>0.803<br>_푖푚퐾푑_and<br>**s**<br>gn an ad<br>sentation<br>n, we div<br>ere the_ 푝_<br> respectiv<br> be noted||||||||||
|~~0~~<br>~~1~~<br>0.715<br>0.720<br>0.725<br>0.730<br>0.735<br>0.740<br><br>~~0~~<br>~~1~~<br>0.775<br>0.779<br>0.783<br>0.787<br>0.791<br>0.795<br>0.799<br>0.803<br>_푖푚퐾푑_and<br>**s**<br>gn an ad<br>sentation<br>n, we div<br>ere the_ 푝_<br> respectiv<br> be noted||~~1~~<br>nd<br>ad<br>on<br>iv<br>_ 푝_<br>tiv<br>ed|~~2~~<br>_ 푠푖_<br>diti<br>s.<br>id<br>_퐾푑_<br>el<br> th|~~3~~<br>~~4~~<br>si<br>_푚퐾푡_<br>onal<br>e aﬃ<br>val<br>y. A<br>at su|~~5~~<br>mKt<br>for<br> exp<br>niti<br>ue 7<br>ﬃni<br>ch d|~~6~~<br>inf<br>eri<br>es<br> an<br>tie<br>ivi|~~7~~<br>erri<br>m<br>int<br>d<br>s b<br>sio|~~8~~<br>n<br>en<br>o<br>th<br>el<br>n|g<br>t<br>t<br>e<br>o<br> i|


|0.284|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|~~0~~<br>~~1~~<br>0.170<br>0.189<br>0.208<br>0.227<br>0.246<br>0.265<br>~~0~~<br>~~1~~<br>0.300<br>0.309<br>0.318<br>0.327<br>0.336<br>0.345<br>0.354<br>unseen no<br>o explor<br>o cluste<br>KIBA sc<br>w the pre<br> conducte||||||||||
|~~0~~<br>~~1~~<br>0.170<br>0.189<br>0.208<br>0.227<br>0.246<br>0.265<br>~~0~~<br>~~1~~<br>0.300<br>0.309<br>0.318<br>0.327<br>0.336<br>0.345<br>0.354<br>unseen no<br>o explor<br>o cluste<br>KIBA sc<br>w the pre<br> conducte||||||||||
|~~0~~<br>~~1~~<br>0.170<br>0.189<br>0.208<br>0.227<br>0.246<br>0.265<br>~~0~~<br>~~1~~<br>0.300<br>0.309<br>0.318<br>0.327<br>0.336<br>0.345<br>0.354<br>unseen no<br>o explor<br>o cluste<br>KIBA sc<br>w the pre<br> conducte||~~1~~|~~2~~|~~3~~<br>~~4~~<br>si<br>r2<br>m|~~5~~<br>mKd<br>|~~6~~|~~7~~|~~8~~||
|~~0~~<br>~~1~~<br>0.170<br>0.189<br>0.208<br>0.227<br>0.246<br>0.265<br>~~0~~<br>~~1~~<br>0.300<br>0.309<br>0.318<br>0.327<br>0.336<br>0.345<br>0.354<br>unseen no<br>o explor<br>o cluste<br>KIBA sc<br>w the pre<br> conducte||||||||||
|~~0~~<br>~~1~~<br>0.170<br>0.189<br>0.208<br>0.227<br>0.246<br>0.265<br>~~0~~<br>~~1~~<br>0.300<br>0.309<br>0.318<br>0.327<br>0.336<br>0.345<br>0.354<br>unseen no<br>o explor<br>o cluste<br>KIBA sc<br>w the pre<br> conducte||~~1~~<br>no<br>or<br>te<br>sc<br>re<br>cte|~~2~~<br>de<br>e th<br>rs t<br>ore<br>deﬁ<br>d|~~3~~<br>~~4~~<br>si<br>s.<br>e re<br>hro<br> 12.<br>ned<br>on th|~~5~~<br>mKt<br>pres<br>ugh<br>1 ar<br> thr<br>e te|~~6~~<br>en<br>pre<br>e s<br>esh<br>st s|~~7~~<br>tat<br>de<br>ele<br>ol<br>ets|~~8~~<br>io<br>ﬁ<br>c<br>d<br> o|n<br>n<br>t<br>a<br>f|


|0.535 0.555|Col2|Col3|Pear|rson|Col6|Col7|Col8|Col9|
|---|---|---|---|---|---|---|---|---|
|~~0~~<br>~~1~~<br>0.415<br>0.435<br>0.455<br>0.475<br>0.495<br>0.515<br><br>~~0~~<br>~~1~~<br>0.560<br>0.567<br>0.574<br>0.581<br>0.588<br>0.595<br>0.602<br>0.609<br>power of<br>d thresh<br>d as thres<br>e classiﬁe<br>he two be|||||||||
|~~0~~<br>~~1~~<br>0.415<br>0.435<br>0.455<br>0.475<br>0.495<br>0.515<br><br>~~0~~<br>~~1~~<br>0.560<br>0.567<br>0.574<br>0.581<br>0.588<br>0.595<br>0.602<br>0.609<br>power of<br>d thresh<br>d as thres<br>e classiﬁe<br>he two be||~~1~~|~~2~~<br>~~3~~<br><br>si<br>Pea|~~5~~<br>mKd<br>rson|~~6~~|~~7~~|~~8~~||
|~~0~~<br>~~1~~<br>0.415<br>0.435<br>0.455<br>0.475<br>0.495<br>0.515<br><br>~~0~~<br>~~1~~<br>0.560<br>0.567<br>0.574<br>0.581<br>0.588<br>0.595<br>0.602<br>0.609<br>power of<br>d thresh<br>d as thres<br>e classiﬁe<br>he two be|||||||||
|~~0~~<br>~~1~~<br>0.415<br>0.435<br>0.455<br>0.475<br>0.495<br>0.515<br><br>~~0~~<br>~~1~~<br>0.560<br>0.567<br>0.574<br>0.581<br>0.588<br>0.595<br>0.602<br>0.609<br>power of<br>d thresh<br>d as thres<br>e classiﬁe<br>he two be||~~1~~<br>of<br>h<br>res<br>ﬁe<br>|~~2~~<br>~~3~~<br><br>s<br>the pr<br>olds pr<br>holds<br>d as w<br>|~~5~~<br>imKt<br>opos<br>ovid<br> for<br>eak<br>|~~6~~<br>ed<br>ed<br>the<br> on<br>|~~7~~<br>mo<br>in<br> D<br>es<br>|~~8~~<br>d<br> t<br>av<br>a<br>|e<br>h<br>i<br>n<br>i|
|~~0~~<br>~~1~~<br>0.415<br>0.435<br>0.455<br>0.475<br>0.495<br>0.515<br><br>~~0~~<br>~~1~~<br>0.560<br>0.567<br>0.574<br>0.581<br>0.588<br>0.595<br>0.602<br>0.609<br>power of<br>d thresh<br>d as thres<br>e classiﬁe<br>he two be||be|nchm|ark d|ata|set|s|s|



Zhaoyang Chu et al.: _Preprint submitted to Elsevier_ Page 11 of 15


HGRL-DTA


setting **S1** . We preserve the trained HGRL-DTA model in the training phase and then extract the learned representations
of testing affinity samples before the final prediction layer.
This experiment analysis is based on an empirical assumption that affinities are expected as close as possible
in the same cluster and as far as possible in different clusters in the affinity representation space. In this study, we
use Silhouette Coefficient (SC) [46], Calinski-Harabasz Index (CHI) [7] and Davies-Bouldin Index (DBI) [11] to
evaluate cluster performance of the affinity representations extracted from various models. Such cluster performance
is positively associated with the representation power of the models.
Table 2 reports the cluster performance of affinity representations of our model and baselines on the two benchmark
datasets. As we can see, our HGRL-DTA model achieves the best cluster performance compared to baseline methods.
Moreover, to analyze the affinity representations more intuitively, we sample weak affinities and strong ones with the
ratio of 1:1 from the test set of the KIBA dataset and project their representations into 2D space using tSNE [55] for
visualization. As illustrated in Figure 4, HGRL-DTA can well distinguish weak affinities (red) and strong ones (blue);
DeepDTA, GraphDTA, and DGraphDTA recognize most of the strong affinities; AttentionDTA differentiates part of
affinities. These results indicate that HGRL-DTA allows more delicate affinity representations, which leads to better
performance for the binding affinity prediction.


**Table 2**

Cluster performance of affinity representations.


Davis KIBA
Architecture

SC↑ CHI↑ DBI↓ SC↑ CHI↑ DBI↓


DeepDTA 0.585 3122.789 0.730 0.305 4325.479 1.711
AttentionDTA 0.303 592.239 1.728 0.176 2200.669 2.470

GraphDTA 0.615 2751.677 0.917 0.313 5589.479 1.593
DGraphDTA **0.643** 2506.929 0.906 0.353 4034.783 1.968
**HGRL-DTA** 0.639 **4330.756** **0.587** **0.410** **10385.635** **1.194**


**Figure 4:** Visualization of affinity representations. Red: weak affinity. Blue: strong affinity.


**5. Conclusion**


In this paper, we propose a novel hierarchical graph representation learning model to learn the representations of
drugs and targets for better drug-target binding affinity prediction. Our model can capture the global-level topological
affinities of drug-target pairs and the local-level molecular properties of drugs/targets synergistically, and incorporate
such hierarchical graph information using a message broadcasting mechanism. To generalize our model for the cold
start situation, we design a similarity-based embedding map to infer the representations of unseen drugs and targets.
Extensive experiments under four different scenarios have demonstrated that integrating the topological information of
affinity relationships into the representations of drugs and targets can significantly improve the predictive capacity of
the models. We also find experimental evidence suggesting that the message broadcasting mechanism is beneficial for
the integration of the hierarchical graph information, and the similarity-based embedding map is an effective strategy to
infer representations for unseen drugs or targets. In the future, we will extend the proposed method to other biological


Zhaoyang Chu et al.: _Preprint submitted to Elsevier_ Page 12 of 15


HGRL-DTA


entity association prediction tasks with hierarchical graph architecture, e.g., drug-drug interaction (DDI) prediction
and protein-protein interaction (PPI) prediction.


**Acknowledgements**


This work was supported by the National Natural Science Foundation of China (Grant No.62072206, Grant No.62102158),
Huazhong Agricultural University Scientific & Technological Self-innovation Foundation. The funders have no role
in study design, data collection, data analysis, data interpretation, or writing of the manuscript.


**References**


[1] K. Abbasi, P. Razzaghi, A. Poso, M. Amanlou, J. B. Ghasemi, A. Masoudi-Nejad, DeepCDA: deep cross-domain compound-protein affinity
prediction through LSTM and convolutional neural networks, Bioinformatics 36 (2020) 4633–4642.

[2] R. Apweiler, A. Bairoch, C. H. Wu, W. C. Barker, B. Boeckmann, S. Ferro, E. Gasteiger, H. Huang, R. Lopez, M. Magrane, M. J. Martin,
D. A. Natale, C. O’Donovan, N. Redaschi, L.-S. L. Yeh, UniProt: the Universal Protein knowledgebase, Nucleic Acids Research 32 (2004)
D115–D119.

[3] Y. Bai, K. Gu, Y. Sun, W. Wang, Bi-Level Graph Neural Networks for Drug-Drug Interaction Prediction, arXiv preprint arXiv:2006.14002
(2020).

[4] P. J. Ballester, J. B. O. Mitchell, A machine learning approach to predicting protein-ligand binding affinity with applications to molecular
docking, Bioinformatics 26 (2010) 1169–1175.

[5] J. Benesty, J. Chen, Y. Huang, I. Cohen, Pearson Correlation Coefficient, in: Noise Reduction in Speech Processing, Springer, 2009, pp.

37–40.

[6] E. E. Bolton, Y. Wang, P. A. Thiessen, S. H. Bryant, PubChem: Integrated Platform of Small Molecules and Biological Activities, Annual
Reports in Computational Chemistry 4 (2008) 217–241.

[7] T. Caliński, J. Harabasz, A dendrite method for cluster analysis, Communications in Statistics 3 (1974) 1–27.

[8] L. Chen, X. Tan, D. Wang, F. Zhong, X. Liu, T. Yang, X. Luo, K. Chen, H. Jiang, M. Zheng, TransformerCPI: improving compound–protein
interaction prediction by sequence-based deep learning with self-attention mechanism and label reversal experiments, Bioinformatics 36
(2020) 4406–4414.

[9] M. Chen, Z. Wei, Z. Huang, B. Ding, Y. Li, Simple and Deep Graph Convolutional Networks, in: H. D. III, A. Singh (Eds.), Proceedings
of the 37th International Conference on Machine Learning, volume 119 of _Proceedings of Machine Learning Research_, PMLR, 2020, pp.
1725–1735. URL: `[https://proceedings.mlr.press/v119/chen20v.html](https://proceedings.mlr.press/v119/chen20v.html)` .

[10] S. Cheng, L. Zhang, B. Jin, Q. Zhang, X. Lu, M. You, X. Tian, GraphMS: Drug Target Prediction Using Graph Representation Learning with
Substructures, Applied Sciences 11 (2021).

[11] D. L. Davies, D. W. Bouldin, A Cluster Separation Measure, IEEE Transactions on Pattern Analysis and Machine Intelligence PAMI-1 (1979)

224–227.

[12] M. I. Davis, J. P. Hunt, S. Herrgard, P. Ciceri, L. M. Wodicka, G. Pallares, M. Hocker, D. K. Treiber, P. P. Zarrinkar, Comprehensive analysis
of kinase inhibitor selectivity, Nature biotechnology 29 (2011) 1046—1051.

[13] M. Fey, J. E. Lenssen, Fast Graph Representation Learning with PyTorch Geometric, in: ICLR 2019 Workshop on Representation Learning
on Graphs and Manifolds, 2019. URL: `[https://arxiv.org/abs/1903.02428](https://arxiv.org/abs/1903.02428)` .

[14] G. K. Ganotra, R. C. Wade, Prediction of Drug–Target Binding Kinetics by Comparative Binding Energy Analysis, ACS Medicinal Chemistry
Letters 9 (2018) 1134–1139. PMID: 30429958.

[15] H. Gao, Z. Wang, S. Ji, Large-Scale Learnable Graph Convolutional Networks, in: Proceedings of the 24th ACM SIGKDD International
Conference on Knowledge Discovery &amp; Data Mining, KDD’18, Association for Computing Machinery, New York, NY, USA, 2018, pp.
1416–1424. URL: `[https://doi.org/10.1145/3219819.3219947](https://doi.org/10.1145/3219819.3219947)` . doi: `[10.1145/3219819.3219947](http://dx.doi.org/10.1145/3219819.3219947)` .

[16] K. Y. Gao, A. Fokoue, H. Luo, A. Iyengar, S. Dey, P. Zhang, Interpretable Drug Target Prediction Using Deep Neural Representation, in:
Proceedings of the 27th International Joint Conference on Artificial Intelligence, IJCAI’18, AAAI Press, 2018, pp. 3371–3377.

[17] A. Gaulton, L. J. Bellis, A. P. Bento, J. Chambers, M. Davies, A. Hersey, Y. Light, S. McGlinchey, D. Michalovich, B. Al-Lazikani, J. P.
Overington, ChEMBL: a large-scale bioactivity database for drug discovery, Nucleic Acids Research 40 (2012) D1100 – D1107.

[18] M. K. Gilson, T. Liu, M. Baitaluk, G. Nicola, L. Hwang, J. Chong, BindingDB in 2015: A public database for medicinal chemistry, computational chemistry and systems pharmacology, Nucleic Acids Research 44 (2015) D1045–D1053.

[19] M. Gönen, G. Heller, Concordance probability and discriminatory power in proportional hazards regression, Biometrika 92 (2005) 965–970.

[20] T. He, M. Heidemeyer, F. Ban, A. Cherkasov, M. Ester, SimBoost: a read-across approach for predicting drug-target binding affinities using
gradient boosting machines, journal of Cheminformatics 9 (2017).

[21] K. Huang, C. Xiao, L. M. Glass, J. Sun, MolTrans: Molecular Interaction Transformer for drug–target interaction prediction, Bioinformatics
37 (2020) 830–836.

[22] M. Jiang, Z. Li, S. Zhang, S. Wang, X. Wang, Q. Yuan, Z. Wei, Drug–target affinity prediction using graph neural network and contact maps,
RSC Adv. 10 (2020) 20701–20712.

[23] J. Jiménez, M. Škalič, G. Martínez-Rosell, G. De Fabritiis, KDEEP: Protein-Ligand Absolute Binding Affinity Prediction via 3DConvolutional Neural Networks, Journal of Chemical Information and Modeling 58 (2018) 287–296. PMID: 29309725.

[24] M. Karimi, D. Wu, Z. Wang, Y. Shen, DeepAffinity: interpretable deep learning of compound–protein affinity through unified recurrent and
convolutional neural networks, Bioinformatics 35 (2019) 3329–3338.


Zhaoyang Chu et al.: _Preprint submitted to Elsevier_ Page 13 of 15


HGRL-DTA


[25] M. Karimi, D. Wu, Z. Wang, Y. Shen, Explainable Deep Relational Networks for Predicting Compound–Protein Affinities and Contacts,
Journal of Chemical Information and Modeling 61 (2021) 46–66. PMID: 33347301.

[26] S. Kim, J. Chen, T. Cheng, A. Gindulyte, J. He, S. He, Q. Li, B. A. Shoemaker, P. A. Thiessen, B. Yu, L. Zaslavsky, J. Zhang, E. E. Bolton,
PubChem 2019 update: improved access to chemical data, Nucleic Acids Research 47 (2018) D1102–D1109.

[27] D. P. Kingma, J. Ba, Adam: A method for stochastic optimization, in: International Conference on Learning Representations (ICLR), 2015.

[28] T. N. Kipf, M. Welling, Semi-Supervised Classification with Graph Convolutional Networks, in: International Conference on Learning
Representations (ICLR), 2017.

[29] G. Landrum, RDKit: Open-Source Cheminformatics Software (2016).

[30] Y. Lecun, Y. Bengio, G. Hinton, Deep learning, Nature Cell Biology 521 (2015) 436–444. Funding Information: Acknowledgements The
authors would like to thank the Natural Sciences and Engineering Research Council of Canada, the Canadian Institute For Advanced Research
(CIFAR), the National Science Foundation and Office of Naval Research for support. Y.L. and Y.B. are CIFAR fellows. Publisher Copyright:
© 2015 Macmillan Publishers Limited. All rights reserved.

[31] I. Lee, J. Keum, H. Nam, DeepConv-DTI: Prediction of drug-target interactions via deep learning with convolution on protein sequences,
PLOS Computational Biology 15 (2019) 1–21.

[32] S. Li, J. Zhou, T. Xu, L. Huang, F. Wang, H. Xiong, W. Huang, D. Dou, H. Xiong, Structure-aware Interactive Graph Neural Networks for the
Prediction of Protein-Ligand Binding Affinity, Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining
(2021).

[33] X. Lin, K. Zhao, T. Xiao, Z. Quan, Z.-J. Wang, P. S. Yu, DeepGS: Deep Representation Learning of Graphs and Sequences for Drug-Target
Binding Affinity Prediction, in: ECAI, 2020.

[34] Y. Luo, X. Zhao, J. Zhou, J. Yang, Y. Zhang, W. Kuang, J. Peng, L. Chen, J. Zeng, A Network Integration Approach for Drug-Target Interaction
Prediction and Computational Drug Repositioning from Heterogeneous Information, Nat Commun 8 (2017) 573.

[35] M. Michel, D. Menéndez Hurtado, A. Elofsson, PconsC4: fast, accurate and hassle-free contact predictions, Bioinformatics 35 (2018)
2677–2679.

[36] T. Nguyen, H. Le, T. P. Quinn, T. Nguyen, T. D. Le, S. Venkatesh, GraphDTA: predicting drug–target binding affinity with graph neural
networks, Bioinformatics 37 (2020) 1140–1147.

[37] T. M. Nguyen, T. Nguyen, T. M. Le, T. Tran, GEFA: Early fusion approach in drug-target affinity prediction, IEEE/ACM transactions on
computational biology and bioinformatics (2021).

[38] H. Öztürk, A. Özgür, E. Ozkirimli, DeepDTA: deep drug–target binding affinity prediction, Bioinformatics 34 (2018) i821–i829.

[39] T. Pahikkala, A. Airola, S. Pietilä, S. Shakyawar, A. Szwajda, J. Tang, T. Aittokallio, Toward more realistic drug–target interaction predictions,
Briefings in Bioinformatics 16 (2014) 325–337.

[40] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Kopf, E. Yang,
Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, S. Chintala, PyTorch: An Imperative Style, High-Performance
Deep Learning Library, in: Advances in Neural Information Processing Systems 32, Curran Associates, Inc., 2019, pp. 8024–8035. URL:
```
  http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.
```

`[pdf](http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf)` .

[41] J. Peng, Y. Wang, J. Guan, J. Li, R. Han, J. Hao, Z. Wei, X. Shang, An end-to-end heterogeneous graph representation learning-based
framework for drug–target interaction prediction, Briefings in Bioinformatics 22 (2021). Bbaa430.

[42] P. Pratim Roy, S. Paul, I. Mitra, K. Roy, On Two Novel Parameters for Validation of Predictive QSAR Models, Molecules 14 (2009) 1660–

1701.

[43] M. Ragoza, J. Hochuli, E. Idrobo, J. Sunseri, D. R. Koes, Protein-Ligand Scoring with Convolutional Neural Networks, Journal of Chemical
Information and Modeling 57 (2017) 942–957. PMID: 28368587.

[44] B. Ramsundar, P. Eastman, P. Walters, V. Pande, Deep Learning for the Life Sciences: Applying Deep Learning to Genomics, Microscopy,
Drug Discovery, and More, O’Reilly Media, 2019.

[45] Y. Rong, W. Huang, T. Xu, J. Huang, DropEdge: Towards Deep Graph Convolutional Networks on Node Classification, in: International
Conference on Learning Representations, 2020. URL: `[https://openreview.net/forum?id=Hkx1qkrKPr](https://openreview.net/forum?id=Hkx1qkrKPr)` .

[46] P. J. Rousseeuw, Silhouettes: A graphical aid to the interpretation and validation of cluster analysis, Journal of Computational and Applied
Mathematics 20 (1987) 53–65.

[47] K. Roy, P. Chakraborty, I. Mitra, P. K. Ojha, S. Kar, R. N. Das, Some case studies on application of “rm2” metrics for judging quality of
quantitative structure–activity relationship predictions: Emphasis on scaling of response data, Journal of Computational Chemistry 34 (2013)

1071–1082.

[48] X. Ru, X. Ye, T. Sakurai, Q. Zou, NerLTR-DTA: drug-target binding affinity prediction based on neighbor relationship and learning to rank,
Bioinformatics (2022). Btac048.

[49] T. Smith, M. Waterman, Identification of common molecular subsequences, Journal of Molecular Biology 147 (1981) 195–197.

[50] C. Sun, P. Xuan, T. Zhang, Y. Ye, Graph convolutional autoencoder and generative adversarial network-based method for predicting drug-target
interactions, IEEE/ACM Transactions on Computational Biology and Bioinformatics (2020) 1–1.

[51] M. Sun, S. Zhao, C. Gilvary, O. Elemento, J. Zhou, F. Wang, Graph convolutional networks for computational drug development and discovery,
Briefings in Bioinformatics 21 (2019) 919–935.

[52] J. Tang, A. Szwajda, S. Shakyawar, T. Xu, P. Hintsanen, K. Wennerberg, T. Aittokallio, Making Sense of Large-Scale Kinase Inhibitor
Bioactivity Data Sets: A Comparative and Integrative Analysis, journal of Chemical Information and Modeling 54 (2014) 735–743. PMID:
24521231.

[53] M. Thafar, A. B. Raies, S. Albaradei, M. Essack, V. B. Bajic, Comparison Study of Computational Prediction Tools for Drug-Target Binding
Affinities, Frontiers in Chemistry 7 (2019) 782.

[54] M. Tsubaki, K. Tomii, J. Sese, Compound-protein interaction prediction with end-to-end learning of neural networks for graphs and sequences,


Zhaoyang Chu et al.: _Preprint submitted to Elsevier_ Page 14 of 15


HGRL-DTA


Bioinformatics 35 (2018) 309–318.

[55] L. Van der Maaten, G. Hinton, Visualizing Data using t-SNE, Journal of Machine Learning Research 9 (2008) 2579–2605.

[56] F. Wan, L. Hong, A. Xiao, T. Jiang, J. Zeng, NeoDTI: neural integration of neighbor information from a heterogeneous network for discovering
new drug–target interactions, Bioinformatics 35 (2018) 104–111.

[57] H. Wang, D. Lian, Y. Zhang, L. Qin, X. Lin, GoGNN: Graph of Graphs Neural Network for Predicting Structured Entity Interactions, in:
Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, IJCAI’20, 2021.

[58] K. Wang, R. Zhou, Y. Li, M. Li, DeepDTAF: a deep learning method to predict protein–ligand binding affinity, Briefings in Bioinformatics
22 (2021). Bbab072.

[59] Y. Wang, Y. Min, X. Chen, J. Wu, Multi-View Graph Contrastive Representation Learning for Drug-Drug Interaction Prediction, Association
for Computing Machinery, New York, NY, USA, 2021, p. 2921–2933. URL: `[https://doi.org/10.1145/3442381.3449786](https://doi.org/10.1145/3442381.3449786)` .

[60] D. Weininger, SMILES, a chemical language and information system. 1. Introduction to methodology and encoding rules, J. Chem. Inf.
Comput. Sci. 28 (1988) 31–36.

[61] Q. Wu, Z. Peng, I. Anishchenko, Q. Cong, D. Baker, J. Yang, Protein contact prediction using metagenome sequence data and residual neural
networks, Bioinformatics 36 (2019) 41–48.

[62] Z. Wu, S. Pan, F. Chen, G. Long, C. Zhang, S. Y. Philip, A comprehensive survey on graph neural networks, IEEE transactions on neural
networks and learning systems 32 (2020) 4–24.

[63] L. Zhao, J. Wang, L. Pang, Y. Liu, J. Zhang, GANsDTA: Predicting Drug-Target Binding Affinity Using GANs, Frontiers in Genetics 10
(2020) 1243.

[64] Q. Zhao, F. Xiao, M. Yang, Y. Li, J. Wang, AttentionDTA: prediction of drug–target binding affinity using attention model, in: 2019 IEEE
International Conference on Bioinformatics and Biomedicine (BIBM), 2019, pp. 64–69. doi: `[10.1109/BIBM47256.2019.8983125](http://dx.doi.org/10.1109/BIBM47256.2019.8983125)` .


Zhaoyang Chu et al.: _Preprint submitted to Elsevier_ Page 15 of 15


