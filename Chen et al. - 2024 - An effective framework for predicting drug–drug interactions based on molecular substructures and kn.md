[Computers in Biology and Medicine 169 (2024) 107900](https://doi.org/10.1016/j.compbiomed.2023.107900)


[Contents lists available at ScienceDirect](https://www.elsevier.com/locate/compbiomed)

# Computers in Biology and Medicine


[journal homepage: www.elsevier.com/locate/compbiomed](https://www.elsevier.com/locate/compbiomed)

## An effective framework for predicting drug–drug interactions based on molecular substructures and knowledge graph neural network


Siqi Chen [a], Ivan Semenov [b], Fengyun Zhang [b], Yang Yang [b], Jie Geng [c], Xuequan Feng [d] [,][∗],
Qinghua Meng [e], Kaiyou Lei [f]


a _School of Information Science and Engineering, Chongqing Jiaotong University, Chongqing, 400074, China_
b _College of Intelligence and Computing, Tianjin University, Tianjin, 300072, China_
c _TianJin Chest Hospital, Tianjin University, Tianjin, 300222, China_
d _Tianjin First Central Hospital, Tianjin, 300192, China_
e _Tianjin Key Laboratory of Sports Physiology and Sports Medicine, Tianjin University of Sport, Tianjin, 301617, China_
f _College of Computer and Information Science, Southwest University, Chongqing, 400715, China_



A R T I C L E I N F O


[Dataset link: https://github.com/SchenLab/MS](https://github.com/SchenLab/MSKG-DDI)

[KG-DDI](https://github.com/SchenLab/MSKG-DDI)


_Keywords:_
Drug–drug interactions

Prediction

Knowledge graph neural network
Knowledge-embedded message-passing neural

network

Deep learning


**1. Introduction**



A B S T R A C T


Drug–drug interactions (DDIs) play a central role in drug research, as the simultaneous administration
of multiple drugs can have harmful or beneficial effects. Harmful interactions lead to adverse reactions,
some of which can be life-threatening, while beneficial interactions can promote efficacy. Therefore, it is
crucial for physicians, patients, and the research community to identify potential DDIs. Although many AIbased techniques have been proposed for predicting DDIs, most existing computational models primarily
focus on integrating multiple data sources or combining popular embedding methods. Researchers often
overlook the valuable information within the molecular structure of drugs or only consider the structural
information of drugs, neglecting the relationship or topological information between drugs and other biological
objects. In this study, we propose MSKG-DDI – a two-component framework that incorporates the Drug
Chemical Structure Graph-based component and the Drug Knowledge Graph-based component to capture
multimodal characteristics of drugs. Subsequently, a multimodal fusion neural layer is utilized to explore the
complementarity between multimodal representations of drugs. Extensive experiments were conducted using
two real-world datasets, and the results demonstrate that MSKG-DDI outperforms other state-of-the-art models
in binary-class, multi-class, and multi-label prediction tasks under both transductive and inductive settings.
Furthermore, the ablation analysis further confirms the practical usefulness of MSKG-DDI.



Most human diseases are pathologically complex and resistant to
a single drug. The use of combination drug therapy can effectively
improve efficacy, reduce side effects, lower drug resistance, and treat
multiple or complex diseases [1]. With the rapid growth of the number
of drug types, it is essential to manage drug safety when multiple
drugs are adopted in the treatment of a disease [2]. Drug–Drug Interactions (DDIs) often occur in the case of simultaneous administration
of multiple drugs, which may result in adverse drug reactions that
cause serious health problems and huge medical costs [3]. Therefore,
identifying potential DDIs is critical for healthcare providers. While
vitro experiments and clinical trials can be performed to identify DDIs,
systematic combinatorial screening of potential DDIs remains challenging and expensive [4], due to the fact that it is incapable to
manage the rapidly increasing volume of biomedical data nowadays.


∗ Corresponding author.
_E-mail address:_ [fengxuequan@nankai.edu.cn (X. Feng).](mailto:fengxuequan@nankai.edu.cn)



The computational methods are thus widely developed for prediction
of DDIs or other relations, because of the inherent advantages that they
hold, e.g., low cost and high efficiency [5–7]. Broadly speaking, these
methods can be divided into two categories: binary-class and multiclass/multi-label prediction task. In the binary-class prediction task, the
goal is to predict whether there is an interaction between a pair of
drugs, not considering the specific type of interaction. In contrast, the
multi-class prediction aims to predict the specific type of interaction
between a pair of drugs, such as synergistic, antagonistic, or additive
interactions. In some cases, multiple types of interactions may occur
between the same pairs of drugs. Therefore, the multi-label task is to
predict multiple types of drug-drug interactions for a given pair of drugs
(please refer to Fig. 1).
Remarkable successes of advanced Artificial Intelligence (AI) techniques have been witnessed in a wide range of domains [8–19]. Most



[https://doi.org/10.1016/j.compbiomed.2023.107900](https://doi.org/10.1016/j.compbiomed.2023.107900)
Received 11 September 2023; Received in revised form 27 November 2023; Accepted 23 December 2023

Available online 29 December 2023
0010-4825/© 2024 Elsevier Ltd. All rights reserved.


_S. Chen et al._


**Fig. 1.** Examples of binary-class prediction task, multi-class and multi-label prediction

task.


existing AI-based approaches for predicting Drug–Drug Interactions
(DDI) rely on the assumption of drug similarity. Specifically, given two
drugs jointly produces a specific biological effect, another drug similar
to one of them is assumed to have the same effect when interacting
with the other one [20]. DDIs prediction can be, therefore, performed
on the basis of drug-related similarity features like chemical structures
and/or other features such as enzymes, targets, genomic similarity, etc.
A pioneering computational model that utilized deep learning techniques to calculate structural similarity for predicting DDIs was DeepDDI [21]. Since then, more advanced computational models have been
developed, and they perform DDIs prediction using more diverse data
sources to obtain multiple drug features or by combining sophisticated
embedding methods to obtain topological information [22–24]. For
the approaches of integrating multiple data sources, various features
of drugs like their chemical structures, pharmacokinetic properties,
and therapeutic indications, are taken into consideration to predict
potential DDIs. Deng et al. [22] proposed the DDIMDL, a multi-modal
deep learning framework that integrates diverse drug features from
multiple data sources. Precisely, the framework constructs four deep
neural network (DNN)-based sub-models using each of the four types of
drug features: chemical substructures, targets, enzymes, and pathways.
The joint DNN framework is then employed to combine the sub-models
and acquire cross-modality representations of drug–drug pairs, enabling
the prediction of DDI events. Lin et al. [25] also integrated multiple
sources of drug-related information to predict potential DDIs. They
employed a transformer self-attention mechanism to process the information and weigh the different drug features, thereby facilitating more
accurate predictions. These studies demonstrate that the integration
of multiple drug sources can improve the accuracy of DDI prediction
models. Nevertheless, the choice of features used to integrate them can
greatly influence the performance of the models, and they are limited
in obtaining the rich features of drugs in structural information and
semantic relations.

In addition to integrating multiple data sources, another research
line relates to utilizing popular embedding methods for predicting
DDIs [26,27]. Zhang et al. [4] developed a graph convolutional network (GCN) with drugs, targets and the side effects, and considered the
DDI prediction as multi-relational link prediction task. [28] designed an
effective framework called KGNN for DDI prediction which can capture
drug and its potential neighborhoods in the Knowledge Graph (KG).
Another example of using KG [29] in a DDI task, it initializes drug
representations using encoder–decoder layer, learns by propagating
and aggregating neighboring information along top-ranked network
paths. In the work of [30] the authors proposed the combinations of
local–global information from biomedical knowledge graph to improve
DDI prediction performance by obtaining chemical local information
on drug sequence semantics and a graph neural network model for
extracting biological local information.
The aforementioned methods have shown promising results, but
still have certain limitations. For example, most of them take little
account of the influence of the rich neighborhood information of each
entity in KG since they directly learn node latent embedding. In fact,
such information can provide different perspectives on drug nodes



a label and  = { _𝑙_ 1 _, 𝑙_ 2 _,_ … _, 𝑙_ _𝑁_ _𝑙_ } denotes the label set ( _𝑁_ _𝑙_ the types

number of events), _𝑦_ _𝑖𝑗_ = _𝑙_ _𝑧_ means that the interaction event _𝑙_ _𝑧_ exists
between drug _𝑑_ _𝑖_ and drug _𝑑_ _𝑗_ .



_Computers in Biology and Medicine 169 (2024) 107900_


and complementary knowledge, which suggests that a more effective
embedding method should be capable of leveraging both to attain
a comprehensive and unified representation in DDIs prediction [25].
However, existing methods either focus on one type of information
or simply integrate the two types, leading to potential information
loss and limiting the model’s performance. Additionally, some methods
may encounter the ‘‘cold start’’ problem, where they are unable to
make predictions for drugs with no prior knowledge. For example,
RANEDDI [5] incorporates multirelational information between drugs
and integrates relation-aware network structure information into a
multirelational DDI network to obtain drug embeddings. However, it
cannot handle drugs with no neighbors, since RANEDDI learns drug
embeddings from the DDI network.

Against this background, this work makes an attempt to solve
these issues by combining the internal information obtained from
drug molecular structure graph as well as rich topological and semantic information obtained from drug knowledge graph. The MSKGDDI framework, a novel deep learning-based method for DDI prediction, is proposed to effectively improve DDI prediction performance.
MSKG-DDI consists of the drug chemical structure graph based module
and drug knowledge graph (DKG) based module. The drug chemical
structure graph based module utilizes knowledge-embedded messagepassing neural networks (KEMPNN) to operate directly on the raw
molecular graph representations of drugs for richer feature extraction.
It breaks the DDI prediction task between two drugs down to identifying pairwise interactions between their respective substructures.
Next, taking inspiration from graph neural networks (GNNs), which
effectively represent diverse structural data, we introduce a GNN layer.
This layer extracts both structural information and semantic relations
from the Drug Knowledge Graph (DKG) to learn drug representations.
Finally, we design a multimodal fusion neural layer. This layer predicts drug–drug interactions (DDIs) by exploring the complementarity
between the multimodal representations of the drug. We evaluate our
model on two real-word datasets: (1) Drugbank dataset with 1573
drugs, 338,304 DDI pairs, and 86 interaction types. (2) KEGG dataset
with 1925 drugs, 56,983 DDI pairs and 1317 interaction types. Moreover, experiments are conducted in two settings: a transductive setting,
where the training and test sets of DDI pairs share the same drugs, and
an inductive setting (also referred to as ‘‘cold start’’), where new drugs
appear in the DDI triplets in the test sets. One can find that the latter
is more sophisticated than the former, as it poses a challenge to the
model’s generalization abilities given no prior knowledge of new drugs.


**2. Problem formulation**


This section formally defines the DDI prediction problem. We start
by providing the fundamental definitions that are necessary for prob
lem formulation.


_2.1. DDI matrix_



Formally, the label matrix with | _𝑁_ _𝑑_ | rows and | _𝑁_ _𝑑_ | columns is
defined as the DDI matrix _𝑌_, where _𝑁_ _𝑑_ is set of drugs and [|] | _𝑁_ _𝑑_ [|] | denotes
the number of drugs and each element _𝑦_ _𝑖𝑗_ denotes the interaction label
between drug i ( _𝑑_ _𝑖_ ) and drug j ( _𝑑_ _𝑗_ ). For binary-class prediction task,
_𝑦_ _𝑖𝑗_ ∈{0 _,_ 1} represents the existence of interaction between _𝑑_ _𝑖_ and
_𝑑_ _𝑗_ interact (i.e., _𝑦_ _𝑖𝑗_ = 1) or the absence of evidence for interaction
(i.e., _𝑦_ _𝑖𝑗_ = 0). For multi-class/multi-label prediction task, _𝑦_ _𝑖𝑗_ ∈  is



a label and  = { _𝑙_ 1 _, 𝑙_ 2 _,_ … _, 𝑙_ _𝑁_ _𝑙_



2


_S. Chen et al._


_2.2. Drug chemical structure graph_


Representation of molecular structure in the form of a graph proved
to be effective both in Quantum Chemistry [31,32] and for drug
research [33]. In our study, we present drug’s corresponding molecular
structure graph set as _𝑆_ = [{] _𝑔_ 1 _, 𝑔_ 2 … _, 𝑔_ | _𝑁_ _𝑑_ | }. For each drug _𝑑_ _𝑖_ ∈ _𝑁_ _𝑑_,
we constructed a molecular graph _𝑔_ _𝑖_ ∈ _𝑆_ according to its SMILES
string, and _𝑔_ _𝑖_ = ( _𝑉, 𝑈_ ) where _𝑣_ ∈ _𝑉_ represent atoms, and _𝑢_ ∈ _𝑈_
represent chemical bonds. We adopted knowledge-embedded messagepassing neural networks (KEMPNN) [34] to generate the structure
representation of _𝑑_ _𝑖_, which is a more efficient way of processing molecular structure than message-passing neural network (MPNN) where
MUFFIN [35] used.


_2.3. Drug knowledge graph_


In addition to the information extracted from the drug chemical
structure graph, we also consider topological structural information and
semantic relations for drug-related entities in the form of a knowledge
graph. Based on Deng et al.’s [22] experimental results, the combination of substructures, targets, and enzymes performs best among all
combinations of features. Therefore, we construct the knowledge graph
using these three features. Here, a special type of knowledge graph for
DDI prediction named drug knowledge graph (DKG) is built, denoted
by _𝐺_ = ( _𝐷, 𝑅, 𝐸_ ), which is comprised of entity-relation-entity triples:


_𝐺_ = [{(] _𝑑, 𝑟_ _𝑑𝑡_ _, 𝑒_ [)] ∣ _𝑑_ ∈ _𝐷_ ∧ _𝑟_ _𝑑𝑡_ ∈ _𝑅_ ∧ _𝑒_ ∈ _𝐸_ ∧ _𝐷_ ∩ _𝐸_ = ∅ [}] (1)


where _𝐷_ and _𝐸_ describe a subset of drug entities and a subset of tail
entities (drug related nodes, e.g. enzymes) respectively, and _𝑅_ is the
relation subset, indicating whether there is an interaction or connection
between drugs and tail entities.


_2.4. DDI prediction_


Given the DDI matrix _𝑌_, the drug knowledge graph _𝐺_ and the
drug chemical structure graph _𝑆_, the task is to predict whether drug
_𝑖_ [(] _𝑖_ ∈ _𝑁_ _𝑑_ ) has potential unknown interaction with drug _𝑗_ ( _𝑗_ ∈ _𝑁_ _𝑑_ _,_
_𝑗_ ≠ _𝑖_ ). Our model aims at learning a prediction function _̂𝑦_ _𝑖𝑗_ = _𝛤_ ( _𝑖, 𝑗_ ∣ _𝛽,_
_𝑌, 𝐺, 𝑆_ ). For binary-class prediction, _̂𝑦_ _𝑖𝑗_ denotes the probability of DDI
occurrence, and _𝛽_ denotes the sigmoid activation function, while for
multi-class and multi-label prediction, _̂𝑦_ _𝑖𝑗_ denotes the probability of
an event (or several events in multi-label) that drug j interacts with
drug i, and _𝛽_ denotes the softmax activation function for multi-class
and sigmoid for multi-label.


**3. Proposed method**


_3.1. Overview of MSKG-DDI_


The architecture of MSKG-DDI is depicted in Fig. 2, consisting of two
main modules: the Graph-based module and the DKG-based module. In
the following, we will explore each module in detail.


_3.2. The Graph-based module_


The Graph-based module takes as input the SMILES string [36] of
a drug, which is a textual representation of its molecular structure,
and the output is a vector representing the drug’s structural features.
The SMILES string representation is converted into a graphical representation using the open-source cheminformatics library RDKit [37]
as illustrated in Fig. 3. And the structure of the transformer encoder,
as shown in Fig. 4, basically consists of the self-attention layer, layer
normalization, residual Connections and feed-forward layer. For each
drug _𝑑_ _𝑖_, a molecular graph _𝑔_ _𝑖_ = ( _𝑉, 𝑈_ ) is generated, where _𝑣_ ∈ _𝑉_ represent atoms, and _𝑢_ ∈ _𝑈_ represent chemical bonds. We represent each
node as vector _𝑥_ _𝑣_ ( _𝑣_ ∈ _𝑉_ ), and edges are represented as _𝑢_ _𝑣𝑤_ ( _𝑣, 𝑤_ ∈ _𝑉_ ).



_𝑠_ [(] ( _[𝑛]_ _𝑑_ [)] _𝑖_ _,𝑟_ _𝑖𝑒_ ) [= sum] ( _𝑊_ [(] _[𝑙]_ [)] [ (] _𝑒_ [(] _𝑑_ _[𝑛]_ _𝑖_ [−1)] _⊙𝑒_ [(] _𝑟_ _[𝑛]_ _𝑖𝑒_ [−1)]



_Computers in Biology and Medicine 169 (2024) 107900_


We adopted knowledge-embedded message-passing neural networks
(KEMPNN) to generate the structure representation of _𝑑_ _𝑖_ . The process
involves a message-passing phase, a knowledge-attention architecture,
and a readout phase. In the message-passing phase, we initialize the
hidden state of node _ℎ_ [0] v [as] _[ ℎ]_ [0] v [= A] [0] [x] [v] [+ b] [0] [, where][ A] [0] [is a matrix]
of the shape, b 0 is a bias vector of shape _𝑛_ _𝑣_ . At the _𝑡_ _[𝑡ℎ]_ step of the
message-passing iteration, the ( _𝑡_ + 1) _[𝑡ℎ]_ step message is calculated as:


_𝑚_ _[𝑡]_ _𝑣_ [+1] = ∑ _𝑀_ [(] _𝑢_ _𝑣,𝑤_ ) _ℎ_ _𝑡𝑤_ (2)

_𝑤_ ∈Neighbor( _𝑣_ )


_ℎ_ _[𝑡]_ _𝑣_ [+1] = GRU [(] _ℎ_ _[𝑡]_ _𝑣_ _[, 𝑚]_ _[𝑡]_ _𝑣_ [+1] ), _𝑡_ + 1 ≤ _𝑇_ (3)


where _𝑀_ is a multilayer perceptron to calculate an _𝑛_ _𝑣_ × _𝑛_ _𝑣_ matrix from
edge feature _𝑢_ _𝑣𝑤_, and GRU is a gated-recurrent-unit cell. This calculation is repeated until _𝑡_ reaches a specified number of iterations _𝑇_, which
is set to 2. After the messaging stage, we add a knowledge attention
architecture, which calculates how each node should be weighted:


_̃𝑚_ _[𝑡]_ _𝑣_ [+1] = ∑ _𝑀_ [(] _𝑢_ _𝑣,𝑤_ ) _𝑎_ _𝑡𝑤_ (4)

_𝑤_ ∈Neighbor( _𝑣_ )


_𝑎_ _[𝑡]_ _𝑣_ [+1] = ReLU [(] _𝐴_ _[𝑡]_ _̃𝑚_ _[𝑡]_ _𝑣_ [+1] + _𝑏_ _[𝑡]_ + _𝑎_ _[𝑡]_ _𝑣_ ), _𝑡_ + 1 ≤ _𝑇_ (5)


_𝑎_ _𝑣_ = _𝐴_ _[𝑇]_ _𝑎_ _[𝑇]_ _𝑣_ [+] _[ 𝑏]_ _[𝑇]_ (6)


where _𝑎_ _[𝑡]_ v [is an attention weight at] _[ 𝑡]_ _[𝑡ℎ]_ [iteration on node] _[ 𝑣]_ [and] _[ 𝑎]_ [0] _𝑣_ [=] _[ ℎ]_ _[𝑇]_ w [.]
_𝐴_ _[𝑡]_ [′] is an _𝑛_ _𝑣_ × _𝑛_ _𝑣_ matrix and _𝑏_ _[𝑡]_ [′] is an _𝑛_ _𝑣_ -dimensional bias vector. The
message-passing operations for calculating _𝑎_ _[𝑡]_ [′+1] above utilize the ReLU
activation function with skip connections and are performed _𝑇_ ′ times.
The final knowledge attention value _𝑎_ v used for embedding is then
calculated without ReLU activation. Finally, the final atom embedding
_ℎ_ _[𝑓]_ v [is obtained by multiplying the output of the message-passing phase]
_ℎ_ _[𝑇]_ v [by the knowledge attention weight] _[ 𝑎]_ [v] [.]


_ℎ_ _[𝑓]_ _𝑣_ [=] _[ 𝑎]_ _[𝑣]_ _[ℎ]_ _[𝑇]_ _𝑣_ (7)


The Graph-based drug representation _𝐷_ _𝑠𝑖_ can be acquired by the readout phase, and the third phase can be described as follows:



( _𝐴_ _𝑟_ _ℎ_ _𝑓𝑣_ [+] _[ 𝑏]_ _[𝑟]_ )
)



_𝐷_ _𝑠𝑖_ = tan _ℎ_



∑
( _𝑣_ ∈ _𝑉_



(8)



Similarly, we can obtain the representation _𝐷_ _𝑠𝑗_ for drug j.


_3.3. The DKG-based module_


The DKG-based module takes the drug knowledge graph as input
data, which is obtained from the DDI matrix, and produces a KG
vector representing the drug’s KG. As depicted in Fig. 2, the DKGbased module employs the GNN to effectively capture high-order drug
neighborhood topologies between drug pairs in the KG. To generate
an embedding for drug neighborhood topologies between drug pairs, it
applies similar convolutions that aggregate and integrate neighborhood
topological information, such as entities and relations, from the local
drug receptive field. This operation can learn to capture local topological structures and simultaneously characterize both semantic KG
information and relations between drugs and related entities. Following
the previous work [38] that simply examines H-hop/order graph neighborhoods, the proposed solution also defines H-Layer neighborhoods
(or equally the depth of receptive field). We set the depth of HLayer neighborhoods to 2, since this value considers the full context
of the observed node and avoids missing information or noise from far
neighbors in the graph.

To incorporate the semantics of relations into drug representation
learning for each drug _𝑑_ _𝑖_, we calculate the semantic feature score _𝑠_ [(] ( _[𝑛]_ _𝑑_ [)] _𝑖_ _,𝑟_ _𝑖𝑒_ )
between drug _𝑑_ _𝑖_ and tail entity _𝑡_ _𝑒_ (which can be any entity in the KG
that has a relation to the observed drug), with relation _𝑟_ _𝑖𝑒_ using the
following equation:



+ _𝑏_ [(] _[𝑙]_ [)] [)]
)



(9)



3


_S. Chen et al._



_Computers in Biology and Medicine 169 (2024) 107900_



**Fig. 2.** Illustration of the proposed MSKG-DDI, consisting of two modules: the Graph-based and the DKG-based. (1) The Graph-based module utilizes knowledge-embedded
message-passing neural networks to operate directly on the raw molecular graph representations of drugs for richer feature extraction. (2) The DKG-based module utilizes the GNN
to extract the topological structural information and semantic relations from the constructed drug knowledge graph. (3) The multimodal fusion neural layer is applied to effectively
assist the joint representation learning of both the structural information and topological information.


where _⊕_ denotes the concatenation operator, _𝑊_ [′] is the trainable weight
matrix and _𝑏_ [′] is the bias vector. _𝑒_ _𝑡_ _𝑒_ is the tail entity _𝑡_ _𝑒_ representation.
Similarly, we can obtain the representation _𝐷_ _𝑗_ for drug j.


_3.4. Multimodal neural fusion layer_



**Fig. 3.** Generate Drug Chemical Structure Graph from SMILES.


**Fig. 4.** The attention module of the fusion layer.


wheresentation obtained from previous message passing steps, which memo- _⊙_ denotes the element-wise product, _𝑒_ [(] _𝑑_ _[𝑛]_ _𝑖_ [−1)] is the drug _𝑑_ _𝑖_ reprerize the messages from its ( _𝑛_ −1)-layer neighbors. _𝑒_ [(] _𝑟_ _[𝑛]_ _𝑖𝑒_ [−1)] is the relation
representation between drug _𝑑_ _𝑖_ and tail entity _𝑡_ _𝑒_ after ( _𝑛_ −1) _[𝑡ℎ]_ GNN
layer. _𝑊_ [(] _[𝑙]_ [)] and _𝑏_ [(] _[𝑙]_ [)] are the trainable weight matrix and the bias vector,
respectively, and _𝑙_ is the number of fully connected layers. sum denotes
element-wise summation across all the vectors.

The performance of network structure features can be improved by
minimizing the influence of neighbors [27]. Hence, instead of using all
the neighbors for each drug _𝑑_ _𝑖_, we uniformly sample a set of fixed size,
denoted as  _𝑠_ ( _𝑑_ _𝑖_ ). The DKG-based drug representation _𝐷_ _𝑖_ for drug i
is obtained as follows:



⎛ ⎛
_𝐷_ _𝑖_ = _𝑒_ [(] _𝑑_ _[𝑛]_ _𝑖_ [)] [= ReLU] ⎜ _𝑊_ [′] ⎜ _𝑒_ [(] _𝑑_ _[𝑛]_ _𝑖_ [−1)] _⊕_ ∑ _𝑠_ [(] ( _[𝑛]_ _𝑑_ [)] _𝑖_ _,𝑟_ _𝑖𝑒_ ) _[𝑒]_ _𝑡_ [(] _𝑒_ _[𝑛]_ [−1)]
⎜⎝ ⎜⎝ _𝑡_ _𝑒_ ∈ _𝑠_ ( _𝑑_ _𝑖_ )



⎞ ⎞
⎟ + _𝑏_ [′] ⎟ (10)
⎟⎠ ⎟⎠



As the Graph-based and the DKG-based components provide complementary information to each other, therefore the fusion of these
features critically affects the quality of the prediction. First, we concatenate all the embeddings of both observed drugs in order to capture
all knowledge about these features. After obtaining the feature vectors,
we use the multi-headed self-attention mechanism to perform a feature
fusion. We choose the most performant way in our experiment which
considers those two components’ coherence and complementarity together in the so-called multimodal neural fusion layer. The embeddings
_𝐷_ _𝑠𝑖_ and _𝐷_ _𝑖_ for drug i, learned from the Graph-based component and the
DKG-based component representations, are encoded by element-wise
product operation as the multimodal embedding _𝐷_ _𝑖_ [′] [of drug i, which]
can be described as:


_𝐷_ _𝑖_ [′] [=] _[ 𝐷]_ _[𝑠𝑖]_ _[⊕𝐷]_ _[𝑖]_ (11)


where _⊕_ denotes the concatenation operator that concatenates the local
and global features. We concatenate the embedding _𝐷_ _𝑗_ [′] [of drug j and]
_𝐷_ _𝑖_ [′] [of drug i to final multi-drug feature] _[ 𝐷]_ _𝑖𝑗_ [′] [(where] _[ 𝐷]_ _𝑖𝑗_ [′] [=] _[ 𝐷]_ _𝑖_ [′] _[⊕𝐷]_ _𝑗_ [′] [), and]
feed into the multi-head attention module (also called the transformer
block) to take advantage of self-attention mechanism to learn the
weight of each vector. The structure of the transformer encoder basically consists of the self-attention layer, layer normalization, residual
Connections and feed-forward layer. The formula for calculating the
multi-head attention is given as follows:


_𝑀𝑢𝑙𝑡𝑖𝐻𝑒𝑎𝑑_ = _𝐶𝑜𝑛𝑐𝑎𝑡_ ( _ℎ𝑒𝑎𝑑_ 1 _, ℎ𝑒𝑎𝑑_ 2 _,_ … _, ℎ𝑒𝑎𝑑_ _𝑛_ ) _𝑊_ [◦] (12)


_ℎ𝑒𝑎𝑑_ _𝑖_ = _𝑠𝑜𝑓𝑡𝑚𝑎𝑥_ ( _𝑄_ _𝑖_ × _𝐾_ _𝑖_ _[𝑇]_ ) _𝑉_ _𝑖_ (13)
~~√~~ _𝑑_ _𝑘_



4


_S. Chen et al._


**Table 1**

The statistic of two widely used datasets.


Drugs DDI Entities Sparsity


DrugBank 1,573 338,304 4,572 27.36%
KEGG 1,925 56,983 129,910 3.08%


where _𝑄_ _𝑖_, _𝐾_ _𝑖_ and _𝑉_ _𝑖_ are the _𝑄_ (Query), _𝐾_ (Key) and _𝑉_ (Value) matrices
derived from the linear transformation of _𝐷_ _𝑖𝑗_ [′] [,] _[ 𝑄]_ _[𝑖]_ [=] _[ 𝐷]_ _𝑖𝑗_ [′] [×] _[ 𝑊]_ _𝑖_ _[𝑄]_ [,] _[ 𝐾]_ _[𝑖]_ [=]
_𝐷_ _𝑖𝑗_ [′] [×] _[𝑊]_ _𝑖_ _[𝐾]_ and _𝑉_ _𝑖_ = _𝐷_ _𝑖𝑗_ [′] [×] _[𝑊]_ _𝑖_ _[𝑉]_ [. The projections are matrices of parameters]
_𝑊_ _𝑖_ _[𝑄]_ ∈ _𝑅_ _[𝑑]_ _[𝑖]_ _[𝑛]_ [×] _[𝑑]_ _[𝑄]_, _𝑊_ _𝑖_ _[𝐾]_ ∈ _𝑅_ _[𝑑]_ _[𝑖]_ _[𝑛]_ [×] _[𝑑]_ _[𝐾]_, _𝑊_ _𝑖_ _[𝑉]_ ∈ _𝑅_ _[𝑑]_ _[𝑖]_ _[𝑛]_ [×] _[𝑑]_ _[𝑉]_ . _𝐶𝑜𝑛𝑐𝑎𝑡_ (∙) is the vector
concatenated operation.


Residual connection [39] can alleviate the problem of gradient
disappearance. The gradient can be directly transmitted through the
residual link during backpropagation, which greatly accelerates the
backpropagation and helps to design the neural network more deeply.
Layer normalization was used after the self-attention layer and after
the feed-forward network layer, in an effort to avoid the problem of
‘‘covariance shift’’ by re-standardizing the computed vector representations. This can also speed up the convergence of the neural network.
Then, the output drug–drug score _𝑋_ from the previews part of encoder
is passing into fully connected layers to predict the DDI probability as

follows:


_̂𝑦_ _𝑖𝑗_ = _𝜎_ ( _𝑀𝐿𝑃_ ( _𝑋_ )) (14)


where _⊕_ denotes the concatenation operator, _𝑀𝐿𝑃_ is the multilayer
perceptron. Here, _̂𝑦_ _𝑖𝑗_ represents the probability of drug pair interaction
in the binary-class prediction task, while in the multi-class/multi-label
task, it represents the prediction score of each relation type. _𝜎_ denotes
the sigmoid function in binary-class and multi-label tasks, but it refers

to the softmax function in multi-class tasks.


**4. Experiments**


In this section, we provide a detailed description of the experimental
setups and present various experimental results to evaluate the overall
performance of our model.


_4.1. Experimental setup_


_4.1.1. Dataset_


In order to demonstrate the effectiveness of our proposed model,
we conducted extensive experiments on the widely used datasets for
the two types of DDI prediction tasks.


_Binary-class DDIs task._ (1) **DrugBank** : DrugBank is arguably the most
popular and frequently used open-source dataset for the research of
drugs and other related studies. The latest version of DrugBank is
[5.1.10, and the dataset is available publicly at https://go.drugbank.](https://go.drugbank.com)
[com. We extracted a list of drug identifier combinations from the veri-](https://go.drugbank.com)
fied DDIs, resulting in 1573 approved drugs and 338,304 pairwise DDIs.
These DDIs were represented as edges between drug nodes, forming
a relatively dense dataset. (2) **KEGG** : We obtained the verified DDI
data from [28], which contains 1925 drugs and a total of 56,983 drug
[pairs with verified DDIs. The KEGG-drug dataset is available from https:](https://www.kegg.jp/kegg/drug)
[//www.kegg.jp/kegg/drug publicly. Fig. 5 shows the distribution of the](https://www.kegg.jp/kegg/drug)
samples between events in the dataset. The frequency distribution of all
types is shown in Fig. 5(a) (all types are sorted by quantity), and their
percentage is given in Fig. 5(b). The statistics of the used datasets are
given in Table 1.



_Computers in Biology and Medicine 169 (2024) 107900_


**Fig. 5.** The difference in the number of samples between events.


_Multi-class DDIs task._ We obtained DDI data with multiple relationships
from GMPNN [40], SSI-DDI [41] which was extracted from DrugBank.
The dataset consists of 191,808 tuples describing 86 types of interactions, indicating how one drug affects the metabolism of another drug.
However, we excluded relationships that have less than 30 samples,
resulting in a total of 1569 drugs and 171,864 drug pairs with 65
distinct relationships. For each drug, its SMILES representation was
converted into a molecular graph using RDKit [37]. Each drug pair
in the dataset is only associated with one type of interaction. Fig. 5
shows the distribution of the samples between events in the dataset.
The frequency distribution of all types is shown in Fig. 5(a) (all types
are sorted by quantity), and their percentage is given in Fig. 5(b).


_Multi-label DDIs task._ We used multi-labels dataset proposed by [40].
The original data source is side effects database TWOSIDES [42]. After
filtering database [4] it contains 645 drugs, 1318 interaction types, and
4,651,131 DDI tupls. In [40] was removed interaction types that have
less than 500 samples, to increase the average DDIs distribution. The
final dataset contains 963 interactions types, and 4,576,287 drug pairs.


_4.1.2. Baselines_

To evaluate the effectiveness, MSKG-DDI is compared with the
following state-of-the-art work as follows:


  - DeepDDI [21] develops a deep learning-based method that reduces the dimension of drug features based on a principal component analysis, and takes as input drug structural similarity.

  - DDIMDL [22] develops a multimodal deep learning model for
DDI prediction. This model constructs similarity matrices based
on chemical substructures, targets, enzymes and pathways, and
it adopts a DNN predictor to perform DDI prediction for each
feature.

  - KGNN [28] adopts graph convolutional network (GCN) to selectively aggregate the neighbor information with high-level layer
information for learning the node representation. When performing the multi-class prediction task, the number of output neurons
is set to the types of available events (65), as KGNN was initially
designed for the binary-class prediction.



5


_S. Chen et al._


  - MUFFIN [35] designs a bi-level crossover strategy with cross-level

and scalar-level components to jointly learn the fusion characterization of drug-self structure information and the KG with
rich bio-medical information from the cross-level and scalar-level

perspectives based on convolutional neural networks.

  - RANEDDI [5] considers the multirelational information between

drugs and integrates the relation-aware network structure information in the topology of a multirelational DDI network to get
the drug embedding. It designs an end-to-end relationship-aware
network embedding model to predict drug interactions.

  - MDF-SA-DDI [25] proposes a DDI prediction model based on

multi-source drug fusion, multi-source feature fusion and transformer self-attention mechanism. It combines a pair of drugs in
four different ways and inputs the combined drug feature representation into four fusion networks to obtain the latent features

of the drug pairs. Then, transformer blocks perform latent feature
fusion.


_4.1.3. Evaluation metrics_


We evaluate the model performance using several classification
evaluation metrics that are commonly used in binary-class, multi-class
and multi-label tasks. These metrics include Recall, Precision, F1 score,
accuracy (ACC), area under the precision recall curve (AUPR) which
measures the ratio of true positives among all positive predictions for
each given recall rate, and area under the receiver operating characteristic curve (AUC), which illustrates the true-positive rate versus the
false-positive rate at different cutoffs. Their definitions are as follows:



(15)


(16)


(17)



Accuracy = [1]

_𝑙_


Precision = [1]

_𝑙_



_𝑙_
∑

_𝑖_ =1


_𝑙_
∑

_𝑖_ =1



_𝑇𝑃_ _𝑖_ + _𝑇𝑁_ _𝑖_

_𝑇𝑃_ _𝑖_ + _𝐹𝑃_ _𝑖_ + _𝑇𝑁_ _𝑖_ + _𝐹𝑁_ _𝑖_


_𝑇𝑃_ _𝑖_

_𝑇𝑃_ _𝑖_ + _𝐹𝑃_ _𝑖_



_Computers in Biology and Medicine 169 (2024) 107900_


_4.1.5. Evaluation_


We randomly divide all approved DDIs as positive samples into
training, validation and testing sets in a 8/1/1 ratio, and randomly
sample the complement set of positive samples as negative samples,
with an equal number of positive and negative samples in all phase.
We adopt Adam algorithm with a learning rate of 0.002 to optimize
all trainable parameters through a random search. To assess our proposed method, we use 5-fold cross validation (5-CV). In this process,
we randomly partition all drug–drug interaction (DDI) pairs into five
subsets for our experiments. The evaluation score is the average result
from all five rounds. We also implement an early-stopping strategy
to prevent over-fitting. This strategy halts the training if there is no
observed improvement after five epochs.


_4.2. Results and analysis_


_4.2.1. Binary-class prediction_

In this section, we compare the performance of the proposed model
with several state-of-the-art DDI prediction methods: DeepDDI, KGNN,
MUFFIN and RANEDDI for binary-class prediction. The results are
shown in Table 2, where the best results are highlighted in bold and the
suboptimal results are underlined. Among others, MSKG-DDI showed
the best performance. Specifically, (i) DeepDDI had a relatively low
performance than that of our model, because it merely utilized chemical substructure similarity information as a feature; (ii) KGNN and
RANEDDI failed to obtain satisfying results, which considered the
semantic information of drugs and the high-order relationship of nodes
without leveraging the auxiliary features based on the molecular structure graph; and (iii) Taking into account drug substructure information and KG information with rich biomedical information, MUFFIN
achieved better performance than other baselines. However, it uses traditional knowledge graph embedding methods, which learn node latent
embedding directly, limiting its ability in acquiring the rich neighborhood information of each entity in KG. MSKG-DDI, by contrast, adopts
the knowledge-embedded message-passing neural networks (KEMPNN)
to overcome the shortcomings of small data sets and molecular representation transparency. In addition, MSKG-DDI introduces attention
mechanism to multimodal learning of attributes and human knowledge,
making GNN more generalized and consistent.


_4.2.2. Multi-class prediction_

For multi-class prediction, we compare MSKG-DDI with state-ofthe-art models, including DDIMDL, KGNN, MUFFIN, and MDF-SA-DDI,
for both transductive and inductive multi-class DDI prediction tasks.
DDIMDL, introduced as a new baseline here, optimizes the DeepDDI
model by utilizing three drug-related features (including chemical substructure, target, and enzyme), which are inputted into three different
constructed sub-models for training. RANEDDI is based on a relationaware network architecture, which learns drug embeddings from the
DDI network. However, it fails to handle drugs that have no neighbors,
making it susceptible to the ‘‘cold start’’ problem and excluded in this
comparison. Therefore, we compare MSKG-DDI with other state-of-theart models, including DDIMDL, KGNN, MUFFIN, and MDF-SA-DDI, for
both transductive and inductive multi-class DDI prediction tasks.

Table 3 depicts the results of the prediction, where the best results
are highlighted in bold and the suboptimal results are underlined.
MSKG-DDI again showed an advantage over other baselines, ranking
in the first place in all metrics and leading the mean performance of
the others by a margin between 3.13% to 13.04%. Furthermore, we
investigate the performance of MSKG-DDI under each event. Fig. 6 displays the AUC, AUPR, and F1 scores for each model under each event.
Generally, the events with higher frequency can gain better performance. Among 65 events, MSKG-DDI outperformed the other methods
by achieving the highest AUC scores in 55 events, the highest AUPR
scores in 58 events, and the highest F1 scores in 57 events.



Recall = [1]

_𝑙_



_𝑙_
∑

_𝑖_ =1



_𝑇𝑃_ _𝑖_

_𝑇𝑃_ _𝑖_ + _𝐹𝑁_ _𝑖_



_𝐹_ 1 = [2 ×][ Precision][ ×][ Recall] (18)

Precision + Recall


where TP and TN represent the number of correctly predicted DDI
pairs and unlabeled drug–drug pairs, respectively, FP and FN are the
number of incorrectly predicted DDI pairs and unlabeled drug–drug
pairs, respectively, _𝑙_ is the number of DDI interaction types (while in
the case of the binary-class prediction, it is defined as 1).

For Recall, Precision, and F1-score, each of them has macro and
micro variants. Macro metrics provide an average assessment of how
different types of interactions perform. For instance, ‘‘macro precision’’
is the mean of the Precision values for various interaction types. In contrast, Micro metrics are similar to metrics used in binary classification,
where the proportions of true positive, false positive, true negative, and
false negative samples are summed across all interaction types. For the
binary-class prediction task, we use micro metrics for AUPR and AUC,
and macro metrics for F1 score. For the multi-class prediction task, we
use macro-precision, macro-recall and macro-f1. For the multi-label, we
use micro metrics for AUC, and macro metrics for precision and recall.


_4.1.4. Parameters settings_

In the experiments, we represented drugs’ KG and structure Graph
as 128-dimensional vectors, the structure Graphs vectors were pretrained by Graph-based module. The batch size was set to 2048, epoch
number to 200; we used an early stop to avoid overfitting the model;
the learning rate is 0.0002. The number of multi-head self-attention
is 8. The size of the last fully connected layers was set to 2048, and
the activation function for fully connected layers was set to ReLU;
the output neuron size was set to 1 for binary classification, 86 for
multiclass, 962 for multilabel. The output layer activation function was
set according to the tasks, binary/multilabel - Sigmoid, multiclass Softmax.



6


_S. Chen et al._



_Computers in Biology and Medicine 169 (2024) 107900_


**Table 2**

Comparative evaluation of binary DDI prediction in the transductive setting. First/second row of each method corresponds to results
on the DrugBank and KEGG dataset respectively. The best results are marked in bold with the suboptimal ones being underlined.


0.8897 0.9641 0.9790 0.9823 **0.9971**
AUC
0.8994 0.8887 0.9491 0.9261 **0.9682**


0.8174 0.9418 0.9511 0.9642 **0.9836**
ACC
0.8229 0.8329 0.8900 0.8626 **0.9235**


0.8205 0.9437 0.9514 0.9651 **0.9834**
F1
0.7966 0.8394 0.8945 0.8661 **0.9248**


0.8532 0.9662 0.9777 0.9828 **0.9967**
AUPR
0.8442 0.8345 0.9336 0.8947 **0.9687**


**Table 3**

Comparative evaluation of multiclass DDI prediction results in the transductive setting. The best results are
highlighted in boldface and the suboptimal results are underlined.


DDIMDL 0.9061 0.8499 0.8272 0.8875

KGNN 0.9206 0.8734 0.8704 0.8862

MUFFIN 0.9274 0.9014 0.8984 0.9155

MDF-SA-DDI 0.9126 0.8493 0.8250 0.8908

MSKG-DDI **0.9614** **0.9403** **0.9386** **0.9537**


**Table 4**

Comparative evaluation of multilabel DDI prediction results in the transductive setting. The best results are highlighted in boldface
and the suboptimal results are underlined.


AUC 0.8512 0.8627 0.8841 0.8578 **0.9107**

Precision 0.7329 0.7471 0.7512 0.7429 **0.7993**

Recall 0.8408 0.8593 0.8789 0.8607 **0.9125**


**Fig. 6.** The AUC, AUPR, and F1 scores of all prediction models for each event.



_4.2.3. Multi-label prediction_


In this section, we compare the performance of the proposed model
MSKG-DDI in the DDI transductive multilabel prediction with other
state-of-the-art models, we select the same candidates for this task as
for multiclass prediction task. To perform the benchmark, the output
layer size was set according to the number of intersection type labels

963 in all models.


The results are shown in Table 4. MSKG-DDI has demonstrate ex
cellent performance, and the best results compared with all competing
methods in all metrics, outperform the best baseline by an average
of 3.61%. Compared to other models, the average performance is exceeded by a margin from 2.66% to 7.17%. The second best result on this



task showed MUFFIN, the same rating was obtained by MUFFIN in the
multiclass transductive setting benchmark. The high performance of the
proposed method and MUFFIN in multi-class/multi-label experiments
assumes use of KG and structural graph features to be an effective
approach.


_4.3. Ablation study_


In order to explore how the Graph-based and DKG-based components improve the performance of the proposed model, we conducted
the ablation study on the following two variants of MSKG-DDI: 1)
MSKG-DDI_dcs where we only explore drug chemical structure graph



7


_S. Chen et al._



_Computers in Biology and Medicine 169 (2024) 107900_


**Fig. 7.** Results of the ablation experiments with the relative performances compared with complete MSKG-DDI in all the metrics.


**Table 5**

Investigating the contributions of raw molecular graph representations and drug knowledge graph representations. The best
results are highlighted in boldface and the suboptimal results are underlined.


ACC AUPR AUC F1 Recall Precision


MSKG-DDI_dcs 0.8859 0.9448 0.9504 0.8893 0.9167 0.8634
MSKG-DDI_dkg 0.9442 0.9837 0.9853 0.9449 0.9574 0.9327

MSKG-DDI **0.9836** **0.9967** **0.9971** **0.9834** **0.9881** **0.9788**



embedding using the raw molecular graph representations, and 2)
MSKG-DDI_dkg where we only consider the topological structures and
semantic relations to learn the embedding of drug–drug pairs from the

DKG.


Fig. 7 demonstrates the importance of both modules for DDI prediction performance in terms of the six metrics. It can be clearly observed
that both variants lagged behind MSKG-DDI in all metrics, and MSKGDDI_dcs performed worse than MSKG-DDI_dkg. Table 5 reports the
performance of two model variants and MSKG-DDI. It can be clearly
observed that both variants lagged behind MSKG-DDI in all metrics,
and MSKG-DDI_dcs performed worse than MSKG-DDI_dkg. Specifically,
MSKG-DDI outperformed MSKG-DDI_dcs by 4.87% to 13.13% and outperformed MSKG-DDI_dkg by 1.16% to 4.73%, while the decrease
of MSKG-DDI_dcs compared to MSKG-DDI_dkg ranged from 3.54% to

7.43%.

To sum up, combining drug chemical structure graph embedding
and drug knowledge graph embedding with topological structures and
semantic relations is effective in enhancing feature extraction, which
can indeed boost the performance.


_4.4. Effect of embedding dimension_


We first conducted experiments to explore the influence of embedding dimension on our model’s performance, varying its value from
16 to 512. The results are showed on Fig. 8(a). It can be found
that the model’s performance increases with embedding dimensions,
and reaches the peak performance at the 128 dimensions. After that,
its performance drops quickly. This is because when the embedding
dimension is too large, the model is more likely to overfit the training
data, leading to worse performance on test data.


_4.5. Effect of GNN layers_


Next, we conducted experiments to evaluate the performance of our
model with different numbers of GNN layers. According to the results
given in Fig. 8(b), the model achieved the highest performance with
a single GNN layer. The reason behind is that larger depths create too



many links and bring massive noise to the model. This is also in line
with our perception, as inferring drug-entity similarities from a long
relation chain makes little sense. Therefore, based on our experimental
results, we conclude that using a single GNN layer is sufficient for
real-world applications.


_4.6. Effect of neighborhood size_


Lastly, we conducted experiments to explore the effectiveness of our
model by varying the size of the sampled neighbors. The results are
presented in Fig. 8(c), which shows that our model achieved the best
performance when the size of the neighborhood was set to 4. When
the neighborhood size was too small, the model could not fully incorporate the structural information, leading to suboptimal performance.
Conversely, when the neighborhood size was too large, the model was
more susceptible to noise, which could cause performance degradation.
These findings demonstrate the importance of selecting an appropriate
neighborhood size to achieve optimal performance in predicting DDIs.
Although the optimal size of the neighborhood may vary depending on
the specific problem being addressed as well as the characteristics of
the dataset, our results suggest that our model can effectively leverage
structural information and balance the trade-off between incorporating
neighborhood information and avoiding noise.


_4.7. Inductive setting experiment_


So far, we have presented the results of experiment in a transductive
setting, that is, the drugs in test phase were also included in the training
phase. The prediction task in transductive setting corresponds to Task 0
in Tables 2 and 3. Predictions in the inductive setting are more complex
than in transductive setting, since there are new drugs in DDI triplets
in the test sets. These tests are also called ‘‘cold start’’ tests, in which
one can test the generalization abilities of the model without any prior
knowledge of the new drugs. There were two types of test strategy:


  - Task 1: the test set contains one known drug (obtained during the
training phase) and one new drug.



8


_S. Chen et al._



_Computers in Biology and Medicine 169 (2024) 107900_


**Fig. 8.** Results of MSKG-DDI with varying values of embedding dimension, GNN layers and neighborhood size.


**Table 6**

Comparative evaluation of binary DDI prediction results in the inductive setting. The best results of each dataset are highlighted in boldface
and the suboptimal results are underlined.


Task Methods DrugBank dataset KEGG dataset


AUC ACC F1 AUPR AUC ACC F1 AUPR



Task 1


Task 2



DeepDDI 0.7368 0.5177 0.6598 0.7368 0.7069 0.4669 0.6063 0.7069

KGNN 0.9099 0.8412 0.8397 0.9153 0.8828 0.8160 0.8085 0.8706

MUFFIN **0.9447** 0.6405 0.5039 0.6244 0.8573 0.7545 0.7204 0.8250

MSKG-DDI 0.9431 **0.8926** **0.8558** **0.9436** **0.9167** **0.8519** **0.8287** **0.9039**


DeepDDI 0.5080 0.4518 0.3077 0.5094 0.4983 0.4250 0.3002 0.4983

KGNN 0.5244 0.5032 0.3347 0.5180 0.4770 0.4936 0.3142 0.4976

MUFFIN 0.5214 0.5011 0.3814 **0.5492** 0.4985 **0.5146** 0.3026 0.4760

MSKG-DDI **0.5464** **0.5172** **0.4179** 0.5368 **0.5109** 0.5075 **0.3714** **0.5186**


**Table 7**

Comparative evaluation of multiclass DDI prediction results in the inductive setting. The best results are highlighted in boldface
and the suboptimal results are underlined.


Task Methods ACC Macro-F1 Macro-Recall Macro-Precision


DDIMDL 0.7833 0.6705 0.6720 0.6784

Task 1 KGNN 0.8121 0.7088 0.6975 0.7431

MUFFIN 0.8157 0.7153 0.7023 0.7609

MDF-SA-DDI 0.8064 0.6799 0.6792 0.7233

MSKG-DDI **0.8552** **0.7605** **0.7596** **0.7965**


Task 2 DDIMDL 0.3588 0.2010 0.2021 0.2479

KGNN 0.3724 0.2131 0.2273 0.2580

MUFFIN 0.3948 0.2595 0.2685 0.3102

MDF-SA-DDI 0.3691 0.2049 0.2195 0.2532

MSKG-DDI **0.4397** **0.3687** **0.3933** **0.4064**




  - Task 2: the test set contains two new drugs.


As RANEDDI learns drug embeddings from the DDI network, it

is unable to handle cases where the drug has no neighbors, causing

the ‘‘cold start’’ problem. Therefore, we only compare MSKG-DDI with

DeepDDI, KGNN, and MUFFIN for binary-class prediction. Table 6

presents the results of the binary DDI prediction task, while Table 7

gives the results of the multiclass DDI prediction. There was a notice
able drop in performance in the inductive setting compared to that in

the transductive setting. Specifically, without prior knowledge about

the new drugs, the performance of all models for Task 1 and Task 2

decreased, especially for Task 2. On the other hand, the experimental

results demonstrate that MSKG-DDI outperformed other methods for

Task 1 and Task 2, which corroborates the efficiency of operating

directly on the raw molecular graph representations of drugs for richer

feature extraction again. Overall, the studies clearly show that deep

learning and raw molecular graph representations are critical for the

DDI prediction, and our framework outperforms the baselines.



**5. Discussion**


MSKG-DDI effectively operates directly on the raw molecular graph
representations of drugs to obtain richer feature extraction by utilizing
knowledge-embedded message-passing neural networks (KEMPNN) on
the drug chemical structure graph. In this way, it can break the DDI
prediction task between two drugs down to identifying pairwise interactions between their respective substructures, which can determine
whether a pair of new drugs would interact. Furthermore, MSKGDDI exploits both the topological information and the semantic relations by leveraging a graph neural network on the drug knowledge
graph. The proposed approach has contributed to drug–drug prediction,
and the experimental results proved the effectiveness of feature processing from multiple representations (e.g. strings, graphs) of entities.
The concept of MSKG-DDI can be applied to other problems in bioinformatics, such as predicting Protein-Protein interactions, predicting
Drug-Target interactions. Besides bioinformatics tasks, this framework
could be useful in tasks requiring the extraction and processing of
various features from entities, with the construction of KGs and their

utilization.



9


_S. Chen et al._


There may be multiple directions for future development. First,
we believe it is worth investing research efforts in enhancing the
interpretability of DDI prediction by developing methods to visualize
and explain the model’s predictions. This would not only increase the
trust and adoption of the model but also provide valuable insights into
the underlying mechanisms of drug interactions, ultimately leading to
more effective and safer drug therapies. Also, it is interesting to identify
the relative importance of each parts of the molecular and knowledge
graphs for DDI predictions.


**CRediT authorship contribution statement**


**Siqi Chen:** Conceptualization, Funding acquisition, Investigation,
Methodology, Resources, Software, Writing – original draft, Writing –
review & editing. **Ivan Semenov:** Data curation, Software, Visualization, Writing – original draft. **Fengyun Zhang:** Data curation, Formal
analysis, Validation. **Yang Yang:** Data curation, Visualization, Writing
– original draft. **Jie Geng:** Software, Writing – review & editing. **Xue-**
**quan Feng:** Conceptualization, Funding acquisition, Writing – review
& editing. **Qinghua Meng:** Data curation, Formal analysis. **Kaiyou Lei:**
Resources, Supervision.


**Declaration of competing interest**


The authors declare that they have no known competing financial interests or personal relationships that could have appeared to
influence the work reported in this paper.


**Data availability**


[The codes and datasets are available online at https://github.com/](https://github.com/SchenLab/MSKG-DDI)
[SchenLab/MSKG-DDI.](https://github.com/SchenLab/MSKG-DDI)


**Acknowledgments**


This work was supported by the National Natural Science Foundation of China [Grant Nos.: 61602391, 82072212]. The authors also
sincerely thank the editors and reviewers for their valuable comments.


**References**


[[1] I.V. Bijnsdorp, E. Giovannetti, G.J. Peters, Analysis of drug interactions, in:](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb1)

[Cancer Cell Culture, Springer, 2011, pp. 421–434.](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb1)

[[2] J. Niu, R.M. Straubinger, D.E. Mager, Pharmacodynamic drug–drug interactions,](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb2)

[Clin. Pharmacol. Ther. 105 (6) (2019) 1395–1406.](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb2)

[[3] S. Vilar, E. Uriarte, L. Santana, T. Lorberbaum, G. Hripcsak, C. Friedman,](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb3)

[N.P. Tatonetti, Similarity-based modeling in large-scale prediction of drug-drug](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb3)

[interactions, Nat. Protoc. 9 (9) (2014) 2147–2163.](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb3)

[[4] M. Zitnik, M. Agrawal, J. Leskovec, Modeling polypharmacy side effects with](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb4)

[graph convolutional networks, Bioinformatics 34 (13) (2018) i457–i466.](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb4)

[[5] H. Yu, W. Dong, J. Shi, RANEDDI: Relation-aware network embedding for](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb5)

[drug-drug interaction prediction, Inform. Sci. 582 (2022) 167–180.](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb5)

[6] Y.P. Zhang, Q. Zou, PPTPP: A novel therapeutic peptide prediction method

using physicochemical property encoding and adaptive feature representation
[learning, Bioinformatics 36 (13) (2020) 3982–3987, http://dx.doi.org/10.1093/](http://dx.doi.org/10.1093/bioinformatics/btaa275)

[bioinformatics/btaa275.](http://dx.doi.org/10.1093/bioinformatics/btaa275)

[7] Y. Wang, Y. Zhai, Y. Ding, Q. Zou, SBSM-pro: Support bio-sequence machine

[for proteins, 2023, http://dx.doi.org/10.48550/arXiv.2308.10275, arXiv preprint](http://dx.doi.org/10.48550/arXiv.2308.10275)

[arXiv:2308.10275.](http://arxiv.org/abs/2308.10275)

[8] Y. LeCun, Y. Bengio, G.E. Hinton, Deep learning, Nat. 521 (7553) (2015)

[436–444, http://dx.doi.org/10.1038/NATURE14539.](http://dx.doi.org/10.1038/NATURE14539)

[9] R. Su, H. Wu, B. Xu, X. Liu, L. Wei, Developing a multi-dose computational

model for drug-induced hepatotoxicity prediction based on toxicogenomics data,
[IEEE ACM Trans. Comput. Biol. Bioinform. 16 (4) (2019) 1231–1239, http:](http://dx.doi.org/10.1109/TCBB.2018.2858756)
[//dx.doi.org/10.1109/TCBB.2018.2858756.](http://dx.doi.org/10.1109/TCBB.2018.2858756)

[10] L. Wei, S. Luan, L.A.E. Nagai, R. Su, Q. Zou, Exploring sequence-based

features for the improved prediction of DNA N4-methylcytosine sites in mul[tiple species, Bioinform. 35 (8) (2019) 1326–1333, http://dx.doi.org/10.1093/](http://dx.doi.org/10.1093/BIOINFORMATICS/BTY824)

[BIOINFORMATICS/BTY824.](http://dx.doi.org/10.1093/BIOINFORMATICS/BTY824)



_Computers in Biology and Medicine 169 (2024) 107900_


[11] L. Wu, S. Chen, X. Gao, Y. Zheng, J. Hao, Detecting and learning against un
known opponents for automated negotiations, in: D.N. Pham, T. Theeramunkong,
G. Governatori, F. Liu (Eds.), PRICAI 2021: Trends in Artificial Intelligence,
[Springer International Publishing, Cham, 2021, pp. 17–31, http://dx.doi.org/10.](http://dx.doi.org/10.1007/978-3-030-89370-5_2)

[1007/978-3-030-89370-5_2.](http://dx.doi.org/10.1007/978-3-030-89370-5_2)

[12] A. Grigoriu, A. Zaveri, G. Weiss, M. Dumontier, SIENA: Semi-automatic semantic

enhancement of datasets using concept recognition, J. Biomed. Semant. 12 (1)
[(2021) 5, http://dx.doi.org/10.1186/S13326-021-00239-Z.](http://dx.doi.org/10.1186/S13326-021-00239-Z)

[13] S. Chen, Y. Yang, R. Su, Deep reinforcement learning with emergent commu
nication for coalitional negotiation games, Math. Biosci. Eng. 19 (5) (2022)
[4592–4609, http://dx.doi.org/10.3934/mbe.2022212.](http://dx.doi.org/10.3934/mbe.2022212)

[14] W. Chao, Z. Quan, A machine learning method for differentiating and predicting

human-infective coronavirus based on physicochemical features and composition
[of the spike protein, Chin. J. Electron. 30 (EN20210502) (2021) 815, http:](http://dx.doi.org/10.1049/cje.2021.06.003)
[//dx.doi.org/10.1049/cje.2021.06.003.](http://dx.doi.org/10.1049/cje.2021.06.003)

[15] S. Chen, R. Su, An autonomous agent for negotiation with multiple communi
cation channels using parametrized deep Q-network, Math. Biosci. Eng. 19 (8)
[(2022) 7933–7951, http://dx.doi.org/10.3934/mbe.2022371.](http://dx.doi.org/10.3934/mbe.2022371)

[16] C. Zhang, D. Fang, S. Sen, X. Li, Z. Feng, W. Xue, D. An, X. Zhao, R. Chen,

Opinion dynamics in gossiper-media networks based on multiagent reinforcement
[learning, IEEE Trans. Netw. Sci. Eng. 10 (2) (2023) 1143–1156, http://dx.doi.](http://dx.doi.org/10.1109/TNSE.2022.3229770)

[org/10.1109/TNSE.2022.3229770.](http://dx.doi.org/10.1109/TNSE.2022.3229770)

[17] R. Su, H. Yang, L. Wei, S. Chen, Q. Zou, A multi-label learning model for

predicting drug-induced pathology in multi-organ based on toxicogenomics data,
[PLoS Comput. Biol. 18 (9) (2022) 1–28, http://dx.doi.org/10.1371/journal.pcbi.](http://dx.doi.org/10.1371/journal.pcbi.1010402)

[1010402.](http://dx.doi.org/10.1371/journal.pcbi.1010402)

[18] S. Chen, Q. Sun, H. You, T. Yang, J. Hao, Transfer learning based agent for

automated negotiation, in: Proceedings of the 2023 International Conference on
Autonomous Agents and Multiagent Systems, AAMAS 2023, London, United King[dom, 2023, ACM, 2023, pp. 2895–2898, http://dx.doi.org/10.5555/3545946.](http://dx.doi.org/10.5555/3545946.3599115)

[3599115.](http://dx.doi.org/10.5555/3545946.3599115)

[19] S. Chen, Y. Yang, H. Zhou, Q. Sun, R. Su, DNN-PNN: A parallel deep neural

network model to improve anticancer drug sensitivity, Methods 209 (2023) 1–9,
[http://dx.doi.org/10.1016/j.ymeth.2022.11.002.](http://dx.doi.org/10.1016/j.ymeth.2022.11.002)

[[20] S. Vilar, R. Harpaz, E. Uriarte, L. Santana, R. Rabadan, C. Friedman, Drug—](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb20)

[drug interaction through molecular structure similarity analysis, J. Am. Med.](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb20)
[Inf. Assoc. 19 (6) (2012) 1066–1074.](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb20)

[[21] J.Y. Ryu, H.U. Kim, S.Y. Lee, Deep learning improves prediction of drug–drug](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb21)

[and drug–food interactions, Proc. Natl. Acad. Sci. 115 (18) (2018) E4304–E4311.](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb21)

[[22] Y. Deng, X. Xu, Y. Qiu, J. Xia, W. Zhang, S. Liu, A multimodal deep learning](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb22)

[framework for predicting drug–drug interaction events, Bioinformatics 36 (15)](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb22)

[(2020) 4316–4322.](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb22)

[[23] T. Lyu, J. Gao, L. Tian, Z. Li, P. Zhang, J. Zhang, MDNN: A multimodal deep](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb23)

[neural network for predicting drug-drug interaction events, in: IJCAI, 2021, pp.](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb23)

[3536–3542.](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb23)

[[24] L. Guo, X. Lei, M. Chen, Y. Pan, Msresg: Using GAE and residual GCN to predict](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb24)

[drug–drug interactions based on multi-source drug features, Interdiscipl. Sci.:](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb24)
[Comput. Life Sci. (2023) 1–18.](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb24)

[[25] S. Lin, Y. Wang, L. Zhang, Y. Chu, Y. Liu, Y. Fang, M. Jiang, Q. Wang, B.](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb25)

[Zhao, Y. Xiong, et al., MDF-SA-DDI: Predicting drug–drug interaction events](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb25)
[based on multi-source drug fusion, multi-source feature fusion and transformer](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb25)
[self-attention mechanism, Brief. Bioinform. 23 (1) (2022) bbab421.](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb25)

[[26] Y. Yu, K. Huang, C. Zhang, L.M. Glass, J. Sun, C. Xiao, SumGNN: Multi-](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb26)

[typed drug interaction prediction via efficient knowledge graph summarization,](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb26)
[Bioinformatics 37 (18) (2021) 2988–2995.](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb26)

[[27] Y.-H. Feng, S.-W. Zhang, J.-Y. Shi, DPDDI: A deep predictor for drug-drug](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb27)

[interactions, BMC Bioinformatics 21 (1) (2020) 1–15.](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb27)

[[28] X. Lin, Z. Quan, Z.-J. Wang, T. Ma, X. Zeng, KGNN: Knowledge graph neural](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb28)

[network for drug-drug interaction prediction., in: IJCAI, 2020, pp. 2739–2745.](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb28)

[[29] X. Su, L. Hu, Z. You, P. Hu, B. Zhao, Attention-based knowledge graph](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb29)

[representation learning for predicting drug-drug interactions, Brief. Bioinform.](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb29)
[23 (3) (2022) bbac140.](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb29)

[[30] Z.-H. Ren, Z.-H. You, C.-Q. Yu, L.-P. Li, Y.-J. Guan, L.-X. Guo, J. Pan, A](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb30)

[biomedical knowledge graph-based method for drug–drug interactions prediction](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb30)
[through combining local and global features with deep neural networks, Brief.](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb30)
[Bioinform. 23 (5) (2022) bbac363.](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb30)

[[31] K.T. Schütt, F. Arbabzadah, S. Chmiela, K.R. Müller, A. Tkatchenko, Quantum-](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb31)

[chemical insights from deep tensor neural networks, Nat. Commun. 8 (1) (2017)](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb31)

[1–8.](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb31)

[32] S. Zhang, Y. Liu, L. Xie, Molecular mechanics-driven graph neural network with

[multiplex graph for molecular structures, 2020, arXiv preprint arXiv:2011.07457.](http://arxiv.org/abs/2011.07457)

[[33] H. Altae-Tran, B. Ramsundar, A.S. Pappu, V. Pande, Low data drug discovery](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb33)

[with one-shot learning, ACS Central Sci. 3 (4) (2017) 283–293.](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb33)

[[34] T. Hasebe, Knowledge-embedded message-passing neural networks: Improving](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb34)

[molecular property prediction with human knowledge, ACS Omega 6 (42) (2021)](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb34)

[27955–27967.](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb34)



10


_S. Chen et al._


[[35] Y. Chen, T. Ma, X. Yang, J. Wang, B. Song, X. Zeng, MUFFIN: Multi-scale](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb35)

[feature fusion for drug–drug interaction prediction, Bioinformatics 37 (17) (2021)](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb35)

[2651–2658.](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb35)

[[36] D. Weininger, SMILES, a chemical language and information system. 1. Intro-](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb36)

[duction to methodology and encoding rules, J. Chem. Inf. Comput. Sci. 28 (1)](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb36)

[(1988) 31–36.](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb36)

[[37] A.P. Bento, A. Hersey, E. Félix, G. Landrum, A. Gaulton, F. Atkinson, L.J. Bellis,](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb37)

[M. De Veij, A.R. Leach, An open source chemical structure curation pipeline](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb37)
[using RDKit, J. Cheminformatics 12 (2020) 1–16.](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb37)

[[38] W. Hamilton, Z. Ying, J. Leskovec, Inductive representation learning on large](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb38)

[graphs, in: Advances in Neural Information Processing Systems, vol. 30, 2017.](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb38)



_Computers in Biology and Medicine 169 (2024) 107900_


[39] K. He, X. Zhang, S. Ren, J. Sun, Deep residual learning for image recognition,

in: 2016 IEEE Conference on Computer Vision and Pattern Recognition, CVPR,
[2016, pp. 770–778, http://dx.doi.org/10.1109/CVPR.2016.90.](http://dx.doi.org/10.1109/CVPR.2016.90)

[[40] A.K. Nyamabo, H. Yu, Z. Liu, J.-Y. Shi, Drug–drug interaction prediction with](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb40)

[learnable size-adaptive molecular substructures, Brief. Bioinform. 23 (1) (2022)](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb40)

[bbab441.](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb40)

[[41] A.K. Nyamabo, H. Yu, J.-Y. Shi, SSI–DDI: Substructure–substructure interactions](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb41)

[for drug–drug interaction prediction, Brief. Bioinform. 22 (6) (2021) bbab133.](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb41)

[[42] N.P. Tatonetti, P.P. Ye, R. Daneshjou, R.B. Altman, Data-driven prediction of](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb42)

[drug effects and interactions, Sci. Transl. Med. 4 (125) (2012) 125ra31.](http://refhub.elsevier.com/S0010-4825(23)01365-3/sb42)



11


