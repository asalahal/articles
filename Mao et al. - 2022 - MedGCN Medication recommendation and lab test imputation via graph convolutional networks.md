## MedGCN: Medication Recommendation and Lab Test Imputation via Graph Convolutional Networks

Chengsheng Mao [1], Liang Yao [1], and Yuan Luo [1]


1 Department of Preventive Medicine, Feinberg School of Medicine,
Northwestern University, Chicago, IL, USA


**Abstract**


Laboratory testing and medication prescription are two of the most
important routines in daily clinical practice. Developing an artificial intelligence system that can automatically make lab test imputations and
medication recommendations can save costs on potentially redundant lab
tests and inform physicians of a more effective prescription. We present
an intelligent medical system (named MedGCN) that can automatically
recommend the patients’ medications based on their incomplete lab tests,
and can even accurately estimate the lab values that have not been taken.
In our system, we integrate the complex relations between multiple types
of medical entities with their inherent features in a heterogeneous graph.
Then we model the graph to learn a distributed representation for each
entity in the graph based on graph convolutional networks (GCN). By the
propagation of graph convolutional networks, the entity representations
can incorporate multiple types of medical information that can benefit
multiple medical tasks. Moreover, we introduce a cross regularization
strategy to reduce overfitting for multi-task training by the interaction
between the multiple tasks. In this study, we construct a graph to associate 4 types of medical entities, i.e., patients, encounters, lab tests,
and medications, and applied a graph neural network to learn node embeddings for medication recommendation and lab test imputation. we
validate our MedGCN model on two real-world datasets: NMEDW and

MIMIC-III. The experimental results on both datasets demonstrate that
our model can outperform the state-of-the-art in both tasks. We believe
that our innovative system can provide a promising and reliable way to
assist physicians to make medication prescriptions and to save costs on
potentially redundant lab tests.


**Keywords:** medication recommendation; lab test imputation; graph convolutional networks; multi-task learning; electronic health records.


### **1 Introduction**

With the increasing adoption of Electronic Health Records (EHRs), more and
more researchers are working to mine clinical data to derive new clinical knowledge with a goal of enabling greater diagnostic precision, better personalized
therapeutic recommendations in order to improve patient outcomes and more
efficiently utilize health care resources [1]. One of the key goals of precision
medicine is the ability to suggest personalized therapies to patients based on
their molecular and pathophysiologic profiles [1]. Much work has been devoted
to pharmacogenomics that investigates the link between patient molecular profiles and drug response [2]. However, the status quo is that patient genomics
data is often limited while clinical phenotypic data is ubiquitously available.
Thus there is great potential in linking patient pathophysiologic profiles to medication recommendation, a direction we termed as “pharmacophenomics” that
is understudied yet will soon become imperative. This approach is particularly
interesting for cancer treatment given the heterogeneous nature of the disease.
In treating cancer patients, physicians typically prescribe medications based on
their knowledge and experience. However, due to knowledge gaps or unintended
biases, oftentimes these clinical decisions can be sub-optimal.
On the other hand, the data quality issue often represents one of the major impediments of utilizing Electronic Health Record (EHR) data [3]. Unlike
experimental data that are collected per a research protocol, the primary role
of clinical data is to help clinicians care for patients, so the procedures for its
collection are often neither systematic nor on a regular schedule but rather
guided by patient condition and clinical or administrative requirements. Thus,
many aspects of patients’ clinical states may be unmeasured, unrecorded and
unknown in most patients at most time points. While this missing data may
be fully clinically appropriate, machine learning algorithms cannot directly accommodate missing data. Accordingly, missing clinical phenotypic data (e.g.,
laboratory test data) can hinder EHR knowledge discovery and medication recommendation efforts.
Accurate lab test imputation can save costs of potentially redundant lab
tests, and precise medication recommendations can assist physicians to formulate effective prescriptions. Developing an artificial intelligence (AI) medical
system that can automatically and simultaneously perform the two medical
tasks can help the medical institutions improve medical efficiency and reduce
medical burden. However, most of the previous studies on intelligent medical
systems performed the two tasks independently and ignored the interaction between them [4, 5, 6, 7, 8, 9, 10]. Although most medication recommendation
methods are based on the imputed lab tests, the error of lab test imputation is
propagated to medication recommendation, increasing the error.
However, as we all know, interactions can exist among various types of medical entities. For example, the lab test values usually indicate the health status of
a patient, thus guide the medication recommendations. On the contrary, taking
certain medications usually affects the lab test values. Since there are close and
complex associations between medical entities, in this article, we incorporate the


2


#### Encounter 1 Encounter 2 Encounter 3 Encounter 4 Lab 1 Lab 2 Lab 3 Lab 4 Lab 5 Lab 6

Figure 1: An example of MedGraph. The solid lines represent there are observed
relations between the two objects; the dash lines represent the relation between
the two objects are unknown.


complex relations between multiple types of medical entities with their inherent
features in a heterogeneous graph, named Medical Graph (MedGraph). Figure
1 shows a MedGraph consisting of four types of nodes (i.e., encounters, patients,
labs and medications), each of which could have their inherent features if they
are available, e.g., demographic features for patients and the chemical composition for medications. Graph edges encode the relations between the medical
entities, for example, a patient may have several encounters, an encounter may
include some lab tests with certain values and some medication prescriptions.
Figure 1 shows scenarios where we may need to impute missing lab test results
(e.g., Encounters 2, 3 and 4) and may need to recommend partial or a full list
of medications (e.g., Encounters 3 and 4).


Recently, Graph Convolutional Networks (GCN) attracted wide attention for
inducing informative representations of nodes and edges in graphs, and could be
effective for tasks that could have rich relational information between different
entities [11, 12, 13, 14, 15, 16, 17]. However, there are two issues that must
be effectively tackled before extending the original GCN to MedGraph: (1) the


3


original GCN was designed for homogeneous graphs where all the nodes are of
the same type and have the same feature names. (2) the original GCN could
not handle missing values in the node features. In medical scenes, a MedGraph
could be heterogeneous and have multiple types of nodes with different features,
and there usually are many missing values in the feature matrix. In this article,
based on the idea of GCN, we develop Medical Graph Convolutional Network
(MedGCN) that is able to tackle the above issues for MedGraph and to learn
informative distributed representations for nodes in MedGraph for better medication recommendation and lab test imputation. Our source code of MedGCN
is available at `https://github.com/mocherson/MedGCN` .
The main contributions of this article are as follows. From the medical perspective, we provide an effective way of medication recommendation and lab
test imputation based on EHR data. Moreover, in our experiments on real
EHR data, our method MedGCN can significantly outperform the state-of-theart methods for both medication recommendation and lab test imputation tasks.
From the method perspective, we innovatively incorporate the complex associations between multiple medical entities into MedGraph, and develop MedGCN
to learn the entity representations based on MedGraph for multiple medical
tasks. From the AI perspective, (1) The developed MedGCN is able to perform
multiple medical tasks within one model, including medication recommendations and lab test imputation. (2) MedGCN extends the GCN model to heterogeneous graphs and missing feature values in medical settings. (3) We introduce
cross regularization, a specific form of multi-task learning to reduce overfitting
for the training of MedGCN. (4) MedGCN is a general inductive model that
could use the learned model to efficiently generate representations for new data.

### **2 Related Work**


In this section, we briefly review the literature on representation learning in
the medical domain. More specifically, we focus on the tasks of medication
recommendation and lab test imputation.


**2.1** **Medical Representation Learning**


Representation learning aims to embed an object into a low-dimensional vector space while preserving the characteristics of the object as much as possible.
Medical representation learning tries to generate vector representations for medical entities, e.g., patients and encounters. eNRBM [18] embedded patients into
into a low-dimensional vector space with a modified restricted Boltzmann machine (RBM) for suicide risk stratification. DeepPatient [19] derived patient
representation from EHR data to facilitate clinical predictive modeling. Sun et
al. [20] derived patient representations for disease prediction based on a medical
concept graph and a patient record graph via graph neural networks. DoctorAI

[21] applied a gated recurrent unit (GRU)-based RNN model to embed patients for predicting the diagnosis codes in the next visit of patients. Med2Vec


4


[22] learned the vector representations of medical codes and visits from large
EHR datasets via a multi-layer perceptron (MLP) for clinical risk prediction.
HORDE [23], a unified graph representation learning framework, embedded patients’ visits into a harmonized space for diagnosis code prediction and patient
severity classification. Decagon [24] generated drug representations by a graph
convolutional encoder to model polypharmacy side effects. HSGNN [25] generated visit representation via graph neural network for diagnosis prediction.
MGATRx [26] generated vector representations for drugs and diseases for drug
repositioning. Most of the previous models focus on representations for only one
type of medical entities (e.g., patients or encounters) and ignore the representation for entities of different types. Some models (e.g., Med2Vec, HORDE and
HSGNN) can generate embeddings of multiple specific types of medical entities,
but cannot be extended to more medical entities like lab test. Also, few of the
previous works are designed for multiple tasks in one training. In our work, we
incorporate multiple types of medical entities and their interactions into MedGraph which can be easily extended by adding new types of medical entities
and their relations to exist entities. Moreover, the proposed MedGCN model
can generate embeddings for all the nodes (medical entities) in MedGraph, thus
feasible for multiple tasks involving the medical entities.


**2.2** **Graph Representation Learning on EHR**


Graph representation learning tries to learn a vector representation for part
or the whole of a graph for certain graph learning tasks, e.g., learning node
representation for node classification. As a simple and effective graph learning
model, GCN is increasingly applied to the medical domain for EHR mining and
analysis. Since EHR usually contains multiple medical entities and multiple
relations, the graphs derived from EHR are usually heterogeneous, thus, the
original GCN cannot be directly applied to heterogeneous graphs. We found a
number of methods in the literature applied GCN to graphs from EHR. HORDE

[23] constructed a multi-modal EHR graph caontaining at least 3 types of nodes,
since node features are not available, HORDE ignored the node types and applied GCN to the EHR graph to learn node representations for diagnosis code
prediction and patient severity classification. Sun et al. [20] constructed a medical concept graph and a patient record graph for disease prediction; both graphs
are a bipartite graph with two types of nodes. They ignored the node types in
the propagation rule, but applied a projection weight to align all node embeddings onto the same space. SMGCN [27] construct multiple graphs (i.e., the
symptom-herb bipartite graph, symptom-symptom graph, and herb-herb graph)
for herb recommendation; it used a bipar-GCN for information propagation on
the symptom-herb bipartite graph. In our work, we decompose a heterogeneous
graph into multiple bipartite graphs or homogeneous graphs where the message
passing to a certain node is from only one type of nodes, then aggregate all the
messages the node received to generate the node embedding. From this view,
the bipar-GCN is a special case of our model for bipartite graphs.
MGATRx [26] constructed a graph with 6 node types from different EHR


5


data, and then applied attention mechanism to extend the multi-view sum pooling for drug repositioning. MGATRx and our model MedGCN are both based
on GCN for heterogeneous graphs. But there are server major differences between our work. (1) We have different medical tasks, MGATRx is for drug
repositioning. MedGCN is for medication recommendation and lab task imputation. (2) The constructed graphs are different. The graph in MGATRx has
6 types of nodes (i.e., drug, disease, target, substructure, side effect, Mesh),
while our graph contains 4 types of nodes (patients, medications, labs, encounters). The only common node type is drug/medication. (3) Our MedGCN
has a wide range of applicability, MGATRx paper also applied MedGCN to its
graph for drug repositioning and achieved good performance, though not better
than MGATRx. But it is not straightforward to apply MGATRx to weighted
heterogeneous graphs which is exactly our case.


**2.3** **Medication Recommendation**


Following the categorization from [4], there are mainly two types of deep learning methods designed for medication recommendation: instance-based methods
and Longitudinal methods. instance based methods perform recommendations
ignoring the longitudinal patient history, e.g. LEAP [6] formulated treatment
recommendation as a reinforcement learning problem to predict combination
of medications given patient’s diagnoses. SMGCN [27] applied GCN to multiple graphs to describe the complex relations between symptoms and herbs
for herb recommendation. Longitudinal methods leverage the temporal dependencies within longitudinal patient history to predict future medication, e.g.,
DMNC [28] considers the interactions between two sequences of drug prescription and disease progression in the model. GAMENet [4] models longitudinal
patient records in order to provide safe medication recommendation. G-BERT

[29] combined GNN and transformer models to generate visit representations for
medication recommendation. However, these work focused on one task while
not considering how to integrate with other tasks. Moreover, most of the previous work for medication recommendation took only the diagnosis code as an
input and were unable to directly accommodate lab tests as input. In this work,
to release the workload of physicians in diagnostic reasoning, our model makes
the recommendation based on a graph containing lab tests and patients rather
than the diagnosis codes.


**2.4** **Lab Test Imputation**


Clinical data often contain missing values for test results. Imputation uses
available data and relationships contained within it to predict point or interval estimates for missing values. Numerous imputation algorithms are available

[8, 30, 9, 31], many of these are designed for cross-sectional imputation (measurements at the same time point). Recent imputation studies have attempted
to model the time dimension [32, 10, 33], but they generally consider all time
points to occur within the same patient encounters where the temporal corre

6


lation between these time points are strong enough to contribute to imputation
accuracy. In our work, the lab test imputation is based on a MedGraph that
incorporates multiple types of medical entities and their relations.

### **3 Methods**


**3.1** **Problem Formulation**


Since an encounter may include multiple medications, we model medication
recommendation as a multi-label classification problem. For each medication,
the model should output the recommendation probability. If an informative
representation can be learned for each encounter, this problem can be tackled
by many traditional machine learning algorithms. Thus the problem is how
to effectively integrate the lab information and patient information into the
representations of encounters.
Missing lab values imputation is another challenge in encounter representation learning. Using an encounter-by-lab matrix, the imputation problem can
be formulated as a matrix completion problem. Common imputation methods such as mean or median imputation will overlook the correlation between
different columns. In the medical setting, the concurrent information such as
patients’ baselines and medications are also useful for lab test imputation. We
integrate all this information in the encounter representations to help impute
the missing labs.
Both the two tasks require informative encounter representations. Thus, we
can conduct the two tasks with one model MedGCN where each node can get
information from its neighbors in a layer, and eventually, the encounter nodes
could learn informative representations that can be used for both medication
recommendation and lab test imputation. In addition, we can train the model
with a cross regularization strategy by which the loss of one task can be regarded
as a regularization item for the other task.


**3.2** **MedGraph**


We define MedGraph as a specialized graph _G_ = ( _V, E_ ) where the nodes _V_
consist of medical entities and the edges _E_ consist of the relations between the
medical entities. MedGraph is a heterogeneous graph where the nodes _V_ consist
of multiple types of medical entities. Our goal is to learn a vector representation
for each node in the graph.
Since GCN cannot directly handle missing values in node features, this leads
to an issue for GCN to consider labs as features of encounters due to many
missing values in labs. Fortunately, a graph can accept an empty edge between
two nodes, thus, we represent the labs as nodes of type “lab” that connect to
the encounter nodes, i.e., we construct a bipartite subgraph between labs and
encounters as a part of MedGraph.


7


E1


E2


E3


E4



E1


E2


E3


E4



E1


E2


E3


E4







E1


E2


E3


E4






|P1|P2|P3|
|---|---|---|
|1|0|0|
|0|1|0|
|0|1|0|
|0|0|1|


|M1|M2|M3|
|---|---|---|
|1|0|0|
|0|1|1|
|0|0|0|
|0|0|0|


|L1|L2|L3|L4|L5|L6|
|---|---|---|---|---|---|
|.3|0|.1|0|0|0|
|0|.1|0|0|.5|0|
|0|0|.1|.8|0|0|
|0|0|.0|0|0|.6|


|L1|L2|L3|L4|L5|L6|
|---|---|---|---|---|---|
|1|1|1|0|0|0|
|0|1|0|0|1|0|
|0|0|1|1|0|0|
|0|0|1|0|0|1|



𝐴𝐴 𝐸𝐸×𝑃𝑃 𝐴𝐴 𝐸𝐸×𝑀𝑀 𝐴𝐴 𝐸𝐸×𝐿𝐿 𝑀𝑀 𝐸𝐸×𝐿𝐿


Figure 2: Adjacency matrices corresponding to MedGraph, _A_ _E×P_ : adjacency
matrix between encounters and patients; _A_ _E×M_ : adjacency matrix between
encounters and medications; _A_ _E×L_ : adjacency matrix between encounters and
labs; _M_ _E×L_ : mask matrix between encounters and labs. Abbreviations: P patient; E - encounter; M - medication; L - lab.


Figure 1 illustrates an example of our MedGraph where the nodes consist
of four types of medical entities, i.e., Encounters, Patients, Labs, Medications,
each node could have inherent features within it. The relations between two

types of nodes correspond to an adjacency matrix. We defined the relations
between two types of nodes as follows, and the example adjacency matrices
corresponding to the MedGraph in Figure 1 are shown in Figure 2.


  - The relations between encounters and patients are defined in matrix _A_ _E×P_
in Figure 2. If an encounter belongs to a patient, there is an edge between
the two nodes, the corresponding position of the adjacency matrix _A_ _E×P_
is set to 1, otherwise, 0. Since one encounter must belong to one and only
one patient, there must be only one “1” in each row of the matrix _A_ _E×P_ .


  - The relations between encounters and labs are defined in matrix _A_ _E×L_
in Figure 2. If an encounter includes a lab test, there is a weighted edge
between the two nodes, the corresponding position of the adjacency matrix
_A_ _E×L_ is set to the test value scaled to [0 _,_ 1] by min-max normalization
as Eq. 11, otherwise, 0. To discriminate the true test values “0” and the
missing values, we introduce the mask matrix _M_ _E×L_ with the same shape
as _A_ _E×L_, where the entries corresponding to true test values are set to 1,
and the entries corresponding to missing values are set to 0. For example,
if the L1 test value of E2 is 0, the (E2, L1) position is 0 in the _A_ _E×L_
matrix, but is 1 in the mask _M_ _E×L_ matrix (Figure 2).


  - The relations between encounters and medications are defined in matrix
_A_ _E×M_ in Figure 2. If an encounter is ordered a medication, there is an
edge between the two nodes, the corresponding position of the adjacency
matrix _A_ _E×M_ is set to 1.


  - The connections between nodes of the same type are all self-connections,
the corresponding adjacency matrices are all set as the identity matrix.


In practice, there could be more nodes, more types of medical entities, and
more relations that can be incorporated into MedGraph. For example, if medi

8


cation similarities are known, we could construct a _N_ _M_ _×N_ _M_ matrix to represent
the relations between medications. However, such similarity information often
requires external knowledge sources that are not always available and may be
subjective, thus we only consider the above 4 types of relations in this study.
Note that we did not use diagnosis codes for two reasons. Diagnosis codes often
are recorded for billing purposes and may not reflect patients’ true pathology.
Moreover, we want to make MedGCN more practically useful by not asking
physicians to do the heavy lifting in diagnostic reasoning. Nevertheless, the
diagnosis codes can be easily added to the MedGraph as a new node type for
medication recommendation and lab test imputation.


**3.3** **Graph Convolutional Networks**


Recently, graph convolutional networks (GCN) attracted wide attention for inducing informative representations of nodes and edges, and could be effective
for tasks that could have rich relational information between different entities

[12, 13, 34, 14, 15, 16, 35].


**3.3.1** **General GCN**


GCN learns node representations based on the node features and their connections with the following propagation rule for an undirected graph [12]:




[1]

2 ˜ _A_ ˜ _D_ _[−]_ 2 [1]



_H_ [(] _[k]_ [+1)] = _φ_ ( _D_ [˜] _[−]_ 2 [1]



2 _H_ [(] _[k]_ [)] _W_ [(] _[k]_ [)] ) (1)



where _A_ [˜] = _A_ + _I_ is the adjacency matrix with added self-connection, _D_ [˜] is a
diagonal matrix with _D_ [˜] _ii_ = [�] _j_ _[A]_ [˜] _[ij]_ [,] _[ H]_ [(] _[k]_ [)] [ and] _[ W]_ [ (] _[k]_ [)] [ are the node representation]

matrix and the trainable parameter matrix for the _k_ th layer, respectively; _H_ [(0)]


_·_
can be regarded as the original feature matrix, _φ_ ( ) is the activation function.
The propagation rule of GCN can be interpreted as the Laplacian smoothing

[36], i.e., the new representation of a node is computed as the weighted average
of itself and its neighbors, followed by a linear transformation. Further, GCN
can be generalized to a graph neural network where each node can get and
integrate messages from its neighborhood to update its representation [37, 38],
i.e., a node _v_ ’s representation is updated by



_H_ [(] _[k]_ [+1)] ( _v_ ) = _f_ 1 _W_ 1 [(] _[k]_ [)]



_H_ [(] _[k]_ [)] ( _v_ ) _, f_ 2 _W_ 2 [(] _[k]_ [)] ( _{H_ [(] _[k]_ [)] ( _w_ ) _|w ∈_ _N_ ( _v_ ) _}_ ) (2)
� �



where _N_ ( _v_ ) is the neighborhood of node _v_, _f_ 2 _[W]_ [2] aggregates over the set of
neighborhood features and _f_ 1 _[W]_ [1] merges the node and its neighborhood’s representations. Both _f_ 1 _[W]_ [1] and _f_ 2 _[W]_ [2] should be arbitrary differentiable functions.
Since a neighborhood is a set that is permutation-invariant, the aggregation
function _f_ 2 _[W]_ [2] should also be permutation-invariant. If _f_ 1 _[W]_ [1] and _f_ 2 _[W]_ [2] are sum
operations, Eq. 2 is simplified as Eq. 3 [39].



_H_ [(] _[k]_ [+1)] ( _v_ ) = _φ_







�
 _w∈N_ ( _v_ )



� _H_ [(] _[k]_ [)] ( _w_ ) _· W_ [(] _[k]_ [)]

_w∈N_ ( _v_ ) _∪{v}_





(3)




9


Eq. 3 is designed for a homogeneous graph where all the samples in _N_ [˜] ( _v_ ) =
_N_ ( _v_ ) _∪{v}_ share the same linear transformation _W_ [(] _[k]_ [)] . It can be shown that
Eq. 3 is equivalent to Eq. 1 except for the normalization coefficients.


**3.3.2** **Heterogeneous GCN**


Since the multiple types of nodes imply multiple types of edges in any heterogeneous graphs, the key problem of applying GCN to heterogeneous graphs is how
to discriminate different types of edges in a heterogeneous graph when doing
the propagation. In our method, we decompose the heterogeneous graph into
multiple subgraphs with each subgraph has only one type of edge which can be
represented by a single adjacency matrix. In each GCN layer, the representations of each node in all the subgraphs are aggregated as the node representation.
Based on this idea, Eq. 3 can be extended to multiple types of edges as Eq. 4.





(4)




_H_ [(] _[k]_ [+1)] ( _v_ ) = _φ_







_n_
�


_i_ =1





_i_ =1



� _H_ [(] _[k]_ [)] ( _w_ ) _· W_ _i_ [(] _[k]_ [)]

_w∈N_ [˜] _i_ ( _v_ )



where _N_ [˜] _i_ ( _v_ ) is the neighborhood of _v_ ( _v_ included) in terms of _i_ th edge type.
Eq. 4 can also be written in matrix form as



� _A_ _ij_ _· H_ _j_ [(] _[k]_ [)] _· W_ _ij_ [(] _[k]_ [)]

_j_ =1



_H_ _i_ [(] _[k]_ [+1)] = _φ_







_n_
�
 _j_ =1





(5)




where _H_ _i_ [(] _[k]_ [)] is the presentation of the _i_ th type nodes in the _k_ th layer, _A_ _ij_ is
the adjacency matrix between nodes type _i_ and _j_, _W_ _ij_ [(] _[k]_ [)] is the learnable linear
transformation matrix indicating how type _j_ nodes contribute to type _i_ nodes’
representation in layer _k_ . [1]

Many existing works have extended GCN to heterogeneous graphs like ([39,
40, 24]), and quite a few works represented EHR as graphs and employed graphbased method for certain medical tasks [25, 26, 29]. However, the MedGraph
involved in our work is a specific heterogeneous graph that can incorporate
weighted edges, and different from general weighted graphs, edge with weight
_≤_ 0 can exist, because a lab value can be 0 or negative (e.g., base excess).
Although many existing GCN models can deal with a general heterogeneous
graph, few of the models can be applied to MedGraph directly. The significance
of our work is to introduce to GCN model for a more specific heterogeneous
graph like MedGraph.


**3.4** **MedGCN**


Eq. 5 is a general formula for GCN propagation in a heterogeneous graph.
Applying the propagation rule Eq. 5 to our MedGraph defined in Section 3.2,


1 This can be scaled to a multi-graph where multiple types of edges between two nodes
exist by aggregating multiple items of the _A · H · W_ .


10


we design our MedGCN architecture as shown in Figure 3; and the propagation
for different types of entities are


_H_ _e_ [(] _[k]_ [+1)] = _φ_ � _A_ _E×P_ _·H_ _p_ [(] _[k]_ [)] _·W_ _p_ [(] _[k]_ [)] + _A_ _E×L_ _·H_ _l_ [(] _[k]_ [)] _·W_ _l_ [(] _[k]_ [)] + _A_ _E×M_ _·H_ _m_ [(] _[k]_ [)] _[·][W]_ [ (] _m_ _[k]_ [)] [+] _[H]_ _e_ [(] _[k]_ [)] _·W_ _e_ [(] _[k]_ [)] �



_H_ _p_ [(] _[k]_ [+1)] = _φ_ ( _A_ _P ×E_ _·H_ _e_ [(] _[k]_ [)] _·W_ _e_ [(] _[k]_ [)] + _H_ _p_ [(] _[k]_ [)] _·W_ _p_ [(] _[k]_ [)] [)]

_H_ _l_ [(] _[k]_ [+1)] = _φ_ � _A_ _L×E_ _·H_ _e_ [(] _[k]_ [)] _·W_ _e_ [(] _[k]_ [)] + _H_ _l_ [(] _[k]_ [)] _·W_ _l_ [(] _[k]_ [)] �

_H_ _m_ [(] _[k]_ [+1)] = _φ_ ( _A_ _m×E_ _·H_ _e_ [(] _[k]_ [)] _·W_ _e_ [(] _[k]_ [)] + _H_ _m_ [(] _[k]_ [)] _[·][W]_ [ (] _m_ _[k]_ [)] [)]



(6)



where _A_ _E×P_, _A_ _E×M_ and _A_ _E×L_ are defined as in Section 3.2, their transposes
are denoted as _A_ _P ×E_, _A_ _L×E_ and _A_ _M_ _×E_, respectively, _H_ _p_ [(] _[k]_ [)] [,] _[ H]_ _e_ [(] _[k]_ [)], _H_ _m_ [(] _[k]_ [)] [, and]
_H_ _l_ [(] _[k]_ [)] are the representations of the corresponding type of nodes at layer _k_ . We
make all the transformations from the same type of nodes share parameters
to reduce model complexity, _W_ _p_ [(] _[k]_ [)], _W_ _e_ [(] _[k]_ [)], _W_ _m_ [(] _[k]_ [)] [, and] _[ W]_ [ (] _l_ _[k]_ [)] are the linear transformation matrix from the corresponding type of nodes. Since the adjacency
matrix between the same node type is identical, we omit it in Eq. 6.
Because encounters have connections to patients, labs and medications, an
encounter node updates its representation based on information from neighbors
of these types and itself. Information propagation on other nodes is similar.
After propagation in every layer, each node representation incorporates information from its 1-hop neighbors. Eventually, in the last layer (assume _k_ th
layer), each node learns an informative distributed representation based on its
_k_ -hop neighborhood information. The representations are then inputted to two
different fully-connected neural networks _f_ _θ_ _M_ and _f_ _θ_ _L_ followed by the sigmoid
activation for medication recommendation and lab test imputation, respectively,
i.e.,
_P_ = _sigmoid_ ( _f_ _θ_ _M_ ( _H_ _e_ ))

(7)
_V_ = _sigmoid_ ( _f_ _θ_ _L_ ( _H_ _e_ ))


where _H_ _e_ is the final encounter representations, the output _P_ is a matrix of size
_N_ _E_ _× N_ _M_ with the entry _p_ _ij_ is the recommendation probability of medication
_j_ for encounter _i_ ; the output _V_ is a matrix of size _N_ _E_ _× N_ _L_ with the entry _v_ _ij_
is the normalized estimated value of lab _j_ for encounter _i_ . Here, _N_ _E_, _N_ _M_, and
_N_ _L_ is the number of encounters, medications and labs, respectively.


**3.5** **Model training**


**Loss function.** For the medication recommendation task, the true label is the
adjacency matrix _A_ _E×M_ . Since _A_ _E×M_ is very sparse, to favor the learning of
positive classes, we give a weight _N_ _n_ _/N_ _p_ to the positive classes, where _N_ _n_ and _N_ _p_
are the counts of ‘0’s and ‘1’s in _A_ _E×M_, respectively. Similar with the previous
work [41, 42], The binary cross entropy loss for medication recommendation is
defined as



_N_ _M_
�

_j_



_N_ _n_

_a_ _ij_ log _p_ _ij_ + (1 _−_ _a_ _ij_ ) log (1 _−_ _p_ _ij_ ) (8)

� _N_ _p_ �


11



_L_ _M_ ( _P, A_ ) = _−_ [1]

_N_ _E_



_N_ _E_
�


_i_


𝑀𝑀 1



















































……



……













𝑀𝑀 𝑁𝑁 𝑀𝑀


𝐿𝐿 1





























𝑚𝑚 𝑁𝑁 𝑀𝑀 𝐻𝐻 𝑚𝑚 𝑁𝑁 𝑀𝑀 𝐻𝐻 𝑚𝑚 𝑁𝑁 𝑀𝑀 𝑚𝑚 𝑁𝑁 𝑀𝑀 EstimationLab Test


MedGCN input and layer 1 MedGCN layer k MedGCN multi-task prediction





𝐿𝐿 𝑁𝑁 𝐿𝐿







Figure 3: Architecture of MedGCN. Shape and color of the nodes indicate different types of nodes. Here we use plate notation where the numbers in the corners
of plates ( _N_ _P_ _, N_ _L_ _, N_ _E_ _, N_ _M_ ) indicate repetitions of the same type of nodes in the
MedGraph. Edges with the same color possess shared weights (parameter sharing). P (patient), E (encounter), M (medication), L (lab) are MedGCN input
and _H_ _p_ [(] _[k]_ [)] _[, H]_ _e_ [(] _[k]_ [)] _, H_ _m_ [(] _[k]_ [)] _[, H]_ _l_ [(] _[k]_ [)] are hidden representations of corresponding nodes at
layer _k_ .


where _a_ _ij_ and _p_ _ij_ are the values at position ( _i, j_ ) in matrix _A_ and _P_ respectively.
For the lab test imputation task, the adjacency matrix _A_ _E×L_ contains the
true values as well as some ‘0’s for the missing values. We use the mask matrix
_M_ _E×L_ to screen the true targets for training. Since the lab value are continuous,
we use the mean square error loss for lab test imputation as



_N_ _L_
� _m_ _ij_ ( _v_ _ij_ _−_ _a_ _ij_ ) [2] (9)

_j_



_L_ _L_ ( _V, A_ ) = _−_ [1]

_N_ _E_



_N_ _E_
�


_i_



where _a_ _ij_, _v_ _ij_ and _m_ _ij_ are the values at position ( _i, j_ ) in matrix _A_, _V_ and _M_
respectively.
To learn a MedGCN model that can perform the two tasks, we use the
following loss
_L_ = _L_ _M_ ( _P, A_ _E×M_ ) + _λL_ _L_ ( _V, A_ _E×L_ ) (10)


where _λ_ is a regularization parameter used to adjust the proportion of the two
losses.

**Cross regularization.** Generally, for the medication recommendation task,
we only need to minimize _L_ _M_ loss, and for lab test imputation, we only need _L_ _L_
loss. For both tasks, we need to minimize both _L_ _M_ and _L_ _L_ . Thus, we achieve
the loss function Eq. 10 where _L_ _L_ can be seen as a regularization item for the
medication recommendation task, and _L_ _M_ can be seen as a regularization item
for the lab imputation task. We call this strategy cross regularization between
multiple tasks. Cross regularization is a specific form of multi-task learning for
our scenario to reduce overfitting. Generally, multi-task learning has a major
task and one or more auxiliary tasks, the auxiliary tasks serve the major task in
the form of regularization. However, in our scenario, the two tasks medication
recommendation and lab test imputation are both major tasks. They server


12


as a regularization item for each other. Beneficially, cross regularization could
make the learned representations more informative, because they would carry
information from other related tasks.

**Inductive learning.** The original GCN is considered transductive, i.e.,
it requires that all nodes including test nodes in the graph are present during
training, only the label to predict is unknown. This will make the model not
be able to generalize to unseen nodes. There were some inductive versions of
GCN that can generalize to unseen new samples [13, 14]. Our MedGCN can
also be implemented inductively based on Eq. 4. For a trained model, all its
node embeddings _H_ _∗_ [(] _[k]_ [)] and the learned parameters _W_ _∗_ [(] _[k]_ [)] are known, if a new
test sample _v_ and its connections to the graph are known, the embedding of
neighbors of _v_ could be retrieved to compute the embedding of _v_ by Eq. 4
without retraining the model.

When a new encounter comes, _H_ _e_ [(0)] has one more row with the original
feature of the new encounter and _A_ _E×P_, _A_ _E×L_, _A_ _E×M_ all have one more row
corresponding to the connection of the new encounter to the available patients,
labs and medications, respectively. Since the transformation matrix _W_ _e_ [(0)] has
been already trained, _H_ _e_ [(0)] _· W_ _e_ [(0)] has one more row corresponding to the new
encounter. Similarly, _A_ _E×P_ _·H_ _p_ [(0)] _[·][W]_ [ (0)] _p_, _A_ _E×L_ _·H_ _l_ [(0)] _·W_ _l_ [(0)] and _A_ _E×M_ _·H_ _m_ [(0)] _[·][W]_ [ (0)] _m_
all have one more row corresponding to the new encounter, thus _H_ _e_ [(1)] has one
more row for the new encounter by Eq. 6. Finally, _H_ _e_ [(] _[k]_ [)] has one more row
corresponding to the embedding of the new encounter.

To perform medication recommendation and lab test imputation for a new
encounter _e_ with certain lab test values unknown, we first add the encounter _e_
to MedGraph based on the known information according to the description in
Section 3.2 and then perform the propagation rules (i.e., Eq. 6) of the trained
MedGCN on the updated MedGraph. Eventually, we get the final representation
_h_ _e_ for the encounter. After processed by the fully-connected layers _f_ _θ_ _M_ and
_f_ _θ_ _L_, we get the output medication probability _p_ = _sigmoid_ ( _f_ _θ_ _M_ ( _h_ _e_ )), and the
estimated lab test value _v_ = _sigmoid_ ( _f_ _θ_ _L_ ( _h_ _e_ )), respectively, where _p_ is a vector
of _N_ _M_ components corresponding to the recommendation probabilities of the
_N_ _M_ medications, and _v_ is a vector of _N_ _L_ components corresponding to the
estimated values of the _N_ _L_ lab tests.

### **4 Experiments**


Our method can be applied to a wide variety of MedGraphs and medical tasks.
Here, we validated it on two real-world EHR datasets, NMEDW and MIMICIII, for medication recommendation and lab test imputation. NMEDW is a
dataset of patients with lung cancer from Enterprise Data Warehouse (EDW) of
Northwestern Medicine [43]. MIMIC-III [44] is a public EHR dataset comprising
information relating to patients admitted to critical care units over 11 years
between 2001 and 2012.


13


**4.1** **Data preparation**


For NMEDW dataset, we prepared the dataset by the following steps. (1) We
identified a cohort of patients with lung cancer using ICD-9 codes 162.* and
231.2 between 1/1/2004 and 12/31/2015, 5017 patients and their 112510 encounters were identified. (2) We considered the list of cancer drugs approved by
National Cancer Institute [2] for cancers or conditions related to cancer for medication recommendation, 518 cancer drugs were considered. (3) We excluded the
encounters that had no cancer drug medication records, 9638 encounters corresponding to 1277 patients were left. (4) We excluded the labs that only a few
( _<_ 1%) encounters took and the labs that were constant for all the encounters,
197 labs were identified. (5) We excluded the encounters that did not take any
of the 197 labs, getting 1260 encounters corresponding to 865 patients. (6) We
excluded the cancer drugs that were never prescribed to any of the remaining
1260 encounters, getting 57 cancer drug medications. Eventually, the NMEDW
dataset consists of 1260 encounters for 865 patients, they have taken 197 labs
and 57 medications related to lung cancer.
For MIMIC-III dataset, we prepared the dataset by the following steps. (1)
The same 518 cancer drugs are also considered on NMEDW dataset. (2) We
excluded the encounters (i.e., visits in MIMIC-III) that had no cancer drug
medication records, 18223 encounters corresponding to 15182 patients were left.
(3) We excluded the labs that only a few ( _<_ 1%) encounters took and the
labs that were constant for all the encounters, 219 labs were identified. (4) We
excluded the encounters that did not take any of the 219 labs, getting 18190
encounters corresponding to 15153 patients. (5) We excluded the cancer drugs
that were never prescribed to any of the remaining 18190 encounters, getting 117
cancer drug medications. Eventually, the MIMIC-III dataset consists of 18190
encounters for 15153 patients, they have taken 219 labs and 117 medications.
For both datasets, each lab value is scaled to [0,1] by min-max normalization

as
_v_ = _[v]_ _[or][g]_ _[ −]_ _[v]_ _[min]_ (11)

_v_ _max_ _−_ _v_ _min_


where _v_ _org_ _, v_ _min_ _, v_ _max_ are the original value, the maximum value and the minimum value of the corresponding lab test, respectively, and _v_ is the normalized
value. Based on the final dataset, we can construct a MedGraph similar to
Figure 1 as well as its adjacency matrix according to the criterion described in
Section 3.2, but we have more nodes and edges. The MedGraph information
constructed on NMEDW and MIMIC-III datasets is summarized in Table 1 and

2, respectively.
For both datasets and both tasks, we randomly split the dataset into a
train-val set and a test set with proportion 8:2, the train-val set is further
randomly split into training set and validation set by 9:1. For the medication
recommendation task, the patients are split into training, validation and test.
To be practical, a model should use previous encounter information to make
recommendations for later encounters, thus in our setting, the last encounters


2 `https://www.cancer.gov/about-cancer/treatment/drugs`


14


Table 1: Summary information of NMEDW dataset and the corresponding MedGraph. E=encounters; P=patients; L=labs; M=medications


#E: 1260; #P: 865; #L: 197, #M: 57

Matrix Size Edges Sparsity Values


_A_ _E×P_ 1260 _×_ 865 1260 99.88% binary: 0, 1
_A_ _E×L_ 1260 _×_ 197 43806 82.35% continuous: 0–1
_A_ _E×M_ 1260 _×_ 57 2475 96.55% binary: 0, 1


Table 2: Summary information of MIMIC-III dataset and the corresponding
MedGraph. E=encounters; P=patients; L=labs; M=medications


#E: 18190; #P: 15153; #L: 219, #M: 117

Matrix Size Edges Sparsity Values


_A_ _E×P_ 18190 _×_ 15153 18190 99.99% binary: 0, 1
_A_ _E×L_ 18190 _×_ 219 1029964 68.96% continuous: 0–1
_A_ _E×M_ 18190 _×_ 117 23395 98.68% binary: 0, 1


of the test patients and the validation patients constitute the test encounter
set and the validation encounter set, respectively; all other encounters are used
for training. Since our datasets contain many patients with only one encounter
(specifically, 141 out of 173 test patients for NMEDW and 2628 out of 3031 test
patients for MIMIC-III), the test performance can represent the generalization
capability of a model to new patients. In the training process, we remove all
edges from the test and validation encounters to any medications in MedGraph.
For inductive MedGCN, we remove all testing encounters and their connections
from MedGraph when training. For lab test imputation task, the edges between
encounters and labs are split into training, validation and test. In the training
process, we remove all validation and testing edges corresponding to the missing
values in MedGraph. For both tasks, in each training epoch we evaluate the
performance on the validation set. The best model for the validation set is saved
for evaluation on the test set.


**4.2** **Settings**


In our study, since all types of nodes in the MedGraph have connections to
encounters, to avoid over-smoothing issues [36], we set a 1-layer MedGCN with
the output dimension 300 and dropout rate 0.1. _f_ _θ_ _M_ and _f_ _θ_ _L_ are both single
layer fully-connected neural networks. The features of P, E, M, L nodes are all
initialized with one-hot vectors. We implement MedGCN with PyTorch 1.0.1,
and train it using Adam optimizer [45] with learning rate 0.001. The max
training epoch is set 300. The cross regularization coefficient _λ_ = 1.


15


**4.3** **Performance for Medication Recommendation**


**4.3.1** **Baselines**


For medication recommendation, besides the regular MedGCN, we also implemented the following models for comparison. To release the workload of physicians in diagnostic reasoning, our model was designed not to require diagnostic
input. Advanced deep learning models that require the diagnosis code as an
input (e.g., GameNet [4], G-BERT [29] and DMNC [28]) cannot be directly applied to our scenario, thus not included in our baselines. Since our MedGraph
only involves lab nodes and patient nodes besides the encounter nodes and medication nodes, we only consider the models that accept lab tests and patients as
input.


  - **MedGCN-ind** : the inductive version of MedGCN where the test encoun
ters and all their connections are removed from MedGraph in the training

process.


  - **MedGCN-Med** : the MedGCN trained without cross regularization using Eq. 8 as the loss function.


  - **Multi-layer perceptron (MLP)** : we implement a 3-layer MLP appending a softmax output layer for multi-label classification. The MLP has a
hidden layer size 100 with a Relu activation and an Adam solver [45]. In
our case, the input dimension is the sum of lab test counts and patient
counts; the output dimension is the number of medications in considering.


  **Gradient Boosting Decision Tree (GBDT)** [46]: GBDT is an accurate and effective procedure that can be used for both regression and
classification problems in a variety of areas. GBDT produces a prediction
model in the form of an ensemble of weak decision trees. In our case, we
implement multiple binary GBDT classifiers corresponding to the medications based on lab test values and patients.


  **Random Forest (RF)** [47]: RF is a classifier that fits a number of
decision tree classifiers on various sub-samples of the dataset and uses
averaging to improve the predictive accuracy and control over-fitting. In
our case, we implement multiple binary RF classifiers corresponding to the
medications based on lab test values and patients. For each RF classifier,
we set 100 decision trees in the forest.


  **Logistic Regression (LR)** [48]: we implement multiple binary LR classifiers corresponding to the medications based on the liblinear solver [48].
For each LR classifier, the input features consist of lab test values and
patients.


  **Support Vector Machine (SVM)** [49]: we implement multiple binary
SVM classifiers corresponding to the medications based on libsvm [49]. For
each SVM classifier, we set the rbf kernel and the input features consist
of lab test values and patients.


16


  **Classifier Chains (CC)** [50]: a popular multi-label learning method that
models the correlation between labels by feeding both input and previous
classification results into the latter classifiers. We leverage SVM as the
base estimator.


For the baselines that cannot handle missing values (MLP, GBDT, RF, LR,
SVM, CC), we simply replaced missing values in lab test with 0. We have also
tried replacing missing values with mean values, the results did not show much
difference. All the baselines were implemented with scikit-learn [51]. Since
the dataset is imbalance, all the baselines are implemented with balanced class
weight.


**4.3.2** **Metrics**


Since the average number of medications for an encounter is about 2 in our
datasets (specifically, 2.0 in NMEDW and 1.3 in MIMIC-III), we used mean
average precision at 2 (MAP@2) to evaluate the performance of medication recommendation. MAP is a well-known metric to evaluate the performance of a
recommender system or information retrieval system [3] . Since the task of medication recommendation can be regarded as a multi-label classification problem
as well, we also used label ranking average precision (LRAP) [4] to evaluate the
classification performance of all the considered methods. Both MAP and LRAP
would be in the range of [0 _,_ 1], and higher LRAP or MAP indicates more accuracy recommendation.


**4.3.3** **Results**


For all the methods, we executed the training and test processes 5 times with
different initializations, and recorded the average performance ( _avg._ ) and standard deviation ( _std._ ) in the form of _avg. ± std._ . The results for medication
recommendation are listed in Table 3 and 4 for NMEDW and MIMIC-III, respectively. Since LR and SVM are not sensitive to the initialization, the _std._
are not listed. The results show that all the three variants of MedGCN (i.e.,
MedGCN, MedGCN-ind and MedGCN-Med) significantly outperform all the
baselines in terms of both LRAP and MAP@2 ( _p <_ 0 _._ 05 with t-test). MedGCN
also significantly outperforms MedGCN-Med in both terms. MedGCN-ind is
comparable to MedGCN on both datasets.
Comparing MedGCN-Med and MedGCN, the only difference between the
two models is the loss function where MedGCN was regularized by the lab test
imputation task, while MedGCN-Med did not, the performance of MedGCN
is much better than MedGCN-Med, which validates the efficacy of the cross
regularization. We also found that MedGCN-ind is comparable to MedGCN,


3 Refer to `http://fastml.com/what-you-wanted-to-know-about-mean-average-precision/`
for details.
4 Refer to `https://scikit-learn.org/stable/modules/model_evaluation.html#`
`multilabel-ranking-metrics` for details.


17


indicating that MedGCN-ind can handle new coming data very well, which
validates the potential of our system for online medication recommendation.


Table 3: Results for medication recommendation on NMEDW dataset. The
best results are bolded. “*” indicates the best results significantly outperform
this result ( _p <_ 0 _._ 05 with t-test).


Methods LRAP MAP@2


MedGCN (ours) **.7588** _±_ **.0028** **.7558** _±_ **.0035**
MedGCN-ind (ours) .7491 _±_ .0067* .7558 _±_ .0073
MedGCN-Med (ours) .7477 _±_ .0032* .7457 _±_ .0046*
MLP .7331 _±_ .0126* .6965 _±_ .0113*
GBDT [46] .7120 _±_ .0018* .6864 _±_ .0023*
RF [47] .6872 _±_ .0072* .7055 _±_ .0068*
LR [48] .5325* .4133*
SVM [49] .4324* .3353*
CC [50] .6276 _±_ .0116* .6182 _±_ .0159*

MedGCN-ind: inductive MedGCN; MedGCN-Med: MedGCN without cross
regularization and only use Eq. 8 as loss function; MLP: Multi-layer
perceptron; GBDT: Gradient Boosting Decision Tree; RF: Random Forest;
LR: Logistic Regression; SVM: Support Vector Machine.


Table 4: Results for medication recommendation on MIMIC dataset. The best
results are bolded. “*” indicates the best results significantly outperform this
result ( _p <_ 0 _._ 05 with t-test).


Methods LRAP MAP@2


MedGCN (ours) **.8349** _±_ **.0008** .8069 _±_ .0022
MedGCN-ind (ours) .8345 _±_ .0007 **.8070** _±_ **.0029**
MedGCN-Med (ours) .8346 _±_ .0005 .8061 _±_ .0020
MLP .8325 _±_ .0003* .8030 _±_ .0030*
GBDT [46] .5793 _±_ .0001* .5019 _±_ .0002*
RF [47] .8215 _±_ .0007* .8030 _±_ .0011*
LR [48] .3367* .1839*
SVM [49] .6642* .6146*
CC [50] .7660 _±_ .0005* .7153 _±_ .0003*


**4.4** **Performance for Lab Test Imputation**


**4.4.1** **Baselines**


For lab test imputation, besides the regular MedGCN, we also implemented the
following models for comparison.


18


  - **MedGCN-ind** : the inductive version of MedGCN where the test encoun
ters and all their connections are removed from MedGraph in the training

process.


  - **MedGCN-Lab** : MedGCN-Lab is the MedGCN trained without cross

regularization using Eq. 9 as loss function.


  **Multivariate Imputation by Chained Equations (MICE)** [30]: MICE
is a popular multiple imputation method used to replace missing data values based on fully conditional specification, where each incomplete variable is imputed by a separate model. In our case, we directly use MICE
to impute the adjacency matrix _A_ _E×L_ .


  **Multi-Graph Convolutional Neural Networks (MGCNN)** [52]: MGCNN
is a geometric deep learning method on graphs for matrix completion by
combining a multi-graph convolutional neural network and a recurrent
neural network. We applied MGCNN to complete the adjacency matrix
_A_ _E×L_ to implement the imputation.


  **Graph Convolutional Matrix Completion (GCMC)** [53]: GCMC
uses a Graph convolutional encoder and a bilinear decoder to predict the
connections among nodes in a bipartite graph, thus, to complete adjacency matrix of the bipartite graph. We applied GCMC to complete the
adjacency matrix _A_ _E×L_ to implement the imputation.


  - **GCMC+FEAT** [53]: GCMC with medication information as side information for encounters.


Since MGCNN, GCMC and GCMC+FEAT are designed for discrete features,
when testing these baselines, we discretized the continuous lab value into 5
ratings for each lab in advance.


**4.4.2** **Metrics**


We use mean square error (MSE) to measure the performance of lab test
imputation, following previous studies [9]. A lower MSE indicates the predicted
values are more close to the true value, thus a better model.

**4.4.3** **Results**


We also repeated the training and test processes 5 times with different initializations for all the methods, and recorded the results of lab test imputation
in the form of _avg. ± std._ in Table 5 and 6 for NMEDW and MIMIC-III, respectively. The results show MedGCN can significantly perform better than
all other methods ( _p <_ 0 _._ 05 with t-test), and MedGCN-ind and MedGCN-Lab
both perform better than the other baselines. Comparing the performance of
MedGCN-Lab and MedGCN, we again validate the efficacy of the cross regularization. Although MedGCN-ind is not comparable to MedGCN and MedGCN

19


Table 5: Results for lab test imputation on NMEDW dataset. The best results
are bolded. “*” indicates the best results significantly outperform this result
( _p <_ 0 _._ 05 with t-test).


Methods MSE


MedGCN (ours) **.0229** _±_ **.0025**
MedGCN-ind (ours) .0264 _±_ .0034*
MedGCN-Lab (ours) .0254 _±_ .0003*
MICE [30] .0474 _±_ .0010*
MGCNN [52] .0369 _±_ .0009*
GCMC [53] .0426 _±_ .0025*
GCMC+FEAT [53] .0359 _±_ .0030*
MedGCN-Lab: MedGCN without cross regularization and only use Eq. 9 as
loss function; MGCNN: Multi-Graph Convolutional Neural Networks; GCMC:
Graph Convolutional Matrix Completion; GCMC+FEAT: GCMC with side
information


Table 6: Results for lab test imputation on MIMIC-III dataset. The best results
are bolded. “*” indicates the best results significantly outperform this result
( _p <_ 0 _._ 05 with t-test).


Methods MSE


MedGCN(ours) **.0140** _±_ **.0002**
MedGCN-ind(ours) .0143 _±_ .0002*
MedGCN-Lab(ours) .0143 _±_ .0001*
MICE [30] .0146 _±_ .0001*
MGCNN [52] .0413 _±_ .0048*
GCMC [53] .0296 _±_ .0004*
GCMC+FEAT [53] .0290 _±_ .0001*


20


Lab on NMEDW for lab test imputation, it can outperform the other baselines
and provide acceptable results.
To validate the efficacy of MedGCN for lab test imputation, we also test our
imputation results of lab tests for medication recommendation. We compare
the results of our imputation with lab values filled with 0 in Table 7. We found
that using labs filled by MedGCN can achieve better performance than that
filled with 0 for most of the classifiers. Only for the GBDT classifier, filled with
0 is better than filled by MedGCN, but the differences are not significant.


Table 7: Results of Medication recommendation on NMEDW dataset using lab
tests filled by MedGCN and with 0.

|Filled with 0 Filled by MedGCN<br>methods<br>LRAP MAP@2 LRAP MAP@2|Filled with 0|Col3|Filled by MedGCN|Col5|
|---|---|---|---|---|
|methods<br>Filled with 0<br>Filled by MedGCN<br>LRAP<br>MAP@2<br>LRAP<br>MAP@2|LRAP|MAP@2|LRAP|MAP@2|



MLP .7331 _±_ .0126 .6965 _±_ .0113 **.7409** _±_ **.0040** **.7448** _±_ **.0165**

GBDT **.7120** _±_ **.0018** **.6864** _±_ **.0023** .6968 _±_ .0105 .6832 _±_ .0215

RF .6872 _±_ .0072 .7055 _±_ .0068 **.6907** _±_ **.0052** **.7303** _±_ **.00167**

LR 0.5325 0.4113 **0.5337** **0.4408**

SVM 0.4324 0.3353 **0.5553** **0.5376**


**4.5** **Parameter Sensitivity**


We have also tested the effect of cross regularization parameter _λ_ of MedGCN
on the results for the two tasks on NMEDW dataset (see Section 3.5 for the
detail of cross regularization). Figure 4 shows the performances of MedGCN
for medication recommendation in terms of LRAP (Figure 4a) and MAP@2
(Figure 4b) with different _λ_ . Figure 5 shows the MSEs of MedGCN for lab
test imputation with different _λ_ . From the variation curves of LRAP (Figure
4a), MAP@2 (Figure 4b), and MSE (Figure 5), all the 3 metrics change very
slowly with the hyper-parameter _λ_ (note that the x tick in Figure 4a and 4b
is exponential), indicating that MedGCN is robust to _λ_ for both medication
recommendation and lab test imputation. From Figure 4a and 4b, the performance of medication recommendation decreases with _λ_ increases exponentially.
From Figure 5, the performance of lab test imputation increases slowly with _λ_
increases. Because _λ_ is to adjust the weights of the two learning tasks, a larger
_λ_ represents the model is more prone to the lab test imputation task, and vice
versa, thus, MedGCN with a greater _λ_ will have a better lab test imputation
performance but a slightly lower medication recommendation performance.
We have also tested multi-layer MedGCNs for the two tasks, the performance
of MedGCN with different number of layers are shown in Figure 6, each layer
has 300 units in the experiment. From Figure 6, the 1-layer MedGCN performs
better than multi-layer MedGCNs for both medication recommendation and lab
test imputation. Because in the constructed MedGraph, an encounter and its
one-hop neighbors are much more important than other neighbors for the two


21


(a) LRAP



(b) MAP@2



Figure 4: Performances of MedGCN with different cross regularization parameters _λ_ for the medication recommendation. x axis is exponential.


Figure 5: MSE performances of MedGCN with different cross regularization
parameters _λ_ for lab test imputation.


learning tasks, stacking multiple MedGCN layers would result in over-smoothing
issues [36].


**4.6** **Case Study**


In our case study, we choose an encounter from NMEDW that has only a few
lab values available from the test set to see what medication our model will

recommend, and what lab values will be imputed. The results of our case study
are shown in Table 8, where the upper part and the middle part respectively list
the recommended medications and estimated lab values based on two available

labs on urine analysis, and the lower part lists the abbreviations of appeared
medications and labs. Of note, albuminuria was shown to associate with a
2.4-fold likelihood of risk of lung cancer [54].
In the upper part of Table 8, the true medications prescribed by physicians
(ground truth) and the top 5 recommended medications by different methods
are listed. Although no priority exists in the ground truth, for the implemented


22


(a) medication recommendation



(b) lab test imputation



Figure 6: Performance of MedGCN with different number of layers for the two
tasks.


recommendation models, medications in the front of the recommendation list
have higher recommendation priority. We found the top 3 recommendations of
MedGCN exactly match the physician prescribed medications. LR and MLP
missed one medication in the top 5 recommendations; GBDT recommends a
lower priority for _gem_ than two other medications (i.e., _car_ and _bev_ ) that physicians did not prescribe.
The middle part of Table 8 lists the imputed values for lab test _urobi_ for
the same encounter, the true value was measured as 0.2, but we removed it in
the training process. We can see that our MedGCN predicts (0.19) closer to the
true value than all other methods. What is more, MedGCN can perform the
two tasks simultaneously.


**4.7** **Discussion**


From the above results, MedGCN works well for both medication recommendation and lab test imputation. The reason may be two-fold. First, because
the constructed MedGraph incorporates the complex relations between different
medical entities, MedGCN built on the informative MedGraph has the potential
to learn a very informative representation for each node. Considering the baselines in Tables 3 and 5, they all ignore the correlations between different medications as well as different labs, while MedGCN takes these relations into account
through their shared encounters in the constructed MedGraph, thus MedGCN
can achieve better performances for both tasks. Furthermore, the cross regularization strategy enhances MedGCN by reducing overfitting. MedGCN uses
a loss function considering multiple tasks to guide the training of the model
by the cross regularization strategy, thus reducing the overfitting for a specific
task and making the learned representation predictive for multiple tasks. The
results in Tables 3 and 5 also validate the efficacy of cross regularization.
Besides the improved performance, MedGCN also has some other features.
(1) MedGCN can incorporate the features of quite different medical tasks in one


23


Table 8: The recommended medications and estimated lab values in our case

study with two labs available.


**Available labs** **Methods** **Recommendations**


_uph_ : 6 Ground Truth _dex_, _gem_, _ond_
_uspg_ : 1.019 MedGCN _ond_, _dex_, _gem_, _peg_, _bev_
LR _ond_, _dex_, _pem_, _bev_, _car_
MLP _ond_, _dex_, _hyd_, _ral_, _den_
GBDT _ond_, _dex_, _car_, _bev_, _gem_


**Available labs** **Methods** **estimated lab** ( _urobi_ ), EU/dL


_uph_ : 6 True value 0.2
_uspg_ : 1.019 MedGCN 0.19
GCMC 0.25

GCMA+FEAT 0.24

MGCNN 1.01


lab abbreviations _uph_ Urine pH
_uspg_ Urine specific gravity
_urobi_ Urobilinogen


med abbreviations _ond_ ondansetron

_dex_ dexamethasone

_gem_ gemcitabine
_peg_ pegfilgrastim
_bev_ bevacizumab

_pem_ pemetrexed
_car_ carboplatin
_hyd_ hydroxyurea
_ral_ raloxifene

_den_ denosumab


model and perform multiple tasks simultaneously. MedGCN takes the advantages of multi-task learning [55, 56] and incorporates multiple medical tasks such
as medication recommendation and lab test imputation in one single model. In
our future work, more medical tasks will be incorporated into MedGCN. (2)
MedGCN can handle missing attribute values. Most previous GCN models
cannot accept missing values. We convert the node attributes as a new type
of nodes incorporated to MedGraph where a missing value corresponds to a
missing edge that GCN can handle well. (3) MedGCN can be applied to heterogeneous graphs. MedGraph is a typical heterogeneous graph that incorporates
many types of medical entities and relations. We split the heterogeneous graph
into multiple subgraphs with each subgraph has only one type of edge. In each
GCN layer, the representations of each node in all the subgraphs are aggregated as the node representation. (4) MedGCN enables inductive learning on


24


MedGraphs. Tables 3 and 5 show the inductive version of MedGCN almost
performs comparable to regular MedGCN, validating the generalization ability
of our model to new data.

Another feature of our work is that we do not involve diagnosis code in our
framework for medication recommendation and lab test imputation. This could
release physicians from heavy lifting in diagnostic reasoning. To recommend
medications for a new patient/encounter, unlike most of the previous work that
needs physicians to provide the diagnosis codes [4, 6, 27, 28, 29], our model just
needs the results of only a few lab tests.
In this paper, MedGCN is evaluated with a simple Medgraph with 4 types
of medical entities and two medical tasks. In practical scenarios, a MedGraph
can be scaled to include more medical entities as well as the relations, such
as diagnosis code, and the MedGCN trained on the MedGraph can be used
to perform other tasks such as diagnosis prediction. Also, the time order of
encounters can be considered to construct a directional MedGraph to enhance
the model. These represent the direction of our future work.

### **5 Conclusion**


In this article, we innovatively incorporated the complex associations between
multiple medical entities into MedGraph, and developed MedGCN, an end-toend graph embedding model, to learn informative medical entity representations
for multiple medical tasks based on MedGraph. We augmented MedGCN to
handle heterogeneous MedGraph by formulating different medical entities different types of nodes. By transforming node features to a new type of nodes in
the graph, MedGCN can also handle missing values in the features of a medical
entity. MedGCN is a general inductive model that could use the learned node
representations and network weights to efficiently generate representations for
new nodes. In addition, we introduce cross regularization to reduce overfittings
by making the learned representation more informative. Experimental results
on a real medical dataset showed that MedGCN outperformed state-of-the-art
methods in both medication recommendation and lab test imputation tasks.

### **Acknowledgements**


The research is supported in part by the following US NIH grants: R21LM012618,
5UL1TR001422, U01TR003528 and R01LM013337.

### **References**


[1] Raimond L Winslow, Natalia Trayanova, Donald Geman, and Michael I
Miller. Computational medicine: translating models to clinical care.
Science translational medicine, 4(158):158rv11–158rv11, 2012.


25


[2] Konrad J Karczewski, Roxana Daneshjou, and Russ B Altman. Pharmacogenomics. PLoS computational biology, 8(12):e1002817, 2012.


[3] Isaac S Kohane. Ten things we have to do to achieve precision medicine.

Science, 349(6243):37–38, 2015.


[4] Junyuan Shang, Cao Xiao, Tengfei Ma, Hongyan Li, and Jimeng Sun.
Gamenet: graph augmented memory networks for recommending medication combination. In Proceedings of the AAAI Conference on Artificial
Intelligence, volume 33, pages 1126–1133, 2019.


[5] Jonathan X Wang, Delaney K Sullivan, Adam J Wells, Alex C Wells, and
Jonathan H Chen. Neural networks for clinical order decision support.
AMIA Summits on Translational Science Proceedings, 2019:315, 2019.


[6] Yutao Zhang, Robert Chen, Jie Tang, Walter F Stewart, and Jimeng Sun.
Leap: learning to prescribe effective and safe treatment combinations for
multimorbidity. In proceedings of the 23rd ACM SIGKDD international
conference on knowledge Discovery and data Mining, pages 1315–1324.
ACM, 2017.


[7] Lishan Yu, Qiuchen Zhang, Elmer V Bernstam, and Xiaoqian Jiang. Predict or draw blood: An integrated method to reduce lab tests. Journal of
Biomedical Informatics, page 103394, 2020.


[8] Akbar K Waljee, Ashin Mukherjee, Amit G Singal, Yiwei Zhang, Jeffrey
Warren, Ulysses Balis, Jorge Marrero, Ji Zhu, and Peter DR Higgins. Comparison of imputation methods for missing laboratory data in medicine.
BMJ open, 3(8):e002847, 2013.


[9] Yuan Luo, Peter Szolovits, Anand S Dighe, and Jason M Baron. Using
machine learning to predict laboratory test results. American journal of
clinical pathology, 145(6):778–788, 2016.


[10] Yuan Luo, Peter Szolovits, Anand S Dighe, and Jason M Baron. 3d-mice:
integration of cross-sectional and longitudinal imputation for multi-analyte
longitudinal clinical data. Journal of the American Medical Informatics
Association, 25(6):645–653, 2017.


[11] Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang,
and S Yu Philip. A comprehensive survey on graph neural networks. IEEE
Transactions on Neural Networks and Learning Systems, 2020.


[12] Thomas N Kipf and Max Welling. Semi-supervised classification with
graph convolutional networks. In International Conference on Learning
Representations, 2017.


[13] Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation
learning on large graphs. In Advances in neural information processing
systems, pages 1024–1034, 2017.


26


[14] Jie Chen, Tengfei Ma, and Cao Xiao. FastGCN: Fast learning with
graph convolutional networks via importance sampling. In International
Conference on Learning Representations, 2018.


[15] Yifu Li, Ran Jin, and Yuan Luo. Classifying relations in clinical narratives using segment graph convolutional and recurrent neural networks
(seg-gcrns). Journal of the American Medical Informatics Association,
26(3):262–268, 2018.


[16] Liang Yao, Chengsheng Mao, and Yuan Luo. Graph convolutional networks
for text classification. In Proceedings of the AAAI Conference on Artificial
Intelligence, volume 33, pages 7370–7377, 2019.


[17] Chengsheng Mao, Liang Yao, and Yuan Luo. Imagegcn: Multi-relational
image graph convolutional networks for disease identification with chest
x-rays. arXiv preprint arXiv:1904.00325, 2019.


[18] Truyen Tran, Tu Dinh Nguyen, Dinh Phung, and Svetha Venkatesh. Learning vector representation of medical objects via emr-driven nonnegative restricted boltzmann machines (enrbm). Journal of biomedical informatics,
54:96–105, 2015.


[19] Riccardo Miotto, Li Li, Brian A Kidd, and Joel T Dudley. Deep patient:
an unsupervised representation to predict the future of patients from the
electronic health records. Scientific reports, 6(1):1–10, 2016.


[20] Zhenchao Sun, Hongzhi Yin, Hongxu Chen, Tong Chen, Lizhen Cui, and
Fan Yang. Disease prediction via graph neural networks. IEEE Journal of
Biomedical and Health Informatics, 25(3):818–826, 2020.


[21] Edward Choi, Mohammad Taha Bahadori, Andy Schuetz, Walter F Stewart, and Jimeng Sun. Doctor ai: Predicting clinical events via recurrent
neural networks. In Machine learning for healthcare conference, pages 301–
318. PMLR, 2016.


[22] Edward Choi, Mohammad Taha Bahadori, Elizabeth Searles, Catherine Coffey, Michael Thompson, James Bost, Javier Tejedor-Sojo, and
Jimeng Sun. Multi-layer representation learning for medical concepts.
In proceedings of the 22nd ACM SIGKDD international conference on
knowledge discovery and data mining, pages 1495–1504, 2016.


[23] Dongha Lee, Xiaoqian Jiang, and Hwanjo Yu. Harmonized representation learning on dynamic ehr graphs. Journal of biomedical informatics,
106:103426, 2020.


[24] Marinka Zitnik, Monica Agrawal, and Jure Leskovec. Modeling polypharmacy side effects with graph convolutional networks. Bioinformatics,
34(13):i457–i466, 2018.


27


[25] Zheng Liu, Xiaohan Li, Hao Peng, Lifang He, and S Yu Philip. Heterogeneous similarity graph neural network on electronic health records. In 2020
IEEE International Conference on Big Data (Big Data), pages 1196–1205.
IEEE, 2020.


[26] Jaswanth Kumar Yella and Anil Jegga. Mgatrx: discovering drug
repositioning candidates using multi-view graph attention. IEEE/ACM
Transactions on Computational Biology and Bioinformatics, 2021.


[27] Yuanyuan Jin, Wei Zhang, Xiangnan He, Xinyu Wang, and Xiaoling Wang.
Syndrome-aware herb recommendation with multi-graph convolution network. In 2020 IEEE 36th International Conference on Data Engineering
(ICDE), pages 145–156. IEEE, 2020.


[28] Hung Le, Truyen Tran, and Svetha Venkatesh. Dual memory neural computer for asynchronous two-view sequential learning. In Proceedings of the
24th ACM SIGKDD International Conference on Knowledge Discovery &
Data Mining, pages 1637–1645. ACM, 2018.


[29] Junyuan Shang, Tengfei Ma, Cao Xiao, and Jimeng Sun. Pre-training
of graph augmented transformers for medication recommendation. In
Proceedings of the Twenty-Eighth International Joint Conference on
Artificial Intelligence, IJCAI-19, pages 5953–5959. International Joint Conferences on Artificial Intelligence Organization, 7 2019.


[30] S van Buuren and Karin Groothuis-Oudshoorn. mice: Multivariate imputation by chained equations in r. Journal of statistical software, pages 1–68,
2010.


[31] Daniel J Stekhoven and Peter B¨uhlmann. Missforest—non-parametric missing value imputation for mixed-type data. Bioinformatics, 28(1):112–118,
2011.


[32] Yulei He, Recai Yucel, and Trivellore E Raghunathan. A functional multiple imputation approach to incomplete longitudinal data. Statistics in
medicine, 30(10):1137–1156, 2011.


[33] Stephanie Kliethermes and Jacob Oleson. A bayesian approach to functional mixed-effects modeling for longitudinal data with binomial outcomes.
Statistics in medicine, 33(18):3130–3146, 2014.


[34] Xi Zhang, Lifang He, Kun Chen, Yuan Luo, Jiayu Zhou, and Fei Wang.
Multi-view graph convolutional network and its applications on neuroimage
analysis for parkinson’s disease. In AMIA Annual Symposium Proceedings,
volume 2018, page 1147. American Medical Informatics Association, 2018.


[35] Kang-Lin Hsieh, Yinyin Wang, Luyao Chen, Zhongming Zhao, Sean Savitz,
Xiaoqian Jiang, Jing Tang, and Yejin Kim. Drug repurposing for covid-19
using graph neural network with genetic, mechanistic, and epidemiological
validation. Research Square, 2020.


28


[36] Qimai Li, Zhichao Han, and Xiao-Ming Wu. Deeper insights into graph convolutional networks for semi-supervised learning. In Thirty-Second AAAI
Conference on Artificial Intelligence, 2018.


[37] Christopher Morris, Martin Ritzert, Matthias Fey, William L Hamilton,
Jan Eric Lenssen, Gaurav Rattan, and Martin Grohe. Weisfeiler and leman
go neural: Higher-order graph neural networks. In Proceedings of the AAAI
Conference on Artificial Intelligence, volume 33, pages 4602–4609, 2019.


[38] William L Hamilton, Rex Ying, and Jure Leskovec. Representation learning
on graphs: Methods and applications. IEEE Data Engineering Bulletin,
40(3):52–74, 2017.


[39] Michael Schlichtkrull, Thomas N Kipf, Peter Bloem, Rianne Van Den Berg,
Ivan Titov, and Max Welling. Modeling relational data with graph convolutional networks. In European Semantic Web Conference, pages 593–607.
Springer, 2018.


[40] Xiao Wang, Houye Ji, Chuan Shi, Bai Wang, Yanfang Ye, Peng Cui, and
Philip S Yu. Heterogeneous graph attention network. In The World Wide
Web Conference, pages 2022–2032, 2019.


[41] Chengsheng Mao, Liang Yao, Yiheng Pan, Yuan Luo, and Zexian Zeng.
Deep generative classifiers for thoracic disease diagnosis with chest x-ray
images. In 2018 IEEE International Conference on Bioinformatics and
Biomedicine (BIBM), pages 1209–1214. IEEE, 2018.


[42] Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri,
and Ronald Summers. Chestx-ray8: Hospital-scale chest x-ray database
and benchmarks on weakly-supervised classification and localization of
common thorax diseases. In 2017 IEEE Conference on Computer Vision
and Pattern Recognition(CVPR), pages 3462–3471, 2017.


[43] Justin B Starren, Andrew Q Winter, and Donald M Lloyd-Jones. Enabling
a learning health system through a unified enterprise data warehouse: the
experience of the northwestern university clinical and translational sciences
(nucats) institute. Clinical and translational science, 8(4):269, 2015.


[44] Alistair EW Johnson, Tom J Pollard, Lu Shen, H Lehman Li-Wei, Mengling
Feng, Mohammad Ghassemi, Benjamin Moody, Peter Szolovits, Leo Anthony Celi, and Roger G Mark. Mimic-iii, a freely accessible critical care
database. Scientific data, 3(1):1–9, 2016.


[45] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015.


[46] Jerome H Friedman. Stochastic gradient boosting. Computational statistics

& data analysis, 38(4):367–378, 2002.


[47] Leo Breiman. Random forests. Machine learning, 45(1):5–32, 2001.


29


[48] Rong-En Fan, Kai-Wei Chang, Cho-Jui Hsieh, Xiang-Rui Wang, and ChihJen Lin. Liblinear: A library for large linear classification. the Journal of
machine Learning research, 9:1871–1874, 2008.


[49] Chih-Chung Chang and Chih-Jen Lin. Libsvm: a library for support vector
machines. ACM transactions on intelligent systems and technology (TIST),
2(3):27, 2011.


[50] Jesse Read, Bernhard Pfahringer, Geoff Holmes, and Eibe Frank. Classifier
chains for multi-label classification. Machine learning, 85(3):333–359, 2011.


[51] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel,
M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. Scikit-learn:
Machine learning in Python. the Journal of machine Learning Research,
12:2825–2830, 2011.


[52] Federico Monti, Michael Bronstein, and Xavier Bresson. Geometric matrix
completion with recurrent multi-graph neural networks. In Advances in
neural information processing systems, pages 3697–3707, 2017.


[53] Rianne van den Berg, Thomas N Kipf, and Max Welling. Graph convolutional matrix completion. In SIGKDD, Deep Learning Day, 2018.


[54] Lone Jørgensen, Ivar Heuch, Trond Jenssen, and Bjarne K Jacobsen. Association of albuminuria and cancer incidence. Journal of the American
Society of Nephrology, 19(5):992–998, 2008.


[55] Sebastian Ruder. An overview of multi-task learning in deep neural networks. arXiv preprint arXiv:1706.05098, 2017.


[56] Yu Zhang and Qiang Yang. A survey on multi-task learning. arXiv preprint

arXiv:1707.08114, 2017.


30


