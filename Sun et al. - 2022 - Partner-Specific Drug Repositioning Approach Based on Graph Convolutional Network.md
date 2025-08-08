IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS, VOL. 26, NO. 11, NOVEMBER 2022 5757

## Partner-Specific Drug Repositioning Approach Based on Graph Convolutional Network


Xinliang Sun, Bei Wang, Jie Zhang, and Min Li _, Member, IEEE_



_**Abstract**_ **—Drug** **repositioning** **identifies** **novel** **thera-**
**peutic potentials for existing drugs and is considered an**
**attractive approach due to the opportunity for reduced de-**
**velopment timelines and overall costs. Prior computational**
**methods usually learned a drug’s representation from**
**an entire graph of drug-disease associations. Therefore,**
**the representation of learned drugs representation are**
**static and agnostic to various diseases. However, for**
**different diseases, a drug’s mechanism of actions (MoAs)**
**are different. The relevant context information should be**
**differentiated for the same drug to target different diseases.**
**Computational methods are thus required to learn different**
**representations corresponding to different drug-disease**
**associations for the given drug. In view of this, we propose**
**an end-to-end partner-specific drug repositioning approach**
**based on graph convolutional network, named PSGCN.**
**PSGCN firstly extracts specific context information around**
**drug-disease pairs from an entire graph of drug-disease**
**associations. Then, it implements a graph convolutional**
**network on the extracted graph to learn partner-specific**
**graph representation. As the different layers of graph**
**convolutional** **network** **contribute** **differently** **to** **the**
**representation of the partner-specific graph, we design**
**a layer self-attention mechanism to capture multi-scale**
**layer information. Finally, PSGCN utilizes sortpool strategy**
**to obtain the partner-specific graph embedding and for-**
**mulates a drug-disease association prediction as a graph**
**classification task. A fully-connected module is established**
**to classify the partner-specific graph representations. The**
**experiments on three benchmark datasets prove that**
**the** **representation** **learning** **of** **partner-specific** **graph**
**can lead to superior performances over state-of-the-art**


Manuscript received 18 April 2022; revised 1 July 2022; accepted 24
July 2022. Date of publication 3 August 2022; date of current version 7
November 2022. This work was supported in part by the National Natural
Science Foundation of China under Grant 61832019, in part by Hunan
Provincial Science and Technology Program under Grant 2019CB1007,
and in part by the Science and Technology innovation Program of Hunan
Province under Grant 2021RC4008. _(Xinliang Sun and Bei Wang are_
_contributed equally to this work.) (Corresponding authors: Jie Zhang;_
_Min Li.)_

Xinliang Sun is with the Hunan Provincial Key Lab on Bioinformatics,
School of Computer Science and Engineering, Central South University,
Changsha 410083, China, and also with the SenseTime, Shanghai
[200233, China (e-mail: xinliang-sun123456@csu.edu.cn).](mailto:xinliang-sun123456@csu.edu.cn)

Bei Wang is with the SenseTime, Shanghai 200233, China (e-mail:
[wangbei1@sensetime.com).](mailto:wangbei1@sensetime.com)

Jie Zhang is with the SenseTime, Shanghai 200233, China, and also
with the Qing yuan Research Institute, Shanghai Jiao Tong University,
[Shanghai 200240, China (e-mail: stzhangjie@hotmail.com).](mailto:stzhangjie@hotmail.com)

Min Li is with the Hunan Provincial Key Lab on Bioinformatics, School
of Computer Science and Engineering, Central South University, Chang[sha 410083, China (e-mail: limin@mail.csu.edu.cn).](mailto:limin@mail.csu.edu.cn)

This article has supplementary downloadable material available at
[https://doi.org/10.1109/JBHI.2022.3194891, provided by the authors.](https://doi.org/10.1109/JBHI.2022.3194891)

Digital Object Identifier 10.1109/JBHI.2022.3194891



**methods. In particular, case studies on small cell lung**
**cancer and breast carcinoma confirmed that PSGCN is**
**able to retrieve more actual drug-disease associations in**
**the top prediction results. Moreover, in comparison with**
**other static approaches, PSGCN can partly distinguish the**
**different disease context information for the given drug.**


_**Index Terms**_ **—Drug repositioning, graph convolutional**
**network, partner-specific graph, layer self-attention.**


I. I NTRODUCTION


RUG repositioning, also known as drug repurposing, is a
# D strategy aiming to investigate existing drugs for new ther
apeutic opportunities [1]. Compared to the laborious and expensive de novo drug discovery process, drug repositioning offers an
effective and efficient way to facilitate potential drugs reaching
the market, since pre-clinical and clinical information of the repurposed drugs are already available. Recently, the repositioned
drugs, such as Remdesivir, Ritonavir, and Ocilizumab, have
provided a rapid response to address the worldwide Coronavirus
disease (COVID-19) [2], [3], suggesting that drug repositioning
is a promising way to fight against diseases with no curative

treatment.


For drug repositioning, computational methods are rising
due to the explosion of biological data. Prior computational
methods can be mainly divided into three categories: (1) matrix factorization and completion based methods, (2) two-stage
machine learning based methods, and (3) deep learning methods,
especially graph learning methods [4]. The basic idea of matrix
factorization based methods is to map the drug-disease association matrix into a low-rank feature space and then minimize the
relation reconstruction error by the latent embeddings, where
the similarity matrices of drugs and diseases are taken as biological side information for additional constraints. For instance,
SCMFDD [5] defined similarity measures based on biological
context as additional constraints for the matrix factorization

method. iDrug [6] constructed a cross-network between the
drug-disease associations and drug-target associations, within
whichmatrixfactorizationwasadoptedtopredictpotentialdrugdisease associations. MLMC [7] proposed a multi-view learning
with matrix completion method to predict the potential associations between drugs and diseases. Although matrix factorization
and completion based methods have shown progressive results,
the high computational complexity poses scalability challenges
for dealing with growing data.



2168-2194 © 2022 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.

See https://www.ieee.org/publications/rights/index.html for more information.


Authorized licensed use limited to: ULAKBIM UASL - GAZI UNIV. Downloaded on November 17,2022 at 09:24:11 UTC from IEEE Xplore. Restrictions apply.


5758 IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS, VOL. 26, NO. 11, NOVEMBER 2022


TABLE I

S UMMARY OF THE T HREE B ENCHMARK D ATASETS



Fig. 1. An illustration of the difference between an entire bipartite
graph of drug-disease associations and partner-specific graph. (a) A
toy example of bipartite drug-disease graph, in which circles represent
drugs and triangles represent diseases. (b) Describing the different
context environment for two drug-disease pairs involving the same
drug(node: _u_ 2 ).


Machine learning based methods first pre-process side information as features and then predict whether a drug-disease pair
is positive or negative based on these features. PREDICT [8]
assembled multiple drug-drug and disease-disease similarity
measurestoconstructfeaturesandfedthemintoalogisticregression classifier. HED [9] employed the metapath2vec [10] algorithmtogeneratefeaturevectorsbasedontheconstructedheterogeneous network with drug-drug similarity, disease-disease similarity and drug-disease association networks. HED then trains a
support vector machine (SVM) model on the generated feature
vectors. The two-stage machine learning methods heavily rely
on arbitrary featurization, which requires domain knowledge
and experience.

Deep learning algorithms are increasingly exploited for drug
repositioning. CBPred [11] employed drug similarity and disease similarity information and considered the multiple paths
information between drug-disease associations to improve the
model performance. deepDR [12] integrated various heterogeneous networks information by a multi-modal deep autoencoder
to infer candidates for approved drugs. LAGCN [13] developed
a layer attention graph convolutional network (GCN) method.
It constructed a heterogeneous network based on drug-disease
associations, drug-drug similarities, and disease-disease similarities, then an attention-based GCN was utilized to encode the
nodes, while a bilinear decoder was applied to reconstruct the
drug-disease adjacency matrix. Such GCN-based methods have
achieved promising performance for drug repositioning.

Nevertheless, in previous studies, GCN-based repositioning
methods usually learn drug or disease representation from an
entire graph of drug-disease associations. As shown in Fig. 1(a),
it is a toy example of drug-disease association bipartite graph.
The circles and triangles represent drugs and diseases, respectively. In this bipartite graph, we have four drugs ( _u_ 1 _, u_ 2 _, u_ 3 _, u_ 4 )
and four diseases ( _v_ 1 _, v_ 2 _, v_ 3 _, v_ 4 ). The solid and dashed lines
represent known and unknown associations between drugs and
diseases, respectively. The two unknown associations between
_u_ 2 and _v_ 2 / _v_ 4 need to be predicted. When predicting _u_ 2 - _v_ 2 and
_u_ 2 - _v_ 4, prior GCN-based methods learn the same representative
embedding for _u_ 2 from the whole graph, which is static and
agnostic to various diseases (such as _v_ 2 and _v_ 4 ). However,
despite the same drug, its mechanism of actions (MoAs) to
different diseases are different. Therefore, when targeting different diseases, a drug needs to differentiate the relevant context



information. As Fig. 1(b) depicts, two different relation graphs
are extracted for the two target drug-disease pairs ( _u_ 2 _, v_ 2 ) and
( _u_ 2 _, v_ 4 ), respectively.

To emphasize the difference of target partners, inspired by the
subgraph-based studies [14], [15], we propose a partner-specific
method based on GCN, termed PSGCN. For drug repositioning,
PSGCN transforms a link prediction problem between a drug
and a disease into a graph classification task. Each extracted
graph collects the one-hop neighborhood information for the
two drug-disease pairs. Implementing a GCN on such partnerspecific graph can differentiate various context information as
message propagation and integration, inducing more refined
local structural features for inferring potential associations. The
[source code are at https://github.com/SenseTime-Knowledge-](https://github.com/SenseTime-Knowledge-Mining/PSGCN)
[Mining/PSGCN.](https://github.com/SenseTime-Knowledge-Mining/PSGCN)

Briefly, our contributions can be summarized as follows.
(1) To the best of our knowledge, PSGCN is the first partnerspecific method based on graph convolutional network for drug
repositioning. (2) PSGCN can automatically learn suitable context information about partner-specific graph and utilize layer
self-attention to capture multi-scale information. (3) The experiments demonstrate the effectiveness of PSGCN, and the case
studies further present the potential of PSGCN for real-world
application.


II. M ATERIALS AND M ETHODS


In this study, we propose PSGCN, a novel learning framework
based on graph convolutional network to capture latent relationships between drugs and diseases for effective drug repositioning. As Fig. 2 depicts, our framework mainly comprises three
components: 1) construction of partner-specific graph around
target drug-disease pairs; 2) structural information encoding of
partner-specific graph with GCN; 3) prediction of the potential
drug-disease associations.


_A. Dataset_


To evalute the performance of PSGCN, we use three public benchmarking datasets: Gdataset [8], Cdataset [16], and
LRSSL [17]. Table I summarizes the detailed information of
the three datasets. Gdataset contains 1,933 verified drug-disease
associations between 593 drugs and 313 diseases, where the
drugs are collected from DrugBank [18] and diseases are collected from Online Mendelian Inheritance in Man (OMIM) [19].
Cdataset contains 663 drugs and 409 diseases with 2,532 known
drug-diseases associations derived from Comparative Toxicogenomics Database (CTD) [20]. LRSSL dataset includes 3,051
validated drug-disease associations involving 763 drugs and 681
diseases.



Authorized licensed use limited to: ULAKBIM UASL - GAZI UNIV. Downloaded on November 17,2022 at 09:24:11 UTC from IEEE Xplore. Restrictions apply.


SUN et al.: PARTNER-SPECIFIC DRUG REPOSITIONING APPROACH BASED ON GRAPH CONVOLUTIONAL NETWORK 5759


Fig. 2. The framework of PSGCN. PSGCN mainly consists of three modules: extraction of partner-specific graphs, representation learning of
partner-specific graphs, and prediction of drug-disease associations. Firstly, target drug-disease associations are taken as centroids and their
h-hop neighbors ( _h_ = 1 in this figure) are collected as context information, constituting partner-specific graphs. Then, representation of the graphs
are obtained by graph convolution model and pooling operation. Finally, potential drug-disease associations are predicted as binary classification
of the graphs.


_v_ 4 in Fig. 3(b). The extracted partner-specific sub-graph around
drug _u_ _i_ and disease _v_ _j_ is denoted as _P_ _ij_ ( _P_ _ij_ _⊆_ _G_ ). The two
nodes _u_ _i_ and _v_ _j_ are called _center nodes_, and the other nodes
(e.g. _u_ 1, _u_ 2, _u_ 4, _v_ 1, _v_ 4 ) are called _context nodes_ .



Fig. 3. An illustration of the partner-specific graph extraction. (a) Bipartite drug-disease network. (b) Extractd partner-specific graph around
target node pair.


Besides, since baseline models need additional information,
for the three datasets, we also collect SMILES [21] of drugs
fromDrugBankdatabase, andutilizetheChemical Development
Kit [22] to measure the similarity of two drugs based on the Tanimoto score of their 2D chemical fingerprints. Disease-disease
similarity is obtained from MimMiner [23], where the similarity
score has been normalized into [0,1]. Please note that Gdataset
is regarded as the gold standard dataset, due to its clinicallyvalidated drug-disease associations. Therefore, we take Gdataset
as the main dataset and conduct comprehensive experiments on
it for evaluation.


_B. Partner-Specific Graph Construction_


Given the known drug-disease associations, we construct
a bipartite drug-disease network, _G_ = _{U, V, E}_, where _U_ =
_{u_ 1 _, u_ 2 _, . . ., u_ _m_ _}_ represents drug nodes, _m_ is the number of
drugs; and _V_ = _{v_ 1 _, v_ 2 _, . . ., v_ _n_ _}_ represents disease nodes, _n_ is
the number of diseases. _E_ = _{_ ( _u_ _i_ _, v_ _j_ ) _|u_ _i_ _∈_ _U, v_ _j_ _∈_ _V }_ is the
edge set representing known associations between drugs and
diseases. The adjacency matrix of the network is _A ∈_ _R_ _[m][×][n]_,
where _a_ _ij_ = 1 ( _a_ _ij_ _∈_ _A_ ) when an edge between drug _u_ _i_ and
disease _v_ _j_ exists, otherwise, _a_ _ij_ = 0. Based on the constructed
drug-disease bipartite network, the partner-specific graphs are
extracted. For a target drug-disease pair to be predicted, e.g.
( _u_ 3 _, v_ 2 ) in Fig. 3, we take the two nodes as centroid and extract
h-hop neighbors ( _h_ = 2) around them, i.e., _u_ 1, _u_ 2, _u_ 4, and _v_ 1,



A node labeling strategy [24] is then adopted on the extracted
partner-specific sub-graphs, to (1) distinguish the center nodes
from context nodes and (2) differentiate the types of nodes,
i.e., drug or disease. Specifically, the node labels of the drug
and disease at center are initialized as 0 and 1, respectively.
The context nodes are then labeled according to their hops. A
drug node at the _h_ -th hop is labeled as 2 _h_ . A disease node at
the _h_ -th hop is labeled as 2 _h_ + 1. The labels are based on the
roles (target or context) and types (drug or disease) of the nodes
in a partner-specific graph ( _P_ _ij_ ), and independent of the entire
drug-disease association graph _G_ . Thus, drug and disease nodes
have distinguishable labels for graph representation learning. It
is worth noting that, as shown in Fig. 3(b), the target drug-disease
association in a partner-specific graph is removed to avoid
information leakage.


_C. Representation Learning of Partner-Specific Graph_


After extracting the partner-specific graphs, we leverage a
graph neural network to learn the partner-specific graph representation for association prediction. Most of the existing GCN
based drug repositioning methods, such as [13], [25], apply a
node-level GCN on the whole drug and disease related networks
to learn the node embeddings, then directly predict the association probability with an inner-product or bilinear operator.
Differently, in this work, we learn _graph-level_ embeddings by
implementing a _graph-level_ GCN model to learn the extracted
partner-specificgraphrepresentation.Thegraph-levelGCNconsists of two components: 1) message propagation layers to produce node representations with context information in partnerspecific graph, and 2) a pooling layer to aggregate the node
embeddings into a comprehensive partner-specific graph-level
representation.

_Message propagation on partner-specific graphs:_ To capture
the context information of the center nodes in the extracted

partner-specific graph, a graph convolutional network [26] is



Authorized licensed use limited to: ULAKBIM UASL - GAZI UNIV. Downloaded on November 17,2022 at 09:24:11 UTC from IEEE Xplore. Restrictions apply.


5760 IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS, VOL. 26, NO. 11, NOVEMBER 2022



applied to learn the node representations. Moreover, to fully
acquire multi-scale structure information of the extracted graph,
we stacked _L_ message passing layers. The convolutional operation for a partner-specific graph _P_ _ij_ at the _l_ -th layer is formed
as (1).



_Z_ _[l]_ = _f_ ( _D_ [�] _[−]_ 2 [1]




[1] �

2 _A_ _[p]_ [ �] _D_ _[−]_ 2 [1]



2 _Z_ _[l][−]_ [1] _W_ _[l]_ ) (1)



where ˆ _y_ indicates the probability of the association between drug
_u_ _i_ and disease _v_ _j_ . _W_ 1 and _W_ 2 are trainable weights, and _b_ 1 and
_b_ 2 are bias.


_D. Model Training_


We transform the drug-disease association prediciton problem to graph classification task, and adopt the cross-entropy
loss function (5) to train the model. The known drug-disease
associations in the dataset are considered as positive samples and
others as negative samples. Accordingly, the classification labels
of extracted partner-specific graphs around positive samples are
1, and 0 otherwise.



where _Z_ _[l]_ _∈R_ _[N]_ _[×][d]_ _[l]_ denotes the output node embeddings at layer
_l_, _N_ is the number of nodes in the partner-specific graph, and _d_ _l_
is the number of output channels of layer _l_ . For _l_ = 1, the initial
node features _Z_ [0] = _X_ is a one-hot encoding of the node labels in
thepartner-specificgraph. _A_ [�] _[p]_ = _A_ _[p]_ + _I_,istheadjacencymatrix
of partner-specific graph _P_ _ij_ with added self-connections. _I_ is
the identity matrix, and _D_ � _ii_ = [�] _j_ _A_ [�] _[p]_ _ij_ . _W_ _[l]_ is trainable parameter matrix at layer _D_ [�] is a diagonal degree matrix with _l_ .

_f_ is a nonlinear activation function. After obtaining the nodes
embeddings from different convolutional layers, inspired by
recent research [27], a learnable layer self-attention vector _[−→]_ _α_
is applied to retain more important information from different
layers based on both graph features and topology. The final node
representation _Z_ [1:] _[L]_ is defined as follows:


_Z_ [1:] _[L]_ = _concat_ ( _α_ 1 _Z_ [1] _, α_ 2 _Z_ [2] _, . . ., α_ _L_ _Z_ _[L]_ )


_−→_
_α_ = ( _α_ 1 _, α_ 2 _, . . ., α_ _L_ ) (2)


where _Z_ [1:] _[L]_ _∈R_ _[N]_ _[×]_ [�] 1 _[L]_ _[d]_ _[l]_ is the concatenated output of _L_ graph

convolutional layers, and each row is a node feature embedding
and each column is a feature channel.


_Graph-level representation learning:_ When acquiring the
final node representations, we integrate them into the graph
feature vector by a sorted pooling layer [28]. Specifically, we
sort all the nodes based on the values of _Z_ [1:] _[L]_ in a descending
order. That is, we first compare the value of two nodes from the
last channel of _Z_ _[L]_ to the first channel of _Z_ [1] until the value is not
equal. After sorting the node features in a consistent order, the
number of nodes is unified into a fixed size, truncating the output
node representations from _N_ to _K_ by deleting the last _N −_ _K_
rows if _N > K_, or extending the output by adding _K −_ _N_ zero
rows if _N < K_ . The sorted pooling layer is formed as (3).


_Z_ _[p]_ = Γ( _Z_ [1:] _[L]_ ) = Γ _l_ : _L→_ 1 Γ _d_ : _d_ _l_ _→_ 1 ( _Z_ _[ld]_ ) (3)



_L_ =



�

( _i,j_ )



ˆ
_−y_ _ij_ _· log_ (ˆ _y_ _ij_ ) + (1 _−_ _y_ _ij_ ) _· log_ (1 _−_ _y_ _ij_ ) (5)



where _Z_ _[p]_ _∈_ _K ×_ [�] _[L]_ 1 _[d]_ _[l]_ [ is the output of the sort pooling layer.]

Γ is descending sort operation and _Z_ _[ld]_ denotes the value of the
_d_ th channel in _l_ layer. Detailedly, Γ _l_ : _L→_ 1 means to sort the final
node embedding from the output of layer _L_ to layer 1, and for the
output of each layer _l_, Γ _d_ : _d_ _l_ _→_ 1 ( _Z_ _[ld]_ ) means to sort node vector
_Z_ _[l]_ from its last channel _d_ _l_ to the first channel. Next, a reshape
operation is utilized to map the partner-specific graph into a
_K_ [�] _[L]_ 1 _[d]_ _[l]_ _[ ×]_ [ 1][ vector. After that, MaxPooling layers and 1-D]

convolutionalal layers are utilized to learn local partner-specific
graph patterns on the node sequence.

_Drug-disease association prediction:_ Finally, fully connected
layers perform classification for each partner-specific graph _P_ _ij_
with the pooled feature vector _Z_ _[p]_ (4).


_y_ ˆ = _W_ 2 _· relu_ ( _W_ 1 _Z_ _[p]_ + _b_ 1 ) + _b_ 2 (4)



where ( _i, j_ ) denotes the pair for drug _u_ _i_ and disease _v_ _j_ . _y_ _ij_ is the
truth label, while ˆ _y_ _ij_ is the predicted association probability of
( _u_ _i_ _, v_ _j_ ). To optimize the model, we use the Adam optimizer [29]
and train the model in a denoising setup by randomly dropping
out all outgoing messages of a particular edge with a fixed
probability. We also apply regular dropout [30] to prediction
layers.


III. R ESULTS AND D ISCUSSION


In this section, we evaluate our proposed PSGCN on three
benchmark datasets against competitive drug repositioning
methods. And we analyse the impact of hop depth on our model.
Furthermore, we conduct a _de novo_ experiment to verify the
performance of our model for identifying potential indications.
Finally, we show specific examples of the drug repositioning
results to illustrate the ability of PSGCN in practical application.


_A. Evaluation Metrics and Baseline Models_


We conduct 10-fold cross validation to evaluate the performance of our approach. Specifically, all the known drug-disease
associationsareconsideredaspositivesamplesandarerandomly
splited into ten subsets with the same size. In each fold, nine
subsets are combined as the positive training set, while the
remaining subset is treated as the positive testing set. Besides,
we randomly select the same number of negative instances as the
positive ones in training and testing sets. We utilize Area Under
the Receiver Operating Characteristic curve (AUROC) and Area
Under thePrecisionRecall curve(AUPR) as themetrics toassess
the performance of models, which are widely used for drug
repositioning prediction tasks. To provide robust estimation of
performance, we repeated the 10-fold cross validation procedure
ten times, and reported the mean results.

We compared the performance of our model against several competitive drug repositioning approaches. For SCMFDD,
iDrug, NRLMF and DRWBNCF, we adopt the same experimental setting as their paper recommended. For GRMF, following [6], the parameters _λ_ _l_ = 0 _._ 5 _, λ_ _d_ = _λ_ _t_ = 10 _[−]_ [3] are chosen.
For NIMCGCN, we set the parameters following that in [31].
r SCMFDD [5] is a matrix factorization method, which

maps the high dimensional drug and disease latent vectors



Authorized licensed use limited to: ULAKBIM UASL - GAZI UNIV. Downloaded on November 17,2022 at 09:24:11 UTC from IEEE Xplore. Restrictions apply.


SUN et al.: PARTNER-SPECIFIC DRUG REPOSITIONING APPROACH BASED ON GRAPH CONVOLUTIONAL NETWORK 5761


TABLE II
P ERFORMANCE C OMPARISON OF 10 T IMES 10-F OLD C ROSS V ALIDATION P REDICTION R ESULTS B ETWEEN O UR M ETHOD AND B ASELINES

O VER G DATASET, C DATASET AND LRSSL D ATASET


The best reported result is bolded and the second best result is underlined.



to low dimensional space for prediction, and incorporates
drug and disease similarities as constraints.
r iDrug [6] is a cross-network framework that integrates

drugrepositioninganddrug-targetpredictionintoaunified
networks with the overlapped drugs as the anchor nodes.
r GRMF [32] uses graph regularization to learn low
dimensional non-linear manifolds. In addition, the method
considers that many of the non-occurring edges in the
network are actually unknown or missing cases, and developed a preprocessing step to enhance prediction.
r NRLMF [33] utilizes a logistic matrix factorization based

method with a nearest neighborhood regularization for
drug-target interaction prediction.
r NIMCGCN [34], a graph convolutional network based

method for miRNA-disease association prediction, which
are demostrated to have great potential for drug repositioning.
r DRWBNCF [35] adopts weighted bilinear graph convo
lution operation to predict potential drug-disease associations by integrating the prior information of drugs and
diseases.


_B. Performance of PSGCN in the Cross-Validation_


For a fair comparison, we reported the average results of ten
times 10-fold cross validation on three datasets, with variance
to show the stability of the results. As shown in Table II, we find
that PSGCN consistently attains the best AUROC and AUPR
values over three datasets. The AUROC values achieved by
PSGCN on Gdataset is 0.9485, which is higher than the second
best method NRLMF 3.88%. It is also observed that PSGCN

achieves 0.9566 and 0.9395 for AUROC values on Cdataset

and LRSSL, respectively, which is an improvement of 2.72%
and 1.63% compared to the iDrug and DRWBNCF. For the
AUPR metric, PSGCN yields the best performance among all
methods,leadingtoanaverageimprovementof2.51%compared
totheDRWBNCFmethodonGdataset, andabout 1.50%relative
improvement over previous state-of-the-art method DRWBNCF
on both Cdataset and LRSSL datasets. This demonstrates the

superiority of our proposed method in drug repositioning task.

Notice that the baseline methods achieve the results in Table II

by using additional information, while our PSGCN is only
based on learning from the bipartite drug-disease network. For
instance, iDrug utilizes drug-target interaction information as



cross-domain bridge. Furthermore, when compared to GCNbased methods, such as NIMCGCN and DRWBNCF, PSGCN
still outperforms them by a large margin on three datasets, which
demonstrates the significance of the partner-specific representation learning.


_C. Parameter Analysis_


To investigate the effect of hop depth when extracting partnerspecificgraphonmodelperformance,wesearchedthenumberof
hop in the range of _{_ 1 _,_ 2 _,_ 3 _,_ 4 _}_ and performed 10-fold cross validation on Gdataset. The mean results of AUROC and AUPRC

are illustrated in Fig. 4. Clearly, extracting only 1-hop induces
the lowest performance, and 2-hop depth is capable of endowing
the model with better representation ability. While continuing
increasingthehopdepth,theimprovementsarenotobvious.This
suggests that the 2-hop information is sufficient to differentiate
partner-specific context information. Thus, we choose the hop
as 2 in this paper. The other two dataset h-hop experiment results
are shown in Supplementary Figure S1 and Figure S2.


_D. Test of PSGCN on Sparse Data_


In this section, we investigate the model performance on
sparsity data, we test the model on Gdataset under different
sparsity levels of the drug-disease associations. Here we randomly retain a part of known associations in the Gdataset
at a ratio _λ ∈{_ 80% _,_ 85% _,_ 90% _,_ 95% _,_ 100% _}_ and implement
10-fold cross validation to evaluate the method. As shown in

Fig. 5, we found that the number of drug–disease associations is
an important factor for the drug-disease association prediction,
and more associations can result in better prediction models.
Besides, we also found that PSGCN can produce the robust and
high performances across different sparsity data and the results
still outperform the baseline models under low associations
scenario.


_E. Discovering Candidates for New Drugs_


To validate the ability of PSGCN on prediction for new
drugs, we performed de novo test on the Gdataset. For a test
drug, we removed all of its known disease associations as the
testing set, and used all the remaining associations as the training
samples. Such setting makes the test drug isolated in the known
drug-disease association graph. Therefore, to relieve the coldstart problem, we conduct a K-Nearest Neighbor preprocessing
step for these new drugs. Specifically, for each novel drug, K



Authorized licensed use limited to: ULAKBIM UASL - GAZI UNIV. Downloaded on November 17,2022 at 09:24:11 UTC from IEEE Xplore. Restrictions apply.


5762 IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS, VOL. 26, NO. 11, NOVEMBER 2022


Fig. 4. Impact of the number of hop on PSGCN model performance on Gdataset. (a) The Area Under the Receiver operating characteristic
(AUROC) curves of 10-fold cross validation results obtained by searching different hops. (b) The Area Under the Precision Recall (AUPR) curves of
10-fold cross validation results obtained by searching different hops.


TABLE III

P ERFORMANCE OF A LL M ETHODS IN P REDICTING P OTENTIAL

I NDICATIONS FOR N EW D RUGS ON G DATASET


the Gdataset as training set and take the missing drug-disease
associations as candidate pairs for SCLC and breast carcinoma.
We subsequently rank all the candidate drugs by the computed
predictionscoresforeachdisease,andverifythepredictedtop10
potential drug-disease associations in acknowledged CTD [36],
PubChem, [1] and DrugCentral [37] databases.



Fig. 5. PSGCN performances for different sparsity ratios on Gdataset.


nearest neighbor drugs of the given drug are picked based on
their drug similarities in descending order. Then we update its
associations in the bipartite drug-disease graph with a part of its
nearest neighbor drugs’ association information. As shown in
Table III, we can find that PSGCN achieves the best results on
both evaluation metrics (AUROC is 0.8970, AURP is 0.3484),
which demonstrates the superiority of PSGCN for indicating
indications for novel drugs.


_F. Case Studies_


To assess the practical use of PSGCN, we take small cell lung
cancer (SCLC) and breast carcinoma as case studies. Specifically, we employ all the known drug-disease associations in



_SCLC:_ SCLC is the most malignant type of lung cancer,
with the characteristics of rapid progression, high metastatic
tendenc and easy recurrence. As shown in Table IV, among
the top 10 predicted candicate drugs by PSGCN, eight drugs
out of the top predicted ten candidate drugs can be confirmed
by authoritative public databases (80% hit rate). For example,
Doxorubicin is the top predicted candidate, which has been
proved that the combination of NGR-hTNF(a vascular-targeting
agent) and Doxorubicin shows manageable toxicity and promising activity in patients with relapsed SCLC [38]. The second
predicted candidate drug is ifosfamide, which is an alkylating
and immunosuppressive agent used in chemotherapy for the
treatment of cancers. The therapeutic effect of ifosfamide on
SCLC has been demonstrated on CTD database [39], [40].


1 [[Online]. Available: https://pubchem.ncbi.nlm.nih.gov](https://pubchem.ncbi.nlm.nih.gov)



Authorized licensed use limited to: ULAKBIM UASL - GAZI UNIV. Downloaded on November 17,2022 at 09:24:11 UTC from IEEE Xplore. Restrictions apply.


SUN et al.: PARTNER-SPECIFIC DRUG REPOSITIONING APPROACH BASED ON GRAPH CONVOLUTIONAL NETWORK 5763


TABLE IV
T OP 10 P REDICTED D RUGS FOR P OTENTIALLY T REATING S MALL C ELL

L UNG C ANCER



TABLE V
T OP 10 P REDICTED D RUGS FOR P OTENTIALLY T REATING B REAST C ANCER


_Breast cancer:_ Similar to SCLC, we also focus on analyzing
the top 10 drug candidates for Breast cancer predicted by PSGCN. Table V shows the Top 10 potential drugs recommended
by PSGCN. These 10 potential drugs have been verified by the
reliable evidence with 100% hit rate. [41] find that simultaneous
liposomal delivery of Vincristine and Quercetin has the ability
to enhance estrogen-receptor-negative breast cancer treatment.
Ethinylestradiol (a synthetic compound) has been used primarily
as contraceptive. However, the recent research [42] finds that
Ethinylestradiol can cure a patient with metastatic breast cancer.

To sum up, such case studies demonstrate the promising
abilityofPSGCNfordiscoveringpotentialdrugsforspecificdiseases. We expect that the predicted candidate drugs by PSGCN
will provide a meaningful reference for clinicians in practical
application.


_G. Visualization_


To intuitively present the characteristic of our PSGCN that
extracts specific contextual information for association prediction, we visualize the embeddings of partner-specific graphs
corresponding to Doxorubicin (a drug) with different targeting
diseases. Specifically, we employ principal component analysis
(PCA) to transform the high dimensional embeddings to two



Fig. 6. Visualization of partner-specific graph representation learned
from PSGCN model. The X- and Y- axes represent the two primary
dimensions after performing PCA, respectively. The nodes represent
the learned embeddings of partner-specific graphs when Doxorubicin
targets different diseases.


primary dimensions for visualization, and we set the same color
for nodes close to each other.


As shown in Fig. 6, these nodes are roughly clustered into
two clusters (green nodes and light blue nodes). As we known,
subject headings in the Mesh Tree [61] reveal the degree of
category similarity between diseases. Thus it is reasonable to
expect that diseases with closer cataloging in MeSH will induce
more similar partner-specific embeddings. In the green cluster, the node _7: Hodgkin Disease_ and node _8: Lymphoblastic_
_Leukemia_ are both associated with lymphoid tissue. Concretely,
_Hodgkin Disease_ belongs to Lymphoma (with subject heading

[C04.557.386]) in Mesh Tree, and _Lymphoblastic Leukemia_ under the category of Leukemia, Lymphoid (with subject heading

[C04.557.337.428]). In the light blue cluster, node _10: Neurob-_
_lastoma_ and _11: Osteosarcoma_, the subject headings of which
in the Mesh Tree are under [C04.557.465] and [C04.557.450],
respectively, both occur most often in young children.

Such examples indicate the potential of PSGCN on capturing
distinguishable representations for different target pairs. Since
thesediseasesarenotstandardforstrictclustering,itisinevitable
to appear a slight bias, (e.g. _13: Turcot syndrome_ and _3: Stomach_
_Neoplasms_ are all under Digestive System Neoplasms (with
subject headings [C04.588.274]) of the Mesh Tree. However,
compared to other methods that learn a static embedding, our
method provides a more differentiated representation for effective prediction. Additionally, such visual descriptions can be
regarded as a reference to assist researchers to find the association among diseases. The more details about visualization are
shown in Supplementary Table S1.


IV. C ONCLUSION


In this paper, we present a novel partner-specific drug repositioning approach based on graph neural network, PSGCN.
Instead of learning general feature for each node, PSGCN emphasizes to learn a summary representation for the graph of a



Authorized licensed use limited to: ULAKBIM UASL - GAZI UNIV. Downloaded on November 17,2022 at 09:24:11 UTC from IEEE Xplore. Restrictions apply.


5764 IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS, VOL. 26, NO. 11, NOVEMBER 2022



specific drug-disease association, which considers the different
roles of an object corresponding to different cases.

Compared to previous drug repositioning methods, our
PSGCN method applies graph convolutional network to
automatically capture partner-specific context information for
expressive feature learning. Extensive experiments have demonstrated the superior performance of PSGCN on the task of drug
repositioning. The partner-graph representation visualization of
a drug with different diseases indicates that our partner-specific
strategy prompts the model to better capture target specific
structural information, which provides high-quality representations for drug repositioning task. Furthermore, case studies
suggest the ability of PSGCN to predict unknown drug-disease
associations in terms of the concrete diseases.


Although PSGCN obtains satisfactory results, compared to
the whole drug space, the known drug–disease associations are
sometimes too sparse to provide abundant context information,
resulting in the performance of PSGCN still having room for
improvement. In the future, we will consider to enrich the
partner-specific graph with more biological entities. That is,
not only to capture the direct association context information,
but also to incorporate the biological knowledge based context information for more robust and effective representation
learning.


R EFERENCES


[1] S. Pushpakom et al., “Drug repurposing: Progress, challenges and recom
mendations,” _Nature Rev. Drug Discov._, vol. 18, no. 1, pp. 41–58, 2019.

[2] C. Harrison, “Coronavirus puts drug repurposing on the fast track,” _Nature_

_Biotechnol._, vol. 38, no. 4, pp. 379–381, 2020.

[3] Y. Zhou, F. Wang, J. Tang, R. Nussinov, and F. Cheng, “Artificial intelli
gencein COVID-19 drug repurposing,” _Lancet Digit.Health_, vol.2, no.12,
pp. e667–e676, 2020.

[4] H.Luo,M.Li,M.Yang,F.-X.Wu,Y.Li,andJ.Wang,“Biomedicaldataand

computational models for drug repositioning: A comprehensive review,”
_Brief. Bioinf._, vol. 22, no. 2, pp. 1604–1619, 2021.

[5] W. Zhang et al., “Predicting drug-disease associations by using similarity

constrained matrix factorization,” _BMC Bioinf._, vol. 19, no. 1, 2018,
Art. no. 233.

[6] H. Chen, F. Cheng, and J. Li, “iDrug: Integration of drug repositioning

and drug-target prediction via cross-network embedding,” _PLoS Comput._
_Biol._, vol. 16, no. 7, 2020, Art. no. e1008040.

[7] Y. Yan, M. Yang, H. Zhao, G. Duan, X. Peng, and J. Wang, “Drug

repositioning based on multi-view learning with matrix completion,” _Brief._
_Bioinf._, vol. 23, no. 3, 2022, Art. no. bbac054.

[8] A. Gottlieb, G. Y. Stein, E. Ruppin, and R. Sharan, “PREDICT: A method

for inferring novel drug indications with application to personalized
medicine,” _Mol. Syst. Biol._, vol. 7, no. 1, 2011, Art. no. 496.

[9] K. Yang, X. Zhao, D. Waxman, and X.-M. Zhao, “Predicting drug-disease

associations with heterogeneous network embedding,” _Chaos: An Inter-_
_discipl. J. Nonlinear Sci._, vol. 29, no. 12, 2019, Art. no. 123109.

[10] Y. Dong, N. V. Chawla, and A. Swami, “metapath2vec: Scalable represen
tation learning for heterogeneous networks,” in _Proc. 23rd ACM SIGKDD_
_Int. Conf. Knowl. Discov. Data Mining_, 2017, pp. 135–144.

[11] P. Xuan, Y. Ye, T. Zhang, L. Zhao, and C. Sun, “Convolutional neural

network and bidirectional long short-term memory-based method for predicting drug–disease associations,” _Cells_, vol. 8, no. 7, 2019, Art. no. 705.

[12] X. Zeng, S. Zhu, X. Liu, Y. Zhou, R. Nussinov, and F. Cheng, “deepDR:

A network-based deep learning approach to in silico drug repositioning,”
_Bioinformatics_, vol. 35, no. 24, pp. 5191–5198, 2019.

[13] Z. Yu, F. Huang, X. Zhao, W. Xiao, and W. Zhang, “Predicting drug–

disease associations through layer attention graph convolutional network,”
_Brief. Bioinf._, vol. 22, no. 4, 2021, Art. no. bbaa243.

[14] Z. Cao, L. Wang, and G. De Melo, “Link prediction via subgraph

embedding-based convex matrix completion,” in _Proc. AAAI Conf. Artif._
_Intell._, 2018, pp. 2803–2810.

[15] M. Zhang and Y. Chen, “Link prediction based on graph neural networks,”

in _Proc. 32nd Int. Conf. Neural Inf. Process. Syst._, 2018, pp. 5171–5181.




[16] H. Luo et al., “Drug repositioning based on comprehensive similarity

measures and Bi-random walk algorithm,” _Bioinformatics_, vol. 32, no. 17,
pp. 2664–2671, 2016.

[17] X. Liang et al., “Lrssl: Predict and interpret drug–disease associations

based on data integration using sparse subspace learning,” _Bioinformatics_,
vol. 33, no. 8, pp. 1187–1196, 2017.

[18] D. S. Wishart et al., “Drugbank: A comprehensive resource for in silico

drug discovery and exploration,” _Nucleic Acids Res._, vol. 34, no. suppl_1,
pp. D668–D672, 2006.

[19] A. Hamosh, A. F. Scott, J. S. Amberger, C. A. Bocchini, and V. A.

McKusick, “Online mendelian inheritance in man (OMIM), a knowledgebase of human genes and genetic disorders,” _Nucleic Acids Res._, vol. 33,
no. suppl_1, pp. D514–D517, 2005.

[20] A. P. Davis et al., “The comparative toxicogenomics database: Update

2017,” _Nucleic Acids Res._, vol. 45, no. D1, pp. D972–D978, 2017.

[21] D. Weininger, “Smiles, a chemical language and information system. 1.

introduction to methodology and encoding rules,” _J. Chem. Inf. Comput._
_Sci._, vol. 28, no. 1, pp. 31–36, 1988.

[22] C. Steinbeck, Y. Han, S. Kuhn, O. Horlacher, E. Luttmann, and E. Wil
lighagen, “The chemistry development kit (CDK): An open-source java
library for chemo-and bioinformatics,” _J. Chem. Inf. Comput. Sci._, vol. 43,
no. 2, pp. 493–500, 2003.

[23] M. A. Van Driel, J. Bruggeman, G. Vriend, H. G. Brunner, and J. A.

Leunissen, “A text-mining analysis of the human phenome,” _Eur. J. Hum._
_Genet._, vol. 14, no. 5, pp. 535–542, 2006.

[24] M. Zhang and Y. Chen, “Inductive matrix completion based on graph

neural networks,” in _Proc. Int. Conf. Learn. Representations_, 2020,
pp. 1–25.

[25] F.Wan,L.Hong,A.Xiao,T.Jiang,andJ.Zeng,“Neodti:Neuralintegration

of neighbor information from a heterogeneous network for discovering
new drug–target interactions,” _Bioinformatics_, vol. 35, no. 1, pp. 104–111,
2019.

[26] T. N. Kipf and M. Welling, “Semi-supervised classification with graph

convolutional networks,” in _proc. Int. Conf. Learn. Representations_, 2017,
pp. 1–14.

[27] J. Lee, I. Lee, and J. Kang, “Self-attention graph pooling,” in _Proc. Int._

_Conf. Mach. Learn._, 2019, pp. 3734–3743.

[28] M. Zhang, Z. Cui, M. Neumann, and Y. Chen, “An end-to-end deep

learning architecture for graph classification,” in _Proc. AAAI Conf. Artif._
_Intell._, 2018, pp. 4438–4445.

[29] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,”

in _Proc. Int. Conf. Learn. Representations_, 2015, pp. 1–41.

[30] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdi
nov, “Dropout: A simple way to prevent neural networks from overfitting,”
_J. Mach. Learn. Res._, vol. 15, no. 1, pp. 1929–1958, 2014.

[31] L. Cai et al., “Drug repositioning based on the heterogeneous information

fusion graph convolutional network,” _Brief. Bioinf._, vol. 22, no. 6, 2021,
Art. no. bbab319.

[32] A. Ezzat, P. Zhao, M. Wu, X.-L. Li, and C.-K. Kwoh, “Drug-target

interaction prediction with graph regularized matrix factorization,”
_IEEE/ACM Trans. Comput. Biol. Bioinf._, vol. 14, no. 3, pp. 646–656,
2016.

[33] Y. Liu, M. Wu, C. Miao, P. Zhao, and X.-L. Li, “Neighborhood regularized

logistic matrix factorization for drug-target interaction prediction,” _PLoS_
_Comput. Biol._, vol. 12, no. 2, 2016, Art. no. e1004760.

[34] J. Li, S. Zhang, T. Liu, C. Ning, Z. Zhang, and W. Zhou, “Neural inductive

matrix completion with graph convolutional networks for miRNA-disease
association prediction,” _Bioinformatics_, vol. 36, no. 8, pp. 2538–2546,
2020.

[35] Y. Meng, C. Lu, M. Jin, J. Xu, X. Zeng, and J. Yang, “A weighted

bilinear neural collaborative filtering approach for drug repositioning,”
_Brief. Bioinf._, vol. 23, no. 2, 2022, Art. no. bbab581.

[36] A. P. Davis et al., “Comparative toxicogenomics database (CTD): Up
date 2021,” _Nucleic Acids Res._, vol. 49, no. D1, pp. D1138–D1143,
2020.

[37] O. Ursu et al., “DrugCentral: Online drug compendium,” _Nucleic Acids_

_Res._, vol. 45, no. D1, pp. D932–D939,2016.

[38] V. Gregorc et al., “NGR-hTNF and doxorubicin as second-line treatment

of patients with small cell lung cancer,” _Oncologist_, vol. 23, no. 10,
pp. 1133–e112, 2018 .

[39] D. Decaudin et al., “In vivo efficacy of STI571 in xenografted human small

cell lung cancer alone or combined with chemotherapy,” _Int. J. Cancer_,
vol. 113, no. 5, pp. 849–856, 2005.

[40] I. Tanaka et al., “A phase II trial of ifosfamide combination with rec
ommended supportive therapy for recurrent SCLC in second-line and
heavily treated setting,” _Cancer Chemotherapy Pharmacol._, vol. 81, no. 2,
pp. 339–345, 2018.



Authorized licensed use limited to: ULAKBIM UASL - GAZI UNIV. Downloaded on November 17,2022 at 09:24:11 UTC from IEEE Xplore. Restrictions apply.


SUN et al.: PARTNER-SPECIFIC DRUG REPOSITIONING APPROACH BASED ON GRAPH CONVOLUTIONAL NETWORK 5765




[41] M.-Y. Wong and G. N. Chiu, “Simultaneous liposomal delivery of

quercetin and vincristine for enhanced estrogen-receptor-negative breast
cancer treatment,” _Anti-Cancer Drugs_, vol. 21, no. 4, pp. 401–410,
2010.

[42] A. Sueta et al., “Successful ethinylestradiol therapy for a metastatic breast

cancer patient with heavily pre-treated with endocrine therapies,” in _Proc._
_Int. Cancer Conf. J._, 2016, vol. 5, pp. 126–130.

[43] N. B. Leighl et al., “A phase I study of pegylated liposomal doxorubicin

hydrochloride (Caelyx) in combination with cyclophosphamide and vincristine as second-line treatment of patients with small-cell lung cancer,”
_Clin. Lung Cancer_, vol. 5, no. 2, pp. 107–112, 2003.

[44] H.Yangetal.,“Pharmaco-transcriptomiccorrelationanalysisrevealsnovel

responsive signatures to HDAC inhibitors and identifies dasatinib as a
synergistic interactor in small-cell lung cancer,” _EBioMedicine_, vol. 69,
2021, Art. no. 103457.

[45] G. Rustin et al., “A phase Ib trial of CA4P (combretastatin A-4 phosphate),

carboplatin, and paclitaxel in patients with advanced cancer,” _Brit. J._
_Cancer_, vol. 102, no. 9, pp. 1355–1360, 2010.

[46] A. Mouri et al., “Combination therapy with carboplatin and paclitaxel for

small cell lung cancer,” _Respir. Investigation_, vol. 57, no. 1, pp. 34–39,
2019.

[47] J. Hardy, T. Noble, and I. Smith, “Symptom relief with moderate dose

chemotherapy (mitomycin-C, vinblastine and cisplatin) in advanced nonsmall cell lung cancer,” _Brit. J. Cancer_, vol. 60, no. 5, pp. 764–766, 1989.

[48] T. Yokoyama, K. Miyazawa, T. Yoshida, and K. Ohyashiki, “Combination

of vitamin K2 plus imatinib mesylate enhances induction of apoptosis in
small cell lung cancer cell lines,” _Int. J. Oncol._, vol. 26, no. 1, pp. 33–40,
2005.

[49] C. M. Rudin et al., “Comprehensive genomic analysis identifies SOX2

as a frequently amplified gene in small-cell lung cancer,” _Nature Genet._,
vol. 44, no. 10, pp. 1111–1116, 2012.

[50] M. Peifer et al., “Integrative genome analyses identify key somatic driver

mutations of small-cell lung cancer,” _Nature Genet._, vol. 44, no. 10,
pp. 1104–1110, 2012.

[51] S. Esmaeili-Mahani, F. Falahi, and M. M. Yaghoobi, “Proapoptotic

and antiproliferative effects of Thymus caramanicus on human breast
cancer cell line (MCF-7) and its interaction with anticancer drug vincristine,” _Evidence-Based Complement. Altern. Med._, vol. 2014, 2014,
Art. no. 893247.




[52] J. K. Woodward, H. L. Neville-Webbe, R. E. Coleman, and I. Holen,

“Combined effects of zoledronic acid and doxorubicin on breast cancer
cell invasion in vitro,” _Anti-Cancer Drugs_, vol. 16, no. 8, pp. 845–854,
2005.

[53] Y. Tian et al., “Valproic acid sensitizes breast cancer cells to hydroxyurea

through inhibiting RPA2 hyperphosphorylation-mediated DNA repair
pathway,” _DNA Repair_, vol. 58, pp. 1–12, 2017.

[54] F. Morales-Vásquez et al., “Adjuvant chemotherapy with doxorubicin

and dacarbazine has no effect in recurrence-free survival of malignant
phyllodes tumors of the breast,” _Breast J._, vol. 13, no. 6, pp. 551–556,
2007.

[55] F. Di Costanzo et al., “Paclitaxel and mitoxantrone in metastatic breast

cancer: A phase II trial of the italian oncology group for cancer research,”
_Cancer Investigation_, vol. 22, no. 3, pp. 331–337, 2004.

[56] H. Qiao et al., “Redox-triggered mitoxantrone prodrug micelles for over
comingmultidrug-resistantbreastcancer,” _J.DrugTargeting_,vol.26,no.1,
pp. 75–85, 2018.

[57] H. E. Daaboul et al., “ _β_ -2-himachalen-6-ol inhibits 4t1 cells-induced

metastatic triple negative breast carcinoma in murine model,” _Chemico-_
_Biol. Interact._, vol. 309, 2019, Art. no. 108703.

[58] R.-J. Ju et al., “Octreotide-modified liposomes containing daunorubicin

and dihydroartemisinin for treatment of invasive breast cancer,” _Ar-_
_tif. Cells, Nanomedicine, Biotechnol._, vol. 46, no. sup1, pp. 616–628,
2018.

[59] L. B. Marks et al., “Impact of high-dose chemotherapy on the ability to

deliver subsequent local–regional radiotherapy for breast cancer: Analysis
of cancer and leukemia group B protocol 9082,” _Int. J. Radiat. Oncol.*_
_Biol.* Phys._, vol. 76, no. 5, pp. 1305–1313, 2010.

[60] M. E. Melisko, M. Assefa, J. Hwang, A. DeLuca, J. W. Park, and H. S.

Rugo, “Phase II study of irinotecan and temozolomide in breast cancer
patients with progressing central nervous system disease,” _Breast Cancer_
_Res. Treat._, vol. 177, no. 2, pp. 401–408, 2019.

[61] I. K. Dhammi and S. Kumar, “Medical subject headings (MeSH) terms,”

_Indian J. Orthopaedics_, vol. 48, no. 5, 2014, Art. no. 443.



Authorized licensed use limited to: ULAKBIM UASL - GAZI UNIV. Downloaded on November 17,2022 at 09:24:11 UTC from IEEE Xplore. Restrictions apply.


