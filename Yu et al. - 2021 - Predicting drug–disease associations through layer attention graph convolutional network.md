_Briefings in Bioinformatics,_ 22(4), 2021, 1–11


**[https://doi.org/10.1093/bib/bbaa243](https://doi.org/10.1093/bib/bbaa243)**
Problem solving protocol

# **Predicting drug–disease associations through layer** **attention graph convolutional network**

## Zhouxin Yu [†], Feng Huang [†], Xiaohan Zhao, Wenjie Xiao and Wen Zhang


Corresponding author: Wen Zhang, College of Informatics, Huazhong Agricultural University, Wuhan, Hubei 430070, China; Hubei Engineering
Technology Research Center of Agricultural Big Data, Wuhan, Hubei 430070, China. E-mail: zhangwen@mail.hzau.edu.cn

- These authors contributed equally to this work.


Abstract


Background: Determining drug–disease associations is an integral part in the process of drug development. However, the
identification of drug–disease associations through wet experiments is costly and inefficient. Hence, the development of
efficient and high-accuracy computational methods for predicting drug–disease associations is of great significance.
Results: In this paper, we propose a novel computational method named as layer attention graph convolutional network
(LAGCN) for the drug–disease association prediction. Specifically, LAGCN first integrates the known drug–disease
associations, drug–drug similarities and disease–disease similarities into a heterogeneous network, and applies the graph
convolution operation to the network to learn the embeddings of drugs and diseases. Second, LAGCN combines the
embeddings from multiple graph convolution layers using an attention mechanism. Third, the unobserved drug–disease
associations are scored based on the integrated embeddings. Evaluated by 5-fold cross-validations, LAGCN achieves an area
under the precision–recall curve of 0.3168 and an area under the receiver–operating characteristic curve of 0.8750, which are
better than the results of existing state-of-the-art prediction methods and baseline methods. The case study shows that
LAGCN can discover novel associations that are not curated in our dataset.

Conclusion: LAGCN is a useful tool for predicting drug–disease associations. This study reveals that embeddings from
different convolution layers can reflect the proximities of different orders, and combining the embeddings by the attention
mechanism can improve the prediction performances.


**Key words:** drug; disease; drug–disease association prediction; layer attention; graph convolutional network



Introduction


Drug development is an extremely lengthy and costly process,
which costs 2.6 billion dollars and takes 12 years on average for a
new drug [1]. Determining the associated diseases of a new drug
(such as off-label indications and side-effects) is an important
part of drug development. Computationally, identifying drug–
disease associations efficiently picks out candidate associations



and guides wet experiments for further validation, and hence
can accelerate drug development. The development of highaccuracy (ACC) computational methods is of far-reaching significance with still great challenges and has attracted continuous
attention.

The previous computational methods for predicting drug–
disease associations can be roughly divided into three categories



**Zhouxin Yu** is a student of the College of Informatics, Huazhong Agricultural University. His research interests include bioinformatics and machine
learning.
**Feng Huang** is a PhD student of the College of Informatics, Huazhong Agricultural University. His research interests include bioinformatics and machine
learning.
**Xiaohan Zhao** is a student of the College of Informatics, Huazhong Agricultural University. Her research interests include bioinformatics and machine
learning.
**Wenjie Xiao** is a student of the Information School, University of Washington. Her research interests include bioinformatics and machine learning.
**Wen Zhang** obtained a bachelor’s degree and a master’s degree in computational mathematics from Wuhan University in 2003, 2006, and got a doctoral
degree in computer science from Wuhan University in 2009. He is now a professor in the College of Informatics, Huazhong Agricultural University, People’s
Republic of China. His research interests include bioinformatics and machine learning.
**Submitted:** 29 June 2020; **Received (in revised form):** 16 August 2020


© The Author(s) 2020. Published by Oxford University Press. All rights reserved. For Permissions, please email: journals.permissions@oup.com


1


2 _Yu et al._


[2, 3], i.e. network diffusion-based methods, machine learningbased methods and deep learning-based methods.
Network diffusion-based methods generally link drugs with
diseases through message propagation on the paths bridging
different networks [4–7]. For example, Wang _et al_ . [8] designed
a triple-layer heterogeneous graph-based inference method (TLHGBI) to infer potential links between drugs and diseases. Luo
_et al_ . applied random walkers respectively on the drug–disease
bipartite network [9] and on a drug–target–disease heterogeneous network [10] to predict novel drug–disease associations.
Although the network diffusion-based methods have advantages of good interpretability, their performances are not satisfactory [2].
Machine learning techniques have been widely adopted to
develop more accurate models for drug–disease association prediction. For example, a number of feature-based classification
methods [11–15] take drug–disease pairs as samples, encode the
side information of drugs and diseases into feature vectors to
characterize the samples and then train classifiers to distinguish
whether the associations exist or not. However, feature-based
classification methods heavily rely on the extraction of features
and the selection of negative samples. Thus, a surge of more
sophisticated techniques, such as sparse subspace learning [16],
semi-supervised graph cut [17], label propagation [18], regularized least squares [19], matrix factorization [20–22] and matrix
completion [23–26], have been applied to the drug–disease association prediction. In particular, matrix factorization and matrix
completion techniques are of great popularity in community,
due to their flexibility in integrating prior information, and have
shown promising results in predicting drug–disease associations, but it is challenging to deploy them on large-scale data
because of high-complexity matrix operations.
Deep learning methods have been demonstrated to be
more effective in many tasks, including but not limited to face
recognition, question answering system, computational biology

[27, 28], and also have successful applications in the drug–
disease association prediction [29–31]. For example, Zeng _et al_ .

[29] recently developed a network-based deep learning approach,
termed deepDR. It firstly calculated positive pointwise mutual
information (PPMI) matrices from 10 drug-related networks and
use them as features, then fused PPMI matrices by multimodal
deep autoencoder and finally exploited the fused features to
infer new applications for existing drugs by collective variational
autoencoder. The advantage of deepDR is taking full use of
topological information of drug similarity networks. However,
deepDR does not take the side information of diseases into
account. DeepDR has two separate components instead of a
full end-to-end framework, and this may have an impact on the
performances of prediction models.
Graph convolutional network (GCN) [32], extending convolutional neural networks for processing graph data, is readily
embedded in end-to-end architectures to perform specific tasks
with graph inputs, captures structural information of graphs
via message passing between the nodes of graphs and retains
high interpretability. Recently, it demonstrates convincing performances on biomedical network analysis, such as microRNA
(miRNA)–disease association prediction [33], polypharmacy sideeffect prediction [34] and miRNA–drug resistance association
prediction [35].
In this paper, we develop a novel end-to-end layer attention
graph convolutional network (LAGCN) method for predicting
drug–disease associations. We first construct heterogeneous
networks by bridging the known drug–disease associations,
drug–drug similarities and disease–disease similarities. Then,



we use the graph convolution operation on the heterogeneous
network to learn the embeddings of drugs and diseases.
Given that the embeddings from multiple convolution layers
reflect proximities of different orders [36] between nodes in
the network, we resort to the attention mechanism [37] to
integrate all useful structural information from multiple graph
convolution layers. Finally, the predictive scores for unobserved
drug–disease associations are given by a well-defined score
function based on the integrated embeddings. According to
the reliable _in silico_ experiments, our proposed method LAGCN
achieves the area under the precision–recall curve (AUPR) score
of 0.3168 and area under the receiver–operating characteristic
curve (AUC) score of 0.8750, and performs better than other
state-of-the-art methods and baseline methods.

The main contributions of this work are summarized as

follows:


 - We propose a full end-to-end graph-based deep learning
method, termed LAGCN, for effectively predicting drug–
disease associations.

 - LAGCN utilizes a GCN to capture structural information
from the heterogeneous network composed of drug–disease
associations, drug–drug similarities and disease–disease
similarities.

 - The attention mechanism is introduced to combine the

embeddings from different convolution layers, which leads
to a more informative representation of drugs and diseases.


Materials


**Datasets**


The data in our previous studies [20, 38] are assembled as the
main dataset in this paper. The main dataset contains 18 416
drug–disease associations between 269 drugs and 598 diseases
derived from Comparative Toxicogenomics Database (CTD) [39].
The comprehensive information about drugs, such as targets,
enzymes, drug–drug interactions, pathways and substructures,
is obtained from DrugBank database [40]. The diseases are normalized through standard terms from Medical Subject Headings
(MeSH). Considering that therapeutic associations may have
special significance for drug discovery, we also extract 6244 therapeutic associations annotated in CTD from the main dataset as
the therapeutic dataset. Detailed information about two datasets
is summarized in Table 1.


**Construction of the heterogeneous network**


_**Drug–drug similarities**_


Drugs usually have different features describing biological or
chemical characteristics. One drug can be encoded as a binary
feature vector where each element means the presence or
absence of features descriptor. Since we have different types
of features, we can convert drugs into multiple types of feature
vectors and calculate various drug–drug similarities based on
these features by using different similarity measures. To the best
of our knowledge, Jaccard index [29, 41] and Cosine similarity [21]
are two prevailing measures for the drug–drug similarities.
Jaccard index between two binary feature vectors _x_ _i_ and _x_ _j_ is
calculated by



_S_ _[r]_ _ij_ [=]



~~�~~ ��� _xx_ _ii_ ∩ ∪ _xx_ _jj_ �� ~~�~~ � (1)


_layer attention GCN_ 3


**Table 1.** Summary of two datasets


Dataset Drugs Diseases Known Drug features
associations


Target Enzyme Drug–drug Pathway Substructure
interactions


Main dataset 269 598 18 416 623 247 2086 465 881

Therapeutic 269 598 6244 623 247 2086 465 881

dataset



where ��� _x_ _i_ ∩ _x_ _j_ ��� denotes the number of cases where both elements

in _x_ _i_ and the corresponding ones of _x_ _j_ are equal to 1, and ��� _x_ _i_ ∪ _x_ _j_ ���

denotes the number of the cases where either the elements of _x_ _i_
or the corresponding ones of _x_ _j_ are equal to 1.
Cosine similarity between two binary feature vectors _x_ _i_ and
_x_ _j_ is calculated by


_S_ _[r]_ _ij_ [=] ~~∥~~ _[x]_ _i_ _x_ ~~∥∥~~ _i_            - _x_ _j_ _[x]_ _j_ ~~∥~~ (2)


where ∥ _x_ _i_ ∥ denotes the L2-norm of _x_ _i_ .
In this work, we adopt the Jaccard index to calculate the drug–
drug similarities for our prediction methods and also consider
Cosine similarity. Jaccard index and Cosine similarity are compared in the section of ‘Results and Discussion’. Since we have
five types of drug features in our datasets, we also calculate
drug–drug similarities based on different features and compare
these similarities.


_**Disease–disease similarities**_


MeSH descriptors of diseases can be represented as hierarchical
directed acyclic graphs (DAGs). As described in [42], disease–
disease similarities can be calculated using the DAG structures.
For a disease _d_, we represent its hierarchical relationship by

DAG( _d_ ) = � _N_ ( _d_ ), _E_ ( _d_ )�, where _N_ ( _d_ ) is the set of nodes containing

_d_ and its ancestors, and _E_ ( _d_ ) denotes the set of direct links from
parent nodes to their child nodes. Based on this DAG structure,
the contribution of a node _n_ in DAG( _d_ ) to the semantic value of
disease _d_ is given by



**Layer attention graph convolutional network**


In this section, we introduce the LAGCN for drug–disease association prediction. The workflow of LAGCN is briefly shown in
Figure 1.


**Method architecture**


GCN [32] is a multilayer connected neural network architecture
for learning low-dimensional representations of nodes from
graph-structured data. Each layer of GCN aggregates neighbor’s
information to reconstruct the embeddings as the inputs of the
next layer through the direct links of graphs.
Specifically, given a network with the corresponding adjacency matrix _G_, the layerwise propagation rule of GCN is formulated as



disease _d_ _j_ ; otherwise _A_ _ij_ = 0. The pairwise similarities between
_N_ drugs are denoted as a similarity matrix _S_ _[r]_ with _S_ _[r]_ _ij_ [as its]
( _i_, _j_ )th entry; the pairwise similarities between _M_ diseases are
denoted as a similarity matrix _S_ _[d]_ with _S_ _[d]_ _ij_ [as its (] _[i]_ [,] _[ j]_ [)th entry.]



− [1]
We normalize the similarity matrices by ∼ _S_ _[r]_ = _D_ _r_ 2




[1] − [1]

2 _S_ _[r]_ _D_ _r_ 2



We normalize the similarity matrices by ∼ _S_ _[r]_ = _D_ _r_ − 2 _S_ _[r]_ _D_ _r_ − 2 and

∼ _S_ _[d]_ = _D_ _d_ − [1] 2 _S_ _[d]_ _D_ _d_ − [1] 2, where _D_ _r_ = diag( [�] _j_ _[S]_ _[r]_ _ij_ [) and] _[ D]_ _[d]_ [ =][ diag(][�] _j_ _[S]_ _[d]_ _ij_ [).]

Finally, we construct the heterogeneous network defined by the
adjacency matrix:




[1] − [1]

2 _S_ _[d]_ _D_ _d_ 2



_j_ _[S]_ _[r]_ _ij_ [) and] _[ D]_ _[d]_ [ =][ diag(][�]



2, where _D_ _r_ = diag( [�]



�



_A_ _H_ =



∼ _S_ _[r]_ _A_
_A_ _[T]_ ∼ _S_ _[d]_
�



∈ R _[(][N]_ [+] _[M][)]_ [×] _[(][N]_ [+] _[M][)]_ (5)



_H_ _[(]_ _[l]_ [+][1] _[)]_ = _f_ � _H_ [(] _[l]_ [)], _G_ � = _σ_ � _D_ [−] 2 [1]



_C_ _d_ ( _n_ ) =



1 if _n_ = _d_
(3)
� max � _�_ ∗ _C_ _d_ _(n_ [′] _)_ | _n_ [′] ∈ children of _n_ } if _n_ ̸= _d_




[1]

2 _G D_ [−] 2 [1]




[1]

2 _H_ [(] _[l]_ [)] _W_ [(] _[l]_ [)] [�] (6)



where _�_ is a contribution factor ranging from 0 to 1, and here
_�_ is set to 0.5. The semantic value of disease _d_ is defined as
DV( _d_ ) = [�] _n_ ∈ _N_ ( _d_ ) _[C]_ _[d]_ [(] _[n]_ [)] _[.]_ [ It is believed that diseases with more]

common ancestors in the DAG are prone to have higher semantic similarities. According to this hypothesis, we calculate the
semantic similarity between the two diseases _d_ _i_ and _d_ _j_ by



_n_ ∈ _N_ _(_ _di_ _)_ ∩ _N_ � _dj_



_)_ ∩ _N_ � _dj_ � _di_ _dj_ (4)

DV _(_ _d_ _i_ _)_ +DV _(_ _d_ _j_ _)_



� � _C_ _di_ ( _n_ )+ _C_ _dj_ ( _n_ )�



where _H_ [(] _[l]_ [)] is the embeddings of nodes at the _l_ th layer, _D_ =
diag( [�] _j_ _[G]_ _[ij]_ [) is the degree matrix of] _[ G]_ [,] _[ W]_ [(] _[l]_ [)] [ is a layer-specific train-]

able weight matrix and _σ_ (· )is a non-linear activation function.
To build a GCN-based encoder for learning the lowdimensional representations of drugs and diseases, we consider
combining node similarities and directly linked association
information through deploying GCN on our constructed
heterogeneous graph _A_ _H_ . First, we introduce a penalty factor
_μ_ to control the contribution of similarities in the propagation
process of GCN. To be specific, we set the input graph _G_ as



_S_ _[d]_ _ij_ [=]


_**Heterogeneous network**_



�



The heterogeneous network is constructed based on drug–
disease associations, disease–diseases similarities and disease–

disease similarities.

We denote drug–disease associations as a binary matrix _A_ ∈
{0, 1} _[N]_ [×] _[M]_, where _M_, _N_ denote the number of diseases and drugs,
respectively. _A_ _ij_ is equal to 1, if a drug _r_ _i_ has association with a



�



_G_ =



_μ_ ∼ _S_ _[r]_ _A_
� _A_ _[T]_ _μ_ ∼ _S_ _[d]_



(7)



Then, we initialize embeddings as


4 _Yu et al._


**Figure 1** . The workflow of layer attention graph convolutional network.



�



_H_ [(0)] =



0 _A_

_A_ _[T]_ 0
�



(8)



the final embeddings of drugs, _H_ _D_ ∈ R _[M]_ [×] _[k]_ is the final embeddings of diseases and _a_ _l_ is auto-learned by neural networks and
initialized as 1 _/_ ( _l_ + 1), _l_ = 1, 2, _. . ._, _L_ .
To reconstruct the adjacency matrix for drug–disease associations, a bilinear decoder _A_ [′] = _f_ ( _H_ _R_, _H_ _D_ ) created by [33] is
adopted:


_A_ [′] = sigmoid � _H_ _R_ _W_ [′] _H_ _[T]_ _D_ � (10)


where _W_ [′] ∈ R _[k]_ [×] _[k]_ is a trainable matrix. The predicted score for
the association between drug _r_ _i_ and disease _d_ _j_ is given by the

corresponding � _i_, _j_ �th entry of _A_ [′], denoted as _a_ [′] _ij_ [.]


**Optimization**


From a dataset with _N_ drugs and _M_ diseases, we take drug–
disease association pairs as positive instances and take other
pairs as negative instances. Herein, the set of positive instances
and the set of negative instances are respectively denoted as
_Y_ [+] and _Y_ [−] . Differentiating two types of drug–disease pairs is a
binary classification problem. However, the number of associations is much less than that of the drug–disease pairs, which
have no observed associations. Here, we adopt the weighted
cross-entropy as loss function:


1
Loss = − _N_ × _M_ � _λ_ × [�] _(_ _[i]_ [,] _[j]_ _)_ [∈] _[Y]_ [+] [ log] _[ a]_ _ij_ [′] [+][ �] _(_ _[i]_ [,] _[j]_ _)_ [∈] _[Y]_ [−] [log] �1 − _a_ [′] _ij_ �� (11)



With the above settings, the first layer of our GCN encoder is
formulated as



_H_ [(1)] = _σ_ _D_ [−] 2 [1]
�




[1]

2 _H_ [(0)] _W_ [(0)] [�] (9)




[1]

2 _G D_ [−] 2 [1]



where _W_ [(0)] ∈ R [(] _[N]_ [+] _[M]_ [)][×] _[k]_ is an input-to-hidden weight matrix, _H_ [(1)] ∈
R [(] _[N]_ [+] _[M]_ [)][×] _[k]_ is the first-layer embeddings of the nodes (drugs and
diseases) of the heterogeneous network _A_ _H_, _k_ is the dimensionality of the embeddings and _G_ is defined in Equation (7). The
subsequent layers of our GCN encoder follow the Equation (6) for
_l_ = 1, 2, _. . ._, _L_ with _W_ [(] _[l]_ [)] ∈ R _[k]_ [×] _[k]_ and _G_ defined in Equation (7). After
_L_ iterations, we can obtain _L k_ -dimensionality embeddings from
different graph convolution layers. Exponential linear unit [43]
is used as the non-linear activation function in all graph convolution layers, which not only accelerates learning procedure but
also significantly enhances generalization performance.
The embeddings at different layers capture different structural information of the heterogeneous network. For example,
the first layer harvests the direct link information, and higher
layers capture multihop neighbor information (high-order proximity) by iteratively updating the embeddings [44, 45]. Considering the contributions of different embeddings at different
layers are inconsistent, we introduce an attention mechanism
to combine these embeddings and obtain final embeddings of



where ( _i_, _j_ ) denotes the pair for drug _r_ _i_ and disease _d_ _j_ . _λ_ =



−
�� _Y_ ��
� �

,
~~��~~ _Y_ + ~~��~~
� �



drugs and diseases as � _HH_ _DR_



� = [�] _a_ _l_ _H_ _[l]_, where _H_ _R_ ∈ R _[N]_ [×] _[k]_ is



_Y_ + and _Y_ − are the number of instances in _Y_ + and _Y_ − . The
��� ��� ��� ���


_layer attention GCN_ 5


**Table 2.** Performances of LAGCN with different drug–drug similarities and without similarities


Similarity measures Drug feature AUPR AUC RE SP ACC F1


Jaccard Target 0.3168 0.8750 0.3600 0.9760 0.9605 0.3150
Enzyme 0.3166 0.8758 0.3567 0.9764 0.9608 0.3149
Drug interaction 0.3163 0.8761 0.3533 0.9772 0.9615 0.3167
Pathway 0.3149 0.8761 0.3603 0.9758 0.9603 0.3141

Substructure 0.3115 0.8765 0.3475 0.9778 0.9619 0.3153

Average similarity 0.3165 0.8763 0.3773 0.9739 0.9588 0.3159
Concatenated feature-based 0.3134 0.8761 0.3505 0.9776 0.9618 0.3162

similarity


Cosine Target 0.3149 0.8737 0.3717 0.9739 0.9587 0.3121
Enzyme 0.3137 0.8739 0.3701 0.9746 0.9593 0.3144
Drug interaction 0.3136 0.8753 0.3464 0.9776 0.9617 0.3130
Pathway 0.3149 0.8746 0.3591 0.9758 0.9603 0.3131

Substructure 0.3110 0.8748 0.3609 0.9755 0.9600 0.3128

Average similarity 0.3113 0.8743 0.3606 0.9756 0.9601 0.3132
Concatenated features-based 0.3131 0.8754 0.3596 0.9757 0.9602 0.3125

similarity


LAGCN-NH 0.2952 0.8455 0.3577 0.9491 0.9342 0.2918


RE, recall; SP, specificity; F1, F1-measure.



weight factor _λ_ imposes the importance of observed associations
to reduce the influence of data imbalance.
All the trainable weight matrices ( _W_ [(] _[l]_ [)] and _W_ [′] ) are initialized
by the Xaiver initialization method [46]. Then, we use Adam
optimizer [47] to minimize the loss function. Adam optimizer
can update the weights of neural network iteratively based on
training data. To prevent over-fitting, we introduce node dropout

[48] and regular dropout [49] to the graph convolution layers. This
node dropout can be considered as training of different models
on various small subnetworks, and unknown drug–disease pairs
are predicted by integrating these small models [50]. Besides, the
cyclic learning rate [51] is used during the optimization. A simple
cyclic learning rate makes a change in the learning rate between
the maximum learning rate and the minimum, helping us to
balance the training speed and ACC.


Results and discussion


**Experimental setting**


In our experiments, we adopted 5-fold cross-validation (5-CV)
to evaluate the performances of prediction methods. All known
drug–disease associations are randomly split into five equalsized subsets. The cross-validation process is repeated fives
times, and every subset is used as the testing set in turn while
the remaining four subsets are used as the training set. In each
fold, a prediction model is constructed on known associations
in the training set and is used to predict associations in the
testing set. We adopt the AUPR and the AUC as primary metrics,
for they can measure the performances of methods without any
specific threshold. Besides, the threshold-based metrics are also
calculated, i.e. recall (also known as sensitivity), specificity, ACC,
precision and F1-measure (F1).
There are several hyperparameters in LAGCN such as
the dimensionality of embeddings _k_, the number of layers
_L_, the initial learning rate of optimizer _lr_, the total training
epochs of LAGCN _α_, two dropout rates (node dropout and
regular dropout) _β_, _γ_ and the penalty factor _μ_ in the heterogeneous network. We consider different combinations of

these parameters from the ranges _α_ ∈ �500, 1000, 2000, 4000�,



_β_, _γ_ ∈ �0.1, 0.2, 0.3, 0.4, 0.5, 0.6�and _μ_ ∈ �2, 4, 6, 8, 10�. By adjusting

the parameters empirically, we set the parameter _k_ = 64, _L_ =
3, _lr_ = 0.008, _α_ = 4000, _β_ = 0.6, _γ_ = 0.4 and _μ_ = 6 for LAGCN in the
following experiments.


**Results of LAGCN**


_**Influence of different heterogeneous networks**_


LAGCN makes use of the drug–disease heterogeneous network
to build the prediction model. The drug–disease heterogeneous
network consists of known drug–disease associations, drug–
drug similarities and disease–disease similarities. Since we consider five types of drug features and two similarity measures,
we can train LAGCN on different heterogeneous networks based
on different drug–drug similarities, and then discuss how these
drug–drug similarities influence the performances of LAGCN.
LAGCN models based on heterogeneous networks with different drug–drug similarities are evaluated by 5-CV on the main
dataset, and the corresponding results are displayed in Table 2.
Jaccard index leads to slightly better results than Cosine similarity measure, and similarities based on different features produce similar performances. These results indicate that LAGCN
is robust, regardless of similarity measures and drug features.
Drug target-based similarities (by both Jaccard index and Cosine
similarity) leads to the highest AUPR score.
Moreover, we only use the drug–disease associations to
construct the network, and then build a reduced version of
LAGCN based on this network, named as LAGCN-NH. According
to Table 2, LAGCN-NH produces lower AUPR score and AUC
score than all LAGCN models, which demonstrates that the
drug–drug similarities and disease–disease similarities in the
heterogeneous network contain useful information and lead to
the improved performances of LAGCN.
We also integrate different drug feature-based similarities
using two simple strategies, and build LAGCN models. The
average similarity strategy calculates the average of drug–drug
similarities based on different features to obtain the integrated
similarities. The concatenated feature-based similarity strategy
firstly concatenate different drug feature vectors and then
calculate drug–drug similarities based on the concatenated


6 _Yu et al._


**Table 3.** Performance of LAGCN based on different embeddings


Models AUPR AUC RE SP ACC F1


LAGCN 0.3168 0.8750 0.3600 0.9760 0.9605 0.3150

LAGCN-AVE 0.2912 0.8675 0.3550 0.9732 0.9576 0.2971

LAGCN-CON 0.3006 0.8738 0.3467 0.9771 0.9612 0.3106

LAGCN-L1 0.2928 0.8765 0.3486 0.9777 0.9618 0.3155

LAGCN-L2 0.2921 0.8629 0.3489 0.9725 0.9568 0.2896

LAGCN-L3 0.2724 0.7319 0.4108 0.7788 0.7686 0.1909


RE, recall; SP, specificity; F1, F1-measure.


feature vectors. The results in Table 2 show that integrating
different drug feature-based similarities do not necessarily
lead to improved performances. The possible reason is that the
known drug–disease associations make the major contribution
to the prediction, and drug features bring supplementary
information but have redundant information between them.

Based on the above discussion, we adopt the target-based
drug–drug similarities calculated by the Jaccard index, MeSHbased disease–disease similarities and drug–disease associations to construct the heterogeneous network and then build the
LAGCN models in the following study.


_**Effect of layer attention mechanism**_



Layer attention is one component of the network architecture of
LAGCN and is in charge of managing and quantifying the interdependence of different convolution layers. Here, we discuss the
effect of the layer attention mechanism.
We only use the embeddings at the _l_ th layer of LAGCN with _l_ =
1, 2, 3 to build LAGCN models, abbreviated as LAGCN-L1, LAGCNL2 and LAGCN-L3. The results of all models evaluated by 5-CV on
the main dataset are shown in Table 3. LAGCN-L1 and LAGCN
L2 produce better results than LAGCN-L3, indicating that the
first-layer embeddings and second-layer embeddings contain
more information than the third-layer embeddings. The results
may be caused by the over-smoothing of GCN [52]. LAGCN that
integrates embeddings at three layers produces better results
than LAGCN-L1, LAGCN-L2 and LAGCN-L3.
It is believed that the _l_ th convolution layer of GCN captures
the _l_ th-order proximity, and the attention weights denote the
contributions of the embeddings at different convolution layers
to the final embeddings. We implement 20 runs of 5-CV for
LAGCN, and visualize the attention weights for three convolution layers in Figure 2. Three layers have different attention
weights, and first layer _>_ second layer _>_ third layer, which meets
our expectation that lower-order proximity has a greater contribution, and higher-order proximity has a lower contribution.
The results also help to explain the performances of LAGCN-L1,
LAGCN-L2 and LAGCN-L3 in Table 3. Therefore, paying different
attention to convolution layers is necessary for building the
high-ACC prediction models.
Furthermore, we consider other approaches to combining
embeddings at different layers. LAGCN-AVE assigns uniform
weights to different embeddings; LAGCN-CON directly concatenates different embeddings. As shown in Table 3, LAGCN produces better results than LAGCN-AVE and LAGCN-CON, showing
the effectiveness of attention mechanism in LAGCN.


**Comparison with other methods**


In this section, we compare LAGCN with five state-of-the-art
drug–disease prediction methods [8, 20, 24, 25, 29] and a baseline
method [35]. We replicate them according to their publications



**Figure 2** . Attention weights for three convolution layers in LAGCN.


or using publicly available programs. The performances of all
methods are shown in Table 4.


  - TL-HGBI [8] constructed a three-layer heterogeneous network, and then an iterative updating algorithm was proposed to infer the probabilities of new drug–disease associations.

 - SCMFDD [20] projected drug–disease association into two
low-rank spaces uncovering latent features for drugs and
diseases, and then introduced similarity constraints to
smooth the features.

 - DRRS [24] deployed a matrix completion-based recommendation system on a drug–disease heterogeneous network to
predict drug–disease associations.

 - BNNR [25] proposed a bounded nuclear norm regularization method to complete a drug–disease heterogeneous
network.

  - DeepDR [29] calculated PPMI matrices based on drug-related
networks as drug features, and then proposed a multimodal
deep autoencoder for fusing the features and a collective
variational autoencoder for mining new associations. Here,
we implement deepDR on our dataset by calculating six
PPMI matrices from a drug–disease network and five drug–
drug similarity networks.

  - NIMCGCN [35] used GCNs to learn latent feature representations of miRNA and disease from the similarity networks,
and then put the learned features into a neural inductive
matrix completion model to obtain a reconstructed association matrix. NIMCGCN is a GCN-based method proposed for
the miRNA–disease association prediction, and we adopt it
as the baseline method.


According to Table 4, LAGCN outperforms all comparison
methods in terms of most evaluation metrics. Compared


_layer attention GCN_ 7


**Table 4.** Performance of comparison methods on two datasets


Dataset Methods AUPR AUC RE SP ACC F1


Main dataset TL-HGBI 0.0665 0.7029 0.2545 0.9284 0.9114 0.1266

SCMFDD 0.2659 0.8727 0.3430 **0.9783** **0.9623** 0.3143

DRRS 0.1321 0.8429 0.3276 0.9468 0.9324 0.2178

DeepDR 0.1353 0.8211 0.2959 0.9567 0.9400 0.1991

BNNR 0.2262 0.8567 0.3403 0.9738 0.9578 0.2894

NIMCGCN 0.2002 0.8533 0.3083 0.9739 0.9572 0.2661

LAGCN **0.3168** **0.8750** **0.3600** 0.9760 0.9605 **0.3150**


1 pt

Therapeutic dataset TL-HGBI 0.0388 0.7401 0.1151 0.9830 0.9761 0.0720

SCMFDD 0.1383 0.8754 0.2774 0.9871 0.9815 0.1934

DRRS 0.1494 0.8873 0.2726 0.9907 0.9849 0.2249

DeepDR 0.1011 0.8572 0.2327 0.9866 0.9806 0.1610

BNNR 0.1832 0.8794 0.2859 **0.9918** **0.9861** **0.2477**

NIMCGCN 0.0899 0.8075 0.2225 0.9859 0.9798 0.1525

LAGCN **0.3431** **0.8902** **0.5825** 0.9549 0.9520 0.1630


RE, recall; SP, specificity; F1, F1-measure.


**Table 5.** Performance of prediction models in deepDR’s dataset


Methods AUPR AUC RE SP ACC F1


DeepDR 0.9201 0.9021 0.7789 0.8919 0.8354 0.8254


LAGCN 0.9487 0.9406 0.9044 0.8417 0.8731 0.8770

0.9480 0.9391 0.9176 0.8294 0.8735 0.8789

0.9447 0.9360 0.9079 0.8255 0.8667 0.8720

0.9393 0.9314 0.8507 0.8787 0.8647 0.8627

0.9404 0.9342 0.8675 0.874 0.8708 0.8703

0.9378 0.9308 0.864 0.8652 0.8646 0.8645

0.9416 0.935 0.872 0.8543 0.8631 0.8643

0.9377 0.9302 0.861 0.8526 0.8568 0.8575

0.9389 0.9311 0.8421 0.8865 0.8643 0.8612

0.9424 0.9316 0.9277 0.8046 0.8661 0.8739


RE, recall; SP, specificity; F1, F1-measure.



with the matrix factorization and completion-based methods
(SCMFDD, DRRS and BNNR), the network diffusion-based
method TL-HGBI performs worse, whereas LAGCN achieves 52%
improvement on average over them in terms of AUPR. GCNbased methods (NIMCGCN and LAGCN) perform better than
deepDR, demonstrating that GCN may lead to better aggregation
of network topological information.
DeepDR depends on their dataset that contains 10 networks,
whereas LAGCN only uses drug–drug similarities, disease–
disease similarities and drug–disease associations. We also
compare LAGCN with deepDR based on the deepDR’s dataset.
Following deepDR’s experimental setting, we randomly select
20% of the observed drug–disease associations and a matching
number of randomly sampled unknown associations as the
testing set, and the remaining 80% associations are used to train
the models. Among 10 drug-related networks used in deepDR’s
dataset, six types of drug–drug similarities are respectively used
as input of our proposed method LAGCN; other networks are
also converted into drug–drug similarities by Jaccard index

∼ _d_
and then respectively used as input of LAGCN. We set _S_ = 0

in LAGCN due to the absence of disease–disease similarities

in deepDR’s dataset. DeepDR is implemented by using the
publicly available source code and the recommended parameter
settings. As shown in Table 5, LAGCN models that use different



similarities produce consistently higher AUC scores and AUPR
scores than deepDR, which further shows the robustness of
LAGCN. Although both LAGCN and deepDR are deep learningbased methods, the full end-to-end framework of LAGCN can
help to improve the performances.
The known drug–disease associations are an essential
resource for predicting potential drug–disease associations. The
number of observed associations could greatly affect method
performance. To test the robustness of LAGCN and comparison
methods, SCMFDD, deepDR, BNNR and NIMCGCN, we randomly
remove a fraction of known associations in the main dataset
at a ratio _r_ ∈ �80%, 85%, 90%, 95%, 100%� and implement 5-CV

to evaluate methods. As displayed in Figure 3, the number of
drug–disease associations is an important factor for the drug–
disease association prediction, and more associations can result
in better prediction models. However, LAGCN can produce the
most robust and highest performances across different data
richness among these methods.


**Case study**


In this section, we build an LAGCN model using all drug–disease
associations and then predict novel associations. Because all
known associations have been used to construct the prediction


8 _Yu et al._


**Figure 3** . Performances of methods based on different fractions of known associations.



**Table 6.** Top 10 drug–disease associations predicted by LAGCN


Drug Disease Evidence


Cocaine Migraine disorders [57]
Tamoxifen Depressive disorder [53]
Clozapine Hyperhomocysteinemia NA
Cimetidine Hypotension, orthostatic NA
Phenytoin Splenomegaly [58]
Clozapine Hyperhidrosis [59]
Cocaine Endomyocardial fibrosis NA
Sirolimus Sinoatrial block NA

Dexamethasone Multiple myeloma [54]
Tamoxifen Dementia NA


model, the predicted associations require verification by public
literature or other available sources. Top 10 drug–disease associations predicted by LAGCN are listed in Table 6, and we can find
evidence to confirm five out of them. For example, tamoxifen is
capable of prolonging the lives of premenopausal women with
breast cancer and decreasing the probability of recurrence, but it
led to acute depression symptoms in a 34-year-old breast cancer
patient [53]. Dexamethasone is a type of corticosteroid medication. It was used in the treatment of many conditions, including rheumatic problems, several skin diseases, severe allergies,
asthma, chronic obstructive lung disease, croup, brain swelling,
ocular pain following ophthalmic surgery and antibiotics in
tuberculosis. It was also used as a direct chemotherapeutic agent
in the treatment of multiple myeloma, in which dexamethasone
is given in combination with lenalidomide [54].
Furthermore, we check upon the top 10 candidate diseases
for carbamazepine and the top 10 candidate drugs for breast
neoplasms. Table 7 shows the results of our experiments, and
some of the predictions can be confirmed. For example, carbamazepine is an anticonvulsant medication used primarily in
the treatment of epilepsy and neuropathic pain, but De Sarro
_et al_ . [55] have proved that carbamazepine also leads to movement disorders by potentiating the anticonvulsant activity in
the DBA/2 mice animal model. Breast cancer is the leading type
of cancer in women, accounting for 25% of all cases according
to Wikipedia, and countless researchers have been devoting
themselves to finding treatment of it. According to Tsai _et al_ . [56],
tamoxifen and fulvestrant are widely used therapeutic agents
and are considered to alter estrogen receptor (ER) signaling in
ER-positive breast cancers.



Disease Drug Evidence


Breast neoplasms Doxorubicin [65]
Vincristine [66]
Dexamethasone [67]

Sirolimus NA

Tretinoin [68]
Indomethacin [69]

Sorafenib NA

Cytarabine [70]
Mitoxantrone [71]
Tamoxifen [56]


Therefore, case studies demonstrate that LAGCN can help
to identify novel associations as well as associated diseases
(associated drugs) for a given drug (disease).


Conclusions


In this paper, we establish an LAGCN for identifying the latent
drug–disease associations. In contrast to existing methods that
utilize the bipartite graphs, LAGCN captures the topological
information of a heterogeneous network constructed from drug–
disease associations, drug–drug similarities and disease–disease
similarities. LAGCN achieves good performances in predicting
drug–disease associations by adaptively combining embeddings
at different convolution layers with an attention mechanism,



**Table 7.** Top 10 associated diseases (associated drugs) for a given drug
(disease) predicted by LAGCN


Drug Disease Evidence


Carbamazepine Intraoperative NA
complications



Movement disorders [55]
Bradycardia [60]
Tremor [61]
Carcinoma, squamous NA

cell


Liver cirrhosis NA

Leukemia, myelogenous, NA
chronic, BCR-ABL
positive

Chest pain [62]
Cocaine-related [63]

disorders


Lethargy [64]


and outperforms other drug–disease association prediction
methods and the baseline method.

In the future, we will consider more biological entities
involved in the drug–disease associations, such as genes,
miRNAs and targets, and build a heterogeneous network with
more types of entities and links for the drug–disease association
prediction. Although GCN is a powerful method for analyzing
the networks, it suffers from the problem of over-smoothing,
and we will use data augmentation techniques to relieve
over-smoothing in deep GCN.


Data availability


The datasets were derived from the following sources in the
[public domain: the drug–disease associations from http://ctdba](http://ctdbase.org/)
[se.org/, the drug features from https://www.drugbank.ca/ and](http://ctdbase.org/)
[the disease MeSH descriptors from https://meshb.nlm.nih.gov/.](https://meshb.nlm.nih.gov/)
The implementation of LAGCN and the preprocessed data is
[available at https://github.com/storyandwine/LAGCN.](https://github.com/storyandwine/LAGCN)


**Key Points**


   - We propose a full end-to-end graph-based deep learning method, termed LAGCN, for predicting drug–
disease associations.

   - LAGCN utilizes a GCN to capture structural information from the heterogeneous network composed of
associations and similarities.

   - The attention mechanism is introduced to combine

the embeddings at different convolution layers, which
leads to more informative representations of drugs
and diseases.


Funding


This work was supported by the National Natural Science Foundation of China [61772381, 62072206, 61572368];
National Key Research and Development Program

[2018YFC0407904]; and Huazhong Agricultural University
Scientific & Technological Self-innovation Foundation. The
funders have no role in study design, data collection, data
analysis, data interpretation or writing of the manuscript.


References


1. Chan HCS, Shan H, Dahoun T, _et al._ Advancing drug discovery via artificial intelligence. _Trends Pharmacol Sci_ 2019;

**40** :592–604.

2. Luo H, Li M, Yang M, _et al._ Biomedical data and computational
models for drug repositioning: a comprehensive review. _Brief_
_Bioinform_ [2020. https://doi.org/10.1093/bib/bbz176.](https://doi.org/10.1093/bib/bbz176)
3. Zhang Z-C, Zhang X-F, Wu M, _et al._ A graph regularized generalized matrix factorization model for predicting
links in biomedical bipartite networks. _Bioinformatics_ 2020;

**36** :3474–81.

4. von Eichborn J, Murgueitio MS, Dunkel M, _et al._ PROMISCUOUS: a database for network-based drug-repositioning.
_Nucleic Acids Res_ 2010; **39** :D1060–6.
5. Wiegers TC, Davis AP, Cohen KB, _et al._ Text mining and
manual curation of chemical-gene-disease networks for the
comparative toxicogenomics database (CTD). _BMC Bioinfor-_
_matics_ 2009; **10** :326.



_layer attention GCN_ 9


6. Wang L, Wang Y, Hu Q, _et al._ Systematic analysis of new drug
indications by drug-gene-disease coherent subnetworks.
_CPT Pharmacometrics Syst Pharmacol_ 2014; **3** :e146.
7. Zickenrott S, Angarica VE, Upadhyaya BB, _et al._ Prediction
of disease-gene-drug relationships following a differential
network analysis. _Cell Death Dis_ 2016; **7** :e2040.
8. Wang W, Yang S, Zhang X, _et al._ Drug repositioning by
integrating target information through a heterogeneous network model. _Bioinformatics_ 2014; **30** :2923–30.
9. Luo H, Wang J, Li M, _et al._ Drug repositioning based on
comprehensive similarity measures and Bi-Random walk
algorithm. _Bioinformatics_ 2016; **32** :2664–71.
10. Luo H, Wang J, Li M _et al._ Computational drug repositioning
with random walk on a heterogeneous network. _IEEE/ACM_
_Trans Comput Biol Bioinform_ 2019; **6** (16):1890–1900.
11. Gottlieb A, Stein GY, Ruppin E, _et al._ PREDICT: a method for
inferring novel drug indications with application to personalized medicine. _Mol Syst Biol_ 2011; **7** :496.
12. Yang L, Agarwal P. Systematic drug repositioning based on
clinical side-effects. _PLoS One_ 2011; **6** :e28025.
13. Wang K, Sun J, Zhou S, _et al._ Prediction of drug-target
interactions for drug repositioning only based on genomic
expression similarity. _PLoS Comput Biol_ 2013; **9** :e1003315.
14. Oh M, Ahn J, Yoon Y. A network-based classification model
for deriving novel drug-disease associations and assessing
their molecular actions. _PLoS One_ 2014; **9** :e111668.
15. Yang K, Zhao X, Waxman D, _et al._ Predicting drug-disease
associations with heterogeneous network embedding. _Chaos_
2019; **29** :123109.
16. Liang X, Zhang P, Yan L, _et al._ LRSSL: predict and interpret drug–disease associations based on data integration using sparse subspace learning. _Bioinformatics_ 2017; **33** :

1187–96.

17. Wu G,Liu J,Wang C.Semi-supervised graph cut algorithm for
drug repositioning by integrating drug, disease and genomic
associations. In: _2016 IEEE International Conference on Bioinfor-_
_matics and Biomedicine (BIBM)_ . Los Alamitos, CA: IEEE Computer SOC, 2016.
18. Zhang W, Yue X, Chen Y _et al._ Predicting drug-disease associations based on the known association bipartite network. In:
_IEEE International Conference on Bioinformatics and Biomedicine_ .
New York, NY: IEEE, 2017. pp. 503–9.
19. Lu L, Yu H. DR2DI: a powerful computational tool for predicting novel drug-disease associations. _J Comput Aided Mol Des_
2018; **32** :633–42.
20. Zhang W, Yue X, Lin W, _et al._ Predicting drug-disease associations by using similarity constrained matrix factorization.
_BMC Bioinformatics_ 2018; **19** :233.
21. Xuan P, Cao Y, Zhang T, _et al._ Drug repositioning through
integration of prior knowledge and projections of drugs and
diseases. _Bioinformatics_ 2019; **35** :4108–19.
22. Zhang P, Wang F, Hu J. Towards drug repositioning: a unified
computational framework for integrating multiple aspects
of drug similarity and disease similarity. _AMIA Annu Symp_
_Proc_ 2014; **2014** :1258–67.
23. Yang M, Luo H, Li Y, _et al._ Overlap matrix completion
for predicting drug-associated indications. _PLoS Comput Biol_
2019; **15** :e1007541.
24. Luo H, Li M, Wang S, _et al._ Computational drug repositioning using low-rank matrix approximation and randomized
algorithms. _Bioinformatics_ 2018; **34** :1904–12.
25. Yang M, Luo H, Li Y, _et al._ Drug repositioning based on
bounded nuclear norm regularization. _Bioinformatics_ 2019;
**35** :i455–63.


10 _Yu et al._


26. Zhang W, Xu H, Li X, _et al._ DRIMC: an improved drug repositioning approach using Bayesian inductive matrix completion. _Bioinformatics_ 2020; **36** :2839–47.
27. Angermueller C, Pärnamaa T, Parts L, _et al._ Deep learning for
computational biology. _Mol Syst Biol_ 2016; **12** :878–8.
28. Yue X, Gutierrez BJ, Sun H. Clinical reading comprehension:
a thorough analysis of the emrQA dataset. In: _ACL_ . Stroudsburg, PA: Assoc Computational Linguistics-ACL, 2020.
29. Zeng X, Zhu S, Liu X, _et al._ deepDR: a network-based deep
learning approach to in silico drug repositioning. _Bioinformat-_
_ics_ 2019; **35** :5191–8.
30. Li Z, Huang Q, Chen X, _et al._ Identification of drug-disease
associations using information of molecular structures and
clinical symptoms via deep convolutional neural network.
_Front Chem_ 2020; **7** :924.
31. Xuan P, Ye Y, Zhang T, _et al._ Convolutional neural network
and bidirectional long short-term memory-based method
for predicting drug-disease associations. _Cells_ 2019; **8** :705.
32. Kipf TN, Welling M. Semi-supervised classification with
graph convolutional networks. In: _International Conference on_
_Learning Representations (ICLR)_ [, 2017. https://iclr.cc/archive/](https://iclr.cc/archive/www/2017.html)
[www/2017.html.](https://iclr.cc/archive/www/2017.html)

33. Y-A H, Hu P, KCC C, _et al._ Graph convolution for predicting
associations between miRNA and drug resistance. _Bioinfor-_
_matics_ 2019; **36** :851–8.
34. Zitnik M, Agrawal M, Leskovec J. Modeling polypharmacy
side effects with graph convolutional networks. _Bioinformat-_
_ics_ 2018; **34** :i457–66.
35. Li J, Zhang S, Liu T, _et al._ Neural inductive matrix completion with graph convolutional networks for miRNA-disease
association prediction. _Bioinformatics_ 2020; **36** (8):2538–46.
36. Yang J-H, Chen C-M, Wang C-J _et al._ HOP-rec: high-order
proximity for implicit recommendation. In: _Proceedings of the_
_12th ACM Conference on Recommender Systems_ . New York, NY:
Assoc Computing Machinery, 2018, pp. 140–4.
37. Vaswani A, Shazeer N, Parmar N _et al._ Attention is all you
need. In: _Advances in Neural Information Processing Systems_ .
La Jolla, California: Neural Information Processing Systems
(Nips), 2017, pp. 5998–6008.
38. Zhang W, Huang F, Yue X _et al._ Prediction of drug-disease
associations and their effects by signed network-based nonnegative matrix factorization. In: _2018 IEEE International Con-_
_ference on Bioinformatics and Biomedicine (BIBM)_ . New York, NY:
IEEE, 2018, pp. 798–802.
39. Davis AP, Grondin CJ, Johnson RJ, _et al._ The comparative
toxicogenomics database: update 2017. _Nucleic Acids Res_
2017; **45** :D972–8.
40. Law V, Knox C, Djoumbou Y, _et al._ DrugBank 4.0: shedding new light on drug metabolism. _Nucleic Acids Res_ 2014;

**42** :D1091–7.

41. Deng Y, Xu X, Qiu Y, _et al._ A multimodal deep learning framework for predicting drug-drug interaction events. _Bioinfor-_
_matics_ [2020. https://doi.org/10.1093/bioinformatics/btaa501.](https://doi.org/10.1093/bioinformatics/btaa501)
42. Wang D, Wang J, Lu M, _et_ _al._ Inferring the human
microRNA functional similarity and functional network
based on microRNA-associated diseases. _Bioinformatics_ 2010;

**26** :1644–50.

43. Clevert D-A, Unterthiner T, Hochreiter S. Fast and accurate
deep network learning by exponential linear units (ELUs).
In: _International Conference on Learning Representations (ICLR)_,

2016.

44. Wang X, He X, Wang M _et al._ Neural graph collaborative
filtering. In: _Proceedings of the 42nd International ACM SIGIR_



_Conference on Research and Development in Information Retrieval_ .
New York, NY: Assoc Computing Machinery, 2019, pp. 165–74.
45. He X, Deng K, Wang X _et al._ LightGCN: simplifying and
powering graph convolution network for recommendation.
In: _SIGIR_ [, 2020. https://sigir.org/sigir2020/accepted-papers/.](https://sigir.org/sigir2020/accepted-papers/)
46. Glorot X, Bengio Y. Understanding the difficulty of training deep feedforward neural networks. In: _Proceedings of the_
_Thirteenth International Conference on Artificial Intelligence and_
_Statistics_ [, 2010, pp. 249–56. http://proceedings.mlr.press/v9/](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
[glorot10a/glorot10a.pdf.](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
47. Kingma DP, Ba J. Adam: a method for stochastic optimization. In: _International Conference on Learning Representations_
_(ICLR)_ [, 2015. https://iclr.cc/archive/www/2015.html.](https://iclr.cc/archive/www/2015.html)
48. van den Berg R, Kipf TN, Welling M. Graph convolutional
matrix completion. In: _KDD_ [, 2018. https://www.kdd.org/](https://www.kdd.org/kdd2018/deep-learning-day)
[kdd2018/deep-learning-day.](https://www.kdd.org/kdd2018/deep-learning-day)
49. Srivastava N, Hinton G, Krizhevsky A, _et al._ Dropout: a simple
way to prevent neural networks from overfitting. _J Mach_
_Learn Res_ 2014; **15** :1929–58.
50. Zhu L, Hong Z, Zheng H. Predicting gene-disease associations via graph embedding and graph convolutional networks. In: _2019 IEEE International Conference on Bioinformatics_
_and Biomedicine (BIBM)_, 2019, pp. 382–9.
51. Smith LN. Cyclical learning rates for training neural networks. In: _2017 IEEE Winter Conference on Applications of_
_Computer Vision (WACV)_ . New York, NY: IEEE, 2017, pp.

464–72.

52. Li Q, Han Z, Wu X-M. Deeper insights into graph convolutional networks for semi-supervised learning. In: _Thirty-_
_Second AAAI Conference on Artificial Intelligence_ . Palo Alto, CA,

2018.

53. Bourque F, Karama S, Looper K, _et al._ Acute tamoxifeninduced depression and its prevention with venlafaxine.
_Psychosomatics_ 2009; **50** :162–5.
54. Jakobsen Falk I, Lund J, Gréen H, _et al._ Pharmacogenetic study
of the impact of ABCB1 single-nucleotide polymorphisms on
lenalidomide treatment outcomes in patients with multiple
myeloma: results from a phase IV observational study and
subsequent phase II clinical trial. _Cancer Chemother Pharmacol_
2018; **81** :183–93.
55. De Sarro G, Paola EDD, Gratteri S, _et al._ Fosinopril and zofenopril, two angiotensin-converting enzyme (ACE) inhibitors,
potentiate the anticonvulsant activity of antiepileptic drugs
against audiogenic seizures in DBA/2 mice. _Pharmacol Res_
2012; **65** :285–96.
56. Tsai C-F, Cheng Y-K, Lu D-Y, _et al._ Inhibition of estrogen
receptor reduces connexin 43 expression in breast cancers.
_Toxicol Appl Pharmacol_ 2018; **338** :182–90.
57. Srikiatkhachorn A, Suwattanasophon C, Ruangpattanatawee U, _et al._ 2002 Wolff Award. 5-HT2A receptor
activation and nitric oxide synthesis: a possible mechanism
determining migraine attacks. _Headache_ 2002; **42** :566–74.
58. Schlaifer D, Arlet P, De la Roque PM, _et al._ Antiepileptic
drug-induced lymphoproliferative disorder associated with
acquired C1 esterase inhibitor deficiency and angioedema.
_Eur J Haematol_ 1992; **48** :274–5.
59. Szymanski SJD, Leipzig R, Masiar S, _et al._ Anticholinergic
delirium caused by retreatment with clozapine. _Am J Psychi-_
_atry_ 1991; **148** :1752.
60. Kumada T, Hattori H, Doi H, _et al._ Postoperative complete atrioventricular block induced by carbamazepine in
a patient with congenital heart disease. _No To Hattatsu_
2005; **37** :257–61.


61. Horvath J, Coeytaux A, Jallon P, _et al._ Carbamazepine
encephalopathy masquerading as Creutzfeldt–Jakob disease. _Neurology_ 2005; **65** :650.
62. Wittchen F, Spencker S, Zinke S, _et al._ Leistungsknick,
Thoraxschmerz und Polyserositis bei einem 35-jährigen
Patienten mit antikonvulsiver Therapie. _Internist_ 2006; **47** :

69–75.

63. Brady KT, Sonne SC, Malcolm RJ, _et al._ Carbamazepine
in the treatment of cocaine dependence: subtyping
by affective disorder. _Exp Clin Psychopharmacol_ 2002; **10** :

276.

64. Reynolds ER, Stauffer EA, Feeney L, _et al._ Treatment with the
antiepileptic drugs phenytoin and gabapentin ameliorates
seizure and paralysis of Drosophila bang-sensitive mutants.
_J Neurobiol_ 2004; **58** :503–13.
65. Wang X, Ji C, Zhang H, _et al._ Identification of a smallmolecule compound that inhibits homodimerization of
oncogenic NAC1 protein and sensitizes cancer cells to anticancer agents. _J Biol Chem_ 2019; **294** :10006–17.



_layer attention GCN_ 11


66. Cassidy J, Merrick MV, Smyth JF, _et al._ Cardiotoxicity of
mitozantrone assessed by stress and resting nuclear ventriculography. _Eur J Cancer Clin Oncol_ 1988; **24** :935–8.
67. Hofstra L, Van Der Graaf W, De Vries E, _et al._ Ataxia following
docetaxel infusion. _Ann Oncol_ 1997; **8** :812–3.
68. Alsafadi S, Even C, Falet C, _et al._ Retinoic acid receptor
alpha amplifications and retinoic acid sensitivity in breast
cancers. _Clin Breast Cancer_ 2013; **13** :401–8.
69. Ackerstaff E, Gimi B, Artemov D, _et al._ Anti-inflammatory
agent indomethacin reduces invasion and alters
metabolism in a human breast cancer cell line. _Neoplasia_
2007; **9** :222–35.
70. Feldman LD, Buzdar AU, Blumenschein GR. High-dose 1beta-D-arabinofuranosylcytosine in advanced breast cancer.
_Oncology_ 1985; **42** :273–4.
71. Di Costanzo F, Manzione L, Gasperoni S, _et al._ Paclitaxel and
mitoxantrone in metastatic breast cancer: a phase II trial of
the Italian Oncology Group for Cancer Research. _Cancer Invest_
2004; **22** :331–7.


