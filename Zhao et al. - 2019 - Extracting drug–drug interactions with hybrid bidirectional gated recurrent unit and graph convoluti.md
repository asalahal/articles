[Journal of Biomedical Informatics 99 (2019) 103295](https://doi.org/10.1016/j.jbi.2019.103295)


[Contents lists available at ScienceDirect](http://www.sciencedirect.com/science/journal/15320464)

## Journal of Biomedical Informatics


[journal homepage: www.elsevier.com/locate/yjbin](https://www.elsevier.com/locate/yjbin)

### Extracting drug–drug interactions with hybrid bidirectional gated recurrent unit and graph convolutional network


Di Zhao, Jian Wang [⁎], Hongfei Lin, Zhihao Yang, Yijia Zhang


_School of Computer Science and Technology, Dalian University of Technology, 116024 Dalian, China_





A R T I C L E I N F O


_Keywords:_
Drug–drug interactions
Graph convolutional network
Bidirectional gated recurrent unit


**1. Introduction**



A B S T R A C T


Drug–drug interactions are critical in studying drug side effects. Thus, quickly and accurately identifying the
relationship between drugs is necessary. Current methods for biomedical relation extraction include only the
sequential information of sentences, while syntactic graph representations have not been explored in DDI extraction. We herein present a novel hybrid model to extract a biomedical relation that combines a bidirectional
gated recurrent unit (Bi-GRU) and a graph convolutional network (GCN). Bi-GRU and GCN are used to automatically learn the features of sequential representation and syntactic graph representation, respectively. The
experimental results show that the advantages of Bi-GRU and GCN in DDI relation extraction are complementary,
and that the utilization of Bi-GRU and GCN further improves the model performance. We evaluated our model on
the DDI extraction-2013 shared task and discovered that our method achieved reasonable performance.



Drug–drug interactions (DDIs) may occur when two or more drugs
are co-administered, thus altering how one or more drugs function in
the human body, which may cause severe adverse drug reactions [15].
A negative consequence may worsen a patient’s condition or lead to
increasing length of hospital stay and healthcare costs [26]. It is estimated that adverse drug reaction (ADR) causes 197,000 deaths in the
EU each year, and the total social cost of ADR is estimated to be 79
billion euros per year [7]. In the United States, approximately 2.2
million people between the ages of 57 and 85 face potential drug mix
risks [19]. Therefore, healthcare professionals are aware that the more
DDI-related information we have, the less medical accidents would
occur. Currently, some drug-related databases, such as DrugBank,
Therapeutic Target Database, and PharmGKB have been provided to
summarize drug and DDI information for researchers and professionals

[20,36,23]. However, the latest reported progress on the interaction of
several drugs has not been updated into the database in time, and it is
still buried in a large number of unstructured biomedical texts. In addition, the volume of biomedical literature has grown considerably over
the past decades. The traditional method requires human experts to
read a large amount of biomedical literature to discover DDIs; furthermore, manually discovering DDIs is time consuming and costly.
Therefore, the automatic extraction of DDI information from biomedical literature has become an important research area.


⁎ Corresponding author.
_E-mail address:_ [wangjian@dlut.edu.cn (J. Wang).](mailto:wangjian@dlut.edu.cn)



Recently, SemEval-2013 has provided an open corpus to evaluate
the performance of various DDI extraction methods, prompting many
researchers to continually improve the extraction of DDI methods [32].
The original methods used manual predefined rules that match DDI
documents to the obtained DDI categories [31,10]. The limitations of
these methods are that they depend on the ability of professionals and
domain experts to develop a large number of rules. Compared with the
rule-based approach, a machine-learning-based approach is more advantageous in terms of performance and generalization capability. A
machine-learning approach has abundant features including word, dependency graph, and parse tree features but requires the manual selection of features [17]. In addition, these methods have other limitations. For instance, linear-based methods primarily construct rules to
extract biomedical text features [28,5]. However, owing to the diversity
of languages, it is often complex to establish standard rules and valid
information cannot be captured. The kernel-based method is another
example; it requires designing appropriate kernel functions, which is
still highly technology dependent [6,35,8].
Currently, deep learning has been widely used in natural language
processing (NLP) and has shown good performance on different NLP
tasks. To extract DDIs, researchers have begun using deep learning. Deep
learning-based methods can be divided into two main categories: the first
is based on a convolutional neural network (CNN) and the other is based
on a recurrent neural network (RNN) [4]. The CNN model has a hierarchical neural network structure, uses a convolution kernel to learn



[https://doi.org/10.1016/j.jbi.2019.103295](https://doi.org/10.1016/j.jbi.2019.103295)
Received 16 May 2019; Received in revised form 14 September 2019; Accepted 23 September 2019

Available online 27 September 2019
1532-0464/ © 2019 Elsevier Inc. All rights reserved.


_D. Zhao, et al._ _Journal of Biomedical Informatics 99 (2019) 103295_


**Table 1**
Example of preprocessing on the sentence, “Digoxin and verapamil use may be rarely associated with ventricular fibrillation when combined with Adenocard” for
each target pair.


Candidate entity1 Candidate entity2 Processed Sentence


Digoxin verapamil **drug1** and **drug2** use may be rarely associated with ventricular fibrillation when combined with drug3
verapamil Adenocard drug3 and **drug1** use may be rarely associated with ventricular fibrillation when combined with **drug2**



local features, and selects the most important information through a
pooling layer, rendering CNN more suitable for processing short sentences. By contrast, the RNN model has a sequential neural network
structure, which preserves long-range relations among words in a long
sequence. The syntactic convolutional neural network (SCNN) is a typical
representative work based on the CNN [42]. It utilizes word embedding,
syntactic embedding, and other semantic information to represent sentences. First, the SCNN uses two-layer convolution for local feature calculation. Subsequently, it employs max pooling to obtain important information representation and finally classifies DDI features by the
softmax function. Joint two long short-term memory networks (Joint ABLSTM) is an RNN-based method [29]; it uses word and position embedding as latent features. Joint AB-LSTM employs max pooling in the
first bidirectional long short-term memory (Bi-LSTM) layer output for
important information and applies an attention pool to the second BiLSTM layer output to assign weights to features. The combination of two
Bi-LSTMs is used for DDI extraction. However, these approaches do not
consider syntactic graph representations, and DDI extraction can be enhanced by the syntactic graph information. Methods exist for using
syntactic graph information. For example, Zheng et al. used the dependency graph to construct context vectors in identifying DDI categories [43]. Shen et al. used dependency trees to represent long-distance
dependencies between entities while using other attribute features of
drugs to perform DDI relation extraction [33]. However, the dependency
features of the method require manual designs.
Recently, a new approach called the graph convolutional network
(GCN) has attracted wide attention. The GCN retains the global structural information of the graph as well as the attribute information of the
node [3,24]. The GCN has been successfully applied to several NLP
tasks, such as text classification and event detection [38,27].
We herein present a novel hybrid model based on the Bi-GRU and
GCN to extract biomedical relations from biomedical texts. We not only
utilize sentence sequences but also integrate syntax dependency graphs
to improve the performance of biomedical relation extraction. Syntactic
GCNs have been used to automatically capture the long-term dependencies between words [37]. Combining syntactic information from
dependency graphs may help in classifying biomedical relation, especially for long and complex sentences. In the GCN, syntactic knowledge
can be modeled as edges in a graph, where words are used as nodes.
To the best of our knowledge, this is the first study that uses a GCN
encoding syntactic graph for biomedical relation extraction.
The contribution of this paper is threefold:

# • [We propose a novel neural method to extract DDIs from biomedical]

texts with syntactic graph representations.
# • [We demonstrate that Bi-GRU and GCN provide complementary ad-]

vantages in modeling power.
# • [We achieve reasonable performance on widely used datasets for]

biomedical relation extraction using the proposed model with the
semi-supervised or supervised approach.


**2. Related materials**


_2.1. Preprocessing_


Because the distributions of positive and negative instances are
extremely unbalanced, the model performance will be affected.



Previous studies have confirmed that filtering out negative examples
can improve the classification ability of the model. Therefore, we refer
the previous studies to filter negative data [42]. The specific rules for
filtering negative data are as follows:


_2.1.1. Rule1_

Two target drugs in the instance are the same drug. They do not
interact with themselves. Consequently, such instances must be filtered.
Two cases exist: the first is that two drugs have the same name and the
other is that the abbreviation of one drug name is the same as that of
the other drug name.


_2.1.2. Rule2_

If two drugs appear in the same coordinate structure, the instance
must be removed as it tends to be a false positive.
Subsequently, to ensure the generalization ability of the model, the
data must be tokenized [34]. We replace the drug names of the target
drug pairs in the sentence with drug1 and drug2, and those of the other
drugs in the example are replaced by drug3, drug4, etc. Table 1 shows
an example of preprocessing the sentence, “Digoxin and verapamil use
may be rarely associated with ventricular fibrillation when combined
with Adenocard.”


_2.2. Word and position embedding_


Word embedding represents the mathematical embedding of a onedimensional space of each word into a continuous vector space with a
lower dimension. When it represents input layer information, word
embedding has been shown to improve the performance of NLP, such as
machine translation [1]. From Pubmed, we obtained more than 1
million biomedical abstracts that contained the keyword “drug” and
employed word2vec to train classic word embeddings [25]. In addition,
we employed position features in the embedded layer. The position
embedding captures the distance information of the drug entity relative
to each word [39].


_2.3. Syntactic graph_


The input of the GCN requires the syntactic information of the node
in the sentence. Fig. 1 shows an illustrative example of a syntactic
graph. We employed the Stanford parser to obtain the dependency
graph of each word in the candidate sentence [18]. For instance,
“conj_and” denotes the syntactic relation between “digoxin” and “verapamil.” To obtain syntactic information regarding the sentence based
on the dependency graph, the Stanford parser maintains the keyword
on the syntactic graph while filtering out the less important additional
words (such as “and” and “with”). In Fig. 1, the sentence consists of
multiple words, but we can accurately determine the relation based on
the syntactic dependency information. Therefore, DDI extraction will
benefit from the syntactic graph information of the sentence, especially
for lengthy and complex sentences.


_2.4. GCN background_


The GCN can perform meaningful representation learning on nodes
based on node features and graph structures. We provide a brief overview of the GCN for graphs of direction and edge types [37]. Given a



2


_D. Zhao, et al._ _Journal of Biomedical Informatics 99 (2019) 103295_


**Fig. 1.** Example of syntactic parsing result produced by the Stanford parser for the sentence, “Digoxin and verapamil use may be rarely associated with ventricular
fibrillation when combined with Adenocard.”


**Fig. 2.** Overview of our model. The model first encodes each word by concatenating word and position embeddings from Bi-GRU, followed by computing syntactic
dependency over sentences. Finally, the sentence of the hidden representation that concatenate the Bi-GRU and GCN is fed to a softmax classifier.



directed graph _=_ ( _V E_, ), _=_ {, _v_ 1 _v_ 2, _…_, _v_ _m_ } denote the vertices and
_=_ {, _e_ 1 _e_ 2, _…_, _e_ _n_ } denote the edges. A pair of nodes _u_ and _v_ with label _l_ _uv_
is represented as _u v l_ (,, _uv_ ) [. For instance, in][ Fig. 1][, nodes] _[ u]_ [ and] _[ v]_ [ re-]
present the word “digoxin” and “verapamil,” respectively. They have a
directed edge with label _l u v_ (, ) _=_ _l digoxin verapamil_ (, ) _=_ _conj and_ _ . In
addition, allowing information to flow in the reverse direction and selfloop, we add a reverse edge _l u v_ (, ) and self-loop edge _l v v_ (, ) . A node
through the GCN’s convolution layer obtains directly related neighbor
information. After a layer of convolution transformation, the representation of the node is as follows:



¯
_g_ _uv_ _=_ ( _h w_ _u_ - ¯ _l_ _uv_ _+_ _b_ _l_ _uv_ ) (3)


Both _w_ ¯ _l_ _uv_ _d_ and _b_ [¯] _l_ _uv_ are trained parameters. is the sigmoid
function. Combined with edge gating settings, after the _k_ layer information transfer, the final node is represented as follows:



_h_ _vk+_ 1 _=_ _f_ _g_ _k_ _×_ ( _W_ _lk_ _h_ _uk_ _+_ _b_ _lk_ )



1 _=_ _f_ _g_ _uvk_ _×_ ( _W_ _lk_ _uv_ _h_ _uk_ _+_ _b_ _lk_ _uv_
_u_ ( ) _v_



_uv_ _uv_ _uv_

( ) _v_



(4)



_h_ _v_ _=_ _f_ ( _W_ _l_ _uv_ _h_ _u_ _+_ _b_ _l_ _uv_ ),

_u_ ( ) _v_



(1)



where ( ) _v_ is the neighbors set of _u_ in the graph. Here, _h_ _v_ _d_ is the
hidden representation of node _v W_, _l_ _uv_ _d d×_ ; _b_ _l_ _uv_ [are the weight matrix]
and bias, respectively; _f_ is a nonlinear activation function. To capture
multihop neighborhoods, we integrate higher-order neighborhood information by stacking multiple GCN layers. The graph convolution
calculation for the _k_ -th layer is as follows:



_h_ _vk+_ 1 _=_ _f_ ( _W_ _lk_ _h_ _uk_ _+_ _b_ _lk_ )



1 _=_ _f_ ( _W_ _lk_ _uv_ _h_ _uk_ _+_ _b_ _lk_ _uv_
_u_ ( ) _v_



_uv_ _uv_
( ) _v_



(2)



**3. Methods**


_3.1. Bi-GRU_


Learning a continuous representation can be effective for managing
sequential data. An RNN is particularly suitable for encoding sequential
data. In this study, we employed a Bi-GRU to learn the features from a
sentence sequence whose outputs were later appended by the GCN for
DDI extraction [40]. Fig. 2 shows the architecture of our approach. The
calculation of the Bi-GRU is divided into two parts: forward and
reverse sequence information transmissions. For a given sentence

_X_ _=_ (, _x_ 1 _x_ 2, _…_, _x_ _n_ ), _x_ _k_, _x_ denotes the concatenating vector of the
current word and position, and the forward GRU is calculated as follows:


_i_ _=_ ( _W x_ _xi_ _t_ _+_ _W h_ _hi_ _t_ 1 _+_ _b_ _i_ ) (5)


_f_ _=_ ( _W x_ _xf_ _t_ _+_ _W h_ _hf_ _t_ 1 _+_ _b_ _f_ ) (6)


_g_ _=_ _tanh W x_ ( _xg_ _t_ _+_ _W_ _hg_ ( _i_ _h_ _t_ 1 ) _+_ _b_ _g_ ) (7)


_h_ _t_ _=_ (1 _f_ ) _h_ _t_ 1 _+_ _f_ _g_, (8)


where _W_ and _b_ are the weight matrix and bias vector, respectively; is
the sigmoid function; and denotes element-wise multiplication. _x_ _t_ is
the input word vector at time step _t_ and _h_ _t_ is the hidden state of the


current time step _t_ . _h_ _i_ and _h_ _i_ represent the output of the forward GRU



In automatically constructed syntactic graphs, not all types of edges are
equally important; noise exist in the generated syntactic structures.
Therefore, these edges must be discarded. Fig. 1 shows that the word
“associated” has six immediate neighbors in the example sentence.
Meanwhile, “combined” and “fibrillation” are crucial to determine the
DDI relation. “May” and “be” do not contribute much information in
this case. Therefore, it is thus not appropriate to weigh the neighbors
uniformly in the GCN for DDI. The GCN’s edge gating settings can effectively alleviate noise edge problems. The importance of the edge can
be computed as follows:



3


_D. Zhao, et al._ _Journal of Biomedical Informatics 99 (2019) 103295_



**Table 2**

Parameters design.


Parameter name Value


Word embedding dimension 200
Dropout for full connected layer 0.5
Recurrent dropout for GRU 0.5
GCN dropout probability 0.5

Batch size 6

RMSprop-learning rate 0.001
RMSprop- 0.95

RMSprop- 1e-8

Hidden state dimension of Bi-GRU 200

Hidden state dimension of GCN 200


and backward GRU, respectively. The Bi-GRU output is denoted as

_h_ _ibi_ _gru_ _=_ [ ; ] _h h_ _i_ _i_ [.]


_3.2. GCN_


Although the Bi-GRU can capture contextual information, it cannot
capture long-range dependencies that can be captured through graph
edges in a sentence. Syntactic features have been utilized in previous
studies to improve relation extraction [13]. Hence, we employ the GCN
to learn syntactic graph representation for DDI extraction. Using the
syntactic graph and _H_ _[bi]_ _gru_ _n b×_ _bi_ _gru_ as input features to the GCN,
each token is represented in a new _b_ _bi_ _gru_ [-dimensional space. The em-]
beddings are updated as follows:



_g_ _Lk_ _iu_ _×_ ( _W_ _Lk_ _iu_ _h_ _ubi_ _gru_ _+_ _b_ _Lk_

_u_ ( ) _i_



_h_ _igcn_ _k+_ 1 _=_ _f_ _g_ _L_ _iu_ _×_ ( _W_ _L_ _iu_ _h_ _u_ _gru_ _+_ _b_ _L_ _iu_ )



_k+_ 1 _L_ _iu_ _iu_ _u_ _iu_

_u_ ( ) _i_



(9)



**4. Results and discussions**


_4.1. Dataset description_


We performed an experiment on the DDI-2013 shared task 9 and
ADR [14,12]. The data are labeled with five types of DDI relations [2],
as follows:

# • [Mechanism: DDI is described as a pharmacokinetic mechanism in]

sentences, e.g., “Products containing **calcium** and other multivalent
cations will likely interfere with the absorption of **alendronate** .”
# • [Int: DDI appears in the text but no other information is provided,]

e.g., “Based on anecdotal reports, there may be an interaction between **buprenorphine** and **benzodiazepines** .”

# •

[Effect: DDI was described as an effect, e.g., “] **[Acetazolamide]** [ may]
**methenamine** .”
prevent the urinary antiseptic effect of
# • [Advice: DDI is described as a recommendation to use two drugs si-]

multaneously, e.g., “Microdosed minipill **progestin** preparations are
not recommended for use with **Soriatane** .”

# • [False: DDI describes two drugs that are not related, e.g., “Diltiazem]

plasma levels were not significantly affected by **lovastatin** or **pra-**
**vastatin** .”


On the source dataset, the training and the test instances are 27,722
and 5716, respectively. After negative filtering, the training and test
instances are 12,841 and 3020, respectively. Table 3 shows the detailed
changes in the number of categories after the data have been processed.
The ADR dataset was derived from Pubmed [11]. The dataset contains
23,516 instances, with 6821 containing ADR mentions and 16,695 not
containing any ADR mentions. For train, test, and validation splits, the
dataset was randomly divided to an 8:1:1 ratio.


_4.2. Parameters design_


In our experiment, we implemented our proposed methods using the
Python language with the TensorFlow library. The Bi-GRU model was
provided by the Tensorflow platform, and a GCN model was implemented
on the Tensorflow platform. Hyperparameters were set based on a small
development dataset. The word vector of the first layer of the model was
embedded, and the dimension of the word vector was 200. The length of
the sentence was set to 100, while shorter sentence required padding. The
hidden state dimension in both the Bi-GRU and GCN layers was 200. The
batch size was set as six for all experiments. To alleviate the overfitting
problem, we employed the dropout technique in our model using different layers. We selected the dropout rates of [0.3, 0.5, 0.7] for different
layers. By performing experiments on the development sets, we found
that a dropout rate of 0.5 produced the best results for DDI relation extraction. RMSprop was used as the optimizer, with a learning rate of
0.001. Table 2 shows the parameters used in the experiments.
We utilized official evaluation metrics to validate the model performance and obtained the F-score (F) by calculating the precision (P)
and recall (R). F was calculated as follows: _F_ _=_ 2 _PR P_ /( _+_ _R_ ) .


**Table 3**
Class distribution before and after negative instance filtering.


Category Training set Testing set


Before After Before After

processing processing processing processing


No. of negative 23,722 8987 4737 2049

No. of advise 826 814 221 221
No. of effect 1687 1592 360 357
No. of mechanism 1319 1260 302 301

No. of int 188 188 96 92

Ratio 1:5.91 1:2.33 1:4.83 1:2.11

No. of total 27,792 12,841 5716 3020



_h_ _igcn_ _k+_ 1 [denotes the hidden layer vector of node] _[ i]_ [ calculated by a] _[ k]_ [-layer]
GCN.specific details of the formula have been introduced in the previous _h_ _ubi_ _gru_ denotes the hidden feature of node _u_ from the Bi-GRU. The
section. _L_ _iu_ denotes the edge type; the syntactic dependency analyzer
has at least 50 types of syntactic relations. Parameterizing all relation
types may cause overparameterization problems. Hence, we utilized
only three types of edges. The edge type for connecting two nodes is
determined by the following method:



, ( _w w_ _i_, )



( _i_, )



_L w w_



_forward_, ( _w w_ _i_, _j_


_reverse_ ( _w w_



, ( _w w_, _i_ )



_i_, _j_


_j_, _i_



_i_, _j_



_=_



_selfloop_, _i_ _=_ _j_



(10)



_=_



,



_L w w_ ( _i_, _j_ ) [represents different edge types of nodes] _[ w]_ _i_ [connecting to] _[ w]_ _j_ [.]
After the GCN transformation, we obtained a new embedding matrix for
the document _H_ _[gcn]_ _n d×_ _gcn_ . We employed max pooling and mean
pooling to integrate the convolution vectors, separately. The entire
sentence information calculated by the pooling layer is denoted as
_h_ _Spooling_ .


_3.3. DDI type classification_


Finally, the sequence features _h_ _Sbi_ _gru_ and syntax dependency
features _h_ _Spooling_ of the entire sentence are concatenated as
_h_ _Scon_ _=_ [ _h_ _Sbi_ _gru_ ; _h_ _Spooling_ ] and fed into a fully connected network; additionally, the category prediction is performed with the softmax.


_DDI_ _type_ _=_ _Softmax W_ ( _T_ _×_ _h_ _Scon_ _+_ _b_ _T_ ) (11)


_T_ indicates the number of DDI types; _W_ _T_ and _b_ _T_ represent the training
parameters. The final result is the distribution of the DDI type.



4


_D. Zhao, et al._ _Journal of Biomedical Informatics 99 (2019) 103295_



**Table 4**

Performance comparison between proposed methods on the DDI test data. The
highest scores are highlighted in bold.


Methods Precision Recall F-score


Bi-RNN 46.0 50.5 48.1

Bi-LSTM 69.6 66.1 67.8

Bi-GRU 72.8 66.4 69.5

GCN1layer 70.8 65.0 67.8
GCN2layer 68.6 67.8 68.2
GCN3layer 67.9 65.5 66.7
Bi-GRU+GCN(mean) **75.2** 65.3 69.9

Bi-GRU+GCN(max) 73.6 **68.2** **70.8**


_4.3. Experimental results_


The performance of the methods are compared in Table 4. As
shown, the Bi-GRU performed better than the Bi-RNN and Bi-LSTM.
Therefore, in this study, we employed the Bi-GRU to learn the sequence
information of a sentence.

As shown in Table 4, our proposed model with two layers of the
GCN performed the best in the independent experiment. The result
shows that the more the GCN layer, the worse is the model performance. Traditional deep neural networks can stack many layers for
better performance; a multilayer structure contains more parameters
and can significantly improve the expression ability of the model.
However, the number of layers in a GCN is always small and most do
not exceed three layers. As shown in the experiment, stacking multiple
GCN layers will result in excessive smoothing, causing all vertices to
converge to the same value and the model incapable of differentiating
features. Table 4 shows that when the Bi-GRU and GCN are applied to
sentence sequences and syntactic information, respectively, the Bi-GRU
performs better than the GCN2layer. In terms of the overall performance, Bi-GRU achieved a precision, recall, and F-score of 72.8, 66.4,
and 69.5, respectively. The GCN2layer achieved a precision, recall, and
F-score of 68.6, 67.8, and 68.2, respectively. By combining the Bi-LSTM
and GCN, our hybrid model demonstrated an improvement in the
overall performance. The precision, recall, and F-score were 73.6, 68.2,
and 70.8, respectively. The combination of sequence structure information and syntactic graph can fully express the overall information
of the sentence, and the loss of the overall information will be caused by
using a single type of information. Thus, the Bi-GRU is complementary
to The GCN for DDI extraction, and the utilization of both would further
improve the performance. We verified the impact of two different
pooling methods experimentally. We found that the max pooling integrated relatively more valuable syntactic graph information.
As shown in Table 5, the comparison methods are divided into three
categories, i.e., based on linear methods, kernel methods, and deep
learning methods. Based on the linear and kernel methods, the support
vector machine is used as the main model framework. Support vector
machines exhibit weaker model generalization capabilities owing to the


**Table 5**

Performance comparison between proposed methods on the test data for DDI
classification. The highest scores are highlighted in bold. “–” denotes that the
value is not provided herein.


Class Methods Precision Recall F-score


Linear Method UTurku [5] 73.2 49.9 59.4

UWM-TRIADS [28] 43.9 50.5 47.0

Kim [17] – – 67.0

Kernel Method NIL UCM [6] 53.5 50.1 51.7

WBI-DDI [35] 64.2 57.9 60.9

FBK-irst [8] 64.6 65.6 65.1

Neural Network Method SCNN [42] 72.5 65.1 68.6

Joint AB-LSTM [29] **74.5** 65.0 69.4
Our study 73.6 **68.2** **70.8**



manual assignment of features. Compared with the method based on
handcrafted feature, the deep-learning method can not only automatically learn the feature representation of the sentence but also
achieves the best performance, as shown experimentally. This shows
that the deep-learning-based DDI extraction method is effective.
The other methods are primarily based on a neural network. Zhao
et al. proposed a DDI extraction method based on the SCNN [42]. The
SCNN combines a novel word and syntax embedding to represent the
overall information of a sentence. Compared to the SCNN, syntactic
graph features are employed in our method as well to yield a performance that is far superior to that of the SCNN (70.8 vs. 68.6). This
confirms the benefits of using GCN encoding syntactic features in DDI
classification. Joint AB-LSTM is a DDI extraction method based on BiLSTM [29]. It combines two independent Bi-LSTMs to generate a sentence embedding. It differs from our models as we utilized a GCN to
encode biomedical sentences. Compared with the Joint AB-LSTM, our
method has a simpler architecture and enables a higher F-score (70.8
vs. 69.4).
To study the generalizability of our proposed model, we performed
experiments on ADRs using the best model structure and the same
hyperparameters. Table 6 shows the results. Kang et al. developed a
knowledge-based relation extraction system that includes a recognition
entity module and a knowledge base module to determine whether a
relation exists between entities [16]. As shown in Table 6, their method
yielded a low F-score. This may be attributed to the manual design rules
and knowledge base limitations. Li et al. proposed a feedforward neural
network that uses knowledge to jointly extract drug-disease entities and
relations [22,21]. Abeed et al. used multicorpus training for ADR detection. Compared with the two knowledge-based models above, this
method incorporates more knowledge bases and manual features [30].
Without using any knowledge base and manual feature, our model can
achieve an F-score of 82.3, which is competitive. This phenomenon
proves that the proposed method is generalizable in biomedical relation

extraction.


_4.4. Semi-supervised learning_


Recently, Zhang et al. employed the semi-supervised variational
autoencoders (Semi-VAE) method for biomedical relation extraction.
We evaluated the effectiveness of our proposed method in semi-supervised learning, and compared it against existing semi-supervised
DDI classification models [41]. Semi-VAEs use the LSTM and CNN as
encoders and decoders, respectively, and a CNN as a classifier; they
constantly calibrate classifiers and encoders/decoders using variational
autoencoders. Pre-train is a semi-supervised learning method based on
the LSTM structure in which a model if first trained through unlabeled
data and then fine tuned by data labelling [9]. The final results are
summarized in Fig. 3. Overall, we observed that our method outperformed all other methods with a significant margin on the DDI datasets. In addition, our method demonstrated good performance even
with little training data, as shown in Table 5, and the results even exceeded those of most supervised learning methods. This shows that our
method can be successfully trained with a small training set. This may
be attributed to two reasons. First, we preprocess the data and eliminate
some noise data such that the model will not be affected by the noise


**Table 6**

Performance comparison with other works for the ADR task. “–” denotes that
the value is not provided herein.


Methods Precision Recall F-score


Kang [16] 42.1 76.3 54.3
Li [22] 64.0 62.9 63.4

Li [21] 67.5 75.8 71.4

Abeed [30] – – 81.2
Our study **82.3** **82.4** **82.3**



5


_D. Zhao, et al._ _Journal of Biomedical Informatics 99 (2019) 103295_


Extraction-2013 shared task was evaluated using our proposed model.
The experimental results indicated that our hybrid model effectively
combined the advantages of the Bi-GRU and GCN to further improve
model performance. In future research, we plan to collaborate with
healthcare professionals to apply our methods to clinical decision
making.


**Declaration of Competing Interest**


The authors declare that they have no known competing financial
interests or personal relationships that could have appeared to influence the work reported in this paper.


**Acknowledgment**



**Fig. 3.** Performance comparison of different methods on DDI datasets as the
labeled training data are varied from 250 to 4000. We randomly selected the
labeled training data and training over 10 runs to reduce selection deviation.


**Fig. 4.** Performance evaluation of our methods on DDI with different-length
sentences.


data even with little train data. Next, our proposed method learns sequence and graph information. The combination of the two can fully
represent the DDI category. In addition, in the biomedical field, developing the model and labeling the training data can enhance the
experimental results. In fact, data preprocessing using NLP can be used.


_4.5. Sentence length analysis_


We used the baseline Bi-GRU model with sentences of different
lengths to evaluate the performance on the DDI corpora. We partitioned
the sentence length into seven classes (< 10, [10, 20), [20, 30), [30,
40), [40, 50), [50, 60), > 60). As shown in Fig. 4, the F-score of the two
methods is similar when the sentence length ranges from 0 to 30, and
our method maintains a slight advantage. As for the sentence length
exceeding 30, our method maintains a certain advantage. However, for
the length at [40,50), the F-score based on our method is low. By
analyzing the test data, we found that for sentence lengths between 40
and 50, many identical sentences exist but the categories are different.
In this case, the syntactic graph information is the same, resulting in the
GCN layer being redundant. As shown in Fig. 4, our method has a
significant margin in long sentences. This phenomenon proves that our
method is more adept at processing long and complex biomedical

sentences.


**5. Conclusion and future work**


We herein proposed a novel hybrid model based on the Bi-GRU and
GCN to extract biomedical relations. We employed the Bi-GRU and GCN
to automatically learn the features of sentence sequences and syntactic
graph information, respectively. The performance of the DDI



This work is supported by the National Natural Science Foundation
of China (Nos. 61572098, 61572102).


**Appendix A. Supplementary material**


Supplementary data associated with this article can be found, in the
[online version, at https://doi.org/10.1016/j.jbi.2019.103295.](https://doi.org/10.1016/j.jbi.2019.103295)


**References**


[[1] M. Artetxe, G. Labaka, E. Agirre, Generalizing and improving bilingual word em-](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0005)
[bedding mappings with a multi-step framework of linear transformations, Thirty-](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0005)
[Second AAAI Conference on Artificial Intelligence, 2018.](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0005)

[[2] M. Asada, M. Miwa, Y. Sasaki, Extracting drug-drug interactions with attention](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0010)
[cnns, BioNLP 2017 (2017) 9–18.](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0010)

[3] P.W. Battaglia, J.B. Hamrick, V. Bapst, A. Sanchez-Gonzalez, V. Zambaldi, M.
Malinowski, A. Tacchetti, D. Raposo, A. Santoro, R. Faulkner, et al., Relational
inductive biases, deep learning, and graph networks, 2018. arXiv preprint arXiv:
[1806.01261.](arxiv:1806.01261)

[[4] Y. Bengio, P. Simard, P. Frasconi, Learning long-term dependencies with gradient](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0020)
[descent is difficult, IEEE Trans. Neural Netw. 5 (1994) 157–166.](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0020)

[5] J. Björne, S. Kaewphan, T. Salakoski, Uturku: drug named entity recognition and
drug-drug interaction extraction using svm classification and domain knowledge,
in: Second Joint Conference on Lexical and Computational Semantics (∗ SEM):
Proceedings of the Seventh International Workshop on Semantic Evaluation
(SemEval 2013), vol. 2, 2013, pp. 651–659.

[6] B. Bokharaeian, A. Díaz, Nil_ucm: Extracting drug-drug interactions from text
through combination of sequence and tree kernels, in: Second Joint Conference on
Lexical and Computational Semantics (∗ SEM): Proceedings of the Seventh
International Workshop on Semantic Evaluation (SemEval 2013), vol. 2, 2013, pp.
644–650.

[7] R. Businaro, Why we need an efficient and careful pharmacovigilance? J.
[Pharmacovigilance 01 (2013), https://doi.org/10.4172/2329-6887.1000e110.](https://doi.org/10.4172/2329-6887.1000e110)

[8] M.F.M. Chowdhury, A. Lavelli, Fbk-irst: a multi-phase kernel based approach for
drug-drug interaction detection and classification that exploits linguistic information, in: Second Joint Conference on Lexical and Computational Semantics (∗ SEM):
Proceedings of the Seventh International Workshop on Semantic Evaluation
(SemEval 2013), vol. 2, 2013, pp. 351–355.

[[9] A.M. Dai, Q.V. Le, Semi-supervised sequence learning, Adv. Neural Inform. Process.](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0045)
[Syst. (2015) 3079–3087.](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0045)

[[10] K. Fundel, Kuffner, R. Rzimmer, Relex - relation extraction using dependency parse](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0050)
[trees, Bioinformatics 23 (2007) 365–371.](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0050)

[[11] H. Gurulingappa, A. Mateen-Rajpu, L. Toldo, Extraction of potential adverse drug](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0055)
[events from medical case reports, J. Biomed. Semantics 3 (2012) 15.](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0055)

[[12] H. Gurulingappa, A.M. Rajput, A. Roberts, J. Fluck, M. Hofmann-Apitius, L. Toldo,](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0060)
[Development of a benchmark corpus to support the automatic extraction of drug-](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0060)
[related adverse effects from medical case reports, J. Biomed. Inform. 45 (2012)](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0060)
[885–892.](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0060)

[13] Z. He, W. Chen, Z. Li, M. Zhang, W. Zhang, M. Zhang, See: Syntax-aware entity
[embedding for neural relation extraction, 2018. arXiv preprint arXiv: 1801.03603.](arxiv:1801.03603)

[[14] M. Herrero-Zazo, I. Segura-Bedmar, P. Martínez, T. Declerck, The ddi corpus: an](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0070)
[annotated corpus with pharmacological substances and drug-drug interactions, J.](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0070)
[Biomed. Inform. 46 (2013) 914–920.](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0070)

[[15] L.E. Hines, J.E. Murphy, Potentially harmful drug-drug interactions in the elderly: a](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0075)
[review, Am. J. Geriatr. Pharmacother. 9 (2011) 364–377.](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0075)

[[16] N. Kang, B. Singh, C. Bui, Z. Afzal, E.M. van Mulligen, J.A. Kors, Knowledge-based](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0080)
[extraction of adverse drug events from biomedical text, BMC Bioinformatics 15](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0080)
[(2014) 64.](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0080)

[[17] S. Kim, H. Liu, L. Yeganova, W.J. Wilbur, Extracting drug-drug interactions from](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0085)
[literature using a rich feature-based linear kernel approach, J. Biomed. Inform. 55](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0085)
[(2015) 23–30.](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0085)

[[18] D. Klein, C.D. Manning, Accurate unlexicalized parsing, Proceedings of the 41st](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0090)
[Annual Meeting on Association for Computational Linguistics-Volume 1,](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0090)



6


_D. Zhao, et al._ _Journal of Biomedical Informatics 99 (2019) 103295_



[Association for Computational Linguistics, 2003, pp. 423–430.](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0090)

[[19] I. Korkontzelos, A. Nikfarjam, M. Shardlow, A. Sarker, S. Ananiadou, G.H. Gonzalez,](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0095)
[Analysis of the effect of sentiment analysis on extracting adverse drug reactions](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0095)
[from tweets and forum posts, J. Biomed. Inform. 62 (2016) 148–158.](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0095)

[[20] V. Law, C. Knox, Y. Djoumbou, T. Jewison, A.C. Guo, Y. Liu, A. Maciejewski,](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0100)
[D. Arndt, M. Wilson, V. Neveu, et al., Drugbank 4.0: shedding new light on drug](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0100)
[metabolism, Nucl. Acids Res. 42 (2013) D1091–D1097.](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0100)

[[21] F. Li, M. Zhang, G. Fu, D. Ji, A neural joint model for entity and relation extraction](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0105)
[from biomedical text, BMC Bioinformatics 18 (2017) 198.](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0105)

[[22] F. Li, Y. Zhang, M. Zhang, D. Ji, Joint models for extracting adverse drug events](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0110)
[from biomedical text, IJCAI, 2016, pp. 2838–2844.](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0110)

[[23] Y.H. Li, C.Y. Yu, X.X. Li, P. Zhang, J. Tang, Q. Yang, T. Fu, X. Zhang, X. Cui, G. Tu,](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0115)
[et al., Therapeutic target database update 2018: enriched resource for facilitating](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0115)
[bench-to-clinic research of targeted therapeutics, Nucl. Acids Res. 46 (2017)](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0115)
[D1121–D1127.](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0115)

[24] X. Liu, Z. Luo, H. Huang, Jointly multiple events extraction via attention-based
graph information aggregation, in: Proceedings of the 2018 Conference on
Empirical Methods in Natural Language Processing, Brussels, Belgium, October
[31–November 4, 2018, 2018, pp. 1247–1256. https://aclanthology.info/papers/](https://aclanthology.info/papers/D18-1156/d18-1156)
[D18-1156/d18-1156.](https://aclanthology.info/papers/D18-1156/d18-1156)

[25] T. Mikolov, K. Chen, G. Corrado, J. Dean, Efficient estimation of word re[presentations in vector space, 2013. arXiv preprint arXiv: 1301.3781.](arxiv:1301.3781)

[[26] C.S. Moura, F.A. Acurcio, N.O. Belo, Drug-drug interactions associated with length](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0130)
[of stay and cost of hospitalization, J. Pharm. Pharmaceutical Sci. 12 (2009)](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0130)
[266–272.](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0130)

[27] T.H. Nguyen, R. Grishman, Graph convolutional networks with argument-aware
pooling for event detection, in: Thirty-Second AAAI Conference on Artificial
Intelligence, 2018.

[28] M. Rastegar-Mojarad, R.D. Boyce, R. Prasad, Uwm-triads: classifying drug-drug
interactions with two-stage svm and post-processing, in: Second Joint Conference
on Lexical and Computational Semantics (∗ SEM): Proceedings of the Seventh
International Workshop on Semantic Evaluation (SemEval 2013), vol. 2, 2013, pp.
667–674.

[[29] S.K. Sahu, A. Anand, Drug-drug interaction extraction from biomedical texts using](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0145)
[long short-term memory network, J. Biomed. Inform. 86 (2018) 15–24.](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0145)

[[30] A. Sarker, G. Gonzalez, Portable automatic text classification for adverse drug re-](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0150)
[action detection via multi-corpus training, J. Biomed. Inform. 53 (2015) 196–207.](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0150)




[[31] I. Segura-Bedmar, A linguistic rule-based approach to extract drug-drug interactions](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0155)
[from pharmacological documents, Bmc Bioinformatics 12 (2011) S1.](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0155)

[[32] I. Segura-Bedmar, P. Martínez, M. Herrero-Zazo, Lessons learnt from the ddiex-](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0160)
[traction-2013 shared task, J. Biomed. Inform. 51 (2014) 152–164.](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0160)

[[33] Y. Shen, K. Yuan, M. Yang, B. Tang, Y. Li, N. Du, K. Lei, Kmr: knowledge-oriented](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0165)
[medicine representation learning for drug–drug interaction and similarity compu-](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0165)
[tation, J. Cheminform. 11 (2019) 22.](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0165)

[[34] K. Sun, H. Liu, L. Yeganova, W.J. Wilbur, Extracting drug-drug interactions from](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0170)
[literature using a rich feature-based linear kernel approach, J. Biomed. Inform. 55](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0170)
[(2015) 23–30.](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0170)

[35] P. Thomas, M. Neves, T. Rocktäschel, U. Leser, Wbi-ddi: drug-drug interaction extraction using majority voting, in: Second Joint Conference on Lexical and
Computational Semantics (∗ SEM): Proceedings of the Seventh International
Workshop on Semantic Evaluation (SemEval 2013), vol. 2, 2013, pp. 628–635.

[[36] C.F. Thorn, T.E. Klein, R.B. Altman, Pharmgkb: the pharmacogenomics knowledge](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0180)
[base, Pharmacogenomics, Springer, 2013, pp. 311–320.](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0180)

[37] S. Vashishth, R. Joshi, S.S. Prayaga, C. Bhattacharyya, P. Talukdar, RESIDE:
Improving distantly-supervised neural relation extraction using side information,
in: Proceedings of the 2018 Conference on Empirical Methods in Natural Language
Processing, Association for Computational Linguistics, 2018, pp. 1257–1266. URL
[http://aclweb.org/anthology/D18-1157.](http://aclweb.org/anthology/D18-1157)

[38] L. Yao, C. Mao, Y. Luo, Graph convolutional networks for text classification, 2018.
[arXiv preprint arXiv: 1809.05679.](arxiv:1809.05679)

[[39] D. Zeng, K. Liu, S. Lai, G. Zhou, J. Zhao, Relation classification via convolutional](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0195)
[deep neural network, Proceedings of COLING 2014, the 25th International](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0195)
[Conference on Computational Linguistics: Technical Papers, 2014, pp. 2335–2344.](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0195)

[[40] S. Zhang, D. Zheng, X. Hu, M. Yang, Bidirectional long short-term memory networks](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0200)
[for relation classification, Proceedings of the 29th Pacific Asia Conference on](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0200)
[Language, Information and Computation, 2015, pp. 73–78.](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0200)

[[41] Y. Zhang, Z. Lu, Exploring semi-supervised variational autoencoders for biomedical](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0205)
[relation extraction, Methods (2019).](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0205)

[[42] Z. Zhao, Z. Yang, L. Luo, H. Lin, J. Wang, Drug drug interaction extraction from](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0210)
[biomedical literature using syntax convolutional neural network, Bioinformatics 32](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0210)
[(2016) 3444.](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0210)

[[43] W. Zheng, H. Lin, Z. Zhao, B. Xu, Y. Zhang, Z. Yang, J. Wang, A graph kernel based](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0215)
[on context vectors for extracting drug–drug interactions, J. Biomed. Inform. 61](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0215)
[(2016) 34–43.](http://refhub.elsevier.com/S1532-0464(19)30214-X/h0215)



7


