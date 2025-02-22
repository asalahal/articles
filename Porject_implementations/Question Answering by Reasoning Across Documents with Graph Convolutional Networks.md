                 Question Answering by Reasoning Across Documents
                               with Graph Convolutional Networks
           Nicola De Cao                             Wilker Aziz                               Ivan Titov
     University of Edinburgh                 University of Amsterdam                  University of Edinburgh
    University of Amsterdam                      w.aziz@uva.nl                       University of Amsterdam
nicola.decao@gmail.com                                                             ititov@inf.ed.ac.uk
                       Abstract
    Most research in reading comprehension has
    focused on answering questions based on in-
    dividualdocumentsorevensingleparagraphs.
    We introduce a neural model which integrates
    and  reasons  relying  on  information  spread
    within documents and across multiple docu-
    ments. Weframeitasaninferenceproblemon
    a graph. Mentions of entities are nodes of this             Figure 1: A sample from WIKIHOP where multi-step
    graph while edges encode relations between                  reasoning and information combination from different
    different  mentions  (e.g.,  within-  and  cross-           documents is necessary to infer the correct answer.
    document coreference).  Graph convolutional
    networks (GCNs) are applied to these graphs
    and trained to perform multi-step reasoning.                relying only on local information cannot achieve
    Our Entity-GCN method is scalable and com-                  competitive performance.
    pact, and it achieves state-of-the-art results on
    a multi-document question answering dataset,                   Even though these new datasets are challeng-
    WIKIHOP (Welbl et al.,2018).                                ingandrequirereasoningwithindocuments,many
1   Introduction                                                question  answering  and  search  applications  re-
                                                                quire aggregation of information across multiple
Thelong-standinggoalofnaturallanguageunder-                     documents.  The WIKIHOP dataset (Welbl et al.,
standing is the development of systems which can                2018)wasexplicitlycreatedtofacilitatethedevel-
acquireknowledgefromtextcollections. Freshin-                   opment of systems dealing with these scenarios.
terestinreadingcomprehensiontaskswassparked                     Each example in WIKIHOP consists of a collec-
by the availability of large-scale datasets, such as            tion of documents, a query and a set of candidate
SQuAD (Rajpurkar et al.,2016) and CNN/Daily                     answers (Figure1).  Though there is no guaran-
Mail (Hermann et al.,2015), enabling end-to-end                 tee that a question cannot be answered by relying
training of neural models (Seo et al.,2017;Xiong                justonasinglesentence,theauthorsensurethatit
et al.,2017;Shen et al.,2017).  These systems,                  is answerable using a chain of reasoning crossing
given a text and a question, need to answer the                 document boundaries.
query relying on the given document.  Recently,                    Though  an  important  practical  problem,  the
it has been observed that most questions in these               multi-hop  setting  has  so  far  received  little  at-
datasets do not require reasoning across the doc-               tention.   The methods reported byWelbl et al.
ument, but they can be answered relying on in-                  (2018) approach the task by merely concatenat-
formation contained in a single sentence (Weis-                 ingalldocumentsintoasinglelongtextandtrain-
senborn  et  al., 2017).    The  last  generation  of           ing a standard RNN-based reading comprehen-
large-scale reading comprehension datasets, such                sion model,  namely,  BiDAF (Seo et al.,2017)
as a NarrativeQA (Ko ˇcisk´y et al.,2018), Trivi-               and FastQA (Weissenborn et al.,2017).   Docu-
aQA (Joshi et al.,2017), and RACE (Lai et al.,                  ment concatenation in this setting is also used in
2017), have been created in such a way as to ad-                Weaver(Raisonetal.,2018)andMHPGM(Bauer
dress this shortcoming and to ensure that systems               et al.,2018).   The only published paper which

goes  beyond  concatenation  is  due  toDhingra                 withmanytechniquesthatuseexpensivequestion-
et al.(2018),  where they augment RNNs with                     aware recurrent document encoders.
jump-links corresponding to co-reference edges.                    Despitenotusingrecurrentdocumentencoders,
Though these edges provide a structural bias, the               the full Entity-GCN model achieves over 2% im-
RNN states are still tasked with passing the infor-             provement over the best previously-published re-
mationacrossthedocumentandperformingmulti-                      sults. As our model is efﬁcient, we also reported
hop reasoning.                                                  results of an ensemble which brings further 3.6%
  Instead,  we  frame  question  answering  as  an              of improvement and only 3% below the human
inference  problem  on  a  graph  representing  the             performance reported byWelbl et al.(2018). Our
document collection.  Nodes in this graph corre-                contributions can be summarized as follows:
spond to named entities in a document whereas
edges encode relations between them (e.g., cross-                  •we present a novel approach for multi-hop
and within-document coreference links or simply                       QA that relies on a (pre-trained) document
co-occurrence in a document).   We assume that                        encoder and information propagation across
reasoning chains can be captured by propagat-                         multiple documents using graph neural net-
ing local contextual information along edges in                       works;
this graph using a graph convolutional network                     •we  provide  an  efﬁcient  training  technique
(GCN) (Kipf and Welling,2017).                                        which relies on a slower ofﬂine and a faster
  The multi-document setting imposes scalabil-                        on-linecomputationthatdoesnotrequireex-
ity challenges.   In realistic scenarios,  a system                   pensive document processing;
needs to learn to answer a query for a given col-
lection (e.g., Wikipedia or a domain-speciﬁc set                   •weempiricallyshowthatouralgorithmisef-
of documents).  In such scenarios one cannot af-                      fective, presenting an improvement over pre-
ford to run expensive document encoders (e.g.,                        vious results.
RNN or transformer-like self-attention (Vaswani
et al.,2017)), unless the computation can be pre-               2   Method
processed both at train and test time.   Even if                In this section we explain our method.  We ﬁrst
(similarly to WIKIHOP creators) one considers a                 introduce  the  dataset  we  focus  on,  WIKIHOP
coarse-to-ﬁneapproach,whereasetofpotentially                    byWelbl et al.(2018), as well as the task ab-
relevant documents is provided, re-encoding them                straction. Wethenpresentthebuildingblocksthat
in a query-speciﬁc way remains the bottleneck. In               make up our Entity-GCN model, namely, an en-
contrasttootherproposedmethods(e.g.,(Dhingra                    titygraphusedtorelatementionstoentitieswithin
etal.,2018;Raisonetal.,2018;Seoetal.,2017)),                    and across documents, a document encoder used
we avoid training expensive document encoders.                  to obtain representations of mentions in context,
  In our approach, only a small query encoder,                  and a relational graph convolutional network that
the GCN layers and a simple feed-forward an-                    propagates information through the entity graph.
swer selection component are learned.   Instead
of training RNN encoders, we use contextualized                 2.1    Dataset and Task Abstraction
embeddings (ELMo) to obtain initial (local) rep-                Data    The WIKIHOP datasetcomprisesoftuples
resentations of nodes.   This implies that only a
lightweight computation has to be performed on-                 ⟨q,Sq,Cq,a⋆⟩where:qis a query/question,Sq is
line, both at train and test time, whereas the rest             asetofsupportingdocuments,Cq isasetofcandi-
is preprocessed. Even in the somewhat contrived                 date answers (all of which are entities mentioned
WIKIHOP setting, where fairly small sets of can-                inSq), anda⋆∈Cq  is the entity that correctly
didates are provided, the model is at least 5 times             answers the question. WIKIHOP is assembled as-
faster to train than BiDAF.1 Interestingly, when                sumingthatthereexistsacorpusandaknowledge
we substitute ELMo with simple pre-trained word                 base (KB) related to each other. The KB contains
embeddings,  Entity-GCN  still  performs  on  par               triples⟨s,r,o⟩wheresisasubjectentity,oanob-
                                                                ject entity, andra unidirectional relation between
    1When compared to the ‘small’ and hence fast BiDAF
model reported inWelbl et al.(2018), which is 25% less ac-      them.Welbletal.(2018)used W                            IKIPEDIA ascor-
curate than our Entity-GCN. Larger RNN models are prob-         pusand WIKIDATA (Vrande ˇci´c,2012)asKB.The
lematic also because of GPU memory constraints.                 KBisonlyusedforconstructingWIKIHOP:Welbl

et al.(2018) retrieved the supporting documents
Sq from the corpus looking at mentions of subject
and object entities in the text. Note that the setSq
(nottheKB)isprovidedtotheQAsystem,andnot
allofthesupportingdocumentsarerelevantforthe
querybutsomeofthemactasdistractors. Queries,
ontheotherhand,arenotexpressedinnaturallan-
guage, but instead consist of tuples⟨s,r,?⟩where
the object entity is unknown and it has to be in-
ferred by reading the support documents.  There-                Figure 2: Supporting documents (dashed ellipses) or-
fore,answeringaquerycorrespondstoﬁndingthe                      ganized as a graph where nodes are mentions of ei-
entitya⋆ thatistheobjectofatupleintheKBwith                     thercandidateentitiesorqueryentities. Nodeswiththe
subjectsand relationramong the provided set of                  same color indicates they refer to the same entity (ex-
candidate answersCq.                                            act match, coreference or both). Nodes are connected
                                                                bythreesimplerelations: oneindicatingco-occurrence
Task    The goal is to learn a model that can iden-             in the same document (solid edges), another connect-
tify the correct answera⋆ from the set of support-              ing mentions that exactly match (dashed edges), and a
ing documentsSq.   To that end, we exploit the                  third one indicating a coreference (bold-red line).
availablesupervisiontotrainaneuralnetworkthat
computesscoresforcandidatesinCq. Weestimate                        To each nodevi, we associate a continuous an-
the parameters of the architecture by maximizing                notation x i∈R D   which represents an entity in
the likelihood of observations. For prediction, we              thecontextwhereitwasmentioned(detailsinSec-
then output the candidate that achieves the high-               tion2.3). We then proceed to connect these men-
est probability.  In the following, we present our              tionsi)iftheyco-occurwithinthesamedocument
model discussing the design decisions that enable               (we will refer to this as DOC-BASED edges), ii)
multi-stepreasoningandanefﬁcientcomputation.                    if the pair of named entity mentions is identical
2.2    Reasoning on an Entity Graph                             (MATCH edges—these may connect nodes across
                                                                and within documents), or iii) if they are in the
Entity graph    In an ofﬂine step, we organize the              same coreference chain, as predicted by the exter-
content of each training instance in a graph con-               nal coreference system (COREF edges). Note that
necting mentions of candidate answers within and                MATCH edges when connecting mentions in the
across supporting documents.  For a given query                 same document are mostly included in the set of
q=⟨s,r,?⟩,weidentifymentionsinSq oftheen-                       edges predicted by the coreference system.  Hav-
titiesinCq∪{s}andcreateonenodepermention.                       ing the two types of edges lets us distinguish be-
This process is based on the following heuristic:               tween less reliable edges provided by the coref-
                                                                erence system and more reliable (but also more
  1.we consider mentions spans in Sq  exactly                   sparse) edges given by the exact-match heuristic.
     matching an element ofCq∪{s}.  Admit-                      We treat these three types of connections as three
     tedly, this is a rather simple strategy which              different types of relations.  See Figure2for an
     may suffer from low recall.                                illustration. Inadditiontothat,andtopreventhav-
                                                                ing disconnected graphs, we add a fourth type of
  2.we use predictions from a coreference reso-                 relation (COMPLEMENT edge) between any two
     lution system to add mentions of elements in               nodes that are not connected with any of the other
     Cq∪{s}beyond exact matching (including                     relations.  We can think of these edges as those
     both noun phrases and anaphoric pronouns).                 in the complement set of the entity graph with re-
     In particular, we use the end-to-end corefer-              spect to a fully connected graph.
     ence resolution byLee et al.(2017).
                                                                Multi-step   reasoning    Our   model   then   ap-
  3.we discard mentions which are ambiguously                   proaches  multi-step  reasoning  by  transforming
     resolved to multiple coreference chains; this              node  representations  (Section2.3for  details)
     may sacriﬁce recall, but avoids propagating                with a differentiable message passing algorithm
     ambiguity.                                                 that  propagates  information  through  the  entity

graph.       The  algorithm  is  parameterized  by              guage model that relies on character-based input
a  graph  convolutional  network  (GCN)  (Kipf                  representation. ELMo representations, differently
and  Welling, 2017),  in  particular,  we  employ               from other pre-trained word-based models (e.g.,
relational-GCNs(Schlichtkrulletal.,2018),anex-                  word2vec (Mikolov et al.,2013) or GloVe (Pen-
tendedversionthataccommodatesedgesofdiffer-                     nington et al.,2014)),  are contextualized since
ent types. In Section2.4we describe the propaga-                each token representation depends on the entire
tion rule.                                                      text excerpt (i.e., the whole sentence).
  Each step of the algorithm (also referred to as                  Wechoosenottoﬁnetunenorpropagategradi-
ahop)updatesallnoderepresentationsinparallel.                   ents through the ELMo architecture, as it would
In particular, a node is updated as a function of               have  deﬁed  the  goal  of  not  having  specialized
messages from its direct neighbours, and a mes-                 RNN encoders.  In the experiments, we will also
sage is possibly speciﬁc to a certain relation.  At             ablate the use of ELMo showing how our model
theendoftheﬁrststep,everynodeisawareofev-                       behaves using non-contextualized word represen-
eryothernodeitconnectsdirectlyto. Besides,the                   tations (we use GloVe).
neighbourhood of a node may include mentions
of the same entity as well as others (e.g., same-               Documents  pre-processing    ELMo  encodings
document relation), and these mentions may have                 are  used  to  produce  a  set  of  representations
occurred in different documents. Taking this idea               {x i}Ni=1  ,wherex i∈R D  denotestheithcandidate
recursively, each further step of the algorithm al-             mention in context.   Note that these representa-
lows a node to indirectly interact with nodes al-               tions do not depend on the query yet and no train-
readyknowntotheirneighbours. AfterLlayersof                     able model was used to process the documents so
R-GCN,informationhasbeenpropagatedthrough                       far,thatis,weuseELMoasaﬁxedpre-traineden-
paths connecting up toL+1   nodes.                              coder. Therefore, we can pre-compute representa-
  We start with node representations{h (0)i}Ni=1  ,             tionofmentionsonceandstorethemforlateruse.
and transform them by applyingL layers of R-                    Query-dependent  mention  encodings   ELMo
GCN obtaining{h (L )i}Ni=1  .  Together with a rep-             encodings are used to produce a query represen-
resentationq  ofthequery,wedeﬁneadistribution                   tation q∈R K   as well.  Here, q  is a concatena-
over candidate answers and we train maximizing                  tion of the ﬁnal outputs from a bidirectional RNN
the likelihood of observations. The probability of              layer trained to re-encode ELMo representations
selecting a candidatec∈Cq as an answer is then                  ofwordsinthequery. Thevectorq  isusedtocom-
                            (                       )           puteaquery-dependentrepresentationofmentions
  P(c|q,Cq,Sq)∝exp             maxi∈Mcfo([q,h (L )i   ]),       {ˆx  i}Ni=1   aswellastocomputeaprobabilitydistri-
                                                        (1)     bution over candidates (as in Equation1). Query-
wherefo  is a parameterized afﬁne transforma-                   dependent mention encodings ˆx  i = fx(q,x i) are
tion, andMc is the set of node indices such that                generated by a trainable functionfx  which is pa-
i∈Mc  only if nodevi is a mention ofc.  The                     rameterized by a feed-forward neural network.
max     operator in Equation1is necessary to select
thenodewithhighestpredictedprobabilitysincea                    2.4    Entity Relational Graph Convolutional
candidate answer is realized in multiple locations                     Network
via different nodes.                                            Our model uses a gated version of the original
2.3    Node Annotations                                         R-GCN propagation rule.   At the ﬁrst layer, all
Keeping in mind we want an efﬁcient model, we                   hiddennoderepresentationareinitializedwiththe
encode words in supporting documents and in the                 query-aware encodings h (0)i    =   ˆx  i. Then, at each
query using only a pre-trained model for contex-                layer 0≤ℓ≤L, the update message u (ℓ)i    to the
tualized word representations rather than training              ithnodeisasumofatransformationfs ofthecur-
ourownencoder. Speciﬁcally,weuseELMo2(Pe-                       rent node representation h (ℓ)i    and transformations
ters et al.,2018), a pre-trained bi-directional lan-            of its neighbours:
    2The use of ELMo is an implementation choice, and, in                                     ∑      ∑
principle, any other contextual pre-trained model could be        u (ℓ)i   = fs(h (ℓ)i  )+     1|Ni|       fr(h (ℓ)j  ),(2)
used (Radford et al.,2018;Devlin et al.,2019).                                                j∈Ni  r∈Rij

whereNi isthesetofindicesofnodesneighbour-                                          Min     Max     Avg.    Median
ingtheithnode,Rij isthesetofedgeannotations                      # candidates          2         79      19.8             14
betweeniandj, andfr  is a parametrized func-                     # documents          3         63      13.7             11
tion speciﬁc to an edge typer∈R.  Recall the                     # tokens/doc.         4    2,046    100.4             91
available relations from Section2.2, namely,    R=
{DOC-BASED,MATCH,COREF,COMPLEMENT}.                             Table 1: WIKIHOP dataset statistics fromWelbl et al.
  Agatingmechanismregulateshowmuchofthe                         (2018): number of candidates and documents per sam-
update message propagates to the next step. This                ple and document length.
provides the model a way to prevent completely
overwriting past information. Indeed, if all neces-             standard (unmasked) one and a masked one. The
saryinformationtoansweraquestionispresentat
alayerwhichisnotthelast,thenthemodelshould                      masked version was created by the authors to test
learn to stop using neighbouring information for                whether methods are able to learn lexical abstrac-
the next steps. Gate levels are computed as                     tion.  In this version, all candidates and all men-
                      (    (              ))                    tions of them in the support documents are re-
           a (ℓ)i   = σ fa   [u (ℓ)i ,h (ℓ)i  ],         (3)    placed by random but consistent placeholder to-
                                                                kens.  Thus, in the masked version, mentions are
whereσ(·)  is  the  sigmoid  function  andfa   a                alwaysreferredtoviaunambiguoussurfaceforms.
parametrized transformation.  Ultimately, the up-               We do not use coreference systems in the masked
dated representation is a gated combination of the              versionastheyrelycruciallyonlexicalrealization
previous representation and a non-linear transfor-              ofmentionsandcannotoperateonmaskedtokens.
mation of the update message:                                   3.1    Comparison
 h (ℓ+1)i       = φ(u (ℓ)i  )⊙a (ℓ)i   +  h (ℓ)i⊙(1−a (ℓ)i  ),(4)In  this  experiment,   we  compare  our  Enitity-
                                                                GCN  against  recent  prior  work  on  the  same
whereφ(·)  is any nonlinear function (we used                   task.      We  present  test  and  development  re-
tanh   ) and⊙stands for element-wise multiplica-                sults  (when  present)  for  both  versions  of  the
tion. Alltransformationsf∗areafﬁneandtheyare                    dataset in Table2.    FromWelbl et al.(2018),
not layer-dependent (since we would like to use                 we list an oracle based on human performance
as few parameters as possible to decrease model                 as well as two standard reading comprehension
complexity promoting efﬁciency and scalability).                models,  namely BiDAF (Seo et al.,2017) and
                                                                FastQA (Weissenborn et al.,2017). We also com-
3   Experiments                                                 pare against Coref-GRU (Dhingra et al.,2018),
Inthissection,wecompareourmethodagainstre-                      MHPGM (Bauer et al.,2018), and Weaver (Rai-
cent work as well as preforming an ablation study               son et al.,2018). Additionally, we include results
using the WIKIHOP dataset (Welbl et al.,2018).                  ofMHQA-GRN(Songetal.,2018),fromarecent
SeeAppendixAinthesupplementarymaterialfor                       arXiv preprint describing concurrent work.  They
adescriptionofthehyper-parametersofourmodel                     jointly train graph neural networks and recurrent
and training details.                                           encoders.  We report single runs of our two best
                                                                single models and an ensemble one on the un-
WIKIHOP    We use WIKIHOP for training, val-                    masked test set (recall that the test set is not pub-
idation/development and test.  The test set is not              licly available and the task organizers only report
publicly available and therefore we measure per-                unmasked results) as well as both versions of the
formance on the validation set in almost all ex-                validation set.
periments.   WIKIHOP has 43,738/ 5,129/ 2,451                      Entity-GCN (best single model without coref-
query-documents samples in the training, valida-                erence edges) outperforms all previous work by
tion and test sets respectively for a total of 51,318           over 2% points.   We additionally re-ran BiDAF
samples.  Authors constructed the dataset as de-                baseline to compare training time: when using a
scribed in Section2.1selecting samples with a                   singleTitanXGPU,BiDAFandEntity-GCNpro-
graph traversal up to a maximum chain length of                 cess 12.5 and 57.8 document sets per second, re-
3 documents (see Table1for additional dataset                   spectively.  Note thatWelbl et al.(2018) had to
statistics).   WIKIHOP comes in two versions, a                 use BiDAF with very small state dimensionalities

                Model                                                         Unmasked       Masked
                                                                             Test    Dev       Test    Dev
                Human (Welbl et al.,2018)                                    74.1      –        –         –
                FastQA (Welbl et al.,2018)                                   25.7      –      35.8      –
                BiDAF (Welbl et al.,2018)                                    42.9      –      54.5      –
                Coref-GRU (Dhingra et al.,2018)                              59.3    56.0       –         –
                MHPGM (Bauer et al.,2018)                                      –      58.2      –         –
                Weaver / Jenga (Raison et al.,2018)                          65.3    64.1       –         –
                MHQA-GRN (Song et al.,2018)                                  65.4    62.8       –         –
                Entity-GCN without coreference (single model)                67.6    64.8       –      70.5
                Entity-GCN with coreference (single model)                   66.4    65.3       –         –
                Entity-GCN* (ensemble 5 models)                              71.2    68.5       –      71.6
Table 2: Accuracy of different models on WIKIHOP closed test set and public validation set.  Our Entity-GCN
outperforms recent prior work without learning any language model to process the input but relying on a pre-
trained one (ELMo – without ﬁne-tunning it) and applying R-GCN to reason among entities in the text.  * with
coreference for unmasked dataset and without coreference for the masked one.
(20), and smaller batch size due to the scalabil-                withoutreadingthecontextatall. Forexample,in
ity issues (both memory and computation costs).                  Figure1, our model would be aware that “Stock-
Wecompareapplyingthesamereductions.3 Even-                       holm”and“Sweden”appearinthesamedocument
tually, we also report an ensemble of 5 indepen-                 but any context words, including the ones encod-
dently trained models.  All models are trained on                ing relations (e.g., “is the capital of”) will be hid-
the same dataset splits with different weight ini-               den. Besides, in the masked case all mentions be-
tializations.  The ensemble prediction is obtained               come‘unknown’tokenswithGloVeandtherefore
               5∏                                                the predictions are equivalent to a random guess.
asargmax  c                                                      Once the strong pre-trained encoder is out of the
              i=1 Pi(c|q,Cq,Sq) from each model.
3.2    Ablation Study                                            way, we also ablate the use of our R-GCN com-
                                                                 ponent,thuscompletelydeprivingthemodelfrom
To help determine the sources of improvements,                   inductive biases that aim at multi-hop reasoning.
we perform an ablation study using the publicly                     Theﬁrstimportantobservationisthatreplacing
available validation set (see Table3).   We per-                 ELMo by GloVe (GloVe with R-GCN in Table3)
form two groups of ablation, one on the embed-                   still yields a competitive system that ranks far
ding layer, to study the effect of ELMo, and one                 abovebaselinesfrom(Welbletal.,2018)andeven
on the edges, to study how different relations af-               above the Coref-GRU ofDhingra et al.(2018), in
fect the overall model performance.                              terms of accuracy on (unmasked) validation set.
Embedding ablation    We argue that ELMo is                      The second important observation is that if we
crucial, since we do not rely on any other context               then remove R-GCN (GloVe w/o R-GCN in Ta-
encoder. However, it is interesting to explore how               ble3), we lose 8.0 points.  That is, the R-GCN
ourR-GCNperformswithoutit. Therefore,inthis                      component pushes the model to perform above
experiment, we replace the deep contextualized                   Coref-GRU still without accessing context,  but
embeddings of both the query and the nodes with                  rather by updating mention representations based
GloVe (Pennington et al.,2014) vectors (insensi-                 on their relation to other ones. These results high-
tivetocontext). Sincewedonothaveanycompo-                        light the impact of our R-GCN component.
nent in our model that processes the documents,                  Graphedgesablation   Inthisexperimentwein-
we expect a drop in performance. In other words,                 vestigate the effect of the different relations avail-
inthisablationourmodeltriestoanswerquestions                     able in the entity graph and processed by the R-
    3Besides, we could not run any other method we com-          GCN module. We start off by testing our stronger
parewithcombinedwithELMowithoutreducingthedimen-                 encoder(i.e.,ELMo)inabsenceofedgesconnect-
sionalityfurtherorhavingtoimplementadistributedversion.          ingmentionsinthesupportingdocuments(i.e.,us-

  Model                      unmasked     masked                ﬁrst thing to note is that the model makes better
  full (ensemble)                68.5             71.6          use of DOC-BASED connections than MATCH or
  full (single)               65.1±  0.11    70.4±  0.12        COREF connections. This is mostly because i) the
                                                                majority of the connections are indeed between
  GloVe with R-GCN               59.2             11.1          mentions in the same document, and ii) without
  GloVe w/o R-GCN                51.2             11.6          connecting mentions within the same document
  No R-GCN                       62.4             63.2          weremoveimportantinformationsincethemodel
  No relation types              62.7             63.9          is unaware they appear closely in the document.
  NoDOC-BASED                    62.9             65.8          Secondly,  we notice that coreference links and
  NoMATCH                        64.3             67.4          complement edges seem to play a more marginal
  NoCOREF                        64.8               –           role. Though it may be surprising for coreference
  NoCOMPLEMENT                   64.1             70.3          edges,recallthattheMATCHheuristicalreadycap-
  Induced edges                  61.5             56.4          turestheeasiestcoreferencecases,andfortherest
                                                                the out-of-domain coreference system may not be
Table 3:  Ablation study on WIKIHOP validation set.             reliable.  Still, modelling all these different rela-
The full model is our Entity-GCN with all of its com-           tionstogethergivesourEntity-GCNaclearadvan-
ponentsandotherrowsindicatemodelstrainedwithout                 tage. This is our best system evaluating on the de-
acomponentofinterest. Wealsoreportbaselinesusing                velopment. Since Entity-GCN seems to gain little
GloVe instead of ELMo with and without R-GCN. For               advantageusingthecoreferencesystem,wereport
the full model we reportmean ±1std   over 5 runs.               testresultsbothwithandwithoutusingit. Surpris-
                                                                ingly, with coreference, we observe performance
ing only self-loops – No R-GCN in Table3). The                  degradation on the test set. It is likely that the test
resultssuggestthatWIKIPHOPgenuinelyrequires                     documentsareharderforthecoreferencesystem.5
multihopinference,asourbestmodelis6.1%and                          Wedoperformonelastablation,namely,were-
8.4% more accurate than this local model, in un-                place our heuristic for assigning edges and their
masked and masked settings, respectively.4 How-                 labels by a model component that predicts them.
ever,italsoshowsthatELMorepresentationscap-                     The last row of Table3(Induced edges) shows
ture predictive context features, without being ex-             model performance when edges are not predeter-
plicitly trained for the task.  It conﬁrms that our             minedbutpredicted. Forthisexperiment,weusea
goal of getting away with training expensive doc-               bilinear functionfe(ˆx  i,ˆx  j) = σ(ˆx⊤i W   eˆx  j)that
ument encoders is a realistic one.                              predicts the importance of a single edge connect-
  We then inspect our model’s effectiveness in                  ing two nodesi,jusing the query-dependent rep-
making use of the structure encoded in the graph.               resentation of mentions (see Section2.3).   The
We  start  naively  by  fully-connecting  all  nodes            performancedropsbelow‘NoR-GCN’suggesting
within and across documents without distinguish-                thatitcannotlearnthesedependenciesonitsown.
ing edges by type (No relation types in Table3).                   Most results are stronger for the masked set-
We observe only marginal improvements with re-                  tingseventhoughwedonotapplythecoreference
spect to ELMo alone (No R-GCN in Table3) in                     resolution system in this setting due to masking.
both the unmasked and masked setting suggest-                   It is not surprising as coreferred mentions are la-
ingthataGCNoperatingoveranaiveentitygraph                       beled with the same identiﬁer in the masked ver-
would not add much to this task and a more infor-               sion, even if their original surface forms did not
mative graph construction and/or a more sophisti-               match (Welbl et al.(2018) used W                  IKIPEDIA links
cated parameterization is indeed needed.                        for masking).  Indeed, in the masked version, an
  Next,  we ablate each type of relations inde-                 entity is always referred to via the same unique
pendently, that is, we either remove connections                surfaceform(e.g.,MASK1)withinandacrossdoc-
of  mentions  that  co-occur  in  the  same  docu-              uments.   In the unmasked setting, on the other
ment (DOC-BASED), connections between men-                      hand, mentions to an entity may differ (e.g., “US”
tions matching exactly (MATCH), or edges pre-                   vs“UnitedStates”)andtheymightnotberetrieved
dicted by the coreference system (COREF). The                   bythecoreferencesystemweareemploying,mak-
    4Recall that all models in the ensemble use the same lo-        5Since the test set is hidden from us, we cannot analyze
cal representations, ELMo.                                      this difference further.

                    Relation                            Accuracy    P@2    P@5    Avg.|Cq| Supports
                    overall (ensemble)                     68.5         81.0     94.1     20.4±  16.6           5129
                    overall (single model)                 65.3         79.7     92.9     20.4±  16.6           5129
                    member      of political  party        85.5         95.7     98.6       5.4±  2.4                 70
          3 best    record   label                         83.0         93.6     99.3      12.4±  6.1              283
                    publisher                              81.5         96.3    100.0      9.6±  5.1                 54
                    place   of  birth                      51.0         67.2     86.8     27.2±  14.5             309
        3 worst     place   of  death                      50.0         67.3     89.1     25.1±  14.3             159
                    inception                              29.9         53.2     83.1     21.9±  11.0               77
Table 4: Accuracy and precision at K (P@K in the table) analysis overall and per query type. Avg.|Cq|indicates
the average number of candidates with one standard deviation.
ingthetaskharderforallmodels. Therefore,aswe                     doesnotseemanysampleswheretherearealarge
rely mostly on exact matching when constructing                  number of candidate entities during training. Dif-
our graphfor themasked case,we aremore effec-                    ferently, we notice that as the number of nodes in
tiveinrecoveringcoreferencelinksonthemasked                      the graph increases, the model performance drops
rather than unmasked version.6                                   but more gently (negative but closer to zero Pear-
4   Error Analysis                                               son’s correlation). This is important as document
                                                                 setscanbelargeinpracticalapplications. SeeFig-
In this section we provide an error analysis for                 ure3in the supplemental material for plots.
our best single model predictions. First of all, we              5   Related Work
look at which type of questions our model per-
forms well or poorly.  There are more than 150                   In  previous  work,  BiDAF  (Seo  et  al., 2017),
query types in the validation set but we ﬁltered                 FastQA   (Weissenborn   et   al.,  2017),   Coref-
the three with the best and with the worst accu-                 GRU  (Dhingra  et  al., 2018),  MHPGM  (Bauer
racy that have at least 50 supporting documents                  et al.,2018), and Weaver / Jenga (Raison et al.,
and at least 5 candidates. We show results in Ta-                2018) have been applied to multi-document ques-
ble4. We observe that questions regarding places                 tionanswering. Theﬁrsttwomainlyfocusonsin-
(birth and death) are considered harder for Entity-              gle document QA andWelbl et al.(2018) adapted
GCN. We then inspect samples where our model                     both of them to work with WIKIHOP.  They pro-
fails while assigning highest likelihood and no-                 cess each instance of the dataset by concatenat-
ticedtwoprincipalsourcesoffailurei)amismatch                     ing alld∈Sq  in a random order adding doc-
betweenwhatiswritteninWIKIPEDIAandwhatis                         ument separator tokens.   They trained using the
annotated in WIKIDATA, and ii) a different degree                ﬁrstanswermentionintheconcatenateddocument
of granularity (e.g., born in “London” vs “UK”                   and evaluating exact match at test time.   Coref-
could be considered both correct by a human but                  GRU, similarly to us, encodes relations between
not when measuring accuracy). See Table6in the                   entity mentions in the document.  Instead of us-
supplement material for some reported samples.                   ing graph neural network layers, as we do, they
   Secondly,westudyhowthemodelperformance                        augment RNNs with jump links corresponding to
degradeswhentheinputgraphislarge. Inparticu-                     pairs of corefereed mentions.   MHPGM uses a
lar, we observe a negative Pearson’s correlation (-              multi-attention mechanism in combination with
0.687)betweenaccuracyandthenumberofcandi-                        external commonsense relations to perform mul-
date answers. However, the performance does not                  tiple hops of reasoning.   Weaver is a deep co-
decreasesteeply. Thedistributionofthenumberof                    encoding model that uses several alternating bi-
candidates in the dataset peaks at 5 and has an av-              LSTMs to process the concatenated documents
erage of approximately 20. Therefore, the model                  and the query.
    6Though other systems do not explicitly link matching           Graph neural networks have been shown suc-
mentions, they similarly beneﬁt from masking (e.g., masks        cessful on a number of NLP tasks (Marcheggiani
essentially single out spans that contain candidate answers).    andTitov,2017;Bastingsetal.,2017;Zhangetal.,

2018a), including those involving document level                      Processing, pages 4220–4230, Brussels, Belgium.
modeling(Pengetal.,2017). Theyhavealsobeen                            Association for Computational Linguistics.
applied in the context of asking questions about                  Jacob  Devlin,  Ming-Wei  Chang,  Kenton  Lee,  and
knowledge contained in a knowledge base (Zhang                        Kristina Toutanova. 2019.BERT: Pre-training of
etal.,2018b). InSchlichtkrulletal.(2018),GCNs                         deep bidirectional transformers for language under-
are used to capture reasoning chains in a knowl-                      standing.   In               Proceedings of the 2019 Conference
edge base. Our work and unpublished concurrent                        of the North American Chapter of the Association
work bySong et al.(2018) are the ﬁrst to study                        for Computational Linguistics:  Human Language
                                                                      Technologies, Volume 1 (Long and Short Papers),
graph neural networks in the context of multi-                        pages 4171–4186, Minneapolis, Minnesota. Associ-
documentQA.Besidesdifferencesinthearchitec-                           ation for Computational Linguistics.
ture,Song et al.(2018) propose to train a combi-                  Bhuwan Dhingra, Qiao Jin, Zhilin Yang, William Co-
nation of a graph recurrent network and an RNN                        hen, and Ruslan Salakhutdinov. 2018.Neural mod-
encoder. We do not train any RNN document en-                         elsforreasoningovermultiplementionsusingcoref-
coders in this work.                                                  erence.  In           Proceedings of the 2018 Conference of
                                                                      the North American Chapter of the Association for
6   Conclusion                                                        ComputationalLinguistics: HumanLanguageTech-
                                                                      nologies,  Volume 2 (Short Papers),  pages 42–48,
We designed a graph neural network that oper-                         New Orleans, Louisiana. Association for Computa-
ates over a compact graph representation of a set                     tional Linguistics.
of documents where nodes are mentions to en-                      Karl   Moritz   Hermann,   Tom´as   Kocisk´y,   Edward
tities and edges signal relations such as within                      Grefenstette,LasseEspeholt,WillKay,MustafaSu-
and  cross-document  coreference.     The  model                      leyman, and Phil Blunsom. 2015.Teaching ma-
learns to answer questions by gathering evidence                      chines to read and comprehend.   In                     Advances in
fromdifferentdocumentsviaadifferentiablemes-                          Neural Information Processing Systems 28: Annual
                                                                      Conference on Neural Information Processing Sys-
sage passing algorithm that updates node repre-                       tems 2015, December 7-12, 2015, Montreal, Que-
sentations based on their neighbourhood.    Our                       bec, Canada, pages 1693–1701.
model outperforms published results where abla-                   Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke
tionsshowsubstantialevidenceinfavourofmulti-                          Zettlemoyer. 2017.TriviaQA: A large scale dis-
stepreasoning. Moreover,wemakethemodelfast                            tantlysupervisedchallengedatasetforreadingcom-
by using pre-trained (contextual) embeddings.                         prehension.    In                   Proceedings  of  the  55th  Annual
                                                                      Meeting of the Association for Computational Lin-
Acknowledgments                                                       guistics  (Volume  1:   Long  Papers),  pages  1601–
                                                                      1611, Vancouver, Canada. Association for Compu-
We would like to thank Johannes Welbl for help-                       tational Linguistics.
ing  to  test  our  system  on  WIKIHOP.     This                 Diederik P. Kingma and Jimmy Ba. 2015.Adam:  A
project is supported by SAP Innovation Center                         method for stochastic optimization.   In                     3rd Inter-
Network,ERCStartingGrantBroadSem(678254)                              national Conference on Learning Representations,
and  the  Dutch  Organization  for  Scientiﬁc  Re-                    ICLR 2015, San Diego, CA, USA, May 7-9, 2015,
search (NWO) VIDI 639.022.518. Wilker Aziz is                         Conference Track Proceedings.
supportedbytheDutchOrganisationforScientiﬁc                       Thomas  N.  Kipf  and  Max  Welling.  2017.Semi-
Research (NWO) VICI Grant nr. 277-89-002.                             supervised classiﬁcation with graph convolutional
                                                                      networks.      In         5th  International  Conference  on
                                                                      Learning  Representations,   ICLR  2017,   Toulon,
References                                                            France, April 24-26, 2017, Conference Track Pro-
                                                                      ceedings. OpenReview.net.
Jasmijn  Bastings,  Ivan  Titov,  Wilker  Aziz,  Diego            Tom´aˇs  Koˇcisk´y,  Jonathan  Schwarz,  Phil  Blunsom,
   Marcheggiani, and Khalil Sima’an. 2017.Graph                       ChrisDyer,KarlMoritzHermann,G´aborMelis,and
   convolutional encoders for syntax-aware neural ma-                 Edward Grefenstette. 2018.The NarrativeQA read-
   chine translation.  In                  Proceedings of the 2017 Con-ing comprehension challenge.                  Transactions of the
   ference on Empirical Methods in Natural Language                   Association for Computational Linguistics, 6:317–
   Processing, pages 1957–1967, Copenhagen, Den-                      328.
   mark. Association for Computational Linguistics.
Lisa Bauer, Yicheng Wang, and Mohit Bansal. 2018.                 Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang,
   Commonsenseforgenerativemulti-hopquestionan-                       and Eduard Hovy. 2017.RACE: Large-scale ReAd-
   swering tasks.  In         Proceedings of the 2018 Confer-         ing comprehension dataset from examinations.   In
   ence on Empirical Methods in Natural Language                      Proceedings of the 2017 Conference on Empirical

   Methods  in  Natural  Language  Processing,  pages                   machine comprehension of text.  In    Proceedings of
   785–794,  Copenhagen,  Denmark. Association for                      the2016ConferenceonEmpiricalMethodsinNatu-
   Computational Linguistics.                                           ralLanguageProcessing,pages2383–2392,Austin,
                                                                        Texas. Association for Computational Linguistics.
KentonLee,LuhengHe,MikeLewis,andLukeZettle-
   moyer. 2017.End-to-end neural coreference reso-                  Michael Schlichtkrull, Thomas N. Kipf, Peter Bloem,
   lution.  In          Proceedings of the 2017 Conference on           Rianne van den Berg, Ivan Titov, and Max Welling.
   Empirical Methods in Natural Language Process-                       2018. Modeling relational data with graph convolu-
   ing, pages 188–197, Copenhagen, Denmark. Asso-                       tional networks.  In The Semantic Web, pages 593–
   ciation for Computational Linguistics.                               607, Cham. Springer International Publishing.
Diego Marcheggiani and Ivan Titov. 2017.Encoding                    Min Joon Seo, Aniruddha Kembhavi, Ali Farhadi, and
   sentences with graphconvolutional networks for se-                   Hannaneh Hajishirzi. 2017.Bidirectional attention
   mantic role labeling.   In              Proceedings of the 2017      ﬂow  for  machine  comprehension.    In                           5th  Inter-
   Conference on Empirical Methods in Natural Lan-                      national Conference on Learning Representations,
   guage Processing, pages 1506–1515, Copenhagen,                       ICLR2017,Toulon,France,April24-26,2017,Con-
   Denmark. Association for Computational Linguis-                      ference Track Proceedings. OpenReview.net.
   tics.
Tom´as Mikolov, Ilya Sutskever, Kai Chen, Gregory S.                Yelong  Shen,   Po-Sen  Huang,   Jianfeng  Gao,   and
   Corrado, and Jeffrey Dean. 2013.Distributed rep-                     Weizhu Chen. 2017.Reasonet:  Learning to stop
   resentations of words and phrases and their com-                     reading in machine comprehension. In                         Proceedings
   positionality.   In                     Advances in Neural Informationofthe23rdACMSIGKDDInternationalConference
   Processing Systems 26: 27th Annual Conference on                     on Knowledge Discovery and Data Mining, Hali-
   Neural Information Processing Systems 2013. Pro-                     fax,NS,Canada,August13-17,2017,pages1047–
   ceedings of a meeting held December 5-8,  2013,                      1055. ACM.
   Lake Tahoe, Nevada, United States, pages 3111–                   Linfeng  Song,  Zhiguo  Wang,  Mo  Yu,  Yue  Zhang,
   3119.                                                                Radu Florian, and Daniel Gildea. 2018.Exploring
Nanyun Peng,  Hoifung Poon,  Chris Quirk,  Kristina                     Graph-structured Passage Representation for Multi-
   Toutanova, and Wen-tau Yih. 2017.Cross-sentence                      hop  Reading  Comprehension  with  Graph  Neural
   n-ary relation extraction with graph LSTMs.             Trans-       Networks.        ArXiv preprint, abs/1809.02040.
   actions of the Association for Computational Lin-
   guistics, 5:101–115.                                             Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky,
                                                                        Ilya  Sutskever,  and  Ruslan  Salakhutdinov.  2014.
Jeffrey Pennington, Richard Socher, and Christopher                     Dropout: A simple way to prevent neural networks
   Manning. 2014.GloVe:  Global vectors for word                        from overﬁtting. The Journal of Machine Learning
   representation.   In                        Proceedings of the 2014 Con-Research, 15(1):1929–1958.
   ference on Empirical Methods in Natural Language
   Processing  (EMNLP),  pages  1532–1543,   Doha,                  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
   Qatar. Association for Computational Linguistics.                    Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt                      Kaiser, and Illia Polosukhin. 2017.Attention is all
   Gardner, Christopher Clark, Kenton Lee, and Luke                     you need.  In        Advances in Neural Information Pro-
   Zettlemoyer. 2018.Deep contextualized word rep-                      cessing Systems 30: Annual Conference on Neural
   resentations.   In                    Proceedings of the 2018 Confer-Information Processing Systems 2017, December 4-
   ence of the North American Chapter of the Associ-                    9, 2017, Long Beach, CA, USA, pages 5998–6008.
   ation for Computational Linguistics:  Human Lan-                 DennyVrandeˇci´c.2012. Wikidata: Anewplatformfor
   guageTechnologies,Volume1(LongPapers),pages                          collaborative data collection.  In Proceedings of the
   2227–2237,  New Orleans,  Louisiana. Association                     21st International Conference on World Wide Web,
   for Computational Linguistics.                                       pages 1063–1064. ACM.
AlecRadford,KarthikNarasimhan,TimSalimans,and                       Dirk Weissenborn,  Georg Wiese,  and Laura Seiffe.
   Ilya Sutskever. 2018.   Improving language under-                    2017.Making neural QA as simple as possible but
   standing with unsupervised learning.  Technical re-                  not simpler.    In             Proceedings of the 21st Confer-
   port, OpenAI.                                                        enceonComputationalNaturalLanguageLearning
Martin  Raison,  Pierre-Emmanuel  Mazar´e,  Rajarshi                    (CoNLL2017),pages271–280,Vancouver,Canada.
   Das, and Antoine Bordes. 2018. Weaver: Deep co-                      Association for Computational Linguistics.
   encoding of questions and documents for machine
   reading.  In Proceedings of the International Con-               Johannes  Welbl,   Pontus  Stenetorp,   and  Sebastian
   ference on Machine Learning (ICML).                                  Riedel. 2018.Constructing datasets for multi-hop
                                                                        readingcomprehensionacrossdocuments.                  Transac-
PranavRajpurkar,JianZhang,KonstantinLopyrev,and                         tions of the Association for Computational Linguis-
   PercyLiang.2016.SQuAD:100,000+questionsfor                           tics, 6:287–302.

Caiming Xiong, Victor Zhong, and Richard Socher.
   2017.Dynamic  coattention  networks  for  ques-
   tion answering.    In                  5th International Conference
   on Learning Representations, ICLR 2017, Toulon,
   France, April 24-26, 2017, Conference Track Pro-
   ceedings. OpenReview.net.
Yuhao Zhang, Peng Qi, and Christopher D. Manning.
   2018a.Graph convolution over pruned dependency
   treesimprovesrelationextraction. In               Proceedingsof
   the 2018 Conference on Empirical Methods in Nat-
   ural Language Processing, pages 2205–2215, Brus-
   sels, Belgium. Association for Computational Lin-
   guistics.
Yuyu Zhang, Hanjun Dai, Zornitsa Kozareva, Alexan-
   der J. Smola,  and Le Song. 2018b.Variational
   reasoning for question answering with knowledge
   graph.   In          Proceedings of the Thirty-Second AAAI
   Conference  on  Artiﬁcial  Intelligence,  (AAAI-18),
   the 30th innovative Applications of Artiﬁcial Intel-
   ligence  (IAAI-18),  and  the  8th  AAAI  Symposium
   on Educational Advances in Artiﬁcial Intelligence
   (EAAI-18),NewOrleans,Louisiana,USA,February
   2-7, 2018, pages 6069–6076. AAAI Press.

A   Implementation and Experiments                               B   Error Analysis
     Details                                                     In Table6, we report three samples from W           IKI-
A.1    Architecture                                              HOPdevelopmentsetwhereoutEntity-GCNfails.
See table5for an outline of Entity-GCN architec-                 In particular, we show two instances where our
tural detail. Here the computational steps                       model presents high conﬁdence on the answer,
  1.ELMo  embeddings  are  a  concatenation  of                  and one where is not. We commented these sam-
      three 1024-dimensional vectors resulting in                ples explaining why our model might fail in these
      3072-dimensional input vectors{x i}Ni=1  .                 cases.
  2.For the query representation   q , we apply 2                C   Ablation Study
      bi-LSTM layers of 256 and 128 hidden units                 In Figure3, we show how the model performance
      to its ELMo vectors.  The concatenation of                 goes when the input graph is large. In particular,
      the forward and backward states results in a               how Entity-GCN performs as the number of can-
      256-dimensional question representation.                   didate answers or the number of nodes increases.
  3.ELMo  embeddings  of  candidates  are  pro-
      jected to 256-dimensional vectors, concate-
      nated to the q , and further transformed with
      a two layers MLP of 1024 and 512 hidden
      units in 512-dimensional query aware entity
      representations{ˆx  i}Ni=1∈R  512  .
  4.All transformations f∗in R-GCN-layers are
      afﬁneandtheydomaintaintheinputandout-
      put dimensionality of node representations                 (a) Candidates set size (x-axis) and accuracy (y-axis). Pear-
      the same (512-dimensional).                                son’s correlation of− 0.687    (p<   10 − 7 ).
  5.Eventually, a 2-layers MLP with [256, 128]
      hiddenunitstakestheconcatenationbetween
      {h (L )i}Ni=1    and q   to predict the probability
      that a candidate nodevi may be the answer
      to the queryq(see Equation1).
   During  preliminary  trials,  we  experimented
with different numbers of R-GCN-layers (in the
range 1-7). We observed that with WIKIHOP, for                   (b) Nodes set size (x-axis) and accuracy (y-axis). Pearson’s
L≥3  models reach essentially the same perfor-                   correlation of− 0.385    (p<   10 − 7 ).
mance, but more layers increase the time required                Figure 3:  Accuracy (blue) of our best single model
totrainthem. Besides,weobservedthatthegating                     with respect to the candidate set size (on the top) and
mechanismlearnstokeepmoreandmoreinforma-                         nodessetsize(onthebottom)onthevalidationset. Re-
tion from the past at each layer making unneces-                 scaleddatadistributions(orange)pernumberofcandi-
sary to have more layers than required.                          date (top) and nodes (bottom).  Dashed lines indicate
                                                                 average accuracy.
A.2    Training Details
We  train  our  models  with  a  batch  size  of  32
for  at  most  20  epochs  using  the  Adam  opti-
mizer (Kingma and Ba,2015) with      β1   =  0 .9,
β2  = 0 .999   and a learning rate of 10−4 . To help
against overﬁtting, we employ dropout (drop rate
∈0,0.1,0.15,0.2,0.25 )  (Srivastava et al.,2014)
and early-stopping on validation accuracy. We re-
port the best results of each experiment based on
accuracy on validation set.

                                                     Input - q,{vi}Ni=1
                               query ELMo 3072-dim                   candidates ELMo 3072-dim
                        2 layers bi-LSTM [256, 128]-dim                    1 layer FF 256-dim
                                                  concatenation 512-dim
                                        2 layer FF [1024, 512]-dim: :{ˆx  i}Ni=1
                                3 layers R-GCN 512-dim each (shared parameters)
                                             concatenation withq  768-dim
                                               3 layers FF [256,128,1]-dim
                                             Output - probabilities overCq
                                                Table 5: Model architecture.
           ID     WH    dev   2257                                      Gold answer         2003 (p= 14 .1)
      Query       inception (of) Derrty Entertainment             Predicted answer          2000 (p= 15 .8)
 Support 1        DerrtyEntertainmentisarecordlabelfoundedby[...]. Theﬁrstalbumreleasedunder
                  Derrty Entertainmentwas Nelly ’s Country Grammar.
 Support 2        Country Grammar is the debut single by American rapper Nelly. The song was pro-
                  duced by Jason Epperson. It was released in2000, [...]
(a) In this example, the model predicts the answer correctly.   However, there is a mismatch between what is written in
WIKIPEDIA and what is annotated in WIKIDATA. In WIKIHOP, answers are generated with WIKIDATA.
           ID     WH    dev   2401                                   Gold answer         Adolph Zukor (p= 7 .1e−4%   )
      Query       producer (of) Forbidden Paradise             Predicted answer          Jesse L. Lask (p= 99 .9%   )
 Support 1        Forbidden Paradiseis a [...] drama ﬁlm produced byFamous Players-Lasky [...]
 Support 2        Famous Players-Lasky Corporation was [...] from the merger of Adolph Zukor’s Fa-
                  mous Players Film Company [..] and theJesse L. LaskyFeature Play Company.
(b) In this sample, there is ambiguity between two entities since both are correct answers reading the passages but only one is
marked as correct. The model fails assigning very high probability to only on one of them.
           ID     WH    dev   3030                               Gold answer          Scania (p= 0 .029%    )
      Query       place   of birth (of) Erik Penser        Predicted answer           Esl¨ov (p= 97 .3%   )
 Support 1        Nils Wilhelm Erik Penser(born August 22, 1942, inEsl¨ov, Sk˚ane) is a Swedish [...]
 Support 2        Sk˚ane County, sometimes referred to as “ Scania County ” in English, is the [...]
(c) In this sample, there is ambiguity between two entities since the city Esl¨ov is located in the Scania County (English name
of Sk˚ane County). The model assigning high probability to the city and it cannot select the county.
        Table 6: Samples from WIKIHOP set where Entity-GCN fails.pindicates the predicted likelihood.

