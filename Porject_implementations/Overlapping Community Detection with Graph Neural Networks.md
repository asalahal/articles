                                   OverlappingCommunityDetection
                                           withGraphNeuralNetworks
                           Oleksandr Shchur                                                          Stephan Günnemann
             Technical Univeristy of Munich, Germany                                       Technical Univeristy of Munich, Germany
                            shchur@in.tum.de                                                          guennemann@in.tum.de
ABSTRACT                                                                          overlapping communities isa requirement not yetmet by existing
Community detection is a fundamental problem in machine learn-                    deep learning approaches for community detection.
ing. While deep learning has shown greatpromise in many graph-                       In thispaper weaddress thisresearch gap andpropose anend-
relatedtasks,developingneuralmodelsforcommunitydetection                          to-enddeep learningmodelcapable ofdetectingoverlapping com-
hasreceivedsurprisinglylittleattention.Thefewexistingapproaches                   munities. To summarize, our main contributions are:
focusondetectingdisjointcommunities,eventhoughcommunities                              •Model:Weintroducea graphneuralnetwork (GNN)based
in real graphs are well known to be overlapping. We address this                          model for overlapping community detection.
shortcoming and propose a graph neural network (GNN) based                             •Data:Weintroduce4newdatasetsforoverlappingcommu-
model for overlapping community detection. Despite its simplicity,                        nity detection that can act as a benchmark and stimulate
our model outperforms the existing baselines by a large margin in                         future research in this area.
thetaskofcommunityrecovery.Weestablishthroughanextensive                               •Experiments:Weperformathoroughevaluationofourmodel
experimental evaluation that the proposed model is effective, scal-                       and show its superior performance compared to established
able and robust to hyperparameter settings. We also perform an                            methodsforoverlappingcommunitydetection,bothinterms
ablationstudythatconfirmsthatGNNisthekeyingredienttothe                                   ofspeedandaccuracy.Wehighlighttheimportanceofthe
poweroftheproposedmodel.Areferenceimplementationaswell                                    GNN component of our model through an ablation study.
as the new datasets are available under www.kdd.in.tum.de/nocd.
ACMReferenceFormat:                                                               2   BACKGROUND
OleksandrShchurandStephanGünnemann.2019.OverlappingCommunity                      Assume that we are given an undirected unweighted graphG, rep-
Detection with Graph Neural Networks. In Proceedings of The First Inter-          resented as a binary adjacency matrixA∈{0, 1}N×N  . We denote
national Workshop on Deep Learning for Graphs (DLG’19). ACM, New York,            as N the number of nodesV ={1, ...,   N}; and as M the number
NY, USA, 7 pages.                                                                 of edges E ={(u,v)∈V×V  : Auv =   1}. Every node might be
                                                                                  associatedwithaD-dimensionalattributevector,thatcanberepre-
1   INTRODUCTION                                                                  sented as an attributematrixX∈R N×D . The goalof overlapping
Graphsprovideanaturalwayofrepresentingcomplexreal-world                           community detection is to assign nodes intoC communities. Such
systems. Community detection methods are an essential tool for                    assignment can be representedas a non-negative community affili-
understandingthe structureandbehaviorofthese systems.Detect-                      ation matrixF∈R N×C≥0   , where Fuc   denotes the strength of node
ingcommunitiesallowsustoanalyzesocialnetworks[14],detect                          u’s membership in communityc (with the notable special case of
fraud [30], discover functional units of the brain [13], and predict              binary assignmentF∈{0, 1}N×C ). Some nodes may be assigned
functions of proteins [32]. The problem of community detection                    to no communities, while others may belong to multiple.
has attracted significant attention of the research community and                    Eventhoughthenotionof"community"seemsratherintuitive,
numerous models and algorithms have been proposed [38].                           there isno universally agreed upon definitionof it in theliterature.
   Intherecentyears,theemergingfieldofdeeplearningforgraphs                       However,mostrecentworkstendto agreewiththestatementthat
hasshowngreatpromiseindesigningmoreaccurateandmorescal-                           a community is a group of nodes that have higher probability to
able algorithms. While deep learning approaches have achieved                     formedgeswitheachotherthanwithothernodesinthegraph[11].
unprecedented results in graph-related tasks like link prediction                 Thisway,theproblemofcommunitydetectioncanbeconsidered
andnodeclassification[5],relativelylittleattentionhasbeendedi-                    in termsof theprobabilistic inferenceframework.Once we posita
cated to their application for unsupervised community detection.                  community-basedgenerativemodelp(G|F)forthegraph,detecting
Severalmethodshavebeenproposed[7,10,43],buttheyallhave                            communities boils down to inferring the unobserved affiliation
a common drawback: they only focus on the special case of dis-                    matrixF given the observed graphG.
joint (non-overlapping) communities. However, it is well known                       Besidesthetraditionalprobabilisticview,onecanalsoviewcom-
that communitiesin real networksare overlapping[41]. Handling                     munitydetectionthroughthelensofrepresentationlearning.The
                                                                                  communityaffiliationmatrixF canbeconsideredasanembedding
Permission to make digital or hard copies of part or all of this work for personal orof nodes intoR C≥0, with the aim of preserving the graph structure.
classroom use isgranted without fee providedthat copies are notmade or distributedGiventherecentsuccessofrepresentationlearningforgraphs[5],
forprofitorcommercialadvantageandthatcopiesbearthisnoticeandthefullcitation
onthefirstpage.Copyrightsforthird-partycomponentsofthisworkmustbehonored.         a question arises: "Can the advances in deep learning for graphs
For all other uses, contact the owner/author(s).                                  be used to design better community detection algorithms?". As we
DLG’19, August 2019, Anchorage, Alaska, USA                                       show in Section 4.1, simply combining existing node embedding
© 2019 Copyright held by the owner/author(s).
                                                                                  approaches with overlapping K-means doesn’t lead to satisfactory

DLG’19,August2019,Anchorage,Alaska,USA                                                                          OleksandrShchurandStephanGünnemann
results. Instead, we propose to combine the probabilistic and repre-                (Section4.2).Also,suchformulationallowsustoseamlesslyincor-
sentation points of view, and learn the community affiliations in                   porate the node features into the model. If the node attributesX
an end-to-end manner using a graph neural network.                                  arenotavailable,wecansimplyuseAasnodefeatures[22].Finally,
                                                                                    with the formulation from Equation 2, it’s even possible to predict
3   THENOCDMODEL                                                                    communities inductively for nodes not seen at training time.
Here, we present the Neural Overlapping Community Detection
(NOCD) model. The core idea of our approach is to combine the                       3.3   Scalability
power of GNNs with the Bernoulli–Poisson probabilistic model.                       One advantage of the BP model is that it allows to efficiently eval-
                                                                                    uate the lossL(F)and its gradients w.r.t.F. By using a caching
3.1   Bernoulli–Poissonmodel                                                        trick [40], we can reduce the computational complexity of these
TheBernoulli–Poisson(BP)model[ 33,40,45]isagraphgenerative                          operations fromO(N2)toO(N +  M). While this already leads to
modelthat allowsforoverlapping communities.According tothe                          large speed-ups due to sparsity of real-world networks (typically
BP model, the graph is generated as follows. Given the affiliations                 M≪ N2 ), we can speed it up even further. Instead of using all
F∈R N×C≥0   , adjacency matrix entriesAuv are sampled i.i.d. as                     entries ofA when computing the loss (Equation 4), we sample a
                  Auv∼Bernoulli(1−exp(−Fu FTv))               (1)                   mini-batch ofS edges and non-edges at each training epoch, thus
                                                                                    approximatelycomputing∇LinO(S).InAppendixEweshowthat
whereFu  istherowvectorofcommunityaffiliationsofnodeu (the                          thisstochasticoptimizationstrategyconvergestothesamesolution
u’srowofthematrixF).Intuitively,themorecommunitiesnodes                             as the full-batch approach, while keeping the computational cost
u andv have incommon (i.e.the higherthe dot productFu FTv is),                      and memory footprint low.
the more likely they are to be connected by an edge.                                   Whilewesubsamplethegraphtoefficientlyevaluatethetraining
   Thismodel hasa numberof desirableproperties:It canproduce                        objectiveL(F), we use the full adjacency matrix inside the GNN.
variouscommunitytopologies(e.g.nested,hierarchical),leadsto                         This doesn’t limit the scalability of our model: NOCD is trained
denseoverlapsbetweencommunities[41]andiscomputationally                             on a graph with 800K+ edges in 3 minutes on a single GPU (see
efficient (Section 3.3). Existing works propose to perform infer-                   Section4.1).ItisstraightforwardtomaketheGNNcomponenteven
ence inthe BPmodel usingmaximum likelihood estimationwith                           more scalable by applying the techniques such as [8, 44].
coordinate ascent [40, 42] or Markov chain Monte Carlo [33, 45].                    4   EVALUATION
3.2   Modeldefinition                                                               Datasets. Weusethefollowingreal-worldgraphdatasetsinour
Instead of treating the affiliation matrixF as a free variable over                 experiments.Facebook [27] is a collection of small (50-800 nodes)
which optimization is performed, we generateF with a GNN:                           ego-networksfromtheFacebookgraph.Largergraphdatasets(10K+
                            F :=  GNNθ(A,X)                                (2)      nodes)withreliableground-truthoverlappingcommunityinforma-
                                                                                    tion and node attributes are not openly available, which hampers
A ReLU nonlinearity is applied element-wise to the output layer                     theevaluationanddevelopmentofnewmethods.Forthisreason
to ensure non-negativity ofF. See Section 4 and Appendix B for                      we havecollected and preprocessed 4 real-worlddatasets, that sat-
details about the GNN architecture.                                                 isfy thesecriteria andcan actas futurebenchmarks.Chemistry,
   The negative log-likelihood of the Bernoulli–Poisson model isÕÕ                  Computer Science, Medicine, Engineering are co-authorship
   −logp(A|F)=−              log(1−exp(−Fu FTv))+              Fu FTv     (3)       networks, constructed from the Microsoft Academic Graph [1].
                    (u ,v)∈E                           (u ,v)< E                    Communities correspond to research areas in respective fields, and
                                                                                    nodeattributesarebasedonkeywordsofthepapersbyeachauthor.
Real-world graphare usually extremelysparse, whichmeans that                        Statistics for all used datasets are provided in Appendix A.
the second term in Equation 3 will provide a much larger contri-                       Modelarchitecture.Forallexperiments,weusea2-layerGraph
bution tothe loss. We counteractthis by balancingthe two terms,                     Convolutional Network (GCN) [22] as the basis for the NOCD
which is a standard technique in imbalanced classification [18]hih    i             model. The GCN is defined as
 L(F)=−E(u ,v)∼P E      log(1−exp(−Fu FTv))+ E(u ,v)∼P N       Fu FTv    (4)                 F :=  GCNθ(A,X)=  ReLU(ˆAReLU(ˆAXW(1))W(2))     (6)
where PE  and PN   denote uniform distributions over edges and                      where  ˆA=   ˜D−1/2 ˜A ˜D−1/2 is thenormalized adjacencymatrix,  ˜A=
non-edges respectively.                                                             A+ IN   is the adjacency matrix with self loops, and  ˜Dii = Í         j  ˜Aij
   Instead of directly optimizing the affiliation matrixF, as done                  is the diagonaldegree matrix of  ˜A. We considered other GNN ar-
by traditional approaches [40, 42], we search for neural network                    chitectures,as wellasdeepermodels, butnoneofthem ledtoany
parametersθ ⋆  thatminimizethe(balanced)negativelog-likelihood                      noticeable improvements. The two main differences of our model
                    θ ⋆  =  argmin                                                  fromstandard GCNinclude(1) batchnormalizationafter thefirst
                               θ L(GNNθ(A,X))                   (5)                 graph convolution layer and (2) L2 regularization applied to all
   UsingaGNNforcommunitypredictionhasseveraladvantages.                             weightmatrices.Wefoundbothofthesemodificationstoleadto
First,duetoanappropriateinductivebias,theGNNoutputssim-                             substantial gains in performance. We optimized the architecture
ilar community affiliation vectors for neighboring nodes, which                     andhyperparameters onlyusingtheComputer Sciencedataset—
improves the quality of predictions compared to simpler models                      noadditionaltuningwasdoneforotherdatasets.Moredetailsabout

                                                                                                                             DLG’19,August2019,Anchorage,Alaska,USA
                Table1:Recoveryofground-truthcommunities,measuredbyNMI(in%).ResultsforNOCDareaveragedover50initializations
                (seeTable4forerrorbars).Bestresultforeachrowinbold.DNF—didnotfinishin12hoursorranoutofmemory.
                themodelconfigurationandthe trainingprocedureareprovided                                  To ensure a fair comparison, all methods were given the true
                inAppendixB.WedenotethemodelworkingonnodeattributesX                                  numberofcommunitiesC.Otherhyperparametersweresettotheir
                asNOCD-X, andthemodelusing theadjacencymatrixAasinput                                 recommended values. An overview of all baseline methods, aswell
                as NOCD-G. In both cases, the feature matrix is row-normalized.                       as their configurations are provided in Appendix C.
                   Assigning nodes to communities.  In order to compare the                               Results:Recovery.  Table 1 shows how well different methods
                detected communities to the ground truth, we first need to con-                       recovertheground-truthcommunities.EitherNOCD-XorNOCD-
                vertthepredictedcontinuouscommunityaffiliationsF intobinary                           G achieve the highest score for 9 out of 10 datasets. We found that
                community assignments. We assign nodeu to communityc if its                           theNMIofbothmethodsisstronglycorrelatedwiththereconstruc-
                affiliationstrengthFuc   isabovea fixedthresholdρ.Wechosethe                          tion loss (Equation 4): NOCD-G outperforms NOCD-X in terms
                thresholdρ=  0.5 like all other hyperparameters — by picking the                      of NMI exactly in those cases, when NOCD-G achieves a lower
                valuethatachievedthebestscoreontheComputerSciencedataset,                             reconstruction loss. This means that we can pick the better per-
                and thenusing itin furtherexperiments without additionaltuning.                       forming of two methods in a completely unsupervised fashion by
                   Metrics.  We found that popular metrics for quantifying agree-                     only considering the loss values.
                mentbetweentrueanddetected communities, suchasJaccardand                                  Results:Hyperparametersensitivity. It’sworthnotingagain
                F1 scores[26,40,42],cangivearbitrarilyhighscoresforcompletely                         that both NOCD models use the same hyperparameter config-
                uninformative community assignments. See Appendix F for an                            uration that was tuned only on the Computer Science dataset
                example and discussion. Instead we use overlapping normalized                         (N  =   22K, M  =   96.8K, D  =   7.8K). Nevertheless, both models
                mutualinformation(NMI)[28],asitismorerobustandmeaningful.                             achieve excellent results on datasets with dramatically different
                                                                                                      characteristics(e.g.Facebook414withN =  150, M =  1.7K, D =  16).
                                                                                                          Results: Scalability.   In addition to displaying excellent re-
                4.1   Recoveryofground-truthcommunities                                               covery results, NOCD is highly scalable. NOCD is trained on the
               We evaluate the NOCD model by checking how well it recovers                            Medicinedataset(63Knodes,810Kedges)usingasingleGTX1080Ti
                communities in graphs with known ground-truth communities.                            GPU in 3 minutes, while only using 750MB of GPU RAM (out of
                   Baselines.  In our selection of baselines, we chose methods                        11GB available). See Appendix D for more details on hardware.
                thatarebasedon differentparadigmsforoverlappingcommunity                                  EPM, SNetOC and CDE don’t scale to larger datasets, since they
                detection:probabilisticinference,non-negativematrixfactorization                      instantiateverylargedensematricesduringcomputations.SNMF
                (NMF)anddeeplearning.Somemethodsincorporatetheattributes,                             and BigCLAM, while being the most scalable methods and having
               while other rely solely on the graph structure.                                        lowerruntimethanNOCD,achievedrelativelylowscoresinrecov-
                   BigCLAM [40], EPM [45] and SNetOC [33] are based on the                            ery.GeneratingtheembeddingswithDeepWalkandGraph2Gauss
                Bernoulli–Poissonmodel.BigCLAMlearns F usingcoordinateas-                             canbedoneveryefficiently.However,overlappingclusteringofthe
                cent,whileEPMandSNetOCperforminferencewithMarkovchain                                 embeddings with NEO-K-Means wasthe bottleneck, which led to
                Monte Carlo (MCMC). CESNA [42] is an extension of BigCLAM                             runtimesexceedingseveralhoursforthelargedatasets.Astheau-
                that additionally models node attributes. SNMF [36] and CDE [26]                      thorsofCESNApointout[42],themethodscalestolargegraphsif
                are NMF approaches for overlapping community detection.                               thenumberofattributesD islow.However,asD increases,whichis
                   We additionally implemented two methods based on neural                            commonformoderndatasets,themethodscalesratherpoorly.This
                graphembedding.First,wecomputenodeembeddingsforallthe                                 is confirmed by our findings — on the Medicine dataset, CESNA
                nodesinthegivengraphusingtwoestablishedapproaches–Deep-                               (parallel version with 18 threads) took 2 hours to converge.
               Walk[29] and Graph2Gauss[4]. Graph2Gauss takesinto account
                bothnodefeaturesandthegraphstructure,whileDeepWalkonly                                4.2   Dowereallyneedagraphneuralnetwork?
                usesthestructure.Then,weclusterthenodesusingNon-exhaustive
                Overlapping(NEO)K-Means[37]—whichallowstoassignthemto                                 Our GNN-based model achieved superior performance in com-
                overlapping communities. We denote the methods based on Deep-                         munityrecovery.Intuitively,itmakessense touseaGNNforthe
               Walk and Graph2Gauss as DW/NEO and G2G/NEO respectively.                               reasonslaidoutinSection3.2.Nevertheless,weshouldaskwhether
Dataset                    BigCLAM    CESNA    EPM    SNetOC    CDE    SNMF    DW/NEO    G2G/NEO    NOCD-G    NOCD-X
Facebook348                    26.0         29.4       6.5          24.0     24.8       13.5             31.2             17.2            34.7          36.4
Facebook414                    48.3         50.3     17.5          52.0     28.7       32.5             40.9             32.3            56.3          59.8
Facebook686                    13.8         13.3       3.1          10.6     13.5       11.6             11.8               5.6            20.6          21.0
Facebook698                    45.6         39.4       9.2          44.9     31.6       28.0             40.1               2.6          49.3            41.7
Facebook1684                   32.7         28.0       6.8          26.1     28.8       13.0           37.2               9.9            34.7            26.1
Facebook1912                   21.4         21.2       9.8          21.4     15.5       23.4             20.8             16.0          36.8            35.6
Chemistry                           0.0         23.3    DNF         DNF    DNF         2.6              1.7             22.8            22.6          45.3
ComputerScience               0.0         33.8    DNF         DNF    DNF         9.4              3.2             31.2            34.2          50.2
Engineering                        7.9         24.3    DNF         DNF    DNF       10.1              4.7             33.4            18.4          39.1
Medicine                             0.0         14.4    DNF         DNF    DNF         4.9              5.5             28.8            27.4          37.8

DLG’19,August2019,Anchorage,Alaska,USA                                                                          OleksandrShchurandStephanGünnemann
Table2:ComparisonoftheGNN-basedmodelagainstsimplerbaselines.Multilayerperceptron(MLP)andFreeVariable(FV)
modelsareoptimizingthesameobjective(Equation4),butrepresentthecommunityaffiliationsF differently.
                                                              Attributes                        Adjacency
                              Dataset                          GNN             MLP             GNN            MLP        Free variable
                              Facebook 348           36.4±2.0     11.7±2.7     34.7±1.5     27.7±1.6       25.7±1.3
                              Facebook 414           59.8±1.8     22.1±3.1     56.3±2.4     48.2±1.7       49.2±0.4
                              Facebook 686           21.0±0.9     1.5±0.7      20.6±1.4     19.8±1.1       13.5±0.9
                              Facebook 698            41.7±3.6      1.4±1.3     49.3±3.4    42.2±2.7       41.5±1.5
                              Facebook 1684          26.1±1.3     17.1±2.0    34.7±2.6    31.9±2.2       22.3±1.4
                              Facebook 1912          35.6±1.3     17.5±1.9    36.8±1.6    33.3±1.4       18.3±1.2
                              Chemistry                 45.3±2.3    46.6±2.9     22.6±3.0     12.1±4.0        5.2±2.3
                              Computer Science    50.2±2.0     49.2±2.0     34.2±2.3     31.9±3.8       15.1±2.2
                              Engineering              39.1±4.5    44.5±3.2     18.4±1.9     15.8±2.1        7.6±2.2
                              Medicine                 37.8±2.8     31.8±2.1     27.4±2.5     23.6±2.1        9.4±2.3
it’spossibleachievecomparableresultswithasimplermodel.To                            5   RELATEDWORK
answer this question, we consider the following two baselines.                      Theproblem ofcommunitydetectioningraphs iswell-established
   Multilayerperceptron(MLP):Instead ofa GCN (Equation 6),                          in the research literature. However, most of the works study detec-
we use a simple fully-connected neural network to generateF.                        tion of non-overlapping communities [3, 35]. Algorithms for over-
                                                                                    lapping community detection can be broadly divided into methods
             F =  MLPθ(X)=  ReLU(ReLU(XW(1))W(2))           (7)                     basedonnon-negativematrixfactorization[23,26,36],probabilistic
                                                                                    inference [24, 33, 40, 45], and heuristics [12, 15, 25, 31].
Thisisrelatedtothemodelproposedby[19].SameasfortheGCN-                                 Deep learning for graphs can be broadly divided into two cat-
basedmodel,weoptimizetheweightsoftheMLP,θ  ={W(1),W(2)},                            egories:graphneuralnetworksandnodeembeddings.GNNs[17,
to minimize the objective Equation 4.                                               22,39]arespecializedneuralnetworkarchitecturesthatcanoper-
                                                                                    ate on graph-structured data. The goal of embedding approaches
                            minθ L(MLPθ(X))                             (8)         [4, 16, 21, 29] is to learn vector representations of nodes in a graph
                                                                                    that can then be used for downstream tasks. While embedding
   Freevariable(FV): Asanevensimplerbaseline,weconsider                             approaches work well for detecting disjoint communities [7, 34],
treatingF as a free variable in optimization and solve                              theyarenotwell-suitedforoverlappingcommunitydetection,as
                                                                                    we showed in our experiments. This is caused by lack of reliable
                                 min                                                and scalable approaches for overlapping clustering of vector data.
                                 F≥0L(F)                                      (9)      Several works have proposed deep learning methods for com-
We optimize the objective using projected gradient descent with                     munity detection. [43] and [6] use neural nets to factorize the
Adam[20],andupdatealltheentriesofF ateachiteration.Thiscan                          modularitymatrix,while[7]jointlylearnsembeddingsfornodes
be seen as an improved version of the BigCLAM model. Original                       and communities. However, neither of these methods can handle
BigCLAMusestheimbalancedobjective(Equation3)andoptimizes                            overlappingcommunities.Alsorelatedtoourmodelistheapproach
F using coordinate ascent with backtracking line search.                            by [19], where theyuse adeepbelief networkto learncommunity
   Setup.  Both for the MLP and FV models, we tuned the hyper-                      affiliations. However, their neural network architecture does not
parameters on the Computer Science dataset (just as we did for                      use the graph, which we have shown to be crucial in Section 4.2;
the GNN model), and used the same configuration for all datasets.                   and,justlikeEPMandSNetOC,reliesonMCMC,whichheavilylim-
Details about the configuration for both models are provided in                     its the scalability of their approach. Lastly, [9] designed a GNN for
AppendixB.Likebefore,weconsiderthevariantsoftheGNN-based                            supervised communitydetection,whichisaverydifferentsetting.
andMLP-basedmodelsthatuseeitherX orAasinputfeatures.We
comparetheNMIscoresobtainedbythemodelsonall11datasets.                              6   DISCUSSION&FUTUREWORK
   Results.  The results for all models are shown in Table 2. The                   WeproposedNOCD—agraphneuralnetworkmodelforoverlap-
twoneuralnetworkbasedmodelsconsistentlyoutperformthefree                            ping community detection. The experimental evaluation confirms
variablemodel.Whennode attributesX areused,the MLP-based                            that the model is accurate, flexible and scalable.
modeloutperformstheGNNversionforChemistryandEngineering                                Besides strong empirical results, our work opens interesting
datasets, where the node features alone provide a strong signal.                    follow-upquestions.Weplantoinvestigatehowthetwoversions
However,MLPachievesextremelylowscoresforFacebook686and                              of our model (NOCD-X and NOCD-G) can be used to quantify the
Facebook 698 datasets, where the attributes are not as reliable. On                 relevance of attributes to the community structure. Moreover, we
the other hand, whenA is used as input, the GNN-based model                         plan to assess the inductive performance of NOCD [17].
always outperforms MLP. Combined, these findings confirm our                           Tosummarize,theresultsobtainedinthispaperprovidestrong
hypothesisthatagraph-basedneuralnetworkarchitectureisindeed                         evidencethatdeeplearningforgraphsdeservesmoreattentionas
beneficial for the community detection task.                                        a framework for overlapping community detection.

                                                                                                                                     DLG’19,August2019,Anchorage,Alaska,USA
ACKNOWLEDGMENTS                                                                                          [32]  Jimin Song and Mona Singh. 2009. How and when should interactome-derived
This research was supported by the German Research Foundation,                                                 clusters be used to predict functional modules and protein function? Bioinfor-
                                                                                                               matics 25 (2009).
Emmy Noether grant GU 1409/2-1.                                                                          [33]  AdrienTodeschini, XeniaMiscouridou,and FrançoisCaron. 2016. Exchangeable
                                                                                                               random measures for sparse and modular graphs with overlapping communities.
                                                                                                               arXiv:1602.02114 (2016).
REFERENCES                                                                                               [34]  AntonTsitsulin,DavideMottin,PanagiotisKarras,andEmmanuelMüller.2018.
                                                                                                               VERSE: Versatile Graph Embeddings from Similarity Measures. In WWW.
 [1][n. d.]. Microsoft Academic Graph. https://kddcup2016.azurewebsites.net/.                            [35]  Ulrike Von Luxburg. 2007.   A tutorial on spectral clustering.   Statistics and
 [2]  Martín Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey                               computing 17 (2007).
      Dean,MatthieuDevin,SanjayGhemawat,GeoffreyIrving,MichaelIsard,etal.                                [36]  Fei Wang, Tao Li, Xin Wang, Shenghuo Zhu, and Chris Ding. 2011. Community
      2016. Tensorflow:Asystemforlarge-scalemachinelearning.In12th{USENIX}                                     discovery using nonnegative matrix factorization. Data Mining and Knowledge
      Symposium on Operating Systems Design and Implementation ({OSDI}16).                                     Discovery 22 (2011).
 [3]  Emmanuel Abbe. 2018.  Community Detection and Stochastic Block Models:                             [37]  Joyce Jiyoung Whang, Inderjit S Dhillon, and David F Gleich. 2015.   Non-
      Recent Developments. JMLR 18 (2018).                                                                     exhaustive, Overlapping k-means. In SDM.
 [4]  Aleksandar Bojchevski and Stephan Günnemann. 2018. Deep Gaussian Embed-                            [38]  Jierui Xie,Stephen Kelley, andBoleslaw K Szymanski.2013. Overlappingcom-
      ding of Graphs: Unsupervised Inductive Learning via Ranking. In ICLR.                                    munitydetectioninnetworks:Thestate-of-the-artandcomparativestudy. CSUR
 [5]  HongyunCai,VincentWZheng,andKevinChang.2018.Acomprehensivesurvey                                         45 (2013).
      of graph embedding: problems, techniques and applications. TKDD (2018).                            [39]  Keyulu  Xu,  Chengtao  Li,  Yonglong  Tian,  Tomohiro  Sonobe,  Ken-ichi
 [6]  Jinxin Cao, Di Jin, Liang Yang, and Jianwu Dang. 2018. Incorporating network                             Kawarabayashi, and Stefanie Jegelka. 2018. Representation learning on graphs
      structure with node contents forcommunitydetection on large networks using                               with jumping knowledge networks. ICML (2018).
      deep learning. Neurocomputing 297 (2018).                                                          [40]  JaewonYangandJureLeskovec.2013. Overlappingcommunitydetectionatscale:
 [7]  SandroCavallari,VincentW Zheng,HongyunCai,KevinChen-ChuanChang,                                          a nonnegative matrix factorization approach. In WSDM.
      and Erik Cambria. 2017.   Learning community embedding with community                              [41]  Jaewon Yang and Jure Leskovec. 2014. Structure and Overlaps of Ground-Truth
      detection and node embedding on graphs. In CIKM.                                                         Communities in Networks. ACM TIST 5 (2014).
 [8]  JianfeiChen,JunZhu,andLeSong.2018. StochasticTrainingofGraphConvolu-                               [42]  JaewonYang,JulianMcAuley,andJureLeskovec.2013. Communitydetectionin
      tional Networks with Variance Reduction.. In ICML.                                                       networks with node attributes. In ICDM.
 [9]  Zhengdao Chen, Xiang Li, and Joan Bruna. 2019. Supervised Community Detec-                         [43]  LiangYang,XiaochunCao,DongxiaoHe,ChuanWang,XiaoWang,andWeixiong
      tion with Hierarchical Graph Neural Networks. In ICLR.                                                   Zhang.2016. ModularityBasedCommunityDetectionwith DeepLearning..In
[10]  Jun Jin Choong, Xin Liu, and Tsuyoshi Murata. 2018.  Learning community                                  IJCAI.
      structure with variational autoencoder. In ICDM.                                                   [44]  RexYing,RuiningHe,KaifengChen,PongEksombatchai,WilliamLHamilton,
[11]  Santo Fortunato and Darko Hric. 2016. Community detection in networks: A                                 and JureLeskovec. 2018. Graph ConvolutionalNeural Networksfor Web-Scale
      user guide. Physics Reports 659 (2016).                                                                  Recommender Systems.
[12]  EstherGalbrun,AristidesGionis,andNikolajTatti.2014.Overlappingcommunity                            [45]  MingyuanZhou.2015. Infiniteedgepartitionmodelsforoverlappingcommunity
      detection in labeled graphs. Data Mining and Knowledge Discovery 28 (2014).                              detection and link prediction. In AISTATS.
[13]  Javier O Garcia, Arian Ashourvan, Sarah Muldoon, Jean M Vettel, and Danielle S
      Bassett. 2018. Applications of communitydetection techniques to braingraphs:
      Algorithmicconsiderationsandimplicationsforneuralfunction. Proc. IEEE 106
      (2018).
[14]  MichelleGirvanandMarkEJNewman.2002. Communitystructureinsocialand
      biological networks. PNAS 99 (2002).
[15]David FGleich andC Seshadhri.2012. Vertex neighborhoods,low conductance
      cuts, and good seeds for local community methods. In KDD.
[16]Aditya Grover and Jure Leskovec. 2016. node2vec: Scalable feature learning for
      networks. In KDD.
[17]  Will Hamilton, Zhitao Ying,and Jure Leskovec. 2017. Inductive representation
      learning on large graphs. In NIPS.
[18]  HaiboHeandEdwardoAGarcia.2008. Learningfromimbalanceddata. TKDE 9
      (2008).
[19]  ChangweiHu,PiyushRai,andLawrenceCarin.2017. DeepGenerativeModels
      for Relational Data with Side Information. ICML.
[20]  Diederik P Kingma and Jimmy Ba. 2015. Adam: A method for stochastic opti-
      mization. ICLR (2015).
[21]  Thomas N Kipf andMax Welling.2016. Variational Graph Auto-Encoders. NIPS
      Workshop on Bayesian Deep Learning.
[22]  ThomasNKipfandMaxWelling.2017. Semi-supervisedclassificationwithgraph
      convolutional networks. ICLR.
[23]  DaKuang,ChrisDing,andHaesunPark.2012. Symmetricnonnegativematrix
      factorization for graph clustering. In SDM.
[24]  PierreLatouche,EtienneBirmelé,ChristopheAmbroise,etal.2011. Overlapping
      stochasticblockmodelswithapplicationtotheFrenchpoliticalblogosphere. The
      Annals of Applied Statistics 5 (2011).
[25]  YixuanLi,KunHe,DavidBindel,andJohnEHopcroft.2015.Uncoveringthesmall
      community structure in large networks: A local spectral approach. In WWW.
[26]  YeLi,ChaofengSha,XinHuang,andYanchunZhang.2018.CommunityDetection
      in Attributed Graphs: An Embedding Approach. In AAAI.
[27]  Julian Mcauley and Jure Leskovec. 2014. Discovering social circles in ego net-
      works. TKDD 8 (2014).
[28]  AaronFMcDaid,DerekGreene,andNeilHurley.2011. Normalizedmutualinfor-
      mation to evaluate overlapping community finding algorithms. arXiv:1110.2515
      (2011).
[29]  BryanPerozzi,RamiAl-Rfou,andStevenSkiena.2014.Deepwalk:Onlinelearning
      of social representations. In KDD.
[30]  CarlosAndréReisPinheiro.2012. Communitydetectiontoidentifyfraudevents
      in telecommunications networks. SAS SUGI proceedings: customer intelligence
      (2012).
[31]  YiyeRuan,DavidFuhry,andSrinivasanParthasarathy.2013. Efficientcommunity
      detection in large networks using content and links. In WWW.

               DLG’19,August2019,Anchorage,Alaska,USA                                                                          OleksandrShchurandStephanGünnemann
               A   DATASETS                                                                         ensurethatitstaysnon-negative:Fuc  =  max{0, Fuc}.We usethe
                                                                                                    same early stopping strategy as for the GNN and MLP models.
                         Table3:Datasetstatistics.K standsfor1000.                                  C   BASELINES
                                                                                                    Table4:Overviewofthebaselines.Seetextforthediscussion
                                                                                                    ofscalabilityofCESNA.
                                                                                                       Method                    Model type            Attributed    Scalable
                                                                                                       BigCLAM [40]         Probabilistic                   ✓
                                                                                                       CESNA [42]             Probabilistic         ✓        ✓ ∗
                                                                                                       SNetOC [33]            Probabilistic
                                                                                                       EPM [45]                  Probabilistic
                                                                                                       CDE [26]                  NMF               ✓
                                                                                                       SNMF [36]               NMF                          ✓
                                                                                                       DW/NEO [29, 37]    Deep learning
               B   MODELCONFIGURATION                                                                  G2G/NEO [4, 37]     Deep learning       ✓
               B.1   Architecture                                                                      NOCD                      Deep learning +probabilistic         ✓        ✓
               We pickedthe hyperparametersand chosethe modelarchitecture
               forall3modelsbyonlyconsideringtheirperformance(NMI)onthe
               ComputerSciencedataset.Noadditionaltuningforotherdatasets                                 •We used the reference C++ implementations of BigCLAM
               has been done.                                                                               and CESNA, that were provided by the authors (https://
                  GNN-basedmodel.(Equation6) Weusea 2-layergraph con-                                       github.com/snap-stanford/snap).Modelswereusedwiththe
               volutionalneuralnetwork,withhiddensizeof128,andtheoutput                                     default parameter settings for step size, backtracking line
               (second) layer of size C (number of communities to detect). We                               search constants, and balancing terms. Since CESNA can
               apply batchnormalization afterthe firstgraph convolution layer.                              only handle binary attributes, we binarize the original at-
               Dropout with 50% keep probability is applied before every layer.                             tributes(setthenonzeroentriesto1)iftheyhaveadifferent
               We add weight decay toboth weight matrices with regularization                               type.
               strengthλ=  10−2.ThefeaturematrixX (orA,incasewearework-                                  •We implemented SNMF ourselvesusing Python. TheF ma-
               ing without attributes) is normalized such that every row has unit                           trix isinitialized by samplingfrom the Uniform[0, 1]distri-
               L2-norm.                                                                                     bution.Werunoptimizationuntil theimprovement inthe
                  We also experimented with the Jumping Knowledge Network                                   reconstructionlossgoesbelow 10−4 periteration,orfor300
               [39]andGraphSAGE[17]architectures,buttheyledtolowerNMI                                       epochs,whicheverhappens first.The resultsfor SNMFare
               scores on the Computer Science dataset.                                                      averaged over 50 random initializations.
                  MLP-basedmodel.(Equation7) WefoundtheMLP modelto                                       •Weuse theMatlabimplementation ofCDE providedby the
               performbestwiththesameconfigurationasdescribedaboveforthe                                    authors.Wesetthehyperparameterstoα =  1,β=  2,κ=  5,
               GCNmodel(i.e.sameregularizationstrength,hiddensize,dropout                                   asrecommendedinthepaper,andrunoptimizationfor20
               and batch norm).                                                                             iterations.
                  Freevariablemodel(Equation9)Weconsideredtwoinitial-                                    •ForSNetOCandEPMweusetheMatlabimplementations
               ization strategiesfor the free variablemodel: (1) Locally minimal                            provided by the authors with the default hyperparameter
               neighborhoods [15] — the strategy used by the BigCLAM and                                    settings. The implementation of EPM provides to options:
               CESNA modelsand (2) initializingF to theoutput of anuntrained                                EPM and HEPM. We found EPM to produce better NMI
               GCN. We found strategy (1) to consistently provide better results.                           scores, so we used it for all the experiments.
               B.2   Training                                                                            •We use the TensorFlow implementation of Graph2Gauss
                                                                                                            provided by the authors. We set the dimension of the em-
               GNN- and MLP-based models.   We train both models using                                      beddings to 128, and only use the µ  matrix as embeddings.
               Adam optimizer [20] with default parameters. The learning rate is                         •WeimplementedDeepWalkourselves:Wesample10random
               set to 10−3. We use the following early stopping strategy: Every                             walks of length 80 from each node, and use the Word2Vec
               50 epochs we compute the full training loss (Equation 4). We stop                            implementation from Gensim (https://radimrehurek.com/
               optimization ifthere was noimprovement inthe lossfor the last                                gensim/)to generatethe embeddings.Thedimension ofem-
               10×50 =  500 iterations, orafter 5000epochs, whichever happens                               beddings is set to 128.
               first.                                                                                    •ForNEO-K-Means,weusetheMatlabcodeprovidedbythe
                  Freevariablemodel.  We use Adam optimizer with learning                                   authors.Welettheparametersα andβbeselectedautomat-
               rate 5·10−2.After everygradient step,we projecttheF matrixto                                 ically using the built-in procedure.
Dataset                 Networktype         N           M        D    C
Facebook348          Social                     224       3.2K        21   14
Facebook414          Social                     150       1.7K        16     7
Facebook686          Social                     168       1.6K         9   14
Facebook698          Social                       61         270         6    13
Facebook1684         Social                     786     14.0K        15   17
Facebook1912         Social                     747     30.0K        29   46
ComputerScience   Co-authorship     22.0K      96.8K    7.8K    18
Chemistry               Co-authorship     35.4K    157.4K    4.9K    14
Medicine                 Co-authorship     63.3K    810.3K    5.5K    17
Engineering            Co-authorship     14.9K      49.3K    4.8K    16

                                                                                                     DLG’19,August2019,Anchorage,Alaska,USA
D   HARDWAREANDSOFTWARE                                                        defined as    Õ                             Õ
The experimentswere performed ona computer runningUbuntu                               1          max                           max
16.04LTS with 2x Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz                          2|S∗|S∗i∈S∗S j∈Sδ(S∗i  , Sj)+     12|S|S j∈SS∗i∈S∗δ(S∗i  , Sj)    (10)
CPUs,256GBofRAMand4xGTX1080TiGPUs.Notethattraining                             whereδ(S∗i  , Sj)is a similarity measure between sets, such as F1-
and inference were done using only a single GPU at a time for                  score or Jaccard similarity.
all models.The NOCD modelwas implemented usingTensorflow                          We discoveredthat thesefrequently usedmeasures canassign
v1.1.2 [2]                                                                     arbitrarily high scores to completely uninformative community
E   CONVERGENCEOFTHESTOCHASTIC                                                 assignments, as you can in see in the following simple example.
     SAMPLINGPROCEDURE                                                         Let the ground truth communities beS∗1 ={v1, ..., vK}andS∗2 =
                                                                                {vN−K   , ..., vN}(K nodesineachcommunity),andletthealgorithm
Instead of using all pairsu,v∈V when computing the gradients                   assignall thenodesto asingle communityS1 =  V ={v1, ..., vN}.
∇θLat every iteration, we sampleS edges andS non-edges uni-                    Whilethispredictedcommunityassignmentiscompletelyuninfor-
formly atrandom. We perform thefollowing experimentto ensure                   mative,itwillachievesymmetricF1-scoreof    2KN  + K   andsymmetric
that our training procedure converges to the same result, as when              Jaccard similarity of K
using the full objective.                                                      willbe 75%and 60%respectively).ThesehighnumbersmightgiveN   (e.g., ifK =  600 and N =  1000, the scores
   Experimental setup.  We train the model on the Computer                     afalseimpressionthatthealgorithmhaslearnedsomethinguseful,
Science dataset and compare the full-batch optimization procedure              whilethatclearlyisn’tthecase.Asanalternative,wesuggestusing
withstochasticgradientdescentfordifferentchoicesofthebatch                     overlappingnormalizedmutualinformation(NMI),asdefinedin
sizeS. Starting from the same initialization, we measure the full              [28].NMIcorrectlyhandlesthedegeneratecases,liketheoneabove,
loss (Equation 4) over the iterations.                                         and assigns them the score of 0.
   Results.Figure1showstrainingcurvesfordifferentbatchsizes
S∈{1000, 2500, 5000, 10000, 20000}, as well as for full-batch train-
ing.Thehorizontalaxisoftheplotdisplaysthenumberofentries
ofadjacencymatrixaccessed.Oneiterationofstochastictraining
accesses 2S entries Aij, and one iteration of full-batch accesses
2N +  2M entries,sinceweareusingthecachingtrickfrom[40].As
we see, the stochastic training procedure is stable: For batch sizes
S =  10K andS =  20K, the loss converges very closely to thevalue
achieved by full-batch training.
Figure1:Convergenceofthestochasticsamplingprocedure.
F   QUANTIFYINGAGREEMENTBETWEEN
     OVERLAPPINGCOMMUNITIES
Apopularchoiceforquantifyingtheagreementbetweentrueand
predicted overlapping communities is the symmetric agreement
score[26,40,42].Giventheground-truthcommunitiesS∗={S∗i}i
andthepredictedcommunitiesS={Sj}j,thesymmetricscoreis

