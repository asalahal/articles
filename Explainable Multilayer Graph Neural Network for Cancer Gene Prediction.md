  EXPLAINABLE MULTILAYER GRAPH NEURAL NETWORK FOR
                                     CANCER GENE PREDICTION
                         Michail Chatzianastasis                                         Michalis Vazirgiannis
                    LIX, École Polytechnique, IP Paris                            LIX, École Polytechnique, IP Paris
                              Palaiseau, France                                             Palaiseau, France
         michail.chatzianastasis@polytechnique.edu                               mvazirg@lix.polytechnique.fr
                                                          Zijun Zhang
                                        Division of Artiﬁcial Intelligence in Medicine
                                       Cedars-Sinai Medical Center, Los Angeles, CA
                                                   zijun.zhang@cshs.org
                                                          ABSTRACT
          Theidentiﬁcationof cancergenesisacriticalyet challengingproblemincancergenomics research.
          Existingcomputationalmethods,includingdeepgraphneuralnetworks,failtoexploitthemultilayered
          gene-gene interactions or provide limited explanation for their predictions.  These methods are
          restricted to a single biological network, which cannot capture the full complexity of tumorigenesis.
          Modelstrainedondifferentbiologicalnetworksoftenyielddifferentandevenoppositecancergene
          predictions, hindering their trustworthy adaptation. Here, we introduce an Explainable Multilayer
          GraphNeuralNetwork(EMGNN)approachtoidentifycancergenes byleveragingmultiplegene-
          gene interaction networks and pan-cancer multi-omics data. Unlike conventional graph learning on a
          single biological network, EMGNN uses a multilayered graphneural network to learn frommultiple
          biological networks for accurate cancer gene prediction. Our method consistently outperforms all
          existing methods, with an average 7.15% improvement in area under the precision-recall curve
          (AUPR)overthecurrentstate-of-the-artmethod. Importantly,EMGNNintegratedmultiplegraphsto
          prioritizenewlypredictedcancergeneswithconﬂictingpredictionsfromsinglebiologicalnetworks.
          For each prediction, EMGNN provided valuable biological insights via both model-level feature
          importanceexplanationsandmolecular-levelgenesetenrichmentanalysis. Overall,EMGNNoffers
          a powerful new paradigm of graph learning through modeling the multilayered topological gene
          relationships and provides a valuable tool for cancer genomics research.
1   Introduction
Understanding the precise function and disease pathogenicity of a gene is dependent on the target gene’s properties,
as well as its interaction partners in a disease-speciﬁc context [36, 47, 12]. High-throughput experiments, such as
whole-genome sequencing and RNA sequencing of bulk and single-cell assays, have enabled unbiased proﬁling of
geneticandmolecularpropertiesforallgenesacrossthegenome. Experimentalmethodstoprobebothphysical[30,3]
andgenetic interactions[27, 8]providevaluableinsights ofthefunctional relevancebetween apairof genes. Basedon
these data, computational methods have been developed to predict gene functions for understudied and uncharacterized
genes by combining the gene’s property with its network connectivity patterns [2, 17, 29]. However, the prediction
ofgenepathogenicityindisease-speciﬁccontextsischallenging. Functionalassaysdescribingthegeneanditsgene
network arerelevantto disease onlyto the degreeto which themeasured property correlateswith disease physiology
[28]; while our understanding of complex disease physiology is poor, even fordiseases with large sample size and data
modalities, such as cancer [24].
As the completeness of known cancer genes is questioned, predicting novel cancer genes remains a crucial task in
cancer genomics research. These genes, which are often mutated or aberrant expressed in cancer cells, play a key

role in the development and progression of the disease [39]. Large-scale cancer sequencing consortia projects have
generated genomic and molecular proﬁling data for a variety of cancer types, providing an information-rich resource
for identifying novel cancer genes. Building on the hypothesis that pan-cancer multi-omic modalities provide critical
information to cancer gene pathogenicity, a pioneering work EMOGI [35] innovatively modeled the multi-omics
features of cancer genes in Protein-Protein interaction (PPI) networks to predict novel cancer genes. To address the
challenge of functional properties irrelevant to cancer disease physiology, EMOGI featurized each gene by a vector
summarizingmulti-omicsdatalevelsacrossvariouscancertypesinTheCancerGenomeAtlas(TCGA)[43]. EMOGI
then modeled the gene-gene interactions from pre-deﬁned generic PPI networks using a Graph Convolution Neural
network (GCN).When trained ona set ofhigh-conﬁdence cancer-and non-cancer genes,EMOGI identiﬁed165   newly
predicted cancer genes without necessarily recurrent alterations, but interact with known cancer genes.
A major limitation of EMOGI is that it didn’t address the disease physiology relevance in the pre-deﬁned graph
topology and connectivity patterns. EMOGI employed six different pre-deﬁned graphs, including genetic-focused
networks suchas Multinet [19], andgeneric protein interactionnetworks such asSTRING-db [40]. Among EMOGI
models trained on different PPI networks, we found an average standard deviation of 25.2%    in unlabelled cancer
genepredictions,demonstratingthenewlypredictedcancergenesweredifferentwhenusingdifferentPPInetworks.
Thus, a trustworthy adaptation of the EMOGI method’s output is challenging because conﬂicting prediction results
are ubiquitous. Furthermore, as cancer disease physiology is complex, using a single predeﬁned graph to represent
the gene-gene relationships cannot fully capture its molecular landscape; therefore, more sophisticated, data-driven
methods are needed to decipher the gene relationships in disease-speciﬁc contexts.
Here,wepropose anovel graphlearningframework, EMGNN(ExplainableMultilayerGraphNeural Network),for
predicting gene pathogenicity based on multiple input biological graphs. EMGNN maximizes the concordance of
functionalgenerelationshipswiththeunknowndiseasephysiologybyjointlymodelingthemultilayeredgraphstructure.
Weevaluatedtheperformanceof EMGNNinpredictingcancergenesusingthe samecompileddatasetsasEMOGIand
showed that our proposed method achieves state-of-the-art performance by combining information from all six PPI
networks. Furthermore, we explained EMGNN’s prediction by both model-level integrated gradients and molecular-
levelgene pathways. By examiningnewly predictedcancergenes identiﬁedby EMGNN,wedemonstrated biological
insightscanberevealedbyleveragingthecomplementaryinformationindifferenttypesofbiologicalnetworks. Overall,
EMGNN provides a powerful new paradigm of graph learning through modeling the multilayered topological gene
relationships. Our key contributions can be summarized as follows:
        •  WedevelopanExplainableMultilayerGraphNeuralNetwork(EMGNN)approachtoidentifycancergenesby
          leveraging multiple protein-protein interaction networks and multi-omics data.
        •  Ourmethoddemonstratessuperiorperformancecomparedtoexistingapproachesasquantiﬁedbyasigniﬁcant
          increasein theAUPRC acrosssix PPInetworks. Theaverage improvementin performanceis 7.15%overthe
          current state-of-the-art method, EMOGI.
        •  We identify the most important multi-omics features for the prediction of each cancer gene, as well as the
          most inﬂuential PPI networks, using model interpretation strategies.
        •  EMGNNidentiﬁes newly predictedcancer genesby integrating multiplePPI networks,providing auniﬁed
          and robust prediction for novel cancer genes discovery. Our code is publicly available on GitHub1
2   Results
2.1   Overview of EMGNN model
Toeffectivelymodelmultilayeredgraphstructuresforcancergeneprediction,wedevelopedagraphneuralnetwork
(GNN) model EMGNN that jointly learned from multiple biological networks as inputs. The extension of GNNs to
handle multiple networks is a non-trivial task, as they are designed to operate on a single graph. The input for EMGNN
is multiple graphs where each graph describes gene-gene relationships by an adjacency matrix, and a feature matrix for
genesthatissharedbymultiplegraphs. Theoutputisabinarynodeclassiﬁcationofputativecancervsnon-cancergene.
EMGNNachievesmultilayeredgraphmodelinginathree-stepapproach(Figure1). Intheﬁrst step,EMGNNupdates
thenoderepresentationwithineachgraphlayer(i.e.,asinglebiologicalnetwork)byasharedmessagepassingoperator.
As different layer-wise graphs have distinct connectivity patterns, node representations will be updated differently in
eachlayer. The sharedmessage passingoperator allows theEMGNN modelto incorporatenewgraphs asinputs, while
keeping the model’s trainable parameters ﬁxed.
   1Code: https://github.com/zhanglab-aim/EMGNN
                                                                  2

                      Figure 1: An illustration of our proposed Explainable Multilayer Graph Neural Network (EMGNN) approach. The
                      model consists of three main steps: (1) Apply a shared GNN to update the node representation matrix of each input
                      graph;(2)Constructametagraphforeachgene,wherethesamegenesacrossallgraphsareconnectedtoametanode,
                      andupdatetherepresentationofthemetanodeswithasecondGNN,MetaGNN;(3)Useamulti-layerperceptronto
                      predict the class of each meta node.
                            Table 1: Test AUPRC values and standard deviation across different PPI networks across ﬁve different runs.
                      EMGNNthenintegratesthelayer-wisenoderepresentationsofthesamegenesacrossmultiplebiologicalnetworksbya
                      metagraph (Methods). For eachgene, themeta-graph consistsof ameta-nodefor thisgene, whichreceives messages
                      acrossindividualnetworksfromthegivengene’srepresentations. Thismessagepassingstepismodeledwithasecond
                      GNN, referred to as the Meta GNN (Figure 1). Meta GNN enables directed message-passing to combine and exchange
                      information from different networks to the meta nodes, which will contain the ﬁnal representations of the genes. In
                      the last step, a multi-layer perceptron (MLP) takes the meta node representations as input, and performs the node
                      classiﬁcation task.
                      Notably, our EMGNN model is a generalized, multilayered form of a single graph GNN. In the special cases where
                      multilayered graphs have identical adjacency matrices, or only one graph is provided as inputs, EMGNN reduces to
                      a standard single-graph GNN, where the shared message passing operators are standard GNN operators, and meta
                      GNN reduces to an identity operation. Thus, EMGNN generalizes single-graph GNN by jointly learning from the
Method                               CPDB             Multinet             PCNet          STRING-db            Iref              Iref(2015)complementary information stored in multiple graphs.
Random                                0.27                  0.18                  0.14                  0.24                  0.17                  0.28
20/20+                                  0.66                  0.62                  0.55                  0.67                  0.61                  0.653
MutSigCV                             0.38                  0.33                  0.27                  0.41                  0.35                  0.43
HotNet2diffusion                  0.62                  0.56                  0.48                  0.50                  0.45                  0.65
DeepWalk+featuresRF           0.74                  0.71                  0.72                  0.71                  0.66                  0.71
PageRank                              0.59                  0.53                  0.54                  0.44                  0.42                  0.62
GCNwithoutomics                0.57                  0.53                  0.47                  0.39                  0.37                  0.64
DeepWalk+SVM                  0.73                  0.51                  0.63                  0.52                  0.62                  0.66
RF                                        0.60                  0.59                  0.51                  0.61                  0.54                  0.62
MLP                                     0.58                  0.63                  0.47                  0.63                  0.55                  0.64
EMOGI[35]                           0.74                  0.74                  0.68                  0.76                  0.67                  0.75
EMOGI[15]                    0.775±0.003      0.732±0.003      0.745±0.002      0.763±0.003      0.701±0.004      0.757±0.001
EMGNN(GCN)             0.809±0.006    0.854±0.007    0.761±0.001    0.856±0.002    0.822±0.002    0.800±0.010
EMGNN(GAT)              0.776±0.018    0.796±0.034    0.730±0.031    0.805±0.030    0.739±0.033    0.773±0.049

                  Table2: TestAUPRCandstandarddeviationofEMGNN(GCN)fordifferentinputperturbationsmethodsacrossthree
                  different runs.
                  2.2   Multilayered graph improves EMGNN performance
                  We applied EMGNN to predict cancer genes due to the wealth of multi-omic proﬁling data available for cancer, yet the
                  underlyingtumorigenesisishighlycomplexandcannotbefullycapturedbyasinglebiologicalnetwork. Toensurefair
                  comparisons,weusedacompileddatasetandkepttheidenticaltraining/testsplitfromapreviousreport[35](Methods).
                  In total, this dataset consisted of 887 labeled cancer genes, 7753 non-cancer genes and 14019 unlabeled genes, along
                  with six PPI networks with multi-omics features. The detailed numbers of labeled positive and negative genes for
                  training the EMGNN model in each PPI network can be found in Supplementary Table 1.
                  The integration of multiple graphs leads to substantial improvements in the performance of EMGNN. We trained
                  EMGNN models and evaluated the testing performance with respect to different numbers of PPI networks. While we
                  added other networks tothe training and validation set, we held-out thesame testing set from the combinedtraining
                  setandkept thetestsetidenticaltoprevious works[15,35]. Anillustration oftheprocessofaddinga newbiological
                  networkand adeﬁnition oftraining andvalidation splits areshown inSupplementary Figure1. Asshown inFigure 2,
                  thetestingperformanceincreasedforeachofthesixtestingdatasetsasthenumberofinputnetworksincreased. For
                  Multinet,STRING-db,IRef,IRef(2015)testsets,theincorporationofmoregraphssteadilyincreasedtheperformance
                  withoutreaching aplateau. For PCNettestset,EMGNNachieved thebesttesting performancebycombiningfour PPI
                  networks. BecausePCNetisthemostdenselyconnectednetwork,thisbehaviorisconsistentwithpreviouslyreported
                  benchmarking results, which suggests that the performance scale with the network size [16].
                  EMGNN trained by incorporating all six graphs achieved state-of-the-art performance for all six test sets (Table 1). As
                  each test set was an independent set of held-out labeled cancer and non-cancer genes from each network and we kept
                  the set identical to previous reports (Methods), the testing performance will inform generalization error of the trained
                  EMGNNs and provide fair comparisons to previous results. EMGNN outperformed the current state-of-the-art method
                  EMOGI by an average margin of 7.15% AUPRC across all test sets, with the largest gain of 11.1% in performance
                  observed in the old version Iref.  The smallest gain was observed in PCNet, likely because PCNet is already an
                  expert-assembled graph combining the information from the other ﬁve graphs [16].  Nevertheless, for PCNet test
                  set, EMGNNcombining six graphsis signiﬁcantly moreaccurate thanEMOGI using PCNet(p-value=0.012,t-test).
                  Therefore,incorporatinginformationfrommultiplenetworksleadstoenhancedpredictivepowerforgenepathogenicity
                  prediction.
                  2.3   Evaluating the performance of different GNN architectures and graph ablations.
                  Tounderstandeachmodelcomponent’scontributiontoEMGNN’ssuperiorperformanceandmodelrobustness,wenext
                  performed an ablation study using different GNN architectures and input perturbations. For GNN architectures, we
                  compared Graph ConvolutionalNetwork (GCN) [21], and Graph Attention Network(GAT)[41]. After identifyingthe
                  mosteffectiveGNN architecture, weproceeded toevaluate itsperformance undervariedinput conditions,includingthe
                  incorporation of random and constant features, and the random removal of 20% and 40% of edges within the input
                  graphs.
                  As shown in Table 1, the GNN architecture played an essential role in EMGNN testing performance. We observed
                  that GCN is the best-performing GNN architecture in all test datasets. Our ﬁndings demonstrated that the choice of
                  GNN architecture has a signiﬁcant impact on the performance of our model. Thus, EMGNN refers to EMGNN(GCN)
                  throughout the paper unless otherwise stated.
                  Biologically, the EMGNN node features and edges are determined using high-throughput assays that inherently have
                  measurementerrors. Tothisend,wesoughttoanswerthetwofollowingquestions: How robustisEMGNNtonode
                  featureandgraphstructureperturbations? Arethenodefeaturesandthegraphstructurebothcrucialfortheprediction
                  of cancer genes? We examined the performance of EMGNN under different types of input perturbations (Table 2),
                  includingremovalofthe multi-omicsnodefeaturesand edges,andaddinguninformativevectors(random orconstant).
                                                                                     4
Method                           CPDB             Multinet             PCNet          STRING-db            Iref              Iref(2015)
RandomFeatures      0.703±0.001    0.727±0.002    0.615±0.009    0.745±0.002    0.674±0.001    0.697±0.005
All-oneFeatures        0.726±0.002    0.769±0.001    0.657±0.010    0.779±0.010    0.710±0.015    0.725±0.013
EdgeRemoval(0.2)    0.800±0.007    0.841±0.016    0.746±0.017    0.841±0.009    0.796±0.005    0.786±0.011
EdgeRemoval(0.4)    0.795±0.004    0.834±0.009    0.743±0.003    0.828±0.004    0.790±0.012    0.802±0.006

Figure 2: Test AUPRC and standard deviation values of EMGNN(GCN) with respect to the number of input PPI
networks. Each line represents a test set of positive and negative labeled genes held out in a speciﬁc PPI network. The
additionofPPInetworkswasconductedusingarandomsamplingapproach,wherethreecombinationsofPPInetworks
weresampled randomlyateachpoint. Notethatthe testingnodesremain thesameas morenetworksare added. We
observethatthe performanceincreasedforthe majorityofthetest datasets,asthenumber ofinputnetworksincreases.
We found that EMGNN decreased in performance for bothrandom and all-one node features, suggesting that the node
featuresderivedfromTCGAconsortiawereinformativeandhighlyrelevanttocancerpathophysiologyandcancergene
pathogenicity. For edgeablations, werandomly removed20% and40% edgesin eachPPI network. Theremoval of
edges slightly decreased EMGNN performance. This is expected, because EMGNN integrates six PPI networks and
demonstrates its robustness towards connectivity pattern by jointly modeling a multilayered graph topology. Overall,
EMGNN effectively leveraged both node features and edges to achieve accurate predictions.
2.4   Explaining EMGNN reveals biological insights of cancer gene pathogenicity
Explainable andtrustworthy modelsare essentialfor understandingthe biological mechanismsof knowncancer genes
andfacilitating thediscovery ofnovelcancer genes. GivenEMGNN’saccurate androbustpredictive performance,we
developed Integrated Gradients methods [22] to explain the node and edge attributions of EMGNN (Methods).
A unique advantage of EMGNN is its integration of multiple PPI networks; therefore, we focused our analysis
on the relative contributions from each PPI network to the known cancer gene predictions (Figure 3A). Each PPI
network’simportance was measured bythecorrespondingmetaedgeimportance(Methods). We examinedthe relative
contributions for two well-known cancer genes(TP53 and BRCA1, predicted cancer gene probability ˆy= 0 .99,0.98
respectively) and two newly predicted cancer genes (COL5A1 and MSLN, predicted cancer gene probability ˆy=
0.98,0.90  respectively; see Figure 3A). Notably, different genes were predicted as cancer genes leveraging evidence
from differentPPI networks. For example, Multinetcontributed to BRCA1,butnot for MSLN.The newly predicted
cancer gene COL5A1 combined information from all six PPI networks.  To systematically examine if some PPI
networks were statistically more important contributors than others, we performed an ANOVA test across all meta edge
importanceforcancergenes(Figure3B).WefoundthatdifferentPPInetworksmadesigniﬁcantlydifferentcontributions
tocancergeneprediction(P-value=1.2e−65 ). ThissuggeststhatcertainPPInetworksweremoreinformativethan
others, likely due to their unique connectivity patterns that were more reﬂective of cancer development and progression.
Amongpairwisecomparisons,theupdatedversionofIref(2015)achievedasigniﬁcantlyhighercontributionthanthe
originalIref(P-value=1.3e−50 ,t-test),whichconsolidatedourobservationthattheincorporationofotherPPInetworks
substantially improved upon the model using only Iref (see Iref performance over input PPI networks in Figure 2).
Thus, EMGNN successfully learned complementary information from the connectivity patterns in each layer of the
multilayer graph, as shown by the overall contributions from individual graphs and gene-speciﬁc variations.
                                                                  5

Figure 3: Explanation ofeach PPI network’s contribution to cancergene predictions. A) Representative PPI network
contributionsinknowncancergenesandnewlypredictedcancergenes. TP53andBRCA1areknowncancergenes;
COL5A1andMSLNarenewlypredictedcancergenes. B)Overalldistributionofmeta-edgefeatureimportanceforall
known cancer genes across six PPI networks. Meta-edge feature importance was normalized to 1 (see Methods for
details). C) An hypothetical illustration of PPI network cancer neighborhood implicated in the variation of meta-edge
importance.  D) Empirical analysis demonstrates a higher correlation between meta-edge importance and cancer
neighborhood for genes with a large meta-edge variance.
We hypothesized that the variation of meta-edge importance was a result of different cancer gene neighborhood in
different PPI networks (Figure 3C). For a target gene, we deﬁne "cancer neighbors" as the number of neighboring
genes that are known cancer genes, which is then normalized by the degree of the target node in the given network.
Hypothetically,forgeneAwhoseneighborswereallcancergenesacrossPPInetworks,themeta-edgeimportanceswere
also comparable. In contrast, for gene B whose cancer gene neighborsvaried across PPI networks, we should observe a
positivecovariancebetweencancergeneneighborsandmeta-edgeimportance. Totestourhypothesis,wecomputedthe
standard deviations of meta-edge importance for each labeled cancer gene, as well as the Pearson correlation between
meta-edge importance andthe cancer gene neighborsacross PPI networks. For genes withlarge standard deviation of
meta-edgeimportance,theimportantPPInetworkstendtohaveahighercancerneighbors,asdemonstratedbyahigher
correlation between meda-edge importance and cancer neighbors (Figure 3D). Due to the complex graph convolutions
that enabled message-passing beyond one-hop neighbors, our cancer neighbors may not fully capture the variations of
PPInetworkimportance. Overall,ourempiricalanalysisdemonstratesthatcancerneighborhoodisimplicatedingenes
with divergent connectivity patterns across PPI networks.
On the individual gene level, a detailed explanation will reveal gene-speciﬁc genetic and molecular aberrations that
the EMGNN model relies on for cancer gene prediction. As EMGNN’s node features were derived by multi-omics
pan-cancerdatasets,we nextassessedif certain typesofomicdata wereinformative tocancergenepredictions. Our
model explanation results ofnode features indicated that Single Nucleotide Variation (SNV) features were found tobe
signiﬁcantly less informative than other types of features, which is consistent with previous reports that CNAs were
moredetrimentaltocancerprogressionthanSNVs[9]. Incontrast,weobservedthatDNAmethylationwassigniﬁcantly
                                                                   6

Figure 4: Explanations of multi-omic node features importance in cancer gene predictions. A) Overall distribution of
node feature importance grouped by omic feature types, including single-nucleotide variants (MF), DNA methylation
(METH),geneexpression(GE)andcopynumberaberrations(CNA),forknowncancergenes. B)Detailednodefeature
importance forthe four genes analyzedin Figure 3B. X-axislabels were color-codedto match theomic feature types in
panel A. Individual tumor types were coded according to TCGA study abbreviations [43].
more important for known cancer gene prediction than other omics data (P-value<0.01 for all three pairwise t-test
ofother omics againstDNAmethylation). Wefurtherexaminedthenodefeature importanceofthesame fourgenes
(TP53, BRCA1, COL5A1, MSLN).As expected, the omics featurecontributionsvaried ondifferent genes and were
highly gene speciﬁc, demonstrating the heterogeneity of tumorigenesis. Point mutations were major contributors to
the prediction of TP53 as a cancer gene, which is consistent with ﬁndings from previous studies [1, 14]. Moreover,
the prediction of BRCA1 correctly identiﬁed gene expression and copy number aberrations as the most signiﬁcant
features [5]. DNAmethylation hada moderatecontributionfor thetwonewly predicted cancergenes, COL5A1and
MSLN(Figure 4B).AsDNAmethylationsare reversible epigeneticmodiﬁcations,thismay suggestpotentialnovel
therapeutic targets for certain cancer genes mediated by DNA methylation [37].
2.5   EMGNN identiﬁes newly predicted cancer genes by integrating multilayer graphs
AkeyutilityofEMGNN,givenitssuperiorperformanceandexplainability,istoidentifynewlypredictedcancergenes
that share similar topological patterns to known cancer genes, but may have been missed by a conventional recurrent
alteration analysis [35, 38]. We applied the trained EMGNN model to predict cancer genes on a total ofn= 14019
unlabeled genes.
By integrating multilayer graphs, EMGNN addressed the divergent and inconsistent cancer gene predictions from
previous models trained using a single PPI network. Prior to EMGNN, models trained using single PPI networks
such as EMOGI had made conﬂicting predictions on which genes were cancerous. Indeed, we observed substantial
variations among the predictions of EMOGI models trained on individual PPI networks, with an average of29%     and
63%     difference between the highest and lowest predictions for the top-100 and all unlabelled nodes, respectively.
Furthermore, we found an average standard deviation of 25.2% in unlabelled cancer gene predictions of EMOGI,
demonstratingthepredictednovelcancergenesweredifferentwhenusingdifferentPPInetworks. Incontrast,EMGNN
resolved these discrepancies using a more accurate and robust representation of the data from multilayer graphs.
Our analysis identiﬁed 435 genes with ahigh probability of being newly predicted cancer genes, with an 88% predicted
cancergeneprobabilitythreshold. Thisthresholdwasselectedbasedonitsabilitytoprovideaprecisiongreaterthan
95%inthelabeledset,indicatingahighdegreeofaccuracyinidentifyingtruecancergenes. Theidentiﬁcationofthese
novel cancer genes may provide new insights into the molecular mechanisms of cancer and offer potential targets for
thedevelopmentofnoveltherapies,demonstratingthatmachinelearningpredictionscanaugmentthecompletenessof
cancer gene catalogs [35]. The complete list of gene predictions from EMGNNcan be found in the project repository
onGitHub.
As a case study, we analyzed the predictions of a newly predicted cancer gene, COL5A1 (Figure 5). For this gene,
EMOGI model trained on STRINGdb predicted a non-cancer gene with high conﬁdence (ˆy= 0 .03 ), EMOGI models
trained on IRefIndex, CPDB and PCNet predicted a cancer gene with high conﬁdence (ˆy>0.98 ), while the models
trained on Multinet and IRefIndex2015 predicted a cancer gene with moderate likelihood (ˆy = 0 .775    and 0.897  ,
respectively). ThefactthatSTRINGdbwasthebest-performingPPInetworkamongmodelstrainedonindividualPPI
networks further complicated the decision making whether COL5A1 should be considered as cancer/non-cancer gene
                                                                   7

Figure 5: EMGNN predicts COL5A1 as a novel cancer gene and reveals biological insights. A) A comparison of
predicted cancer gene probabilityfrom EMGNNand EMOGImodels trainedon singlePPI networks. Asa probability
of50%equaledrandomguessingbetweencancervsnon-cancergene,thebarheightsreﬂectedthepredictionconﬁdence.
B) Three cancer hallmark genesets were signiﬁcantly enriched in the important neighboring genes of COL5A1 as
revealed by interpreting EMGNN model.  C) Enrichment of apical junction cancer hallmark geneset in COL5A1
neighboringgenes. TheneighboringgenesofCOL5A1wererankedbytheirEMGNNnodeimportanceonthex-axis,
where each blue bar represented a gene in the apical junction geneset.  A strong left-shifted curve demonstrates
enrichment of apical junction geneset in the top important genes to predict COL5A1 as a cancer gene.
(Table 1). This level of divergence in predictions hinders a trustworthy adaptation of model predictions in clinical and
pragmaticsettings. Incontrast,EMGNNintegratedtheinformationfromeachindividualPPInetworkinadata-driven
approachand providedmoreaccurate, uniﬁedpredictionsof cancergenes (Table1). EMGNN predictedCOL5A1as a
cancergenewithhighconﬁdence(Figure5A).Importantly,wealsofoundallindividualPPInetworkswerecontributing
similarlytotheﬁnalEMGNNprediction(Figure3A),andrevealedthemulti-omicsfeaturesimplicatedinthisprediction
(Figure 4B).
LeveragingtheexplanationresultsofnodecontributionsforCOL5A1acrossitsneighboringgenes,wefurtherillustrated
the potential biological mechanisms of COL5A1 using a gene set enrichment analysis (Methods). These neighboring
genes formed a ranked gene list based on their explained EMGNN contributions to the prediction of COL5A1 as
a cancer gene or not.  We discovered that three cancer hallmark gene sets, i.e.  apical junction, coagulation, and
complement system (part of the innate immune system), were signiﬁcantly enriched in COL5A1 neighboring genes
(Figure5B).Forexample,theapicaljunctioncancerhallmarkgenesetcontainedgenesannotatedtofunctionincell-cell
adhesion among epithelial cells, many of those were enriched in the top contributors (Figure 5C). This was further
supported by other studies,where COL5AI was associated with skin cancer, the type ofcancer with a strong epithelial
cell origin [25, 45, 13]. Therefore, we demonstrated how molecular mechanisms of newly predictedcancer genes could
be interpreted and discovered using the explainable EMGNN framework.
3   Discussion
The biomedical and biological domain contains a wealth of information, which is often represented and analyzed
using graph structures to reveal relationships and patterns in complex data sets. Various gene interaction and protein-
proteininteractionnetworksdescribethefunctionalrelationshipsofgenesandproteins,respectively. Thegene-gene
relationships were often described in generic cellular contexts and/or by integrating different, heterogeneous sources of
information. Therefore, asingle graphoften struggles tobest matchdisease-speciﬁcconditions, anddifferent graph
constructionand integrationmethods render distinctpredictive powers[4,6]. Substantialeffortshave beendevotedto
developintegrated [4,6] and tissue-speciﬁc graphs  [46]. Here, we tooka complementary approach anddeveloped a
newgraphlearningframework,EMGNN,tojointlymodelmultilayeredgraphs. ApplyingEMGNNtopredictcancer
genesdemonstrateditssuperiorperformanceoverpreviousgraphneuralnetworkstrainedonsinglegraphs. Wealso
employed model explanation techniques to assess both node and edge feature importance. Our results showed that
EMGNNleveragedthecomplementaryinformationfromdifferentgraphlayersandomicsfeaturestopredictcancer
genes. Importantly,wefoundthatcancergenesthathaveconﬂictingpredictionsbasedondifferentsinglegraphs,or
are missed by previous state-of-the-art predictors, can be recovered effectively using EMGNN. This demonstrates the
robustness of EMGNN predictions by joint modeling the multilayered graphs.
                                                                  8

The EMGNN model can be viewed as a data-driven, gradient-enabled integration method for multiple graphs. By
providing multiple PPI networks as input, EMGNN learns from the different connectivity patterns that represent
complementary informationto predict cancer genes. Sinceall PPI networks share thesame type of nodes andedges
(thoughnotnecessarilythesamesetofnodesineachnetwork),EMGNNcurrentlyintegrateshomogeneous,undirected
graphs;however,theEMGNNframeworkcanbeextendedtovarioustypesofgraphsandtoperformcross-datamodality
integration.  In biology and biomedicine, hierarchical graphs and heterogeneous graphs are particularly prevalent,
such as Gene Ontology[7]. For example, biomedical data is often organized in hierarchical levels, starting with genes
and molecules, moving on to cells and tissues, and ﬁnally reaching the level of individual patients and populations.
Therefore, an interesting future direction is to apply EMGNN to model multiple graphs with more heterogeneous node
and edge types, and with more complex inter-graph structures.
DNA methylation and gene expression aberrations are major contributors to EMGNN’s cancer gene predictions,
whichareconsideredimportantfeatureswhenexplainingitsomicsnodefeatures. Unlikesinglenucleotidevariation
and copy numberalteration that introducedpermanent mutations toDNAs, epigeneticand transcriptomic alterations
of cancer genes are potentially reversible by targeted therapies.  Model explanations for EMGNN revealed these
molecular aberrations that may be leveraged for screening and re-purposing of drugs, especially for previously less
well-characterized, newly predicted cancer genes.  This highlights the importance of model explanations to gain
biological and biomedical insights when developing deep learning models to predict gene pathogenicity.
In summary, we present a novel deep learning approach for the prediction of cancer genes by integrating multiple
gene-geneinteractionnetworks. Byapplyinggraphneuralnetworkstoeachindividualnetworkandthencombining
therepresentationsof thesamegenesacross networksthroughameta-graph, ourmodelisableto effectivelyintegrate
informationfrommultiplesources. Wedemonstratetheeffectivenessofourapproachthroughexperimentsonbenchmark
datasets, achieving state-of-the-art performance. Furthermore, the ability to interpret the model’s decision-making
process through the use of integrated gradients allows for a better understanding of the contribution of different
multi-omicfeaturesandPPInetworks. Overall,ourapproachpresentsapromisingavenueforthepredictionofnovel
cancer genes.
4   Methods
4.1   Datasets
Weusethedatasetsandtrain/testsplitscompiledbySchulte-Sasseetal. [35]toensureafaircomparison. Speciﬁcally,
we trained our proposed model with six PPI Networks: CPDB [18], Multinet [19], PCNet [16], STRING-db [40],
Iref[31]anditsnewestversionIref(2015). ThepreprocessingofthesesixPPInetworkswasdoneinSchulte-Sasseetal.
[35]. Weprovideabriefdescriptionofthepreprocessingstepshereforclarityandself-containment. Dependingonthe
source of the data, different conﬁdence thresholds were applied to ﬁlter out low-conﬁdence interactions. Interactions
with ascore higher than0.5 in CPDBand 0.85 inSTRING-db were included. For Multinetand IRefIndex(2015), the
datawasobtained fromtheHotnet2github repository [33]. In thecaseofthe recentversionofIRefIndex,analysiswas
restricted to binary interactions between two human proteins. No further processing was performed on the PCNet.
Asnodefeatures,weusedsingle-nucleotidevariants(MF),copynumberaberrations(CNA),DNAmethylation(METH)
andgeneexpression(GE)dataof29,446   samplesfromTCGA[43],from16  differentcancertypes. TheSNVfrequency
was calculated for each gene in each cancer type by dividing the number of non-silent SNVs in that gene by its exonic
genelength. Gene-associatedCNAswerecollectedfromTCGA,wherethecopynumberrateforeachgenewasdeﬁned
asthenumber oftimesitwasampliﬁed ordeletedinaspeciﬁc cohort. DNA methylation foreachgenein acancertype
was computed as the difference in methylation signal between tumor and matched normal samples. The expression
levelofeachgeneineachsamplewasquantiﬁedusingRNA-seqdata[42],anddifferentialexpressionwascomputedas
alog  2 fold change between cancer and matched normal samples, averaged across samples.
The positive examples included expert-curated cancer genes, high-conﬁdence cancer genes mined from PubMed
abstracts,andgeneswithalteredexpressionandpromotermethylationinatleastonecancertype[32,39]. Thenegative
exampleswerecompiledbyremovinggenesnotassociatedwithcancerfromasetofallgenesandwereselectedbased
on their absence in the positive set and various cancer databases [32, 26]. For further information regarding the data
collection and processing methods, refer to the study by Schulte-Sasse et al. [35].
4.2   Multilayer Graph Neural Network
GNNs. Let a graph be denoted byG = (V,E),whereV ={v1,...,vN}is the set of vertices andE is the set of
edges. LetA∈RN×N denotethe adjacency matrix,X   = [x1,...,xN]T∈RN×dIdenotethe nodefeatures wheredI
is the features dimensions andY  = [y1,...,yN]T∈NN denote the label vector. Graph neural networks have been
                                                                  9

successfully applied to many graph-structured problems[34, 11], as they can effectively leverage both the network
structure andnode features. They typically employ a message-passingscheme, which constitutes of thetwo following
steps. In the ﬁrst step, every node aggregates the representations of its neighbors using a permutation-invariant function.
In the second step, each node updates its own representation by combining the aggregated message from the neighbors
with its own previous representation,
                                                                                          })
                                         m (l)u  =   Aggregate(l)({h (l−1)v      :v∈N(u)       ,                                                  (1)
                                                                                )
                                         h (l)u  =   Combine(l)(h (l−1)u  ,m  (l)u,                                                                   (2)
whereh(l)u  representsthehiddenrepresentationofnodeuatthelth  layeroftheGNNarchitecture. Manychoicesforthe
AggregateandCombinefunctionshavebeenproposedintherecentyears,astheyhaveahugeimpactintherepresentation
power of the model[44]. Amongthe most popular architectures, are GraphConvolution Networks (GCNs) [21], and
GraphAttentionNetworks(GAT)[41]. InGCN,eachnodeaggregatesthefeaturevectorsofitsneighborswithﬁxed
weights inversely proportional to the central and neighbors’ node degrees,h′u =  W ⊤∑                                    √hv
                                                                                                           v∈N(u)∪{u}       ˆdvˆdu, with
ˆdi= 1+ ∑     j∈N(i) 1. InGAT,eachnodeaggregatesthemessagesfromitsneighborusing learnableweightedscores:
h′u= αu,uWh  u+∑          v∈N(u)αu,vWh  v, where the attention coefﬁcientsαu,vare computed as
αu,v=               exp  (LeakyReLU       (a⊤ [Wh  u∥Wh  v]))∑
            k∈N    (u)∪{u} exp(LeakyReLU(             a⊤ [Wh  u∥Wh  k])).
MultilayerGraphConstruction. Extendinggraphneuralnetworkstohandlemultiplenetworksisnotatrivialtask,
as they are designed to operate on a single graph. Next, we describe our method, which can accurately learn node
representations using graph neural networks, from multilayer graphs.
LetN  be the total number of genes, each associated with a feature vectorxj∈RdI. Let alsoK  be the number of
gene-geneinteractionnetworks. WerepresenteachnetworkG(i) withanadjacencymatrixA(i)∈ZNi×Niandfeature
matrixX (i)∈RNi×dI,whereNiisthenumberofgenesinthei-thnetwork. Sincesomegenesarenotpresentedinall
the graphs, the following equation holdsNi≤N ,i∈{0,1,...,K−1}.
In the ﬁrst step, for each graphGiwe apply a graph neural networkf1  that performs message-passing and updates
the node representation matrixH (i) = f1(X (i),A(i)) of each graphi∈{0,1,...,K−1}. We setf1  to be shared
across all graphs. Thisdesign allows us to handle a variable number of graphs while keeping the number of trainable
parameters ﬁxed.
Next,to aggregateand shareinformation between eachgraph, we constructa metagraphGmeta,jfor eachgene/nodej,
wherethesamegenesjacrossallgraphsareconnectedtoametanodevj. Weinitializethefeaturesofthemetanodevj
with the initial features of the corresponding genej. We apply a second GNNf2 to update the representation of the
metanodevj,Hmeta,j= f2(Xmeta,j,Ameta,j),whereXmeta,jcontainsthefeatures ofgenejfromall the networks
andAmeta,jistheadjacencymatrixofthemetagraphGmeta,j,j∈{0,1,...,N}. Wesetf2 tobesharedacrossall
genes. Therefore, in this stage, the model combines and exchanges information between the different networks. Finally,
a multi-layer perceptronf3 predicts the class of the meta nodej, ˆyj= f3(Hmeta,j). An illustration of the proposed
model can be found in Figure 1.
ExperimentalDetails. Toensureafaircomparisonwithpreviouswork,weutilizedthesameexperimentalsetupasin
Schulte et al. [35]. Inparticular, we divided the data for each testing graph into a75% training set and a 25% testing
setusingstratiﬁedsamplingtomaintainequalproportionsofknowncancerandnon-cancergenesinbothsets. Since
ourmodeltakes multiplegraphsasinputforeach experiment,weretainedthe testnodesofonegraphas thetestset,
andweallocated90%oftheremainingnodesfromtheothergraphstothetrainingsetand10%tothevalidationset.
When adding other PPI networks tothe training and validation set, we held-outthe same testing setfrom the combined
training set and kept the test set identical to previous works [15, 35]. An illustration of the process of adding a new
biological network and a deﬁnition of training and validation splits are shown in Supplementary Figure 1. The model
was trained for2000    epochs, using the cross-entropy loss function, and the ADAM optimizer [20] with a learning rate
of0.001  . TheinitialGNNhadthreelayerswithahiddendimensionof64 ,whilethemeta-GNNhadasinglelayerwith
ahiddendimensionof64 . TheSupplementaryTable1providesinformationonthePPInetworks,includingstatistics,
as well as the number of positive and negative genes used for training.
4.3   Model interpretation
Weusedtheintegratedgradient(IG)moduleinCaptum,toassignanimportancescoretoeachinputfeature. Captum
is atool forunderstanding and interpretingthe decision-makingprocess of machine learning models[22]. It offersa
                                                                  10

range of interpretability methods that allow users to analyze the predictions made by their models and understand how
different input variables contribute to these predictions. IG interprets the decisions of neural networks by estimating
the contribution of each input feature to the ﬁnal prediction. The integrated gradient approximates the integral of
gradients of the model’s output with respect to the inputs along a straight line path from a speciﬁc baseline input
to the current input.  The baseline input is typically chosen to be a neutral or a meaningless input, such as an all-
zero vector or a random noise. Formally, letF(x) be the function of a neural network, wherexis the input and ˆx
is the baseline input.  The integrated gradients for inputxand baselinex0  along thei-th dimension is deﬁned as:
IntegratedGradsi(x) =( xi−ˆxi)∫1                 ∂F(ˆx+α(x−ˆx)
                                           α=0        ∂xi  dα,wheretheintegralistakenalongthestraightlinepathfrom
ˆxtoxand∂F(x)/∂xiis the gradient ofF(x) along thei-th dimension.
However,thetraditionalintegratedgradientmethod,whichisdesignedforsingleinputmodels,isnotdirectlyapplicable
to graph neural networks as they have two distinct inputs, namely node features and network connectivity.  This
necessitatesthedevelopmentofamodiﬁedapproachforcomputingintegratedgradientsingraphneuralnetworksthat
considers both inputs. To this end, we propose a decomposition of the problem into two parts: identifying the most
important node featuresand identifying the most crucialedges in the network separately. Since we predict theclass of
eachgenebycombiningallthegraphs,fromthemeta-noderepresentations,weapplytheinterpretationanalysisonlyto
the meta-nodes.
Node feature interpretationanalysis. WeanalyzethecontributionofnodefeaturestothepredictionsoftheGNNby
usingthetraditionalintegratedgradientmethodwhilekeepingtheedgesinthenetworkﬁxed. Speciﬁcally,weinterpolate
between the current node features input and a baseline input where the node features are zero:Attributionxi =
(xi−ˆxi)∫1       ∂F(ˆx+α(x−ˆx,A)
            α=0         ∂xi   dα,whereA aretheadjacencymatricesofthegraphs. Sincethepredictionforeachgene
isalsobasedonthefeaturesofsurroundinggenesinthegraphs,weextractattributionvaluesforthek-hopneighbor
genes as well, wherekis equal to the number of message-passing layers in the ﬁrst GNN. Therefore, the output of
the attribution method for each nodeu, is a matrixK  (u )∈RN×d. Each entryKijof the matrix, corresponds to the
attributionofthefeaturejofnodeitothetargetnodeu. Fromthismatrix,weselecttherowthatcorrespondstothe
feature attributions of the corresponding meta node.
Edge feature interpretation analysis. To analyze the contribution of edges in the meta-graph to the predictions of
theGNN,weusetheintegratedgradientmethodfortheedgeswhilekeepingthenodefeaturesﬁxed. Speciﬁcally,we
interpolatebetweenthecurrentedgeinputandabaselineinputwheretheweightsoftheedgesarezero:Attributionei=
∫1     ∂F(X,Aα)
 α=0     ∂wei dα, whereAα corresponds to the graphs with the edge weights equal toα. We further normalize the
attribution values of each meta node by dividing them by their maximum value, resulting in a range of [0, 1] for each
edge. This explanationtechnique allowsus tounderstand which edgesin themeta-graphare crucialforthe model’s
decision-making process, and therefore which input PPI networks are important for each gene prediction.
4.4   Newly predicted cancer gene discovery
We applied the trained EMGNN model that combined all six individual PPI networks to predict novel cancer genes
in then = 14019       unlabeled genes. We ranked these genes by their predicted cancer gene probability for potential
newpredictedcancergenes(NPCG)inthisstudy. Foreachunlabeledgene,wealsoappliedEMOGImodelstrained
on individual PPI networks to predict the probability of it being a cancer gene. The complete list of the predicted
probabilities for all the unlabeled genes, can be found in the project repository onGitHub.
4.5   Gene set enrichment analysis
To understand the biological mechanisms of EMGNN’s cancer gene prediction, we employed gene set enrichment
analysis(GSEA)toanalyzethefunctionalenrichmentofimportantgenefeaturesincuratedcancerpathwayannotations.
Speciﬁcally,todeterminetheimportanceofneighboringgenenodes,weaggregatedthemaximumfeatureimportance
of each node using Captum’s feature explanation results. Genes with zero importance were excluded in this analysis as
theydidnotcontributetothepredictionofthistargetgene. Wethenrankedtheneighboringgenenodesbasedontheir
importanceandusedthisrankedgenelistasinputforGSEA.Theenrichmentp-valueandmultipletestingcorrected
FDR were computed by GSEA python package [10] against cancer hallmark gene sets [23].
Acknowledgements
WethankallmembersoftheZhanglaboratoryforhelpfuldiscussions. Thisworkwasperformedatthehigh-performance
computing resources at Cedars-Sinai Medical Center and the Simons Foundation.
                                                                  11

Funding
This work has been supported by an institutional commitment fund from Cedars-Sinai Medical Center to ZZ.
References
 [1]  LO Almeida, AC Custódio, GR Pinto, MJ Santos, JR Almeida, CA Clara, JA Rey, Cacilda Casartelli, et al.
      Polymorphisms and dna methylation of gene tp53 associated with extra-axial brain tumors.  Genet Mol Res,
      8(1):8–18, 2009.
 [2]  TanyaZBerardini,SuparnaMundodi,LeonoreReiser,EvaHuala,MargaritaGarcia-Hernandez,PeifenZhang,
      Lukas A Mueller, Jungwoon Yoon, Aisling Doyle, Gabriel Lander, et al. Functional annotation of the arabidopsis
      genome using controlled vocabularies. Plant physiology, 135(2):745–755, 2004.
 [3]  AnnaBrückner,CécilePolge,NicolasLentze,DanielAuerbach,andUweSchlattner. Yeasttwo-hybrid,apowerful
      tool for systems biology. International journal of molecular sciences, 10(6):2763–2788, 2009.
 [4]  MengfeiCao,ChristopherMPietras,XianFeng,KathrynJDoroschak,ThomasSchaffner,JisooPark,HaoZhang,
      Lenore J Cowen, and Benjamin J Hescott.  New directions for diffusion-based network prediction of protein
      function: incorporating pathways with conﬁdence.Bioinformatics, 30(12):i219–i227, 2014.
 [5]  Hui-Ju Chang, Ueng-Cheng Yang, Mei-Yu Lai, Chen-Hsin Chen, and Yang-Cheng Fann.  High brca1 gene
      expression increases the risk of early distant metastasis in er+ breast cancers. Scientiﬁc reports, 12(1):77, 2022.
 [6]  HyunghoonCho, BonnieBerger,andJian Peng. Compactintegration ofmulti-networktopology forfunctional
      analysis of genes. Cell systems, 3(6):540–548, 2016.
 [7]  GeneOntologyConsortium. Thegene ontology(go)databaseandinformatics resource. Nucleicacidsresearch,
      32(suppl_1):D258–D261, 2004.
 [8]  MichaelCostanzo,Anastasia Baryshnikova,JeremyBellay,Yungil Kim,EricDSpear,CarolynSSevier,Huim-
      ing Ding, Judice LY Koh, Kiana Touﬁghi, Sara Mostafavi, et al.  The genetic landscape of a cell. science,
      327(5964):425–431, 2010.
 [9]  HaithamA Elmarakeby, JustinHwang,Rand Arafeh,JettCrowdis,SydneyGang, DavidLiu,Saud HAlDubayan,
      Keyan Salari, Steven Kregel, Camden Richter, et al.  Biologically informed deep neural network for prostate
      cancer discovery. Nature, 598(7880):348–352, 2021.
[10] Zhuoqing Fang, Xinyuan Liu, and Gary Peltz.  Gseapy:  a comprehensive package for performing gene set
      enrichment analysis in python. Bioinformatics, 39(1):btac757, 2023.
[11] VictorFung,JiaxinZhang,EricJuarez,andBobbyGSumpter. Benchmarkinggraphneuralnetworksformaterials
      chemistry. npj Computational Materials, 7(1):1–8, 2021.
[12] Casey S Greene, Arjun Krishnan, Aaron K Wong, Emanuela Ricciotti, Rene A Zelaya, Daniel S Himmelstein,
      RanZhang, BorisMHartmann,Elena Zaslavsky,Stuart CSealfon,etal. Understandingmulticellularfunction
      and disease with human tissue-speciﬁc networks.Nature genetics, 47(6):569–576, 2015.
[13] Sujie Gu, Zesheng Peng, Yuxi Wu, Yihao Wang, Deqiang Lei, Xiaobing Jiang, Hongyang Zhao, and Peng Fu.
      Col5a1 serves as a biomarker of tumor progression and poor prognosis and may be a potential therapeutic target
      in gliomas. Frontiers in Oncology, page 4749, 2021.
[14]DP Guimaraes and P Hainaut. Tp53: a key gene in human cancer.        Biochimie, 84(1):83–93, 2002.
[15] Chenyang Hong, Qin Cao, Zhenghao Zhang, Stephen Kwok-Wing Tsui, and Kevin Y Yip. Reusability report:
      Capturingpropertiesofbiologicalobjectsandtheirrelationshipsusinggraphneuralnetworks. NatureMachine
      Intelligence, 4(3):222–226, 2022.
[16] Justin K Huang, Daniel E Carlin, Michael Ku Yu, Wei Zhang, Jason F Kreisberg, Pablo Tamayo, and Trey Ideker.
      Systematic evaluation of molecular networks for discovery of disease genes. Cell systems, 6(4):484–495, 2018.
[17] RobertIetswaart, BenjaminM Gyori,JohnA Bachman,PeterK Sorger,andL StirlingChurchman. Genewalk
      identiﬁesrelevantgenefunctionsforabiologicalcontextusingnetworkrepresentationlearning.Genomebiology,
      22(1):1–35, 2021.
[18] Atanas Kamburov, Konstantin Pentchev, Hanna Galicka, Christoph Wierling, Hans Lehrach, and Ralf Herwig.
      Consensuspathdb: towardamorecompletepictureofcellbiology. Nucleicacidsresearch,39(suppl_1):D712–
      D717, 2011.
[19] Ekta Khurana, Yao Fu, Jieming Chen, and Mark Gerstein. Interpretation of genomic variants using a uniﬁed
      biological network approach. PLoS computational biology, 9(3):e1002886, 2013.
                                                                 12

[20] DiederikPKingmaandJimmyBa. Adam: Amethodforstochasticoptimization. arXivpreprintarXiv:1412.6980,
      2014.
[21] ThomasN.KipfandMaxWelling. Semi-supervisedclassiﬁcationwithgraphconvolutionalnetworks. InICLR,
      2017.
[22] Narine Kokhlikyan, Vivek Miglani, Miguel Martin, Edward Wang, Bilal Alsallakh, Jonathan Reynolds, Alexander
      Melnikov, Natalia Kliushkina, Carlos Araya, Siqi Yan, and Orion Reblitz-Richardson. Captum: A uniﬁed and
      generic model interpretability library for pytorch, 2020.
[23] ArthurLiberzon,ChetBirger,HelgaThorvaldsdóttir,MahmoudGhandi,JillPMesirov,andPabloTamayo. The
      molecular signatures database hallmark gene set collection. Cell systems, 1(6):417–425, 2015.
[24] Jianfang Liu, Tara Lichtenberg, Katherine A Hoadley, Laila M Poisson, Alexander J Lazar, Andrew D Cherniack,
     Albert J Kovatich, Christopher C Benz, Douglas A Levine, Adrian V Lee, et al. An integrated tcga pan-cancer
      clinical data resource to drive high-quality survival outcome analytics. Cell, 173(2):400–416, 2018.
[25] MichaelBMann,MichaelABlack,DevinJJones,JerroldMWard,ChristopherChinKuanYew,JustinYNewberg,
     AdamJDupuy,AlistairGRust,MarcusWBosenberg,MartinMcMahon,etal. Transposonmutagenesisidentiﬁes
      genetic drivers of braf v600e melanoma. Nature genetics, 47(5):486–495, 2015.
[26] VictorAMcKusick. Mendelianinheritanceinmananditsonlineversion,omim. TheAmericanJournalofHuman
     Genetics, 80(4):588–604, 2007.
[27] ThomasMNorman, MaxAHorlbeck,JosephM Replogle,AlexYGe, Albert Xu,MarcoJost,Luke AGilbert,
      andJonathanSWeissman. Exploringgeneticinteractionmanifoldsconstructedfromrichsingle-cellphenotypes.
      Science, 365(6455):786–793, 2019.
[28] Keith Nykamp,MichaelAnderson,MartinPowers,JohnGarcia,Blanca Herrera, Yuan-YuanHo, YuyaKobayashi,
      Nila Patil, Janita Thusberg, Marjorie Westbrook, et al. Sherloc: a comprehensive reﬁnement of the acmg–amp
      variant classiﬁcation criteria.Genetics in Medicine, 19(10):1105–1117, 2017.
[29] BastianPfeifer,AnnaSaranti,andAndreasHolzinger. Gnn-subnet: diseasesubnetworkdetectionwithexplainable
      graph neural networks. Bioinformatics, 38(Supplement_2):ii120–ii126, 2022.
[30] Wei Qin, Kelvin F Cho, Peter E Cavanagh, and Alice Y Ting. Deciphering molecular interactions by proximity
      labeling. Nature methods, 18(2):133–143, 2021.
[31] SabryRazick, George Magklaras,and IanMDonaldson. ireﬁndex: a consolidatedprotein interaction database
     with provenance. BMC bioinformatics, 9(1):1–19, 2008.
[32] Dimitra Repana, Joel Nulsen, Lisa Dressler, Michele Bortolomeazzi, Santhilata Kuppili Venkata, Aikaterini
      Tourna,AnnaYakovleva,TommasoPalmieri,andFrancescaDCiccarelli. Thenetworkofcancergenes(ncg): a
      comprehensivecatalogueofknownandcandidatecancergenesfromcancersequencingscreens. Genomebiology,
      20:1–12, 2019.
[33] Matthew A Reyna, Mark DM Leiserson, and Benjamin J Raphael. Hierarchical hotnet: identifying hierarchies of
      altered subnetworks. Bioinformatics, 34(17):i972–i980, 2018.
[34] FrancoScarselli,MarcoGori,AhChungTsoi,MarkusHagenbuchner,andGabrieleMonfardini. Thegraphneural
      network model. IEEE Trans. Neural Netw., 20(1):61–80, 2009.
[35] RomanSchulte-Sasse,StefanBudach,DenesHnisz,andAnnalisaMarsico. Integrationofmultiomicsdatawith
      graph convolutional networks toidentify new cancergenes and theirassociated molecular mechanisms. Nature
     Machine Intelligence, 3(6):513–526, 2021.
[36] RachelSGSealfon,AaronKWong,andOlgaGTroyanskaya. Machinelearningmethodstomodelmulticellular
      complexity and tissue speciﬁcity.Nature Reviews Materials, 6(8):717–729, 2021.
[37] ShikharSharma,TheresaKKelly,andPeterAJones. Epigeneticsincancer. Carcinogenesis,31(1):27–36,2010.
[38] Maxwell ASherman, Adam UYaari, Oliver Priebe, FelixDietlein, Po-Ru Loh,and Bonnie Berger. Genome-wide
      mapping of somatic mutation rates uncovers drivers of cancer. Nature Biotechnology, 40(11):1634–1643, 2022.
[39] Zbyslaw Sondka, Sally Bamford, Charlotte G Cole, Sari A Ward, Ian Dunham, and Simon A Forbes.  The
      cosmiccancergenecensus: describinggeneticdysfunctionacrossallhumancancers. NatureReviewsCancer,
     18(11):696–705, 2018.
[40] Damian Szklarczyk, Annika L Gable, David Lyon, Alexander Junge, Stefan Wyder, Jaime Huerta-Cepas, Milan
      Simonovic, Nadezhda T Doncheva, John H Morris, Peer Bork, et al. String v11: protein–protein association
      networkswithincreasedcoverage,supportingfunctionaldiscoveryingenome-wideexperimentaldatasets. Nucleic
      acids research, 47(D1):D607–D613, 2019.
                                                                13

[41] Petar Veliˇckovi´c, GuillemCucurull, Arantxa Casanova, AdrianaRomero, Pietro Liò, andYoshua Bengio. Graph
      attention networks. InICLR, 2018.
[42] Qingguo Wang, Joshua Armenia, Chao Zhang, Alexander V Penson, Ed Reznik, Liguo Zhang, Thais Minet,
      Angelica Ochoa, Benjamin E Gross, Christine A Iacobuzio-Donahue, et al.  Unifying cancer and normal rna
      sequencing data from different sources. Scientiﬁc data, 5(1):1–8, 2018.
[43] John N Weinstein, Eric A Collisson, Gordon B Mills, Kenna R Shaw, Brad A Ozenberger, Kyle Ellrott, Ilya
      Shmulevich, Chris Sander, and Joshua M Stuart. The cancer genome atlas pan-cancer analysis project. Nature
      genetics, 45(10):1113–1120, 2013.
[44] Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka.  How powerful are graph neural networks?  In
      International Conference on Learning Representations, 2019.
[45] HuaZhu,XinyaoHu,ShiFeng,ZhihongJian,XimingXu,LijuanGu,XiaoxingXiong,etal. Thehypoxia-related
      genecol5a1isaprognosticandimmunologicalbiomarkerformultiplehumantumors. OxidativeMedicineand
      Cellular Longevity, 2022, 2022.
[46] Marinka Zitnik and Jure Leskovec. Predicting multicellular function through multi-layer tissue networks. Bioin-
      formatics, 33(14):i190–i198, 2017.
[47] Marinka Zitnik, Francis Nguyen, Bo Wang, Jure Leskovec, Anna Goldenberg, and Michael M Hoffman. Machine
      learning for integrating data in biology and medicine: Principles, practice, and opportunities. Information Fusion,
      50:71–91, 2019.
                                                                  14

