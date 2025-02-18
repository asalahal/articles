 MGNN:AMultimodalGraphNeuralNetworkforPredictingthe
                                              SurvivalofCancerPatients
                 Jianliang Gao                                        Tengfei Lyu                                          Fan Xiong
       School of Computer Science and                     School of Computer Science and                      School of Computer Science and
    Engineering,CentralSouthUniversity                 Engineering,CentralSouthUniversity                  Engineering,CentralSouthUniversity
                 Jianxin Wang                                         Weimao Ke                                              Zhao Li
       School of Computer Science and                   College of Computing & Informatics,                              Alibaba Group
   Engineering,CentralSouthUniversity                               Drexel University
ABSTRACT                                                                           prediction[2].InmedicalIRarea,cancerprognosispredictionhas
Predictingthesurvivalofcancerpatientsholdssignificantmeaning                       significantmeaningsince cancerhasbecome oneoftheworldwide
for public health, and has attracted increasing attention in med-                  health problems. Furthermore, predicting the survival expectancy
ical information communities. In this study, we propose a novel                    ofcancerpatientsisakeyproblemamongcancerprognosispredic-
frameworkforcancersurvivalpredictionnamedMultimodalGraph                           tion[3].Cancersurvivalpredictioncanbeformulatedasacensored
NeuralNetwork(MGNN),whichexploresthefeaturesofreal-world                           survivalanalysisproblem,predictingbothifandwhenanevent(i.e.
multimodaldata suchasgeneexpression,copynumberalteration                           patient death) occurs within a given time period [4]. In medical IR
and clinical data in a unified framework. In order to explore the                  area,thefive-yearsurvivalrateisacommonusedindexforcancer
inherent relation, we first construct the bipartite graphs between                 prognosis. For example, only slightly more than half of the newly
patients andmultimodal data.Subsequently,graph neuralnetwork                       diagnosed oral cancer patients will survive over five years [5].
is adopted to obtain the embedding of each patient on different                       Accuratepredictionofthe cancersurvivalcouldhelpclinicians
bipartite graphs. Finally, a multimodal fusion neural layer is de-                 make effective decisions and establish appropriate therapy pro-
signed to fuse the features from different modal data. The output                  grams. Meanwhile, it can spare a significant number of patients
of our method is the classification of short term survival or long                 from receiving unnecessary treatment and its related expensive
termsurvival foreachpatient. Experimentalresultson onebreast                       medicalcosts[6].Priorworkshavetriedtosolvetheproblemofcan-
cancer dataset demonstrate that MGNN outperforms all baselines.                    cerprognosisprediction. Supportvectormachine-basedrecursive
Furthermore,wetestthetrainedmodelonlungcancerdataset,and                           featureeliminationapproachisproposedforprognosisprediction
the experimental results verify the strong robust by comparing                     byrelyingongeneexpressiondata[7].Arecentinterdisciplinary
with state-of-the-art methods.                                                     effortistoapproachthisproblemfromadeeplearningperspective.
                                                                                   A novel multimodal deep neural network by integrating multi-
KEYWORDS                                                                           dimensionaldataforhumanbreastcancerprognosispredictionis
Medical informationretrieval,Cancer survivalprediction, Graph                      proposed[6].Recently,deeplearninghasbeenanemergingmethod-
neural networks, Multimodal                                                        ologyandprovidedapossibleapproachtoimprovetheaccuracyof
                                                                                   cancersurvivalprediction[8].However,therearetwochallenges
ACMReferenceFormat:                                                                to adopt deep learning for cancersurvival prediction: (1)Howto
JianliangGao,TengfeiLyu,FanXiong,JianxinWang,WeimaoKe,andZhao                      utilize the features of medical data from different modalities (e.g.,
Li. 2020. MGNN: A Multimodal Graph Neural Network for Predicting the               geneexpressionandclinical).Mostofpriorworksdirectlycombine
Survival of Cancer Patients. In Proceedings of the 43rd International ACM          differenttypesof dataintodeeplearningmodel withoututilizing
SIGIR Conference on Research and Development in Information Retrieval              the rich information of multimodal data. (2) How to utilize the
(SIGIR’20),July25–30,2020,VirtualEvent,China.  ACM,NewYork,NY,USA,                 structure information between patients and multimodal medical
4 pages. https://doi.org/10.1145/3397271.3401214
                                                                                   data.Mostofpreviousworksignoretheinherentrelationsbetween
1   INTRODUCTION                                                                   patients and medical data.
MedicalInformationRetrieval(IR)playsanimportantroleinpublic                           To cope with the above challenges, we design a Multimodal
health [1], and has attracted increasing attention in many appli-                  Graph Neural Network (MGNN) to predict the survival of patients.
cations such as cancer prognosis prediction and epidemiological                    We utilize gene expression profile or the DNA copy number alter-
                                                                                   ation(CNA)profiletoconstructabipartitegraphs,whichexplore
Permission to make digital or hard copies of all or part of this work for personal ortherelationamongthem.Basedonthebipartitegraphs,anewgraph
classroom use isgranted without fee providedthat copies are notmade or distributed neuralnetworkisproposedtoobtaintheembeddingrepresentation
forprofitorcommercialadvantageandthatcopiesbearthisnoticeandthefullcitation        ofeachpatient.Finally,theoutputistheclassificationofshortterm
onthefirstpage.CopyrightsforcomponentsofthisworkownedbyothersthanACM               survival orlong termsurvival based ona giventhreshold (usually
mustbehonored.Abstractingwithcreditispermitted.Tocopyotherwise,orrepublish,
topostonserversortoredistributetolists,requirespriorspecificpermissionand/ora      adopt5-yearsurvivalinthecommunity).Themaincontributions
fee.Requestpermissions frompermissions@acm.org.                                    of this paper are summarized as follows:
SIGIR ’20, July 25–30, 2020, Virtual Event, China
©2020Associationfor Computing Machinery.
ACMISBN978-1-4503-8016-4/20/07...$15.00
https://doi.org/10.1145/3397271.3401214

Figure1:TheoverviewoftheproposedMGNN.MGNNiscomposedofthreemainparts:bipartitegraph,GNNlayer,multimodal
fusion neural layer. With the input multimodal data such as gene expression and copy number, (1) bipartite graph further
expressestheirpotentialrelation(e.g,theedgesbetweenpatient𝑝𝑖andgeneexpression𝑔𝑗);(2)theGNNlayer,whichaggregates
informationofneighborsinbipartitegraphbygraphneuralnetworkmodel;and(3)themultimodalfusionneurallayer,which
fusesthefeaturesofmultimodaldata.Fortheoutputlabelsb𝑦𝑝,theshorttermsurvivors(lessthan5yearsurvival)arelabeled
as0andlongtermsurvivors(morethan5yearsurvival)arelabeledas1.
     •We highlight the critical importance of explicitly exploit-                number of patients. We first build bipartite graphs via𝑋𝑔∈R𝑁×𝑚
       ing the multimodal data, and the inherent relation between                and𝑋𝑐∈R𝑁×𝑛, and then the node of bipartite graph aggregates
       patients and multimodal data.                                             information of neighbors by GNN model. Specifically, the final
     •We propose a novel Multimodal Graph Neural Network                         embedding of the patient can be shown as below:
       (MGNN)forcancersurvivalprediction,whichutilizesmulti-                                     E𝑝={𝐸𝑔,𝐸𝑐,𝐸𝑐𝑙𝑖𝑛}∈R𝑁×(𝑑1+𝑑2+𝑘),                 (2)
       modal data in a unified framework. In the framework, the
       problemofcancersurvivalpredictionissolvedasalearning                      where𝐸𝑔∈R𝑁×𝑑1,𝐸𝑐∈R𝑁×𝑑2 and𝐸𝑐𝑙𝑖𝑛∈R𝑁×𝑘 respectively
       task of patient classification.                                           represent the patient embedding from multimodal data,𝑑1 and𝑑2
     •We conductexperiments on real-world datasets. Theresults                   represent the size of the embedding dimension.𝑒′𝑝∈E𝑝 denotes
       demonstrate the state-of-art performance of MGNN and its                  the multimodal embedding vector for each patient.
       effectiveness and robustness for cancer survival prediction.                 The goal ofthe study is to distinguish shortterm survivors and
2   PROPOSEDMGNNMETHOD                                                           long term survivors (usually adopt less 5-year survival), and the
                                                                                 formal definition of the problem is as follows:
In this section, we first define the problem formulation. Then, the                 Input: multimodal dataX𝑝={𝑋𝑔,𝑋𝑐,𝑋𝑐𝑙𝑖𝑛}.
detailed design of MGNN, which learns classification information                    Output:thepredictiveclassificationofshorttermsurvivorsor
from multimodal data, is shown in Figure 1. The overall structure                long term survivors.
is composed of three parts: the bipartite graph, the GNN layer, and                 In order to predict the survival of cancer patients from multi-
the multimodal fusion neural layer.                                              modal data,we design anovel Multimodal Graph Neural Network
                                                                                 (MGNN) as illustrated in Figure 1.
2.1   ProblemFormulation
In this study, multimodal data are composed of the gene expres-                  2.2   MGNN
sion profile data, the CNA profile data and the clinical data. It is             In this subsection, we introduce the detailed design of the MGNN
expressed as follows:                                                            for predicting the survival of cancer patients.
                X𝑝={𝑋𝑔,𝑋𝑐,𝑋𝑐𝑙𝑖𝑛}∈R𝑁×(𝑚+𝑛+𝑘),                  (1)                   BipartiteGraph.Inordertoestablishthelinkbetweenpatients
                                                                                 andmultimodal,weutilizethegeneexpressionprofileandthecopy
where𝑋𝑔∈R𝑁×𝑚,𝑋𝑐∈R𝑁×𝑛 and𝑋𝑐𝑙𝑖𝑛∈R𝑁×𝑘 stand for the                                 numberalteration(CNA)whichbelongtothemultimodaldatato
geneexpressionprofile data, theCNAprofiledataand theclinical                     construct bipartite graph. According to the previous method [9],
data respectively.𝑚,𝑛and𝑘represent the dimension of the gene                     the gene expression features are normalized and processed into
data,theCNAdataandtheclinicaldatarespectivelyand𝑁 isthe                          threecategories:under-expression(-1),over-expression(1),baseline

(0). For each patient, an edge will be built between the patient                       2.3   Optimization
and the gene, only if the gene is not properly expressed (under-                       Formodeloptimization,our method couldbetrainedwithsuper-
expressionorover-expression).Finallyweconstructapatient-gene                           vised setting. Besides, we use L2 regularization to prevent over-
bipartite graph. Obviously, we can intuitively understand the gene                     fitting of our model. Based on the cross-entropy loss, the objective
expressionaffectingpatientsfromthepatient-genebipartitegraph.                          function could be defined as follows:
   For CNA data, we directly utilize the original data with five                                                   𝑁Õ
discrete values: homozygous deletion (-2); hemizygous deletion                                𝐿(𝑦𝑝,ˆ𝑦𝑝)=−1            [𝑦𝑝𝑖𝑙𝑜𝑔ˆ𝑦𝑝𝑖−(1−𝑦𝑝𝑖)𝑙𝑜𝑔(1−ˆ𝑦𝑝𝑖)]
(-1); neutral/no change (0); gain (1); high level amplification (2). A                                         𝑁  𝑖=0
patient-CNA bipartite graph is constructed by using the similar                                      𝐿Õ  𝑇Õ   𝑑𝑙Õ𝑑𝑙+1Õ                                              (8)
approach of the construction of patient-gene bipartite.                                        +1                     𝑤𝑙𝑡𝑖𝑗2,
   GNNLayer.Aftergetting patient-genebipartitegraph, theini-                                      𝜆 𝑙=1  𝑡=1 𝑖=1  𝑗=1
tial representation matrix of patient-gene bipartite𝐸is as follows:                    where𝑦𝑝𝑖 is the actual label, ˆ𝑦𝑝𝑖 is the predictive scores and𝑁 is
             𝐸=[𝑒(0)𝑝1,𝑒(0)𝑝2,···,𝑒(0)𝑝𝑁 ,𝑒(0)𝑔1,𝑒(0)𝑔2,···,𝑒(0)𝑔𝑀],            (3)    the batch size, L and T is the number of the embedding layers and
                   |                 {z                  }|                 {z                  }trainable weight matrices𝑊 , respectively.𝑊𝑙𝑡 ={𝑤𝑡𝑙𝑖𝑗}𝑑𝑙×𝑑𝑙+1  is
                    𝑝𝑎𝑡𝑖𝑒𝑛𝑡𝑠𝑒𝑚𝑏𝑒𝑑𝑑𝑖𝑛𝑔𝑔𝑒𝑛𝑒𝑠𝑒𝑚𝑏𝑒𝑑𝑑𝑖𝑛𝑔                                    the𝑙𝑡ℎweight matrix.
where𝑁 and𝑀 standforthe numberofpatientandgene,respec-
tively,𝑒(0)𝑝  ,𝑒(0)𝑔   areservedastheinitializationofpatientembedding                  3   EXPERIMENTS
and gene embedding respectively.                                                       In this section, we describe the experimental setups and results.
   In order to get the genetic information of patients more effi-                      The results obtained with our method are compared with four of
ciently, we aggregate the messages propagated from the neighbor-                       the state-of-art methods for cancer survival prediction.
hood of𝑝to refine the embedding of𝑝. Specifically, we define the
aggregation function as below:                                                         3.1   ExperimentalSetup
                                          Õ                                            Datasets.Weutilizewell-establishedbenchmarkdatasets1,which
           𝑒(𝑙+1)𝑝     =𝜎(𝑓𝑝→𝑝(𝑒(𝑙)𝑝)+          𝑓𝑔→𝑝(𝑒(𝑙)𝑝,𝑒(𝑙)𝑔)),         (4)        arewidelyusedforpredictingthesurvivalofbreastcancerpatients.
                                         𝑔∈N𝑝                                          Thedatasetisextractedfrom1903validbreastcancerpatient’sdata
where𝑒(𝑙+1)𝑝      denotes the embedding of patient𝑝obtained at the                     and it contains multimodal data including gene expression profile,
(𝑙+1)𝑡ℎGNN layer,N𝑝is neighbor set of patient𝑝,𝜎(·)denotes                             CNA profile and clinical data. In this work, the gene expression
activation function𝐿𝑒𝑎𝑘𝑦𝑅𝑒𝐿𝑈, and𝑓(·)is the representation en-                         profile data include approximately 24369 genes and CNA profile
coding function that can be shown as below:                                            datacontainapproximately22544genesinthebreastcancerdataset.
                                                                                           Each patient has 35 clinical features, including age at diagnosis,
          𝑓𝑝→𝑝(𝑒(𝑙)𝑝)=𝑊(𝑙)                                                         lymphnodesexaminedpositive,cancertypedetailed,radiotherapy
                              1 𝑒(𝑙)𝑝                                       (5)        etc. Finally, we utilize 28 clinical features in our experiment.
          𝑓𝑔→𝑝(𝑒(𝑙)𝑝,𝑒(𝑙)𝑔)=(𝑊(𝑙)1 𝑒(𝑙)𝑔+𝑊(𝑙)2(𝑒(𝑙)𝑔⊙𝑒(𝑙)𝑝))                            To comprehensively evaluate our proposed method, we adopt
                                                                                       ten-fold cross validation in our experiments. Specifically, we ran-
where𝑊(𝑙)1 ,𝑊(𝑙)2 ∈R𝑑𝑙×𝑑𝑙+1  arethetrainableweightmatrices,and                         domlydivideallpatientsintotensubsets.Foreachroundoftraining,
⊙denotes the element-wise product.𝑒(𝑙)𝑝   and𝑒(𝑙)𝑔   are the represen-                 eachsubsetwillbeusedasatestset,andtheremainingninesubsets
tation of patients and genes in𝑙layer of GNN.                                          are divided into a training set (80%) and a validation set (20%). The
   Theembedding ofpatients basedon thepatient-gene bipartite                           prediction score is the average of the output of ten rounds.
graph, through the GNN layer, is𝑒𝑔𝑝. Similarly, the embedding of                           Parameter Settings. The proposed method is implemented
patients𝑒𝑐𝑝 can be obtained by patient-CNA bipartite graph. The                        with Tensorflow and optimized by Adam with a learning rate of
clinical data after standard normalization is denoted as𝑒𝑐𝑙𝑖𝑛𝑝   .                     0.001. The parameters in Section 2 is set as𝑚 =  64, 𝑛=  64, 𝑘=  28,
   MultimodalFusionNeuralLayer.Aftergettingtherepresen-                                and𝑑𝑙=𝑑2 =  128.Theembedding  𝑒(0)𝑔   and𝑒(0)𝑐   are64dimensional
tation𝑒𝑔𝑝,𝑒𝑐𝑝 and𝑒𝑐𝑙𝑖𝑛𝑝    for patient𝑝, these embedding are linked                    and initialized randomly. After obtaining the embedding𝑒′𝑝, it is
together as the final multimodal embedding𝑒′𝑝.                                         filled into the full connection layers with 3 layers, and each hidden
                                                                                       layer contains 200, 100 and 100 units.
                             𝑒′𝑝=𝑒𝑔𝑝||𝑒𝑐𝑝||𝑒𝑐𝑙𝑖𝑛𝑝                                       (6)EvaluationMetrics. We evaluate the MGNN with five metrics
   Then, the multimodal fusion embedding𝑒′𝑝with multiple fully                         including theArea Under theCurve (AUC) ofthe receiver operat-
connected layers is used to predict the survival of cancer patients:                   ing characteristics (ROC) curve,𝑆𝑒𝑛𝑠𝑖𝑡𝑖𝑣𝑖𝑡𝑦(𝑆𝑛),𝐴𝑐𝑐𝑢𝑟𝑎𝑐𝑦(𝐴𝑐𝑐),
                                                                                       𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛(𝑃𝑟𝑒)and Matthew’s correlation coefficient (𝑀𝑐𝑐).
                  ˆ𝑦𝑝=𝑠𝑜𝑓𝑡𝑚𝑎𝑥(𝜌(𝑊(𝑙)3 𝑒′(𝑙)𝑝 +𝑏(𝑙))),                  (7)             3.2   ResultsandAnalysis
where𝑊(𝑙)3    is the trainable weight matrix,𝜌denotes activation                       Weanalyzetheexperimentalresultsonbreastcancerdatasettover-
function𝑡𝑎𝑛ℎ,𝑒′(𝑙)𝑝    denotesthefinalmultimodalrepresentationfor                      ify the effectiveness of MGNN, including performance comparison,
patient𝑝at the layer𝑙, and𝑏(𝑙)is the bias vector. Finally we use                       robustness verification, and ablation tests.
softmax function and obtain the final prediction score ˆ𝑦𝑝.                            1Datasets areavailable athttps://www.cbioportal.org/

Table1:ComparisonofAcc,Pre,Sn,MccandAUCbetween                                            model anddescribedas only usinggene expressionand copy num-
LR,RFSVM,MDNNMDandMGNNmethods(Trainandtest                                                ber profile to make cancer survival prediction, respectively. As
onbreastcancerdataset)                                                                    shown in Figure 3, all the evaluation metrics of MGNN achieves
                                                                                          thebestperformance,whichshowsthatMGNNcaneffectivelyfuse
       Methods       Acc       Pre        Sn        Mcc      AUC                          multimodal data from different dimensions.
           LR           0.760     0.549     0.183     0.209     0.663
           RF           0.791     0.766     0.226     0.337     0.801
          SVM         0.805     0.708     0.365     0.407     0.810
      MDNNMD    0.826     0.749     0.450     0.486     0.845
        MGNN      0.940    0.953    0.969    0.837    0.970
   Performance Comparison. We compare the results of our
method for breast cancer survival prediction with that of four
state-of-the-art models including MDNNMD[6], SVM[7], RF[10]
and LR[11]. We directly use the experimentalresults of MDNNMD.
Table 1 show the experimental results of these models with the
evaluation metrics including𝐴𝑐𝑐,𝑃𝑟𝑒,𝑆𝑛,𝑀𝑐𝑐and𝐴𝑈𝐶 respec-                                      Figure3:AblationtestresultsoftheproposedMGNN.
tively.Comparedtothesebaselines,itcanbeseenthattheproposed
MGNN achieves the best performance among all methods. For ex-
ample, the accuracy of our method reaches 94%, which is higher                            4   CONCLUSIONS
11.4% than the second high accuracy of𝑀𝐷𝑁𝑁𝑀𝐷   method. It                                 In this paper, we propose a novel method MGNN for the cancer
shows the power of MGNN method to extract multimodal data                                 survival prediction of patients. The MGNN method utilizes the
features.Besides,graphneuralnetworkscaneffectivelyassemble                                features of multimodaldata with a unified framework.The experi-
bipartite graph structure interaction information for predicting the                      mental results showed the consistent performance improvement
survival of cancer patients.                                                              by the proposed method over the state-of-art baseline methods on
                                                                                          real-world breast cancer dataset and lung cancer dataset.
                                                                                          ACKNOWLEDGMENTS
                                                                                          TheworkissupportedbytheNationalNaturalScienceFoundation
                                                                                          of China under Grant No.: 61873288,61836016.
                                                                                          REFERENCES
                                                                                           [1]  Xiaoli Wang, Rongzhen Wang, Zhifeng Bao, Jiayin Liang, and Wei Lu. Effective
                                                                                               medicalarchivesprocessingusingknowledgegraphs. InSIGIR,pages1141–1144,
                                                                                               2019.
                                                                                           [2]  Yuexin Wu, YimingYang,Hiroshi Nishiura, andMasaya Saitoh. Deeplearning
                                                                                               forepidemiologicalpredictions. In SIGIR,pages1085–1088,2018.
                                                                                           [3]  Yawen Xiao, Jun Wu, Zongli Lin, and Xiaodong Zhao. A deep learning-based
                                                                                               multi-model ensemble method for cancer prediction.  Computer methods and
  Figure2:Robustnessverificationonlungcancerdataset.                                           programs in biomedicine,153:1–9,2018.
                                                                                           [4]  Anika Cheerlaand Olivier Gevaert. Deep learning withmultimodal representa-
                                                                                               tion forpancancerprognosisprediction. Bioinformatics,35(14):i446–i454, 2019.
   RobustnessVerification.To verifythe robustness ofthe pro-                               [5]  DongWookKim,SanghoonLee,SunmoKwon,andWoongNam. Deeplearning-
                                                                                               based survival prediction of oral cancer patients.  Scientific Reports, 9(1):1–10,
posedMGNN, we applyour modelto survival prediction for lung                                    2019.
cancer patients. The format of the lung cancer dataset is the same                         [6]  Dongdong Sun, MinghuiWang,and Ao Li. A multimodaldeep neural network
                                                                                               for human breast cancer prognosis prediction by integrating multi-dimensional
asthebreastcancerdataset.ThelungcancerdatasetcontainsCAN                                       data. TCBB,16(3):841–850,2018.
profileandclinicaldataof917lungcancerpatients.Ascanbeseen                                  [7]  XiaoyiXu,YaZhang,LiangZou,MinghuiWang,andAoLi. Agenesignaturefor
in Table 1,𝑀𝐷𝑁𝑁𝑀𝐷   method achieves the best performance ex-                                   breastcancerprognosisusingsupportvectormachine.InInternationalConference
                                                                                               on BioMedical Engineering and Informatics,pages928–931, 2012.
ceptourMGNN.Therefore,weselectitasthe baseline tocompare                                   [8]  Xiang Wang,XiangnanHe,MengWang,Fuli Feng,andTat-Seng Chua. Neural
the robustness. As shown in Figure 2, only CNA data is adopted                                 graphcollaborative filtering. InSIGIR,pages 165–174, 2019.
to construct a bipartite graph, and MGNN is used to obtain the                             [9]  PC Stone and S Lund. Predicting prognosis in patients with advanced cancer.
                                                                                               Annals of oncology,18(6):971–976,2007.
embedding representation of each patient on the bipartite graph.                          [10]  Cuong Nguyen, Yong Wang, and Ha Nam Nguyen.  Random forest classifier
Compared to MDNNMD, our proposed method MGNN has best                                          combined with feature selection for breast cancer diagnosis and prognostic.
                                                                                               Journal of Biomedical Science and Engineering,6:551–560,2013.
performance in terms of all evaluation metrics.                                           [11]  MilesFJefferson,NeilPendleton,SamBLucas,andMichaelAHoran.Comparison
   AblationTest. To further explore the effect of multimodal fu-                               of a genetic algorithm neural network with logistic regression for predicting
sion in our proposed model, we compare MGNN with GNN (gene)                                    outcomeaftersurgeryforpatientswithnonsmallcelllungcarcinoma. Cancer:
                                                                                               Interdisciplinary International Journal of the American Cancer Society,79(7):1338–
and GNN (CNA), where GNN (gene), GNN (CNA) are parts of our                                    1342,1997.

