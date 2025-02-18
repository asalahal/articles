AMA-GCN: Adaptive Multi-layer Aggregation Graph Convolutional Network for
                                                       Disease Prediction
                Hao Chen1 ,2 , Fuzhen Zhuang3 ,6 ∗ , Li Xiao1 ,2 ,5 ∗ , Ling Ma1 ,  Haiyan Liu4 ,
                                   Ruifang Zhang4 ,  Huiqin Jiang1 ∗  and Qing He1 ,2
                                          1Zhengzhou University, Zhengzhou, China
   2Key Lab of Intelligent Information Processing of Chinese Academy of Sciences (CAS), Institute of
                                  Computing Technology, CAS, Beijing 100190, China
                  3Institute of Artiﬁcial Intelligence, Beihang University, Beijing 100191, China
                    4The First Afﬁliated Hospital of Zhengzhou University, Zhengzhou, China
         5Ningbo Huamei Hospital, University of the Chinese Academy of Sciences, Ningbo, China
                               6Xiamen Data Intelligence Academy of ICT, CAS, China
   ielma@zzu.edu.cn, yanmai8023@126.com, zhangruifang999@hotmail.com, iehqjiang@zzu.edu.cn,chenhao199503@gs.zzu.edu.cn, zhuangfuzhen@buaa.edu.cn, andrew.lxiao@gmail.com,
                                                           heqing@ict.ac.cn
                           Abstract                                      landmarks in craniomaxillofacial [Lang et al., 2020], bone
     Recently, Graph Convolutional Networks (GCNs)                       age assessment [Gong et al., 2020], representation learning
     have proven to be a powerful mean for Computer                      for medical fMRI images [Gadgil et al., 2020] and so on. In
     Aided Diagnosis (CADx).  This approach requires                     this work, we focus on Computer Aided Diagnosis (CADx)
     building a population graph to aggregate structural                 [Kazietal.,2019b],whichusescomputertechnologytoassist
     information,wherethegraphadjacencymatrixrep-                        physicians in disease prediction.  At present, deep learning
     resents the relationship between nodes. Until now,                  methods have been widely used in disease prediction tasks
     this adjacency matrix is usually deﬁned manually                    and achieve great results, but in order to further improve di-
     based on phenotypic information.   In this paper,                   agnosticaccuracy,itisnecessarytomakefulluseofcomplex
     we propose an encoder that automatically selects                    medical multimodal data to extract the effective information
     the appropriate phenotypic measures according to                    hidden in it. These data usually include medical imaging and
     their spatial distribution, and uses the text simi-                 corresponding non-imaging phenotypic measures (e.g.  sub-
     larity awareness mechanism to calculate the edge                    ject’s age, height, or acquisition site), which are usually non-
     weightsbetweennodes. Theencodercanautomati-                         Euclidean and difﬁcult to be processed by traditional deep
     cally construct the population graph using pheno-                   learningmethods. Moreover,noteveryphenotypicmeasureis
     typic measures which have a positive impact on                      helpfulfordiseasepredictionanditisstillatediousandtime-
     the ﬁnal results, and further realizes the fusion of                consumingtaskforpeopletoselectphenotypicmeasuresthat
     multimodalinformation. Inaddition,anovelgraph                       can have a positive effect on classiﬁcation results.
     convolution network architecture using multi-layer                     Inspired by the success of GCNs in social network anal-
     aggregation mechanism is proposed. The structure                    ysis and recommendation systems [Bronstein et al., 2017],
     can obtain deep structure information while sup-                    graph-based methods are usually used in multimodal data
     pressing over-smooth, and increase the similarity                   processing.  At present, a large number of researchers have
     between the same type of nodes. Experimental re-                    made  great  contributions  to  apply  GCNs  in  CADx.    At
     sults on two databases show that our method can                     ﬁrst, the population graph constructed by the features ex-
     signiﬁcantly improve the diagnostic accuracy for                    tracted from the multimodal data is used as the input of
     Autism spectrum disorder and breast cancer, in-                     GCNs [Sarah et al., 2018].  However, the phenotypic mea-
     dicating its universality in leveraging multimodal                  sures selected in this method contribute the same to the edge
     data for disease prediction.                                        weights, but should actually be different.   Recently, there
                                                                         are two methods to solve this problem,  which are multi-
1   Introduction                                                         graph fusion methods [Vivar et al., 2018; Kazi et al., 2019a;
There is a growing body of researchers that have realized the            Kazi et al., 2019b] and single comprehensive graph methods
potentialforgraphconvolutionalnetworksinmedical-related                  [HuangandChung,2020]. Althoughthesemethodsdealwith
ﬁelds. Recently, GCNs have been widely used to solve a va-               the phenotypic measures differently, they both assign appro-
riety of medical problems already, including localization of             priate weights to it. For example, Huang et al. propose that
                                                                         an encoder can be built to calculate the connection between
   ∗ corresponding authors                                               nodesdirectlyusingmultimodaldata,andtheonlygraphcon-

structed in the end can be used as the input of GCNs.                       methods of different diseases are usually diverse (e.g., ultra-
  Although the above two methods can achieve high accu-                     sound imaging for the breast and fMRI for the brain), which
racy for disease prediction, their common limitation is that                makes this method difﬁcult to apply to the diagnosis of other
there is still no effective method to screen out phenotypic                 diseases.  Moreover, model learning only relies on medical
measuresthathaveanegativeimpactonclassiﬁcationresults.                      imaging, and does not take into account the rich non-image
Moreover,theparameterquantityofmulti-graphfusionmeth-                       phenotypic data, which does not conform to the diagnostic
ods will become very large with the increase in phenotypic                  habits of professional doctors in the actual situation.
measures, which seriously affects the real-time performance
of the model.  In addition, although existing methods, such                 2.2    Multimodal Data Based Approach
as the JK-GCN [Xu et al., 2018] or EV-GCN [Huang and                        In order to further improve the classiﬁcation accuracy, re-
Chung, 2020], have tried to apply layer aggregation mech-                   searchers study patients’ non-imaging phenotypic data while
anism to GCNs to learn more structural information, due to                  studyingmedicalimaging. Thiscomplexdataiscalledmulti-
theactualmedicalsituationbeingtoocomplex,itisstillchal-                     modaldata,whichisusuallynon-Euclideananddifﬁculttobe
lenging to determine an appropriate aggregation strategy.                   processed by traditional deep learning methods.  At present,
  In order to address the above challenges,  we present a                   GCNs are usually used to process multimodal data in the
new similarity-aware adaptive calibrated multi-layer aggre-                 medical ﬁeld.  This method needs to deﬁne the population
gation GCN structure called Adaptive Multi-layer Aggrega-                   graph where nodes represent patients, edges represent con-
tion GCN(AMA-GCN). As shown in Figure 1, this struc-                        nections between patients, and edge weights represent simi-
ture contains a specially designed encoder to select effective              laritybetweenpatients. Speciﬁcally,phenotypicmeasuresare
phenotypic measures and calculate the edge weights, namely                  usuallyusedtocalculatesimilaritybetweenpatients,andim-
phenotypic measure selection and weight encoder (PSWE).                     agefeaturesextractedfrommedicalimagingarestoredinthe
Besides, we propose two separately designed GCN models                      correspondingnodes. Furthermore,accordingtothedifferent
to aggregate the deep structural information and increase the               processing methods of phenotypic measures, GCNs methods
similarity between different objects of the same type, respec-              can be divided into multi-graph fusion methods and single
tively. This structure can improve the accuracy of the model                comprehensive graph methods. Multi-graph fusion methods
and its robustness.  The main contributions of our work can                 usually construct graphs for each phenotypic measure and
be summarized as follows:                                                   process them separately, and design different multi-graph fu-
   • We design PSWE to automatically select the best com-                   sion methods to assign weights to each phenotypic measure,
     binationofphenotypicmeasurestointerpretthesimilar-                     such as RNN [Vivar et al., 2018], self-attention [Kazi et al.,
     ity between subjects and calculate their scores.                       2019a],orLSTM[Kazietal.,2019b]. Singlecomprehensive
   • A multi-layer aggregation graph convolutional network                  graph methods directly extract features from the multimodal
     with multiple aggregation modes is introduced to con-                  data to build a comprehensive graph as input.  For example,
     sider more appropriate structural information for each                 Huang et al. [Huang and Chung, 2020] propose to use an en-
     node. And adynamic updatingmechanism is devisedto                      coder to process multimodal data.  The above methods have
     increase the similarity between nodes of the same type.                achievedgreatresults,buttherearestillsomelimitations: not
                                                                            every phenotypic measure has a positive effect on classiﬁca-
   • We test AMA-GCN on real-world dataset.  The results                    tion results. In fact, although our proposed method is also to
     showthatourmethodissuperiortoothermodelsknown                          buildasinglecomprehensivegraphtopredictdisease,itdoes
     at present in terms of validation set accuracy.                        not need to determine the appropriate phenotypic measures
                                                                            through a large number of experiments as in the past, but is
2   Related Work                                                            automatically completed by the proposed encoder.
In the past, disease classiﬁcation based on deep learning was               3   Methodology
usually achieved using medical imaging.  Recently, in order                 3.1    Preliminaries
to further improve the classiﬁcation accuracy, non-imaging
phenotypicdatahasalsobeenincludedintheresearchscope,                        Inourstudy,thepopulationgraphisdeﬁnedasanundirected
that is, medical multimodal data.                                           graphG = (V,E,A),whereV isthesetofverticesandeach
2.1    Medical Imaging Based Approach                                       vertex represents a patient;E  denotes the set of edges;A
                                                                            denotes the adjacency matrix of population graphG, whose
This method analyzes the medical imaging of patients from                   elements are the edge weights. As shown in Figure 1, the ad-
different perspectives, so as to obtain as many image fea-                  jacency matrixA is obtained by using the proposed PSWE,
tures as possible to improve diagnostic accuracy.   For ex-                 whichcanautomaticallyﬁndtheappropriatephenotypicmea-
ample, MSE-GCN [Yu et al., 2020] is proposed to extract                     sures and calculate the corresponding phenotypic measure
temporalandspatialinformationrespectivelyfromfMRIand                        selection scores (PMS-scores), and then calculate the edge
DTI to comprehensively analyze medical imaging. Zhang et                    weights. NotethattraditionalGCNsusuallyartiﬁciallyselect
al.  have achieved great result performance for nodule-level                phenotypic measures.
malignancy prediction by using the transfer learning method                    The node feature matrixX  is also the input of our model.
[Zhangetal.,2020]. Thismethodrequiresthattheinputdata                       We deﬁneX  asX  = (x1,x2,...,xn )∈R n×m  , wherende-
mustbespeciﬁctypesofmedicalimaging,whiletheimaging                          notes the number of samples,m  represents the dimension of

Figure1: Overallframeworkoftheproposedmethod. PSWE:phenotypicmeasureselectionandweightencoder. GC:graphconvolution. LA:
informationaggregationlayer. Colorsinthegraphs: greenandorange-labeleddiagnosticvalues(e.g.,healthyordiseased),grey: unlabeled.
the features. We select feature extracted from medical imag-                 graph’s adjacency matrixA is deﬁned as follows:
ingofthetwodatasetsinvolvedinourexperimentasthenode                                                     H∑
information.   For the ABIDE database, we use functional                                  A(v,w) =          αh∗γ(Kh (v),Kh (w))             (1)
connectivity derived from resting-state functional Magnetic                                            h =1
ResonanceImaging(rs-fMRI)forAutismSpectrumDisorder                           whereαh  is the PMS-scores of phenotypic measureKh ;γ
(ASD) classiﬁcation.  Rudie et al. [Rudie et al., 2013] pro-                 is a measure of the distance between the value of phenotypic
posed that ASD is linked to disruptions in the functional and                measureKh  of two graph nodes.αh  is deﬁned as follows:
structural organization of the brain, Abraham et al.  [Abra-                                                                       H∑
ham et al., 2016] further demonstrated this assertion.  More                        αh  =     H∗    n K h∑ H                                     (2)
accurately,weusethevectorizedfunctionalconnectivityma-                                               h =1  n K h, H∗nK  h≥h =1 nK  h,
trices as feature vectors.  For our collection of breast cancer                               0,         otherwise.
massultrasoundimagedata,weuseResNet-50toextractfea-                          wherenK  h  is a measure of the number of samples in which
tures of medical imaging and directly verify the performance                 phenotypic measureKh  meets the requirements, and it is de-
of these features on ResNet-50 and Ridge classiﬁer.                          ﬁned differently depending on the type of phenotypic mea-
3.2    Phenotypic Measure Selection and Weight                               sures.  WhenKh  is a non-quantitative phenotypic measure
       Encoder                                                               (e.g.  calciﬁcation or edema), we deﬁnenK  h  as a function
                                                                             with respect to a thresholdθ:
Edges are the channels through which the node ﬁlters in-                                           1   U∑   P∑                    u
formation from its neighbors and represent relationships be-                           nK  h  =    U               u,   n K hu −n K hpn K hpu  <θ,(3)
tween nodes.  Our hypothesis is that some appropriate non-                                           u =1 p=1 nK  hp
imaging phenotypic data can provide critical information as                                        0,          otherwise.
complement to explain the associations between subjects.                     wherenK  hp u  is a measure of the number of samples with the
Therefore,selectingthebestcombinationofphenotypicmea-                        valueuand categorypin the phenotypic measureKh ;nK  hu
sures to interpret the similarity between subjects and assign-               isameasureofthenumberofsampleswiththevalueuinthe
ing appropriate weights to them are the keys to our experi-                  phenotypicmeasureKh . Meanwhile,whenKh  isaquantita-
ment.  In our studies, it will be done by PSWE, and the im-                  tivephenotypicmeasure(e.g. subjectageorBMI),wedeﬁne
plementation process of PSWE as shown in Figure 2.                           nK  h  as a function with respect to a thresholdδ:
  Considering a set ofH  non-imaging phenotypic measures                                              P∑
K ={Kh},includingquantitative(e.g. subject’sage,height,                                                                       s
or BMI) and non-quantitative (e.g. subject’s calciﬁcation or                               nK  h  =    p=1 nK  hps,   n K hp −n K hpn K hps  <δ, (4)
capillary distribution) phenotypic measures. The population                                           0,     otherwise.

                                                                             vationfunction. TheﬁnaloutputZ∈R n×P denotesthelabel
                                                                             predictionforalldatainwhicheachrowZi denotesthelabel
                                                                             prediction for thei−thnode. The optimal weight matrices
                                                                             trained by minimizing the cross-entropy loss function as:
                                                                                                                   P∑
                                                                                               Lsemi    =−∑            Yij lnZij                     (6)
                                                                                                             i∈L  j=1
                                                                             whereL indicatesthesetoflabelednodes.Yij representsthe
                                                                             label information of the data.
                                                                                In addition,  we introduce an auxiliary dynamic update
                                                                             GCN to increase the similarity between nodes of the same
                                                                             types while encouraging competition. As shown in Figure 1,
Figure 2:  Overview of the proposed PSWE. u  i:  values of non-              this model’s output is also calculated by the softmax activa-
quantitative phenotypic measures corresponding to subject i.   s i:          tion function. The ﬁnal outputT∈R n×P  denotes the clas-
valuesofquantitativephenotypicmeasurescorrespondingtosubject                 siﬁcation score for all data in which each rowTi denotes the
i. p i: the category corresponding to subjecti.                              classiﬁcationscoreforthei−thnode. NotethatT istheout-
                                                                             put of the auxiliary model, which is different from the output
wherenK  hp     is a measure of the number of samples with the               Z ofthesemi-supervisedclassiﬁcationmodel. Thesimilarity
categorypinthephenotypicmeasureKh ;Wedeﬁneaclosed                            loss function as:
intervalDfromα toβ, whereD∈value(Kh ), andnK  hp                                                       ∑       ∑  P
                                                                   s  is               Lsim   = tanh       i∈L    j=1  (Yij−Tij)2 + ξ
a measure of the number of samples with the categorypand                                                               2σ2                              (7)
values/∈Din the phenotypic measureKh .                                       whereξis a minimal constant;σdetermines the width of the
  γis also deﬁned differently depending on the type of phe-                  kernel.
notypic measure. For non-quantitative phenotypic measures                       Then, the joint representation is used to compute a fusion
such as family history, we deﬁneγ as the Kronecker delta                     loss. Itposesextraregularizationthatcanhelpgeneralization
function,meaningthatthesimilarityhighlybetweentwosub-                        [Liuet al., 2020]. The ﬁnal loss function is deﬁned as:
jects if their values of phenotypic measureKh  are the same.
For quantitative phenotypic measures such as subject’s age,                                         L=Lsemi   + λLsim                              (8)
we deﬁneγas a function with respect to a closed intervalD                    whereLsemi    andLsim    are deﬁned in Eq.(6) and Eq.(7), re-
fromαtoβ, whereD∈value(Kh ):                                                 spectively.  Parameterλ≥0  is a tradeoff parameter with a
    γ(Kh (v),Kh (w)) =                                                    default value of 1.
       1,          Kh (v),Kh (w)/∈D,                                (5)
                1                                                            4   Experiments and Results
     e  3√|K h (v )−K h (w )|,|Kh (v)−Kh (w)|<β−α,                         4.1    Dataset
       0,          otherwise.                                                To verify the effectiveness of our model, we evaluate it on
  Theinﬂuenceofeffectivephenotypicmeasuresselectedby                         theAutismBrainImagingDataExchange(ABIDE)database
PSWE on the classiﬁcation results will be investigated in our                [Martino et al., 2014].   The ABIDE publicly shares fMRI
experiments, so as to visually demonstrate the performance                   and the corresponding phenotypic data (e.g., age and gen-
of PSWE.                                                                     der)  of  1112  subjects,  and  notes  whether  these  subjects
3.3    Model Structure Design                                                have Autism Spectrum Disorder (ASD). In order to com-
In order to make nodes from different dense blocks to obtain                 pare fairly with state-of-the-art [Huang and Chung, 2020]
sufﬁcienteffectiveinformationoneachlayerwhilesuppress-                       on the ABIDE, we select the same 871 subjects consist-
ing over-smooth, we reconstruct the architecture of the net-                 ing of 403 normal and 468 ASD individuals, and perform
work with multi-layer aggregation mechanism so that the in-                  the same data preprocessing steps [Huang and Chung, 2020;
formation from different layers can be adaptively fused into                 Sarah et al., 2018]. Then we delete the subjects with empty
the ﬁnal expression of the node.  As shown in Figure 1, in                   values, and ﬁnally select the 759 subjects consisting of 367
order to get the key features of structural information, the                 normal and 392 ASD individuals. Besides, approved by the
ﬁrst two aggregation layers that directly integrate the infor-               local Institutional Review Board, the local hospital provides
mation from graph convolution layers are aggregated in the                   ultrasound images of breast nodules and the corresponding
wayofmaxPooling. Besides,theﬁnalaggregationlayeruses                         phenotypic data (e.g., age, gender, and calciﬁcation) of 572
thewayofconcatingtofullysummarytheinformationaggre-                          sets, and follows the diagnostic results given by the radiol-
gated by each layer. The rebuilt model enables each node to                  ogist for each subject to note whether these subjects have
automatically integrate the appropriate information.                         Breast Cancer.  These data are acquired from 121 different
                                                                             patientsconsistingof55adenopathyand66breastcancerin-
  Theinputgraphdatah(0)   isprocessedbytheabovemodel,                        dividuals, and each set contains a total of six images (two
andtheoutputh(final   ) isthencalculatedbythesoftmaxacti-                    ultrasound static imaging and four ultrasound elastography).

          Hyperparameter description             Value                                                       ABIDE                 BCD
          Layer number of the MLA-GCN         5                                                          ACC     AUC     ACC     AUC
          Layer number of the ADU-GCN         2                                 ResNet-50                0.626    0.679    0.897    0.955
          Chebyshev polynomial                        3                         Ridge classiﬁer          0.636    0.688    0.901    0.961
          Number of node features                  2000                         GCN                      0.705    0.731    0.916    0.950
          Graph convolution kernel                   16                         JK-GCN                   0.722    0.736    0.947    0.972
          Learning rate of MLA-GCN            0.005                             GLCN                     0.707    0.725    0.932    0.965
          Learning rate of ADU-GCN              0.05                            EV-GCN                   0.829    0.876    0.967    0.987
          Regularization parameter                0.0005                        EV-GCN+PS                0.936    0.949    0.971    0.989
          Dropout probability                            0.3                    AMA-GCN                  0.984    0.983    0.994    0.998
          Number of training epoch                  300                         AMA-GCN(noP)             0.724    0.747    0.956    0.979
          Tradeoff parameterλ                           1                       AMA-GCN(noW)             0.955    0.974    0.974    0.987
          Optimizer                                         Adam                AMA-GCN(noA)             0.972    0.986    0.976    0.983
                                                                                AMA-GCN(noS)             0.981    0.981    0.990    0.992
Table 1:  Experiment hyperparameter setting.  MLA-GCN: multi-
layer  aggregation  GCN.  ADU-GCN:  auxiliary  dynamic  update              Table 2:  Quantitative comparisons between different methods on
GCN.                                                                        ABIDE and BCD.
Weremovecaseswithoutcompletephenotypicmeasuresand                           AMA-GCN(noA) is a model that uses PSWE to build the
adjustallimagesto256×256,andthenorganizethecollected                        population graph, but uses a two-layer GCN for training.
dataintothebreastcancerdetectiondataset(BCD)andverify                       AMA-GCN(noS)isamodelwithoutthesimilaritylossfunc-
the universality of proposed model on it.                                   tion.
4.2    Baseline Methods and Settings                                          In order to ensure a fair comparison, when we do not use
We compare the AMA-GCN with the following baselines:                        PSWE to select the appropriate phenotypic measures,  we
ResNet-50[Heetal.,2016]: Asinglemodalityclassiﬁcation                       choosegenderandacquisitionsitesasthebasisforconstruct-
approach using only images.                                                 ingthepopulationgraphofABIDEdatabase,andchooseage
Ridge classiﬁer [Abraham et al., 2016]:  A single modal-                    as the basis for constructing the population graph of BCD
ityclassiﬁcationapproach using onlyfeaturesextractedfrom                    dataset. This setting applies to all baseline models. The hy-
medical imaging data.                                                       perparameters of the experiment are shown in Table 1.  We
GCN [Sarah et al., 2018]: A model extracting features con-                  employ 10-fold cross-validation to evaluate the performance
tained in medical multimodal data, which is usually used to                 of the model and implement our model using TensorFlow.
deal with non-Euclidean data.                                               In order to evaluate the performance of models, we choose
JK-GCN [Xu et al., 2018]:  A model using an aggregation                     overall accuracy (ACC) and area under the curve (AUC) as
layer before the output layer to integrate information.                     the evaluation indicators.
GLCN [Jiang et al., 2020]:  A model using graph learning                    4.3    Results and Analysis
mechanismtoconstantlyoptimizegraphstructuretoimprove                        We compare our AMA-GCN with the six baseline methods
the classiﬁcation effect.                                                   forpredictiveclassiﬁcationofdiseaseontheABIDEdatabase
EV-GCN [Huang and Chung, 2020]: A model using a cus-                        and BCD dataset, as shown in Table 2.  It can be observed
tom encoder to obtain the association between nodes from                    thatsingle-modemodels(i.e. ResNet-50andRidgeclassiﬁer)
non-imaging phenotypic data, and JK-GCN is used to aggre-                   only use medical imaging data as the basis for classiﬁcation,
gate information.                                                           and their overall performance is poor. Comparatively, graph-
EV-GCN+PS: A model using the phenotypic measures se-                        based methods (GCN, JK-GCN, GLCN, EV-GCN and ours)
lected by proposed PSWE as the basis for constructing the                   yieldlargerperformancegains,beneﬁtingfromanalyzingas-
populationgraph,andusingEV-GCNtoextractstructuralin-                        sociations between nodes in the population graphs. The pro-
formation.                                                                  posed method, AMA-GCN, obtains an average accuracy of
Ablation Study                                                              98.4%   and99.4%   onABIDEandBCDdatasets,respectively,
To investigate how the PSWE, the multi-layer aggregation                    outperformingtherecentSoTAmethodEV-GCN,whichem-
mechanism and similarity loss function improve the perfor-                  ploysanadaptivepopulationgraphwithvariationaledgesand
mance of the proposed model, we conduct the ablation study                  uses JK-GCN to aggregate structure information. We notice
on the following variants of AMA-GCN:                                       that the performance of graph-based methods is highly sen-
AMA-GCN(noP) is a model that uses the same phenotypic                       sitive to the phenotypic measures used to construct graphs,
measureswithbaselinemethodstobuildthepopulationgraph                        where the phenotypic measuresKh  ={gender, acquisition
as input and uses proposed model for training.                              sites}on ABIDE database used by EV-GCN yields an aver-
AMA-GCN(noW) is a model that only uses PSWE to select                       ageaccuracyof82.9%  . Toprovetheeffectivenessofthephe-
effectivephenotypicmeasures,andusesatwo-layerGCNfor                         notypic measures selected by our PSWE, we train EV-GCN
training.  Note that the weight ratio calculated by PSWE is                 using the same phenotypic measures as those in our model.
not used.                                                                   AsdepictedinTable2,itresultsin10.7%   accuracyand7.3%

  (a) ACC of the ABIDE dataset      (b) AUC of the ABIDE dataset               (a) ACC of the ABIDE dataset        (b) ACC of the BCD dataset
                                                                             Figure 4: Effects of phenotypic measures not selected by PSWE on
                                                                             results, the evaluation indicators is Accuracy (ACC).
                                                                             in Figure 3, using the single phenotypic measure selected by
                                                                             PSWEtoconstructthepopulationgraphasinput,theirperfor-
                                                                             mances on ABIDE and BCD datasets are respectively more
                                                                             than 3%   and 1%   better than that of randomly constructed
   (c) ACC of the BCD dataset         (d) AUC of the BCD dataset             graph.  Comparatively, as shown in Figure 4, using pheno-
                                                                             typic measures not selected by PSWE as the basis for con-
Figure 3:   Effects of effective phenotypic measures selected by             structing graphs, the ﬁnal performance is basically the same
PSWEonresults,theevaluationindicatorsareAccuracy(ACC)and                     as that of randomly generated graphs, and even worse than
Area Under Curve(AUC).                                                       the latter, which indicates that these phenotypic measures se-
                                                                             lectedbyPSWEarereasonable. Meanwhile,PSWEachieves
AUC improvement on ABIDE database respectively, indi-                        the best performance by combining multiple effective phe-
cating that the appropriate phenotypic measures are indeed                   notypic measures, and its accuracy on ABIDE database is
the key to improving the performance of disease prediction.                  even 33.8%   higher than that of randomly constructed graph.
Meanwhile, the universality in leveraging multimodal data                    This indicates that there is hidden complementary informa-
for disease prediction of the proposed AMA-GCN architec-                     tion among phenotypic measures,  and how to learn these
tureisrelativelyvalidatedaccordingtothecomparisonresults                     complementary information is the key to disease prediction.
on BCD dataset in Table 2.                                                   5   Conclusion
Ablation Experiments                                                         In  this  paper,  we  have  proposed  a  generalizable  graph-
As an ablation study, we test whether removing the PSWE,                     convolutional framework that combines multimodal data to
the  multi-layer  aggregation  mechanism  or  similarity  loss               predict disease.   We designed the population graph struc-
function affect model performance, and the results as shown                  ture according to the spatial distribution and text similarity
in Table 2. It can be observed that the phenotypic measures                  of phenotypic measures, while allowing each effective phe-
selected by PSWE can signiﬁcantly improve the classiﬁca-                     notypic measure to contribute to the ﬁnal prediction. We re-
tion accuracy.  Using only these phenotypic measures with-                   constructed the graph convolution model by using the multi-
outconsideringPMS-scores,theaverageaccuracyofthecon-                         layer aggregation mechanism to automatically ﬁnd the opti-
structedgraphsontheABIDEandBCDdatasetsisimproved                             mal feature information from each layer while suppressing
by25.0%   and5.8%  ,respectively. Theclassiﬁcationaccuracy                   over-smooth, and introduce another channel to increase the
is further improved by 1.7%   and 0.2%   on ABIDE and BCD                    similarity between different objects in the same type. Exper-
datasets respectively after calculating PMS-scores to the se-                imental results show that the proposed method achieves su-
lected phenotypic measures. Although the latter can also im-                 perior performance on brain analysis and breast cancer pre-
prove the classiﬁcation accuracy, the improvement is far less                diction. We believe that such an extensible method can have
than the former, which indicates that ﬁnding the appropriate                 a better use of helping people with medical multimodal data
phenotypicmeasuresisthekeytodiseaseprediction. Inaddi-                       for clinical computer-aided diagnosis.
tion, the effectiveness of multi-layer aggregation mechanism
in improving classiﬁcation accuracy is also validated accord-                Acknowledgments
ing to the comparison results in Table 2.
4.4    Phenotypic Measures Analysis                                          This work was supported by the National Natural Science
                                                                             Foundation of China (31900979,U1604262),  CCF-Tencent
Inordertodemonstrateourconclusionthatﬁndingtheappro-                         Open Fund, the National Key Research and Development
priate phenotypic measures is the key to solving the problem                 Program of China under Grant (2017YFB1002104), the Na-
of disease classiﬁcation, and also to prove the effectiveness                tional  Natural  Science  Foundation  of  China  under  Grant
of proposed PSWE, we further explore the impact of each                      (U1811461),andZhengzhouCollaborativeInnovationMajor
phenotypic measure on the classiﬁcation results.  As shown                   Project (20XTZX11020).

References                                                                    imaging data exchange:  towards a large-scale evaluation
[Abrahamet al., 2016] Alexandre  Abraham,  Michael  Mil-                      of the intrinsic brain architecture in autism.   Molecular
   ham, Adriana Di Martino, Cameron Craddock, Dimitris                        Psychiatry, 2014.
   Samaras, Bertrand Thirion, and Gael Varoquaux.  Deriv-                  [Rudieet al., 2013] Jeffrey David Rudie, Jesse Brown, Devi
   ing reproducible biomarkers from multi-site resting-state                  Beck-Pancer,  Leanna  Hernandez,  Emily  Dennis,  Paul
   data:  An autism-based example.  NeuroImage, 147:736,                      Thompson, Susan Bookheimer, and Mirella Dapretto. Al-
   2016.                                                                      tered functional and structural brain network organization
[Bronsteinet al., 2017] Michael   Bronstein,   Joan   Bruna,                  in autism. NeuroImage : Clinical, 2, 2013.
   Yann LeCun, Arthur Szlam, and Pierre Vandergheynst.                     [Sarahet al., 2018] Parisot Sarah, Ktena Soﬁa Ira, Ferrante
   Geometric deep learning:  Going beyond euclidean data.                     Enzo, Lee Matthew, Guerrero Ricardo, Glocker Ben, and
   IEEE Signal Processing Magazine, 34(4):18–42, 2017.                        RueckertDaniel. Diseaseprediction usinggraphconvolu-
[Gadgilet al., 2020] SohamGadgil,QingyuZhao,AdolfPf-                          tional networks: Application to autism spectrum disorder
   efferbaum, Edith Sullivan, Ehsan Adeli, and Kilian Pohl.                   and alzheimer’s disease.  Medical Image Analysis, pages
   Spatio-temporalgraphconvolutionforfunctionalmrianal-                       S1361841518303554–, 2018.
   ysis. arXiv, 2020.                                                      [Vivaret al., 2018] Gerome Vivar,  Andreas Zwergal,  Nas-
[Gonget al., 2020] Ping Gong,  Zihao Yin,  Yizhou Wang,                       sir Navab, and Seyed-Ahmad Ahmadi.  Multi-modal Dis-
   and Yizhou Yu.  Towards Robust Bone Age Assessment:                        easeClassiﬁcationinIncompleteDatasetsUsingGeomet-
   Rethinking Label Noise and Ambiguity.   Medical Image                      ric Matrix Completion. Springer, Cham, 2018.
   Computing and Computer Assisted Intervention – MIC-                     [Xu et al., 2018] Keyulu Xu, Chengtao Li, Yonglong Tian,
   CAI 2020, 2020.                                                            Tomohiro Sonobe, Ken Ichi Kawarabayashi, and Stefanie
[He et al., 2016] Kaiming  He,  Xiangyu  Zhang,  Shaoqing                     Jegelka. Representation learning on graphs with jumping
   Ren, and Jian Sun.   Identity mappings in deep residual                    knowledge networks. arXiv, 2018.
   networks. Springer, Cham, 2016.                                         [Yu et al., 2020] Shuangzhi Yu,  Shuqiang Wang,  Xiaohua
[Huang and Chung, 2020] Yongxiang   Huang   and   Albert                      Xiao, Jiuwen Cao, Guanghui Yue, Dongdong Liu, Tianfu
   Chung. Edge-variationalgraphconvolutionalnetworksfor                       Wang, Yanwu Xu, and Baiying Lei. Multi-scale enhanced
   uncertainty-aware disease prediction.   In Medical Image                   graph convolutional network for early mild cognitive im-
   ComputingandComputerAssistedIntervention–MICCAI                            pairment detection. Springer, Cham, 2020.
   2020, 2020.                                                             [Zhang et al., 2020] Hanxiao  Zhang,  Yun  Gu,  Yulei  Qin,
[Jianget al., 2020] Bo Jiang, Ziyan Zhang, Doudou Lin, Jin                    Feng Yao, and Guang-Zhong Yang.  Learning with sure
   Tang, and Bin Luo. Semi-supervised learning with graph                     data for nodule-level lung cancer prediction.  In Medical
   learning-convolutional  networks.     In  2019  IEEE/CVF                   Image Computing and Computer Assisted Intervention –
   Conference on Computer Vision and Pattern Recognition                      MICCAI 2020, 2020.
   (CVPR), 2020.
[Kaziet al., 2019a] Anees  Kazi,  Arvind  Krishna,  Shayan
   Shekarforoush,KarstenKortuem,andNassirNavab. Self-
   attention equipped graph convolutions for disease predic-
   tion.    In 2019 IEEE 16th International Symposium on
   Biomedical Imaging (ISBI), 2019.
[Kaziet al., 2019b] Anees   Kazi,   Shayan   Shekarforoush,
   Arvind Krishna, Hendrik Burwinkel, and Nassir Navab.
   Graph Convolution Based Attention Model for Personal-
   ized Disease Prediction. Springer, Cham, 2019.
[Langet al., 2020] Yankun Lang, Chunfeng Lian, Deqiang
   Xiao, Hannah Deng, and Dinggang Shen.  Automatic Lo-
   calizationofLandmarksinCraniomaxillofacialCBCTIm-
   ages Using a Local Attention-Based Graph Convolution
   Network.  Medical Image Computing and Computer As-
   sisted Intervention – MICCAI 2020, 2020.
[Liuet al., 2020] Shaoteng Liu, Lijun Gong, Kai Ma, and
   Yefeng Zheng.   Green:  a graph residual re-ranking net-
   work for grading diabetic retinopathy. arXiv, 2020.
[Martinoet al., 2014] Adriana Di Martino, Chao-Gan Yan,
   Qingyang Li, Erin Denio, Francisco Xavier Castellanos,
   Kaat  Alaerts,  Jason  Anderson,  Michal  Assaf,  Susan
   Bookheimer,  and Mirella Dapretto.    The autism brain

