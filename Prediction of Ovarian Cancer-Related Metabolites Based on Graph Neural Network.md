fcell-09-753221        September 29, 2021        Time: 16:52        # 1
                                                                                                                                                     ORIGINAL RESEARCH
                                                                                                                                                  published: 05 October 2021
                                                                                                                                               doi: 10.3389/fcell.2021.753221
                                                      Prediction of Ovarian
                                                      Cancer-Related Metabolites Based
                                                      on Graph Neural Network
                                                      Jingjing Chen1†, Yingying Chen1†, Kefeng Sun1†, Yu Wang1, Hui He1, Lin Sun2, Sifu Ha2,
                                                      Xiaoxiao Li3, Yifei Ou3, Xue Zhang4* and Yanli Bi5*
                                                      1Department of Obstetrics and Gynecology, First Afﬁliated Hospital, Heilongjiang University of Chinese Medicine, Harbin,
                                                      China, 2Department of Reproductive Medicine, Dalian Maternal and Children’s Centre, Dalian, China, 3Graduate School of
                                                      Heilongjiang University of Chinese Medicine, Harbin, China, 4Department of General Practice, Beijing Friendship Hospital,
                                                      Capital Medical University, Beijing, China, 5Department of Reproductive Medicine, The First Afﬁliated Hospital, Henan
                                                      University of Chinese Medicine, Zhengzhou, China
                                                      Ovarian cancer is one of the three most malignant tumors of the female reproductive
                                                      system. At present, researchers do not know its pathogenesis, which makes the
                                                      treatment effect unsatisfactory. Metabolomics is closely related to drug efﬁcacy, safety
                                     Editedby:        evaluation, mechanism of action, and rational drug use. Therefore, identifying ovarian
                                       Lei Deng,      cancer-relatedmetabolitescouldgreatlyhelpresearchersunderstandthepathogenesis
                  Central South University, China
                                                      and develop treatment plans. However, the measurement of metabolites is inaccurate
                                  Reviewedby:
                                       Fei Shen,      and greatly affects the environment, and biological experiment is time-consuming and
            South China University of Technology,     costly. Therefore, researchers tend to use computational methods to identify disease-
                                          China       related metabolites in large scale. Since the hypothesis that similar diseases are related
                                       Sheng Li,
                         Wuhan University, China      to similar metabolites is widely accepted, in this paper, we built both disease similarity
                             *Correspondence:         networkandmetabolitesimilaritynetworkandusedgraphconvolutionalnetwork(GCN)
                                      Xue Zhang       to encode these networks. Then, support vector machine (SVM) was used to identify
                              Vowwzx@163.com          whether a metabolite is related to ovarian cancer. The experiment results show that the
                                         Yanli Bi
                          biyanli.mary@163.com        AUC and AUPR of our method are 0.92 and 0.81, respectively. Finally, we proposed an
                †These authors have contributed       effective method to prioritize ovarian cancer-related metabolites in large scale.
                              equally to this work
                                                      Keywords: ovarian cancer, metabolite, Graph convolutional network, support vector machine, prediction
                             Specialtysection:
                    This article was submitted to
                Molecular and Cellular Pathology,    INTRODUCTION
                          a section of the journal
              Frontiers in Cell and Developmental    Ovarian cancer is a common gynecological malignancy and one of the deadliest female diseases.
                                         Biology     Becausetheunderlyingsymptomsarenotobvious,about70%ofovariancancerpatientsarealready
                     Received: 04 August 2021        at an advanced stage when they are diagnosed (Liu et al.,2020). The survival rates of patients
                     Accepted: 27 August 2021        with ovarian cancer at diﬀerent stages are very diﬀerent, and the mortality rate of patients with
                    Published: 05 October 2021       advancedstagesexceeds75%(Hussain,2020).Therefore,thereisanurgentneedtoﬁndmetabolites
                                       Citation:     related to ovarian cancer to improve the prognosis of ovarian cancer and improve the eﬃciency of
                 Chen J, Chen Y, Sun K, Wang Y,      individualized treatment of patients (Perrone et al.,2020). Many life activities in cells occur at the
                   He H, Sun L, Ha S, Li X, Ou Y,    metabolite level, so metabolomics has become one of the current research hotspots in the ﬁeld
              Zhang X and Bi Y (2021) Prediction     of omics (Blimkie et al.,2020). The research of metabolomics in the early diagnosis of malignant
                       of Ovarian Cancer-Related     tumorshasshownitsadvantages(Agakidouetal.,2020).Ovariancancerisadiseasewithaveryhigh
             Metabolites Based on Graph Neural
                                       Network.      mortality rate of gynecological malignancies. There is an urgent need for a method to diagnose the
                  Front. Cell Dev. Biol. 9:753221.   diseaseearly.Theapplicationofmetabolomicsinovariancancercanprovideideasforthediagnosis
                  doi:10.3389/fcell.2021.753221      and prevention of ovarian cancer.
           Frontiers in Cell and Developmental Biology|www.frontiersin.org                                         1                                                                October2021|Volume 9|Article 753221

fcell-09-753221        September 29, 2021        Time: 16:52        # 2
           Chen et al.                                                                                                                                                                               Predict Ovarian Cancer-Related Metabolites
               The analysis of the metabolites caused by the disease will help                 or deep learning methods to solve biological problems (Chen
           us to more comprehensively grasp the process of disease changes                     etal.,2019,2020;Zhaoetal.,2020b).Disease-relatedgenes(Peng
           and the metabolic pathway of substances in the body, so as to                       and Zhao,2020;Zhao et al.,2021), RNAs (Gebauer et al.,2021),
           make the clinical diagnosis more accurate.Zhou et al.(2010)                         proteins  (Zhao  et  al., 2020c),  and  drugs  (Tianyi  et  al., 2020)
           collected 44 serous papillary ovarian cancer (stage I–IV) and                       have all been identiﬁed by computational methods in large scale,
           50 healthy women and found that histamine, purine nucleotide,                       whichsigniﬁcantlyincreasesthespeedofdiscoveringknowledge.
           glycine, serine, and sarcosine were the diﬀerential metabolites,                    Predictingdisease-relatedmetabolitesbycomputationalmethods
           and   alanine,   serine,   cysteine,   threonine,   and   glycine   were            has become a hot issue in recent years.Hu et al.(2018)used
           overexpressed.Fongetal.(2011)foundthatthereare364kindsof                            random walk to identify disease-related metabolites by similarity
           biochemicalsubstancesinhumanovarianmetabolictissuebygas                             network in 2018. Following this research,Wang et al.(2019)
           chromatography–massspectrometryandliquidchromatography                              fused  text  mining  technology  with  random  walk  to  further
           tandem mass spectrometry. Ovarian transformation can cause                          infer relationship between metabolites and diseases. Then,Peng
           changes in energy utilization, resulting in glycolysis and fatty                    and Zhao(2020)developed “MDBIRW,” which is an improved
           acids (such as carnitine, acetylcarnitine, and butyrylcarnitine)                    random  walk  method  to  identify  disease-related  metabolites.
           β -oxidation changes. Based on the non-targeted metabonomics                        However, these methods all traverse network by random walk,
           method  of  LC/MS,Chen  et  al.(2011) analyzed  the  serum                          which  did  not  fully  extract  the  topological  relationship  of
           samples  of  27  healthy  women,  28  cases  of  benign  ovarian                    similarity network. In 2020,Zhao et al.(2020a)proposed “Deep-
           tumor, and 29 cases of epithelial ovarian cancer. β -Cholestane-                    DRM,”  which  used  Graph  convolutional  network  (GCN)  to
           3,7,12,24,25,  pentose  glucoside,  phenylalanine,  glycine  cholic                 encode similarity network and achieved high accuracy. In this
           acid,  and  propionyl  carnitine  are  potential  biomarkers  for                   paper,wefollowedthisresearchandfocusedonovariancancerto
           epithelial ovarian cancer.                                                          providesupportforthetreatmentanddiagnosisofovariancancer
               Garcia et al.(2011)used the 1H NMR method to analyze                            by prioritizing metabolites.
           the concentration of alanine, valine, phospholipid choline, etc.,
           from the serum of 170 healthy women of appropriate age and
           182 ovarian cancer stage I/II patients, while β -hydroxybutyrate,                   MATERIALS AND METHODS
           acetone,   and   acetoacetic   acid   have   higher   concentrations.               Deep-DRM is a method that fuses GCN, principal component
           These  can  be  qualitatively  compared  with  the  changes  in                     analysis (PCA), and deep neural network (DNN). Considering
           the  concentration  distribution  of  serum  samples  of  cancer                    we only focus on ovarian cancer, the sample set would be much
           patients   studied   by   other   NMR-based   metabolomics.   This                  smaller. Therefore, we used support vector machine (SVM) to
           proves that early diagnosis of ovarian cancer can signiﬁcantly                      replaceDNNtobuildamodelwithasmallsample.Theworkﬂow
           aﬀect  the  clinical  outcome  of  patients  with  ovarian  cancer.                 of our method is shown in Figure 1.
           Chen   et   al.(2011)  analyzed   the   serum   samples   of   27
           healthy  women,  28  cases  of  benign  ovarian  tumors,  and  29
           cases  of  epithelial  ovarian  cancer  using  LC/MS  combined                      Metabolite and Disease Similarity
           analysis, liquid chromatography selective ion monitoring mass                       Network
           spectrometry   technology   combined   with   PCR,   and   other                    We  used  “PaDEL-Descriptor”  (Yap, 2011)  to  estimate  the
           pattern recognition techniques. The study found that 27-nor-                        chemical property of metabolites by their chemical structure.
           5β -cholestane-3,7,12,24,25 pentanol glucuronide can be used in                     The  output  of  this  tool  includes  1D  and  2D  descriptors  and
           the early diagnosis of epithelial ovarian cancer. It is elevated in                 ﬁngerprints. Each metabolite could be represented as a vector of
           the serum of early epithelial ovarian cancer (stage I).Gaul et al.                  2,325 dimension in this way.
           (2015)used ultra-high performance liquid chromatography and                                                   mi=[v1,v2,....,v2325]                            (1)
           high-resolution mass spectrometry from 46 early (I/II) serous
           epithelial ovarian cancer (EOC) patients and 49 age-matched                         The similarity of each of the two metabolites could be calculated
           normal healthy female controls. UPLC-MS and tandem mass                             by vector cosine.
           spectrometry (MS/MS) methods found that 16 metabolites in                                                                   ∑   2325
           lipids and fatty acids have 100% accuracy in the diagnosis of                                                                  k ˆmki×ˆmkj
           early-stage ovarian cancer patients.Woo et al.(2009)tested the                                 sim(ˆmi,ˆmj)=√        ∑   2325          √  ∑   2325
           metabolitesinurineof10breastcancerpatients,9ovariancancer                                                               k   (ˆmki)2×         k   (ˆmkj)2          (2)
           patients, 12 cervical cancer patients, and 22 normal controls.
           They  found  that  1-Methyladenosine  is  a  powerful  biomarker                    Using the similarities of metabolites, we could build a metabolite
           for diagnosing ovarian cancer.Zhang et al.(2012)found that                          similaritynetwork.Inthenetwork,eachnodeisametaboliteand
           2-Piperidinone could be used to distinguish epithelial ovarian                      each edge is the similarity between the two metabolites.
           cancer (EOC) and benign ovarian tumor (BOT).                                            Cheng et al.(2014)proposed SemFunsim to calculate disease
               Although multiple metabolites have been found to be related                     similarity.  We  used  their  results  to  build  an  ovarian  cancer
           to ovarian cancer, the time and money cost of this discovery                        similarity network. All the nodes in this network are diseases
           is  huge.  With  the  development  of  computational  method,                       similar to ovarian cancer. The edges are the similarities between
           increasing number of researchers try to use machine learning                        ovarian cancer and other diseases.
           Frontiers in Cell and Developmental Biology|www.frontiersin.org                                         2                                                                October2021|Volume 9|Article 753221

fcell-09-753221        September 29, 2021        Time: 16:52        # 3
            Chen et al.                                                                                                                                                                               Predict Ovarian Cancer-Related Metabolites
              FIGURE 1 | Workﬂow of GPS-OCM (the fusion of GCN, PCA, and SVM to identify ovarian cancer-related metabolites).
            Feature Encoding by Graph                                                             Feature Dimensionality Reduction
            Convolutional Network                                                                 Since the dimension of metabolites and ovarian cancer features
            Graph   convolutional   network   was   implemented   on   both                       are large, we used PCA to reduce the dimension.
            metabolite  and  disease  similarity  networks,  respectively.  The                       Principal  component  analysis  reduces  the  n-dimensional
            GCN-based network feature extraction method can convert the                           input data to r-dimensional, where r <   n. PCA is essentially
            network  structure  into  a  vector  output  through  a  non-linear                   a basis transformation, so that the transformed data have the
            function:                                                                             largest  variance,  that  is,  by  rotating  the  coordinate  axis  and
                                         H(l+1)=f(H(l),A)                                  (3)    translatingtheoriginofthecoordinate,thevariancebetweenone
                                                                                                  oftheaxes(mainaxis)andthedatapointisminimized.Afterthe
            H(0)=X which is the initial feature of each node.                                     coordinate conversion, the orthogonal axis with high variance is
                First, we need to perform Laplacian changes on the network,                       removed, and the dimensionality reduction data set is obtained.
            andthecorrespondingLaplacianmatrixcalculationformulaisas                                  The SVD method is used to perform PCA dimensionality
            follows:                                                                              reduction.  Assuming  that  there  are  p×n-dimensional  data
                                              L=D−A                                          (4)  samples  X,  there  are  p  samples  in  total,  and  each  row  is
                                                                                                  n-dimensional. The p×n real matrix can be decomposed into:
            Among them, D is the degree matrix of the graph, which can be                                                         X=U∑    VT                                       (8)
            solved by formula 5. A is the adjacency matrix.
                                             Dii=∑ˆ      Aij                                         (5)ˆHere, the dimension of the orthogonal matrix U is p×n, the
                                                      j                                           dimension of the orthogonal matrix V is n×n (orthogonal
                                                                                                  matrix satisﬁes:  UUT=VVT=1), and 6   is a diagonal matrix
            Since D is a diagonal matrix, only its diagonal elements need to                      of n×n. Next, divide6   into r columns, denoted as6  r; use U
            be solved, and the remaining elements are all 0.                                      and V to get the dimensionality reduction data point Yr:
                Then, we need to normalize the Laplacian matrix:                                                                    Yr=U∑                                        (9)
                              Lsym=D−12LD−12=I−D−12AD−12                    (6)                                                                 r
            The ﬁnal formula of GCN would be:                                                     AfterPCA,99%ofthefeatureinformationarepreservedforboth
                                                                                                  metabolites and ovarian cancer.
                                H(l+1)=σ(D−12AD−12H(l)W(l))                       (7)             Identify Ovarian Cancer-Related
            σ() is the activation function,W(l) the parameter to be trained.                      Metabolites
                Finally,  we  obtained  the  encoded  feature  of  metabolites                    After extracting and reducing the features of metabolites and
            and ovarian cancer.                                                                   ovarian cancer, we need to combine features of metabolites and
            Frontiers in Cell and Developmental Biology|www.frontiersin.org                                         3                                                                October2021|Volume 9|Article 753221

fcell-09-753221        September 29, 2021        Time: 16:52        # 4
               Chen et al.                                                                                                                                                                               Predict Ovarian Cancer-Related Metabolites
                  FIGURE 2 | Process of building the SVM model.
                  FIGURE 3 | (A) ROC curve of “SP.” Experiment. (B) ROC curve of “SM” experiment. (C) PR curve of “SP” experiment. (D) PR curve of “SM” experiment.
               ovarian cancer to make ovarian cancer–metabolite pairs. If the                                                   There are ﬁve steps to build the SVM model. The process is
               metabolite has relationship with ovarian cancer, the label of this                                          shown in Figure 2.
               pair would be 1; otherwise, the label is 0.
               TABLE 1 | Comparison experiment of GPS-OCM and other methods on                                             EXPERIMENT RESULTS
               the “SP” test.
               Method                                                   AUC                                            AUPRSince   we   only   focus   on   ovarian   cancer,   we   divided   our
                                                                                                                           experiments into two classes. One is to identify ovarian cancer-
               GPS-OCM                                               0.92                                             0.81 related  metabolites  from  known  disease-related  metabolites,
               RWPS-OCM                                            0.83                                             0.70   which is named as “SP.” The other one is to identify ovarian
               GPR-OCM                                               0.87                                             0.73 cancer-related metabolites from metabolites associated with no
               GPD-OCM                                               0.90                                             0.81 disease, which is names as “SM”.
               Frontiers in Cell and Developmental Biology|www.frontiersin.org                                         4                                                                October2021|Volume 9|Article 753221

fcell-09-753221        September 29, 2021        Time: 16:52        # 5
            Chen et al.                                                                                                                                                                               Predict Ovarian Cancer-Related Metabolites
                We   did   10-cross   validation   on   both   “SP”   and   “SM”                     addition,  metabolites  in  blood  and  urine  have  shown  strong
            experiments.  The  AUC  and  AUPR  of  these  experiments  are                           power  in  diagnosing  cancer  in  early  stage  as  biomarkers.
            shown in Figure3.                                                                        However,   few   metabolites   associated   with   ovarian   cancer
                The AUC of “SP” and “SM” experiments is 0.9168 and 0.9282,                           have   been   found   at   present.   In   order   to   speed   up   the
            respectively.TheAUPRof“SP”and“SM”experimentsis0.81and                                    study  of  metabolites  related  to  ovarian  cancer,  we  proposed
            0.83, respectively.                                                                      a    calculation    method    “GPS-OCM”    based    on    similarity
                To show the superiority of GPS-OCM, we compared GPS-                                 of   metabolites   and   diseases.   This   method   is   fusion   of
            OCM  with  RWPS-OCM,  GPR-OCM,  and  GPD-OCM.  We                                        GCN,  PCA,  and  SVM.  GCN  was  used  to  extract  network
            replacedGCNbyRandomWalk(RW)toconstructRWPS-OCM.                                          topology   features   and   PCA   was   implemented   to   reduce
            GPR-OCMisthefusionofGCN,PCR,andRandomForest(RF).                                         the   dimension   of   disease   and   metabolite   features.   SVM
            GPD-OCM is to replace SVM by deep neural network (DNN).                                  was  applied  to  do  classiﬁcation.  The  experiments  show  the
            The results are shown in Table 1. GPS-OCM performed best                                 high  accuracy  of  our  method  with  high  AUC  and  AUPR.
            among these  methods.                                                                    In   addition,   three   of   the   top   ﬁve   metabolites   that   are
                After verifying the eﬀectiveness of our method, we used all                          identiﬁed as ovarian cancer-related metabolites by our method
            known ovarian cancer-related metabolites as positive samples                             have  been  proven  by  previous  studies,  which  proved  the
            and  randomly  selected  equal  number  of  other  metabolites  as                       accuracy of our results.
            negative samples to build a ﬁnal GPS-OCM model. We totally
            identiﬁed 257 more metabolites that are associated with ovarian
            cancer. To verify whether these metabolites are associated with                          DATA AVAILABILITY STATEMENT
            ovarian cancer, we chose the top ﬁve of these metabolites and
            did case studies.                                                                        The  datasets  presented  in  this  study  can  be  found  in  online
                Three of the top ﬁve metabolites have been reported to be                            repositories.   The   names   of   the   repository/repositories   and
            related to ovarian cancers.Niemi et al.(2017)used morning                                accession number(s) can be found in the article/supplementary
            urine  samples  from  23  women  with  benign  ovarian  tumors                           material.
            and  37  women  with  malignant  ovarian  tumors  and  found
            that  N1,  N12-Diacetylspermine  showed  signiﬁcant  statistical
            diﬀerences,  and  found  that  it  can  help  distinguish  benign                        AUTHOR CONTRIBUTIONS
            and malignant ovarian tumors as well as early and advanced
            stage,  and  low  malignant  potential  and  high-grade  ovarian                         JC and YC wrote this manuscript. YW and HH did experiments.
            cancers from each other, respectively.Fahrmann et al.(2021)                              KS,  SH,  XL,  and  YO  contributed  to  software  analysis.  LS
            found  that  3-acetamidopropyl  can  signiﬁcantly  increase  the                         provided   important   ideas.   XZ   and   YB   guided   the   whole
            sensitivity of  ovarian cancer diagnosis  by 116 ovarian  cancer                         work. All authors contributed to the article and approved the
            patients  and  143  controls.Dessources  et  al.(2017)collected                          submitted version.
            samples from 16 patients with benign ovarian pathology and
            21 patients with malignant pathology and found that multiple
            metabolites  are  signiﬁcantly  associated  with  ovarian  cancer                        FUNDING
            including N-acetylation, acyl carnitines, and tryptophan.
                                                                                                     This  work  was  supported  by  Scientiﬁc  Research  Projects  of
            CONCLUSION                                                                               National Clinical Research Base of Chinese Medicine of Health
                                                                                                     Commission  of  Henan  Province  2021JDZY101  and  Doctoral
            Identifying ovarian cancer-related metabolites can help better                           researchfundoftheFirstAﬃliatedHospitalofHenanUniversity
            understand  pathogenic  mechanism  and  disease  process.  In                            of Chinese Medicine 2021BSJJ003 to YB.
            REFERENCES                                                                                  screened by weighted gene co-expression network analysis. Curr. Gene Ther.
                                                                                                        20,5–14.doi:10.2174/1566523220666200516170832
            Agakidou,E.,Agakidis,C.,Gika,H.,andSaraﬁdis,K.(2020).Emergingbiomarkers                 Chen, X. G., Shi, W. W., and Deng, L. (2019). Prediction of disease comorbidity
                for prediction and early diagnosis of necrotizing enterocolitis in the era of           using hetesim scores based on multiple heterogeneous networks. Curr. Gene
                metabolomics and proteomics. Front. Pediatr. 8:838.doi: 10.3389/fped.2020.              Ther.19, 232–241.doi:10.2174/1566523219666190917155959
                602255                                                                              Cheng, L., Li, J., Ju, P., Peng, J., and Wang, Y. (2014). SemFunSim: a new method
            Blimkie, T., Lee, A. H. Y., and Hancock, R. E. (2020). MetaBridge: an integrative           for measuring disease similarity by integrating semantic and gene functional
                multi-omicstoolformetabolite−enzymemapping.Curr.Protoc.Bioinformatics                   association.PLoS One9:e99415.doi: 10.1371/journal.pone.0099415
                70:e98.doi: 10.1002/cpbi.98                                                         Dessources, K., Cohen, J., Sen, K., Ramadoss, S., and Chaudhuri, G. (2017). N-
            Chen,J.,Zhang,X.,Cao,R.,Lu,X.,Zhao,S.,Fekete,A.,etal.(2011).Serum27-nor-                    acetylation and ovarian cancer: a study of the metabolomic proﬁle of ovarian
                5β -cholestane-3,7,12,24,25pentolglucuronidediscoveredbymetabolomicsas                  cancer compared to benign counterparts. Gynecol. Oncol. 147, 223–224.doi:
                potential diagnostic biomarker for epithelium ovarian cancer. J. Proteome Res.          10.1016/j.ygyno.2017.07.089
                10, 2625–2632.doi:10.1021/pr200173q                                                 Fahrmann, J. F., Irajizad, E., Kobayashi, M., Vykoukal, J., Dennison, J. B., Murage,
            Chen, S. C., Liu, Z. M., Li, M., Huang, Y. H., Wang, M., Wang, W., et al. (2020).           E., et al. (2021). A MYC-driven plasma polyamine signature for early detection
                Potential  prognostic  predictors  and  molecular  targets  for  skin  melanoma         of ovariancancer.Cancers 13:913.doi: 10.3390/cancers13040913
            Frontiers in Cell and Developmental Biology|www.frontiersin.org                                         5                                                                October2021|Volume 9|Article 753221

fcell-09-753221        September 29, 2021        Time: 16:52        # 6
               Chen et al.                                                                                                                                                                               Predict Ovarian Cancer-Related Metabolites
               Fong, M. Y., McDunn, J., and Kakar, S. S. (2011). Identiﬁcation of metabolites in                         Yap,  C.  W.  (2011).  PaDEL−descriptor:  an  open  source  software  to  calculate
                   the normal ovary and their transformation in primary and metastatic ovarian                               molecular descriptors and ﬁngerprints.  J. Comput. Chem. 32, 1466–1474.doi:
                   cancer.PLoSOne6:e19963.doi:10.1371/journal.pone.0019963                                                   10.1002/jcc.21707
               Garcia, E., Andrews, C., Hua, J., Kim, H. L., Sukumaran, D. K., Szyperski, T., et al.                     Zhang,  T.,  Wu,  X.,  Yin,  M.,  Fan,  L.,  Zhang,  H.,  Zhao,  F.,  et  al.  (2012).
                   (2011). Diagnosis of early stage ovarian cancer by 1H NMR metabonomics                                    Discrimination  between  malignant  and  benign  ovarian  tumors  by  plasma
                   of serum explored by use of a microﬂow NMR probe.  J. Proteome Res. 10,                                   metabolomic proﬁling using ultra performance liquid chromatography/mass
                   1765–1771.doi: 10.1021/pr101050d                                                                          spectrometry.  Clin.   Chim.   Acta   413,   861–868.doi:   10.1016/j.cca.2012.
               Gaul, D. A., Mezencev, R., Long, T. Q., Jones, C. M., Benigno, B. B., Gray, A., et al.                        01.026
                   (2015). Highly-accurate metabolomic detection of early-stage ovarian cancer.                          Zhao, T., Hu, Y., Peng, J., and Cheng, L. (2020b). DeepLGP: a novel deep learning
                   Scientiﬁc reports 5:16351.doi: 10.1038/srep16351                                                          method for prioritizing lncRNA target genes. Bioinformatics 36, 4466–4472.
               Gebauer, F., Schwarzl, T., Valcárcel, J., and Hentze, M. W. (2021). RNA-binding                               doi:10.1093/bioinformatics/btaa428
                   proteins in human genetic disease. Nat. Rev. Genet. 22, 185–198.doi: 10.1038/                         Zhao, T., Hu, Y., Zang, T., and Wang, Y. (2020c). Identifying protein biomarkers
                   s41576-020-00302-y                                                                                        in blood for Alzheimer’s disease. Front. Cell Dev. Biol. 8:472.doi: 10.3389/fcell.
               Hu, Y., Zhao, T., Zhang, N., Zang, T., Zhang, J., and Cheng, L. (2018). Identifying                           2020.00472
                   diseases-related metabolites using random walk. BMC Bioinformatics 19:116.                            Zhao,   T.,   Hu,   Y.,   and   Cheng,   L.   (2020a).   Deep-DRM:   a   computational
                   doi: 10.1186/s12859-018-2098-1                                                                            method  for  identifying  disease-related  metabolites  based  on  graph  deep
               Hussain,S.M.A.(2020).Molecular-basedscreeningandtherapeuticsofbreastand                                       learning   approaches.   Brief.   Bioinform.   22:bbaa212.doi:   10.1093/bib/b
                   ovarian cancer in low-and middle-income countries. Cancer Res. Stat. Treat.                               baa212
                   3:81.doi:10.4103/CRST.CRST_88_20                                                                      Zhao, T., Lyu, S., Lu, G., Juan, L., Zeng, X., Wei, Z., et al. (2021). SC2disease:
               Liu,  T.,  Wei,  Q.,  Jin,  J.,  Luo,  Q.,  Liu,  Y.,  Yang,  Y.,  et  al.  (2020).  The  m6A                 a manually curated database of single-cell transcriptome for human diseases.
                   reader YTHDF1 promotes ovarian cancer progression via augmenting EIF3C                                    NucleicAcids Res.49, D1413–D1419.doi:10.1093/nar/gkaa838
                   translation.Nucleic AcidsRes. 48,3816–3831.doi: 10.1093/nar/gkaa048                                   Zhou, M., Guan, W., Walker, L. D., Mezencev, R., Benigno, B. B., Gray, A., et al.
               Niemi,  R.  J.,  Roine,  A.  N.,  Häkkinen,  M.  R.,  Kumpulainen,  P.  S.,  Keinänen,                        (2010). Rapid mass spectrometric metabolic proﬁling of blood sera detects
                   T. A., Vepsäläinen, J. J., et al. (2017). Urinary polyamines as biomarkers for                            ovarian cancer with high accuracy. Cancer Epidemiol. Biomarkers Prev. 19,
                   ovarian  cancer.  Int.  J.  Gynecol.  Cancer  27,  1360–1366.doi:  10.1097/IGC.                           2262–2271.doi: 10.1158/1055-9965.EPI-10-0126
                   0000000000001031
               Peng,  J.,  and  Zhao,  T.  (2020).  Reduction  in  TOM1  expression  exacerbates                         Conﬂict of Interest: The authors declare that the research was conducted in the
                   Alzheimer’s disease.Proc. Natl. Acad. Sci. U.S.A. 117, 3915–3916.doi: 10.1073/                        absence of any commercial or ﬁnancial relationships that could be construed as a
                   pnas.1917589117                                                                                       potential conﬂictofinterest.
               Perrone,E.,Lopez,S.,Zeybek,B.,Bellone,S.,Bonazzoli,E.,Pelligra,S.,etal.(2020).
                   Preclinical  activity  of  sacituzumab  govitecan,  an  antibody-drug  conjugate                      Publisher’sNote:Allclaimsexpressedinthisarticlearesolelythoseoftheauthors
                   targeting  trophoblast  cell-surface  antigen  2  (Trop-2)  linked  to  the  active                   and do not necessarily represent those of their aﬃliated organizations, or those of
                   metabolite of irinotecan (SN-38), in ovarian cancer. Front. Oncol. 10:118.doi:                        the publisher, the editors and the reviewers. Any product that may be evaluated in
                   10.3389/fonc.2020.00118                                                                               this article, or claim that may be made by its manufacturer, is not guaranteed or
               Tianyi, Z., Yang, H., Valsdottir, L. R., Tianyi, Z., and Jiajie, P. (2020). Identifying                   endorsed bythe publisher.
                   drug–targetinteractionsbasedongraphconvolutionalnetworkanddeepneural
                   network.Brief.Bioinform. 22:bbaa044.doi: 10.1093/bib/bbaa044                                          Copyright © 2021 Chen, Chen, Sun, Wang, He, Sun, Ha, Li, Ou, Zhang and Bi.
               Wang, Y., Juan, L., Peng, J., Zang, T., and Wang, Y. (2019). Prioritizing candidate                       This is an open-access article distributed under the terms of theCreativeCommons
                   diseases-related metabolites based on literature and functional similarity. BMC                       AttributionLicense(CC BY). The use, distribution or reproduction in other forums
                   Bioinformatics 20:574.doi:10.1186/s12859-019-3127-4                                                   is permitted, provided the original author(s) and the copyright owner(s) are credited
               Woo, H. M., Kim, K. M., Choi, M. H., Jung, B. H., Lee, J., Kong, G., et al. (2009).                       andthattheoriginalpublicationinthisjournaliscited,inaccordancewithaccepted
                   Mass spectrometry based metabolomic approaches in urinary biomarker study                             academic practice. No use, distribution or reproduction is permitted which does not
                   ofwomen’scancers.Clin.Chim.Acta400,63–69.doi:10.1016/j.cca.2008.10.014                                comply withthese terms.
               Frontiers in Cell and Developmental Biology|www.frontiersin.org                                         6                                                                October2021|Volume 9|Article 753221

