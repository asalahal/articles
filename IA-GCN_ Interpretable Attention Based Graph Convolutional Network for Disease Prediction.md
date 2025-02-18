            HHS Public Access
            Author manuscript
            Mach Learn Med Imaging. Author manuscript; available in PMC 2023 October 18.
Published in final edited form as:
 Mach Learn Med Imaging. 2023 October ; 14348: 382–392. doi:10.1007/978-3-031-45673-2_38.
IA-GCN: Interpretable Attention based Graph Convolutional
Network for Disease Prediction
Anees Kazi     ⋆,1,2,3, Soroush Farghadani          ⋆,4,5, Iman Aganj      2,3, Nassir Navab       1,6
1Computer Aided Medical Procedures, Technical University of Munich, Germany
2Radiology Department, Martinos Center for Biomedical Imaging, Massachusetts General
Hospital, USA
3Harvard Medical School, USA
4Sharif University of Technology, Tehran, Iran
5University of Toronto, Canada
6Whiting School of Engineering, Johns Hopkins University, Baltimore, USA
Abstract
     Interpretability in Graph Convolutional Networks (GCNs) has been explored to some extent in
     general in computer vision; yet, in the medical domain, it requires further examination. Most of
     the interpretability approaches for GCNs, especially in the medical domain, focus on interpreting
     the output of the model in a post-hoc fashion. In this paper, we propose an interpretable attention
     module (IAM) that explains the relevance of the input features to the classification task on a GNN
     Model. The model uses these interpretations to improve its performance. In a clinical scenario,
     such a model can assist the clinical experts in better decision-making for diagnosis and treatment
     planning. The main novelty lies in the IAM, which directly operates on input features. IAM
     learns the attention for each feature based on the unique interpretability-specific losses. We show
     the application of our model on two publicly available datasets, Tadpole and the UK Biobank
     (UKBB). For Tadpole we choose the task of disease classification, and for UKBB, age, and
     sex prediction. The proposed model achieves an increase in an average accuracy of 3.2% for
     Tadpole and 1.6% for UKBB sex and 2% for the UKBB age prediction task compared to the
     state-of-the-art. Further, we show exhaustive validation and clinical interpretation of our results.
Keywords
     Interpretability; Graph Convolutional Network; Disease prediction
1.   Introduction
                 Graph Convolutional Networks (GCNs) have shown great impact in the medical domain
                 [2] such as brain imaging [5], ophthalmology [16], breast cancer [12], and thorax disease
akazi1@mgh.harvard.edu .
⋆Equal contribution. This work was done when A. Kazi and S. Farghadani were affiliated to the Technical University of Munich.

Kazi et al.                                                                                                                                                                                                  Page 2
                       diagnosis [18]. Recently many methodological advances have also been made, especially
                       for medical tasks, such as dealing with missing [6]/imbalanced [8] data, out-of-sample
                       extension [9], handling the multiple-graphs [28] and, graph learning [7] to name a few. In
                       spite of their great success, GCNs are still less transparent than other models. Interpreting
                       the model’s outcome with respect to input (graph and node features) and task (loss) is
                       essential. Interpretability techniques dealing with the analysis of GCNs have been gaining
                       importance in the last couple of years [20]. GNNExplainer [30], for example, is one of the
                       pioneer works in this direction. The paper proposes a post hoc technique to generate an
                       explanation for the outcome of Graph Convolution (GC) based models with respect to the
                       input graph and features. This is obtained by maximizing the mutual information between
                       the pre-trained model output and the output with selected input sub-graph and features.
                       Further conventional gradient-based and decomposition-based techniques have also been
                       applied to GCN [4]. Another work [11] proposes a local interpretable model explanation
                       for graphs. It uses a nonlinear feature selection method leveraging the Hilbert-Schmidt
                       Independence Criterion. However, the method is computationally complex as it generates a
                       nonlinear interpretable model.
                       Deploying non-interpretable Graph-based deep learning models in medicine could lead to
                       incorrect diagnostic decisions [3]. Therefore, adopting interpretability in machine learning
                       (ML) models is important, especially in healthcare [23]. Recently, the main efforts have
                       been towards creating transparent and explainable ML models [26]. One recent work [14]
                       proposes a post hoc approach similar to the GNNexplainer applied to digital pathology.
                       Another method, Brainex-plainer [19], proposes an ROI-selection pooling layer (R-pool)
                       that highlights ROIs (nodes in the graph) important for the prediction of neurological
                       disorders. Interpretability for GCNs is still an open challenge in the medical domain.
                       In this paper, we target the interpretability of GCNs and design a model capable of
                       incorporating the interpretations in the form of attentions for better model performance.
                       We propose an interpretability-specific loss that helps in increasing the confidence of the
                       interpretability. We show that such an interpretation-based feature selection enhances the
                       model performance. We also adapt a graph learning module from [7] to learn the latent
                       population graph thus our model does not require a pre-computed graph. We show the
                       superiority of the proposed model on 2 datasets for 3 tasks, Tadpole [21] for Alzheimer’s
                       disease prediction (three-class classification) and UK Biobank (UKBB) [22] for sex (2
                       classes) and age (4 classes) classification task. We provide several ablation tests and
                       validations supporting our hypothesis that incorporating the interpretation with in the model
                       is beneficial. In the following sections, we first provide mathematical details of the proposed
                       method, then describe the experiments, conclude the paper by a discussion.
   2.   Method
                       Given the dataset Z  =       X ,Y   , where X  ∈ℝ   N  × D    is the feature matrix with dimension D
                       for N   subjects, with Y∈ℝN         , being the labels with c classes, the task is to classify each
                       patient into the respective class from 1,2,…,c. To achieve this, we design an end-to-end
                       model, mathematically defined as yˆ  =f θ           ℎ M  X  ,gϕ  X   . Here yˆ  is the model prediction, fθ        .
                     Mach Learn Med Imaging. Author manuscript; available in PMC 2023 October 18.

Kazi et al.                                                                                                                                                                                                  Page 3
                       is the classification module, ℎM          is the Interpretable Attention Module (IAM) designed to
                       learn the attentions for features, and gϕ          is the model to learn the latent graph. In the
                       following paragraphs, we explain ℎM,gϕ              , the proposed loss, and the base model used for the
                       classification task.
   2.1.   Interpretable Attention Module (IAM):                        ℎM
                       For ℎM   , we define a differentiable and continuous mask M   ∈ℝ   1× D                  that learns an attention
                       coefficient for each feature element from D               features. IAM can be mathematically defined
                            ′=ℎ M                                 ′∈ℝ   1× D
                       as x i        x i=σ    M   × x i where, x i              is the masked output, σ        is the sigmoid function.
                       σ  M   represents the learned mask M  ′⊂             0,1 . The mask M       is continuous as the aim is to learn
                       the interpretable attentions while the model is training. Conceptually, the corresponding
                       weights in the mask M′ should take a value close to zero when a particular feature is not
                       significant towards the task. In effect, mi corresponding to di may improve or deteriorate the
                       model performance based on the importance of di towards the task. The proposed IAM is
                       trained by a customized loss discussed in section 2.4.
   2.2.   Graph Learning Module (GLM):                       gϕ
                       Inspired by DGM [7] we define our GLM mathematically denoted as gϕ                            . Given the input
                       X , GLM predicts an optimal graph G ′∈ℝ   N  × N                which is then used in the GCN for the
                       classification, as shown in Figure 1. GLM consists of 2 layered multilayer perceptron
                       (MLP) followed by a graph construction step and a graph pruning step. MLP takes the
                       feature matrix X  ∈ℝ   N  × D       as input and produces Xˆ        embedding specific for the optimal
                       latent graph as output. A fully connected graph is computed with continuous edge values
                       (shown as graph construction in Figure 1) using the Euclidean distance metric between the
                       feature embedding xˆi and xˆj where           xˆ i,xˆ j∈  Xˆ. Sigmoid function is used for soft thresholding
                                                                    ′                      ′ =              1
                       keeping the GLM differentiable. gij is computed as gij                          t∥ xˆ i−  xˆ j ∥ 2+T with T  being the
                                                                                                1+e
                       threshold parameter and t          >0     the temperature parameter pushing values of gij′              to either 0 or
                       1. Both t and T      are optimized during training. Thus, G′ is obtained.
   2.3.   Classification Module with joint optimization of GLM and IAM
                       As mentioned before, the primary goal is to classify each patient xi into the respective class
                       yi. The classification model can be mathematically defined as Yˆ  =f θ                    X ′,G ′  where fθ is the
                       classification function with learnable parameters θ,G′ the learned latent population graph
                       structure, and X′ the output of IAM. We define fθ as a generic GCN targeted towards node
                       classification. The whole model is trained end to end using a customized loss focusing more
                       on interpretability. This loss is discussed below.
   2.4.   Interpretability-focused loss functions
                       Empirically we observed that training the model with only softmax cross-entropy loss
                       Lc was sub-optimal and, specifically, 1) the performance was not the best, 2) the mask
                       learned average values for all the features reflecting uncertainty, and 3) unimportant features
                      Mach Learn Med Imaging. Author manuscript; available in PMC 2023 October 18.

Kazi et al.                                                                                                                                                                                                  Page 4
                       would take considerable weight in the mask. In order to optimize the whole network in an
                       end-to-end fashion, we define the loss L             as
                                                                              D  −1                            D  −1
                                             L  =    1−  α   L  c+α *    α 1*         −  m  i′log  2m  i′+α 2*        m  i′                  (1)
                                                                               i =0                            i =0
                       where Lc is the softmax cross-entropy loss, α              is the weighting factor chosen experimentally,
                       the next two terms being FMEL          : the feature mask entropy loss, and FMSL            : the feature mask
                       size loss respectively with respective weights factors α1 and α2.FMEL                   and FMSL     are used
                       to regularize Lc. Firstly, F  MSL   = ∑   i =0D  −1 m  i′ lowers the sum of the values of individual mi′.
                       Otherwise, all the features would get the highest importance with all the m′ s taking up the
                                                                       D  −1  −  m  i′log  2′
                       value 1. On the other hand, F  MEL   = ∑   i =0                 m  i pushes the values away from 0.5, which
                       makes the model more confident about the importance of the feature di.FMEL                          and FMSL    are
                       used only by ℎM       for the back propagation.
   3.   Experiments:
                       Two publicly available datasets were used for three tasks. Tadpole [21] for Alzheimer’s
                       disease prediction and UK Biobank (UKBB) [22] for age and sex prediction. The task
                       in the Tadpole dataset was to classify 564 subjects into three categories (Normal, Mild
                       Cognitive Impairment, Alzheimer’s) that represent their clinical status. Each subject had 354
                       multi-modal features that included cognitive tests, MRI ROIs measures, PET imaging, DTI
                       ROI measures, demographics, etc. On the other hand, the UKBB dataset consisted of 14,503
                       subjects with 440 features per individual, which were extracted from MRI and fMRI images.
                       Two classification tasks were considered for this dataset: 1) sex prediction, 2) categorical
                       age prediction. In the second task, subjects’ ages were quantized into four decades as the
                       classification targets. Table 1 shows the results of the classification task for both datasets.
                       We performed an experiment with a linear classifier (LC) to see the complexity of the task
                       as well as with the Chebyshev polynomial-based spectral-GCN [25], and Graph Attention
                       Network (GAT) [27], which is a spatial method. We compared with these two methods as
                       they require a pre-defined graph structure for the classification task, whereas our method
                       and DGM [17] do not. Our reasoning behind learning the graph is that pre-computed/
                       preprocessed graphs can be noisy, irrelevant to the task, or unavailable. Depending on
                       the model, learning the population graph is much more clinically semantic. Unlike Spectral-
                       GCN and GAT, DGCNN [29] constructs a KNN graph at each layer dynamically during
                       training. This removes the requirement for a pre-computed graph. However, the method
                       still lacks the ability to learn the latent graph. DGM [7] and the proposed method, on the
                       other hand, do not require any graph structure to be defined and they only utilize the given
                       features. Implementation Details: M               is initialized either with Gaussian normal distribution
                       or constant values. Experiments were performed using Google Colab with a Tesla T4 GPU
                       with PyTorch 1.6. Number of epochs was 600. Same 10 folds with the train:test split of
                       90:10 were used in all the experiments. We used two MLP layers (16→8) for GLM and
                       two Conv layers followed by a FC layer (32→16→# classes) for the classification network.
                       ReLU was used as the activation function.
                      Mach Learn Med Imaging. Author manuscript; available in PMC 2023 October 18.

Kazi et al.                                                                                                                                                                                                  Page 5
   Classification performance:
                       For Tadpole, the proposed method performed best for all the three measures (Accuracy,
                       AUC, F1). The overall lower F1-score indicates that the task was challenging due to
                       the class imbalance        2  4  1  present in the dataset. The proposed IAM adds interpretable
                                                  7, 7, 7
                       attention to features, which improves the model performance by 3.16% compared to the
                       state-of-the-art (DGM). The low variance shows the stability of the proposed method.
                       The UKBB was chosen due to its much larger dataset size. Sex prediction covers the
                       challenge of larger dataset size, whereas age prediction deals with both large size and
                       imbalance. The results are shown in Table 3. For sex prediction, our method shows
                       superior performance and AUC reconfirms the consistency of the model’s performance.
                       For age prediction, results demonstrate that the overall task is much more challenging
                       than the sex prediction. Lower F1-score shows the existence of class imbalance. Our
                       method outperforms the DGM by 2.02% and 1.65% in accuracy for the sex and age task,
                       respectively. Moreover, the performance trend of other comparative methods can be seen
                       similar to be similar to Tadpole. The above results indicate that the incorporation of graph
                       convolutions helps in better representation learning, resulting in more accurate classification.
                       Further, GAT requires full data in one batch along with the affinity graph, which causes
                       the out-of-memory issue in UKBB experiments. Moreover, DGCNN and DGM achieve
                       higher accuracy compared to Spectral-GCN and GAT. This confirms our hypothesis that a
                       pre-computed graph might not be optimal. Between DGCNN and DGM, the latter performs
                       better, confirming that learning a graph is beneficial to the final task and for getting latent
                       semantic graph as output.
   Analysis of the loss function:
                       Next, we investigated the contribution of all the loss terms, toward the optimization of the
                       task. We report the accuracy of classification, the average attention for the top four features
                       (Avg.4) selected by the model and other features (Avg.O). Table 2 (top) shows changes in
                       the performance and the average of attention values (Avg.4 and Avg.O) with respect to α                          .
                       The performance drops significantly with α=0. Best accuracy at α=0.6 shows that both
                       loss terms are necessary for the optimal performance of the model. Avg.4 and Avg.O surge
                       dramatically each time α        increases. This proves the importance of FMEL              and FMSL    in shrinking
                       the attention values of features.
                       In the second experiment shown in Table 2 (bottom), two specific cases were investigated.
                       While α    is at its optimum value of 0.6, the contribution of α1 and α2 was investigated, to
                       show the contribution of FMEL          and FMSL    in the model. α1 and α2 were set to 0 respectively.
                       FMEL   seems to have more importance in the certainty of interpretable attentions for important
                       features. However, FMSL        pushes the attention values to 0 which helps us in distinguishing
                       more and less important features. The combination of all three terms leads to the best
                       performance as shown by optimal α            s.
                       Interpretability: Here, we show validation experiments to prove the relevance of features
                       selected by the IAM to the clinical task and model performance. We measured the
                       classification performance by manually adding and removing the features from the input
                     Mach Learn Med Imaging. Author manuscript; available in PMC 2023 October 18.

Kazi et al.                                                                                                                                                                                                  Page 6
                       for two traditional methods of GCN [25] and DGM [7]. Table 3 presents experiments with
                       different input features, including a) method trained traditionally on all available features, b)
                       conventional feature selection technique using Ridge classifier applied to the input features
                       at the pre-processing step, c) method trained conventionally with all input features except
                       the features selected by the IAM, and d) model trained on only features selected by the
                       proposed method. Overall, the feature selection approach (b and d) was advantageous, with
                       the proposed IA-based feature selection (d) performing the best. When models were trained
                       with features other than the selected ones, their performance drastically dropped. Similar
                       experiments were repeated on the UKBB dataset for age and sex classification using DGM,
                       and the Ridge classifier with feature selection during preprocessing performed the best,
                       indicating the necessity of feature selection. Further discussion of these results will be
                       provided later in the limitations section.
   Clinical interpretation:
                       In the Tadpole dataset, our model selects four cognitive features CDR, CDR at baseline,
                       MMSE and MMSE at baseline. It is reported in the clinical literature that the cognitive
                       measure of Clinical Dementia Rating Sum of Boxes (CDRSB) compares well with
                       the global CDR score for dementia staging. Cognitive tests measure the decline in a
                       straightforward and quantifiable way with the disease condition. Therefore, these are
                       important in Alzheimer’s disease prediction [             13], in particular, CDRSB and the Mini-Mental
                       State Examination (MMSE) [24, 1]. MMSE is the best-known and the most often used
                       short screening tool for providing an overall measure of cognitive impairment in clinical,
                       research, and community settings. Apart from cognitive tests, the Tadpole dataset includes
                       other imaging features. We observed that the Pearson correlation coefficient with respect to
                       the ground truth and the attention value computed by IAM are roughly linearly related. For
                       the UKBB sex classification task, in the order of importance, our model selected volume
                       features of peripheral cortical gray matter (normalized for (1) head size, (2) white matter,
                       (3) brain, gray+white matter, (4) cortical gray matter, and (5) peripheral cortical gray matter)
                       which is also supported by [15]. For age prediction, the most relevant features selected
                       by our network were (1) volume of peripheral cortical gray matter, mean (2) MD and (3)
                       L2 in fornix on FA skeleton, (4) mean L3 in anterior corona radiata on FA skeleton right
                       and (5) mean L3 in anterior corona radiata on FA skeleton left which are also supported
                       by [15]. For both the Tadpole and UKBB datasets, it is observed that the set of selected
                       features are different depending on the task. Our interpretation of the model not selecting
                       the MRI features in the Tadpole experiments is that attention is distributed over 314 features,
                       which are indistinct compared to cognitive features. MRI features may nevertheless be more
                       valuable when two scans taken over time between two visits to the hospital are compared to
                       check the loss in volume (atrophy). However, in our case, we only considered scans at the
                       baseline.
   Limitations:
                       The model fails in the case of UKBB in Table 3. The best performance is shown by
                       feature selection by the Ridge classifier (b). Intuitively, UKBB is much large data with
                       a difficult task. A more complex attention module design could be helpful. In the case
                       of clinical interpretation for the age classification task, we observed a much larger set of
                     Mach Learn Med Imaging. Author manuscript; available in PMC 2023 October 18.

Kazi et al.                                                                                                                                                                                                  Page 7
                       features were given higher attention (Only the top 4 shown due to page limit), confirming
                       the task complexity. Further, the values of α              s are empirically chosen. They could be learned
                       automatically.
   4.   Discussion and Conclusion
                       We developed a GCN-based model featuring an interpretable attention module (IAM)
                       and a distinct loss function. The IAM learns feature attention and aids model training.
                       Our experiments reveal strong feature correlations via IAM for Tadpole and UKBB sex
                       classification, and our model outperforms state-of-the-art methods in disease and age
                       classification.To address the issue of ignoring important features, we marginalized overall
                       feature subsets and used a Monte Carlo estimate to sample from empirical marginal
                       distribution for nodes during training. Our proposed method handles class imbalance well
                       and achieved higher accuracy and F1-score than DGM in both tasks for UKBB. In terms of
                       results, the method exceeded the highest accuracy by 3.5% for the disease classification task.
                       Furthermore, the proposed method’s F1-score was 3.2% higher than that of the state-of-the-
                       art methods, which shows that it handles class imbalance well. For both tasks in UKBB, the
                       accuracy and F1-score of the proposed method was 1.7% and 1.8% higher than the DGM
                       method, respectively. For the UKBB age prediction task, we observed ∼2% gain in accuracy
                       and F1-score.
   Acknowledgment:
                       Anees Kazi’s financial support was provided by BigPicture (IMI945358) from the Technical University of Munich
                       during this project. Support for this research was partly provided by the National Institutes of Health (NIH),
                       specifically the National Institute on Aging (RF1AG068261).
   References
                       1. A-Rodriguez I, Smailagic N, i Figuls MR, Ciapponi A, Sanchez-Perez E, Giannakou A, Pedraza
                           OL, Cosp XB, Cullum S: Mini-mental state examination (mmse) for the detection of alzheimer’s
                           disease and other dementias in people with mild cognitive impairment (mci). CDSR (3) (2015)
                       2. Ahmedt-Aristizabal D, Armin MA, Denman S, Fookes C, Petersson L: Graph-based deep learning
                           for medical diagnosis and analysis: past, present and future. Sensors 21(14), 4758 (2021) [PubMed:
                           34300498]
                       3. Amann J, Blasimme A, Vayena E, Frey D, Madai VI: Explainability for artificial intelligence in
                           healthcare: a multidisciplinary perspective. BMC Medical Informatics and Decision Making 20(1),
                           1–9 (2020) [PubMed: 31906929]
                       4. Baldassarre F, Azizpour H: Explainability techniques for graph convolutional networks. arXiv
                           preprint arXiv:1905.13686 (2019)
                       5. Bessadok A, Mahjoub MA, Rekik I: Graph neural networks in network neuro-science. IEEE
                           Transactions on Pattern Analysis and Machine Intelligence (2022)
                       6. Chang YW, Natali L, Jamialahmadi O, Romeo S, Pereira JB, Volpe G, Initiative ADN, et al. :
                           Neural network training with highly incomplete medical datasets. Machine Learning: Science and
                           Technology 3(3), 035001 (2022)
                       7. Cosmo L, Kazi A, Ahmadi SA, Navab N, Bronstein M: Latent-graph learning for disease prediction.
                           In: MICCAI. pp. 643–653. Springer (2020)
                       8. Ghorbani M, Kazi A, Baghshah MS, Rabiee HR, Navab N: Ra-gcn: Graph convolutional network
                           for disease prediction problems with imbalanced data. MedIA 75, 102272(2022)
                       9. Hamilton W, Ying Z, Leskovec J: Inductive representation learning on large graphs. In: Proc. NIPS
                           (2017)
                      Mach Learn Med Imaging. Author manuscript; available in PMC 2023 October 18.

Kazi et al.                                                                                                                                                                                                  Page 8
                        10. Hoerl AE, Kennard RW: Ridge regression: Biased estimation for nonorthogonal problems.
                             Technometrics 12, 55–67 (1970)
                        11. Huang Q, Yamada M, Tian Y, Singh D, Yin D, Chang Y: Graphlime: Local interpretable model
                             explanations for graph neural networks. arXiv preprint arXiv:2001.06216 (2020)
                        12. Ibrahim M, Henna S, Jennings B, Butler B, Cullen G: Multi-graph convolutional neural network
                             forbreast cancer multi-task classification (2022)
                        13. Jack CR Jr, Holtzman DM: Biomarker modeling of alzheimer’s disease. Neuron
                        14. Jaume G, Pati P, Foncubierta-Rodriguez A, Feroce F, Scognamiglio G, Anniciello AM, Thiran
                             JP, Goksel O, Gabrani M: Towards explainable graph representations in digital pathology. arXiv
                             preprint arXiv:2007.00311 (2020)
                        15. Jiang H, Lu N, Chen K, Yao L, Li K, Zhang J, Guo X: Predicting brain age of healthy adults based
                             on structural mri parcellation using convolutional neural networks. Frontiers in neurology 10, 1346
                             (2020) [PubMed: 31969858]
                        16. Joshi A, Sharma K: Graph deep network for optic disc and optic cup segmentation for glaucoma
                             disease using retinal imaging. Physical and Engineering Sciences in Medicine 45(3), 847–858
                             (2022) [PubMed: 35737221]
                        17. Kazi A, Cosmo L, Ahmadi SA, Navab N, Bronstein M: Differentiable graph module (dgm) for
                             graph convolutional networks. IEEE Transactions PAMI (2022)
                        18. Lee YW, Huang SK, Chang RF: Chexgat: A disease correlation-aware network for thorax
                             disease diagnosis from chest x-ray images. Artificial Intelligence in Medicine 132, 102382 (2022)
                             [PubMed: 36207088]
                        19. Li X, Duncan J: Braingnn: Interpretable brain graph neural network for fmri analysis. bioRxiv
                             (2020)
                        20. Liu N, Feng Q, Hu X: Interpretability in graph neural networks. In: Graph Neural Networks:
                             Foundations, Frontiers, and Applications
                        21. Marinescu RV, Oxtoby NP, Young AL, Bron EE, Toga AW, Weiner MW, Barkhof F, Fox NC, Klein
                             S, Alexander DC, et al. : Tadpole challenge: Prediction of longitudinal evolution in alzheimer’s
                             disease. arXiv preprint arXiv:1805.03909 (2018)
                        22. Miller KL, Alfaro-Almagro F, Bangerter NK, Thomas DL, Yacoub E, Xu J, Bartsch AJ, Jbabdi S,
                             Sotiropoulos SN, Andersson JL, et al. : Multimodal population brain imaging in the uk biobank
                             prospective epidemiological study. Nature neuroscience 19(11), 1523 (2016) [PubMed: 27643430]
                        23. Molnar C: Interpretable machine learning. Lulu. com (2020)
                        24. O’Bryant SE, Waring SC, Cullum CM, Hall J, Lacritz L, Massman PJ, Lupo PJ, Reisch JS, Doody
                             R: Staging dementia using clinical dementia rating scale sum of boxes scores: a texas alzheimer’s
                             research consortium study. Archives of neurology 65(8), 1091–1095 (2008) [PubMed: 18695059]
                        25. Parisot S, Ktena SI, Ferrante E, Lee M, Moreno RG, Glocker B, Rueckert D: Spectral graph
                             convolutions for population-based disease prediction. In: MICCAI. pp. 177–185. Springer (2017)
                        26. Stiglic G, Kocbek P, Fijacko N, Zitnik M, Verbert K, Cilar L: Interpretability of machine
                             learning-based prediction models in healthcare. Wiley Interdisciplinary Reviews: Data Mining
                             and Knowledge Discovery 10(5), e1379 (2020)
                        27. Velickovic P, Cucurull G, Casanova A, Romero A, Lio P, Bengio Y: Graph attention networks. stat
                             1050, 20 (2017)
                        28. Vivar G, Kazi A, Burwinkel H, Zwergal A, Navab N, Ahmadi SA: Simultaneous imputation
                             and disease classification in incomplete medical datasets using multigraph geometric matrix
                             completion (mgmc). arXiv preprint arXiv:2005.06935
                        29. Wang Y, Sun Y, Liu Z, Sarma SE, Bronstein MM, Solomon JM: Dynamic graph cnn for learning
                             on point clouds. Acm (TOG) 38(5)
                        30. Ying Z, Bourgeois D, You J, Zitnik M, Leskovec J: Gnnexplainer: Generating explanations for
                             graph neural networks. In: NeurIPs. pp. 9244–9255 (2019)
                       Mach Learn Med Imaging. Author manuscript; available in PMC 2023 October 18.

Kazi et al.                                                                                                                                                                                                  Page 9
                           Fig. 1.
                           IA-GCN consists of three main components: 1) Interpretable Attention Module (IAM):
                           ℎM,2     Graph Learning Module (GLM): gϕ                      , and 3) Classification Module: fθ. These are
                           trained in an end-to-end fashion. In backpropagation, two loss functions are playing roles
                           which are demonstrated in blue and red arrows.
                         Mach Learn Med Imaging. Author manuscript; available in PMC 2023 October 18.

Kazi et al.                                                                                                                                                                                                Page 10
                                                                             Table 1.
Performance of the proposed method (mean ± pm STD) compared with several state-of-the-art and baseline
methods on the Tadpole and UKBB dataset for classification.
  Dataset       Task          Method            Accuracy             AUC                 F1
                               LC[10]          70.22±06.32       80.26±04.81       68.73±06.70
                              GCN[25]          81.00±06.40       74.70±04.32        78.4±06.77
  Tadpole      Disease        GAT[27]          81.86±05.80       91.76±03.71       80.90±05.80
                            DGCNN[29]          84.59±04.33       83.56±04.11       82.87±04.27
                              DGM[7]           92.92±02.50       97.16±01.32        91.4±03.32
                              IA-GCN           96.08±02.49        98.6±01.93       94.77±04.05
                                 LC            81.70±01.64       90.05±01.11       81.62±01.62
                              GCN[25]          83.70±01.06       83.55±00.83       83.63±00.86
  UKBB           Sex        DGCNN[29]          87.06±02.89       90.05±01.11       86.74±02.82
                              DGM[7]           90.67±01.26       96.47±00.66       90.65±01.25
                              IA-GCN           92.32±00.89       97.04±00.59       92.25±00.87
                                 LC            59.66±01.17       80.26±00.91       48.32±03.35
                              GCN[25]          55.55±01.82       61.00±02.70       40.68±02.82
  UKBB           Age        DGCNN[29]          58.35±00.91       76.82±03.03       47.12±03.95
                              DGM[7]           63.62±01.23       82.79±01.14       50.23±02.52
                              IA-GCN           65.64±01.12       83.49±01.04       51.73±02.68
                         Mach Learn Med Imaging. Author manuscript; available in PMC 2023 October 18.

Kazi et al.                                                                                                                                                                                                Page 11
                                                                             Table 2.
Performance of IA-GCN on the Tadpole dataset in different settings w.r.t α                                      . Here, we show the model
performance for classification. We report the accuracy of classification, the average attention for the top 4
features (Avg.4), and other features (Avg.O). We also show the model performance when the values of α1 and
α2 are changed.
    α        Accuracy          Avg.4     Avg.O
    0       57.00±09.78         10−6       10−6
   0.2      94.20±03.44         0.12      0.002
   0.4      95.10±02.62         0.29      0.001
   0.6      96.10±02.49         0.74       0.0
   0.8      95.80±02.31         0.78       0.23
   1.0      95.40±02.32         0.82       0.42
  α1=0     95.60 ± 02.44        0.54       0.13
  α2=0     95.10 ± 03.69        0.86       0.26
                         Mach Learn Med Imaging. Author manuscript; available in PMC 2023 October 18.

Kazi et al.                                                                                                                                                                                                Page 12
                                                                             Table 3.
Performance for the classification task. We show results for four baselines on GCN [25] for Tadpole and
UKBB and DGM [7] for Tadpole with different input feature settings.ACC represents accuracy.
  Data         Task         Method        Measure       a                  b                 c                  d
                                          ACC           77.4±02.41         81.00±06.40       74.50±3.44         82.4±04.14
                            GCN           AUC           79.79±04.75        74.70±04.32       72.11±08.24        83.89±09.06
  Tadpole      Disease                    F1            74.70±05.32        78.4±06.77        65.23±08.46        78.73±07.60
                                          ACC           89.2±05.26         92.92±02.50       79.70±04.22        95.09±03.15
                            DGM           AUC           96.47±02.47        97.16±01.32       90.66±02.64        98.33±02.07
                                          F1            88.60±05.32        91.4±03.32        77.9±6.38          93.36±03.28
                                          ACC           62.10±01.45        63.62±01.23       61.54±01.83        59.45±03.15
               Age          DGM           AUC           76.57±02.47        76.82±03.03       81.40±04.73        77.23±02.17
  UKBB                                    F1            46.80±04.83        50.23±02.52       47.31±03.54        47.46±03.19
                                          ACC           89.93±01.3         90.67±01.26       89.04±01.84        87.46±03.32
               Sex          DGM           AUC           95.83±00.76        96.47±00.66       95.02±00.92        93.98±02.43
                                          F1            89.83±01.34        90.65±01.25       89.01±01.75        87.4±03.23
                         Mach Learn Med Imaging. Author manuscript; available in PMC 2023 October 18.

