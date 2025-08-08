Received: 23 October 2019 Revised: 13 July 2020 Accepted: 1 August 2020


DOI: 10.1002/hbm.25175


R E S E A R C H A R T I C L E

# Identifying resting-state effective connectivity abnormalities in drug-naïve major depressive disorder diagnosis via graph convolutional networks


Eunji Jun [1] | Kyoung-Sae Na [2] | Wooyoung Kang [3] | Jiyeon Lee [1] |

Heung-Il Suk [1,4] | Byung-Joo Ham [5]



1 Department of Brain and Cognitive

Engineering, Korea University, Seoul, Republic

of Korea


2 Department of Psychiatry, Gachon University

Gil Medical Center, Incheon, Republic of Korea


3 Department of Biomedical Sciences, Korea

University College of Medicine, Seoul,

Republic of Korea


4 Department of Artificial Intelligence, Korea

University, Seoul, Republic of Korea


5 Department of Psychiatry, Korea University

Anam Hospital, Korea University College of

Medicine, Seoul, Republic of Korea


Correspondence

Heung-Il Suk, Department of Brain and

Cognitive Engineering, Korea University,

Seoul, Republic of Korea.

[Email: heungilsuk@gmail.com](mailto:heungilsuk@gmail.com)


Byung-Joo Ham, Department of Psychiatry,

Korea University Anam Hospital, Korea

University College of Medicine, Seoul 02841,

Republic of Korea.

[Email: byungjoo.ham@gmail.com](mailto:byungjoo.ham@gmail.com)


Funding information

Institute of Information and Communications

Technology Planning and Evaluation, Grant/

Award Number: 2019-0-00079; National

Research Foundation of Korea, Grant/Award

Number: NRF-2017R1A2B4002090


1 | INTRODUCTION



Abstract


Major depressive disorder (MDD) is a leading cause of disability; its symptoms inter

fere with social, occupational, interpersonal, and academic functioning. However, the


diagnosis of MDD is still made by phenomenological approach. The advent of


neuroimaging techniques allowed numerous studies to use resting-state functional


magnetic resonance imaging (rs-fMRI) and estimate functional connectivity for


brain-disease identification. Recently, attempts have been made to investigate effec

tive connectivity (EC) that represents causal relations among regions of interest. In


the meantime, to identify meaningful phenotypes for clinical diagnosis, graph-based


approaches such as graph convolutional networks (GCNs) have been leveraged


recently to explore complex pairwise similarities in imaging/nonimaging features


among subjects. In this study, we validate the use of EC for MDD identification by


estimating its measures via a group sparse representation along with a structured


equation modeling approach in a whole-brain data-driven manner from rs-fMRI. To


distinguish drug-naïve MDD patients from healthy controls, we utilize spectral GCNs


based on a population graph to successfully integrate EC and nonimaging phenotypic


information. Furthermore, we devise a novel sensitivity analysis method to investi

gate the discriminant connections for MDD identification in our trained GCNs. Our


experimental results validated the effectiveness of our method in various scenarios,


and we identified altered connectivities associated with the diagnosis of MDD.


K E Y W O R D S


effective connectivity, deep learning, graph convolutional networks (GCNs), major depressive


disorder (MDD), resting-state functional magnetic resonance imaging (rs-fMRI), Sparse Group


LASSO


mental disorder that is prevalent worldwide (American Psychiatric


Association, 2013). The lifetime prevalence of MDD was estimated to



Major depressive disorder (MDD), characterized by depressed mood,


loss of interest, vegetative symptoms, and cognitive impairment, is a



be 10.8% (American Psychiatric Association, 2013). The symptoms of


MDD substantially interfere with social, occupational, interpersonal,



[This is an open access article under the terms of the Creative Commons Attribution License, which permits use, distribution and reproduction in any medium,](http://creativecommons.org/licenses/by/4.0/)

provided the original work is properly cited.

© 2020 The Authors. Human Brain Mapping published by Wiley Periodicals LLC.


Hum Brain Mapp. 2020;41:4997–5014. [wileyonlinelibrary.com/journal/hbm](http://wileyonlinelibrary.com/journal/hbm) 4997


4998 JUN ET AL .



and academic functioning (American Psychiatric Association, 2013).


Globally, the total years lived with disability (YLD) of depressive disor

ders was 7.5% among all YLD, which has been ranked the highest of


all disease (World Health Organization, 2017). Hence, depressive dis

orders are the leading cause of disability.


Despite the debilitating effects of MDD, the diagnosis of MDD is


still made by phenomenological approach. Given the proximity to the


psychiatric symptoms in terms of mood and cognitive dysregulation,


brain MRI has been used to investigate the neural mechanisms of


MDD (Kempton et al., 2011). Specifically, resting-state functional


magnetic resonance imaging (rs-fMRI) has been widely used for the


diagnosis of MDD by investigating altered functional networks while


a subject is at rest (Anand et al., 2005; Craddock, Holtzheimer, Hu, &


Mayberg, 2009; Greicius et al., 2007). In the meantime, more recently,


the investigation of dynamic changes between connections beyond


simple correlations has been attracting increasing interest (Geng, Xu,


Liu, & Shi, 2018; Rolls et al., 2018). The notion of effective connectivity


(EC) describes the influence of one neural system on another (Friston,


Ungerleider, Jezzard, & Turner, 1994), in contrast to functional connec

tivity (FC) that denotes intrinsic correlations.


Several studies have revealed that the EC may be used as an effi

cient biomarker for the diagnosis of MDD. Specifically, (Schlösser


et al., 2008) found that adolescents suffering from MDD exhibited a


significant difference in EC between the amygdala and subgenual ante

rior cingulate cortex (ACC) during an emotion-relevant task. In addition,


Geng et al. (2018) directly utilized both FC and EC measures as features


for the diagnosis of MDD and established that the discriminative power


of EC features is higher than that of FC features. More recently, using a


large sample size (336 patients with MDD and 350 control subjects),


Rolls et al. (2018) identified significantly altered EC measures in MDD,


such as reduced connectivity from temporal lobe areas to the medial


orbitofrontal cortex. These findings imply that the EC measures are


beneficial for determining if it is altered in neurological disorders, in


addition to FC in the resting-state paradigm in neuroimaging.


Several approaches such as dynamic causal modeling (DCM)


(Park & Friston, 2013) and Granger causality (GC) (Granger, 1969)


have been suggested for estimating EC. DCM is a commonly used


approach; however, it requires the selection of seed regions of inter

est (ROIs) that are widely known as discriminant biomarkers in rele

vant literature rather than the whole brain connectivity due to


computational complexity (Geng et al., 2018). GC, owing to its simplic

ity and ease of implementation, has been widely used to estimate the


EC (Hamilton, Chen, Thomason, Schwartz, & Gotlib, 2011; Liao


et al., 2011; Wu & Marinazzo, 2015). However, studies have shown


that EC estimations given by GC cannot correctly determine the


intensity of the actual causality in the time domain (Hu et al., 2012). In


the meantime, structural equation modeling (SEM) (McIntosh, Rajah, &


Lobaugh, 1999) has been successfully used as a statistical approach


for investigating the EC (Büchel & Friston, 1997; Penny, Stephan,


Mechelli, & Friston, 2004; Suk, Wee, Lee, & Shen, 2015; Wee, Yap,


Zhang, Wang, & Shen, 2014; Zhuang, Peltier, He, LaConte, &


Hu, 2008). The original work of SEM requires a large sample size to


model complex relationships between brain activities.



In recent years, beyond the group-level analyses, there has been


growing interest in using machine learning (ML) techniques to identify


clinically meaningful phenotypes for clinical diagnosis. A typical ML


pipeline for the diagnosis of MDD can be summarized as follows: fea

ture extraction, feature selection, model training, classification, and


performance evaluation. In studies that differentiate MDD patients


from healthy controls (HC), the following have been used as features


extracted from rs-fMRI: spatial independent components (Ramasubbu


et al., 2016; Wei et al., 2013), the Hurst exponent (Jing et al., 2017),


degree centrality (Li et al., 2017), and regional homogeneity (Ma, Li,


Yu, He, & Li, 2013). In addition, many previous studies also applied


graph theory approaches (Bhaumik et al., 2017; Cao et al., 2014; Dry

sdale et al., 2017; Guo et al., 2014; Lord, Horn, Breakspear, &


Walter, 2012; Sundermann et al., 2017; Wang, Ren, & Zhang, 2017;


Yoshida et al., 2017; Zeng, Shen, Liu, & Hu, 2014; Zhong et al., 2017)


to the preestimated FC for investigating the disrupted functional brain


networks in MDD patients. A small number of MDD classification


studies have utilized EC as the feature. In Geng et al. (2018), EC was


estimated using spectral DCM with predefined ROIs, and then, it was


used as the feature for MDD classification; in this case, four super

vised learning classifiers are used: linear support vector machine


(SVM), nonlinear SVM, linear regression, and k-nearest neighbor.


Nonetheless, SVM (Bhaumik et al., 2017; Cao et al., 2014; Drysdale


et al., 2017; Lord et al., 2012; Sundermann et al., 2017; Wang


et al., 2017; Zhong et al., 2017) remains the most commonly used


classifier, but other ML classifiers such as partial least squares regres

sion (Yoshida et al., 2017), maximum margin clustering (Zeng


et al., 2014), linear discriminant analysis (Ma et al., 2013), and neural


networks (Guo et al., 2014) have also been applied for the diagnosis


of MDD.


Recently, graph-based approaches have gained popularity in med

ical applications owing to their ability to accommodate complex


pairwise similarities in imaging/nonimaging features between subjects


(Parisot et al., 2018). They model individuals as vertices and associa

tions or similarities between them as edges, which have been widely


used for supervised (e.g., classification (Tong et al., 2017)) and


unsupervised tasks (e.g., manifold learning (Brosch & Tam, 2013; Wolz


et al., 2012) and clustering (Parisot et al., 2016)). In this study, we


focus on disease classification using a graph-based model. In particu

lar, a generalization of convolutional neural networks (CNNs) to an


irregular graph domain, called spectral graph convolutional networks


(GCNs), has been successfully applied to perform brain disease classi

fication (Parisot et al., 2018). Specifically, (Parisot et al., 2018) utilized


a population graph for GCNs, where a vertex represents a subject and


an edge encodes pairwise similarities of phenotypic data and/or imag

ing features between subjects. This combines imaging and nonimaging


data in a single framework and delivers competitive classification


performance.


In this study, we go beyond the FC toward an EC-based approach


using a group sparse representation leveraged with SEM in an


unsupervised manner. Specifically, this group-constrained sparsity


imposes similar connectional patterns among subjects but maintains


individual differences in correlation weights. To identify MDD,


JUN ET AL . 4999



inspired by Parisot et al. (2018), we exploit the spectral GCNs based


on the population graph to successfully integrate our EC features and


nonimaging demographic features. Furthermore, we devise a sensitiv

ity analysis (SA) method for our learned GCNs to investigate discrimi

nant EC measures for MDD identification. Through various scenarios,


our experimental results validate the effectiveness of the proposed


method in terms of extracted features, feature selection, and classi

fiers. Our main contributions can be summarized in two aspects as


follows:


- We estimated EC by using a whole-brain data-driven approach


with low computational costs through group-constrained sparsity


leveraged with SEM-like mechanism and used it for the diagnosis


of MDD via GCNs for the first time.


- In addition to superior experimental results for MDD identification,


through an SA for our learned GCNs, we successfully identified


meaningful connectivities associated with the diagnosis of MDD


that have been reported in psychiatry literature.


2 | MATERIALS


2.1 | Participants


We collected the rs-fMRI from 29 drug-naïve MDD patients recruited


from the outpatients of the Korea University Anam Hospital (Seoul,


Republic of Korea). These patients included 8 males and 21 females;


their ages ranged from 19 to 60 years, and the mean age was


43.79 years (±13.06). The outpatients were prospectively recruited as


participants who agreed to visit the clinic after 4 weeks, 8 weeks, and


6 months. We defined drug-naïve MDD patients based on the follow

ing two criteria: (a) those who were consistently diagnosed with MDD


over the visits, and (b) those who had no record of prescribed medi

cine due to depressive symptoms at their first visit. The diagnosis was


determined by board-certified psychiatrists based on the Structured


Clinical Interview from the Diagnostic and Statistical Manual of Men

tal Disorders, Fourth Edition (DSM-IV) Axis I disorders. Basic demo

graphic and clinical information such as family history of MDD and



education level were acquired during the psychiatric interview at the


clinic. The severities of depressive symptoms in all the participants


were assessed using the 17-item Hamilton Depression Rating Scale


(HDRS-17) (Hamilton, 1960) that reflects the degree of depression.


The participants, at each visit, were assessed using the HDRS-17, and


MRI scanning was performed at the first visit.


A total of 44 HCs (17 males; 27 females) were recruited from the


community; their ages ranged from 21 to 58 years. The recruitment


was made with the help of an advertisement for those who voluntarily


responded. The similar psychiatric diagnosis was carried out for HCs


who were confirmed with none of any current symptoms and past his

tory of psychiatric disorders. For both the groups, the participants


who satisfied the criteria such as comorbidity of any other major psy

chiatric disorders, expressing psychotic features (i.e., delusion, halluci

nation), having a history of a serious or unstable medical illness


including any primary neurological illness, and exhibited any contrain

dication to MRI scanning (e.g., metal implants) were considered inap

plicable to the study. The protocol of the study was approved by the


Institutional Review Board of Korea University Anam Hospital. In


accordance with the Declaration of Helsinki, all the 73 participants


signed a written informed consent prior to participating in the study.


All participants were acknowledged thoroughly to drop out of the


study at any stage, but there was no participant who dropped out.


The demographic information is summarized in Table 1.


There have been consistent evidences that patients with MDD


had lower educational attainment as compared to HCs (Lorant


et al., 2003). This means that lower educational level is one of the


essential components of MDD which could not be separable from the


diagnosis of MDD. So, in regard to the significant difference


(p-value = .018) between two groups in the education level, the distri

bution of the educational level between the two groups seems to


appropriately reflect real-world clinical situations. The unbalanced dis

tribution of the educational level between the two groups would


influence the classification results. However, there is no reason not to


utilize nonneuroimaging data with neuroimaging data in one classifica

tion model. In clinical psychiatry, ML-based approach primarily aims to


build pragmatic model so that it can help psychiatrists to diagnose and


treat mental disorders (Steele & Paulus, 2019). Hence, it is important



TABLE 1 Demographic information,
psychiatric diagnosis and their statistical
significance of MDD patients and HCs



MDD (n = 29) HC (n = 44) p-Value (t, χ [2] )


Age (years) 43.79 ± 13.06 39.68 ± 11.91 .169 (t = 1.389) [a]


Gender (female/male) 21/8 27/17 .33 (χ [2] = 0.948) [b]


Education level .018 (χ [2] = 8.035) [b]


Elementary and middle school 7 2


High school or college/university 21 35


Above graduate school 1 7


HDRS-17 score 14.48 ± 4.82 1.98 ± 2.11 <.001 (t = 13.166) [a]


Note: Data presented as mean ± standard deviation or n, unless otherwise indicated.

Abbreviations: HC, healhy control; HDRS, Hamilton Depression Rating Scale; MDD, major depressive

disorder.

a Independent sample t test.
b Pearson chi-square.


5000 JUN ET AL .



to take full advantage of available data and maximize the performance


of the classification model. In our method, we combine imaging and


phenotypic data such as educational level in a single framework by


constructing GCNs to enhance the classifying performance.


2.2 | Data acquisition


Volumetric structural MRI scans were acquired using a 3.0 Tesla Sie

mens Trio whole-body imaging system (Siemens Medical Systems,


Iselin, NJ). A T1-weighted magnetization-prepared rapid gradient

echo MP-RAGE was used (repetition time [TR] = 1900 ms, echo time


[TE] = 2.6 ms, field of view = 220 mm, matrix size = 256 × 256;


176 coronal slices without gap, voxel size = 0.9 × 0.9 × 1 mm [3],


flip angle = 9 [∘], and number of excitations = 1). Functional images were


obtained using a single-shot echo planer imaging sequence


(TR = 2,000 ms, TE = 30 ms, flip angle = 90 [∘], number of slices = 42,

matrix = 80 × 80, resolution = 3.0 × 3.0 × 3.0 mm [3] ).


2.3 | Preprocessing


We preprocessed data samples using the Data Processing Assistant


for Resting-State fMRI, a convenient plug-in software based on SPM


and REST. Among the 180 collected rs-fMRI volumes, we initially dis

carded the first 10 volumes of each subject before any further


processing to allow for magnetization equilibrium. Then, the remaining


170 volumes were slice-timing corrected, head motion corrected, and


spatially normalized to the standard Montreal Neurological Institute



space with a resolution of 3 × 3 × 3 mm [3] . To further reduce the


effects of nuisance signals, we performed the regressions of ventricle


and white matter signals as well as six head-motion profiles. Due to


the controversy of removing the global signal in the postprocessing of


rs-fMRI data, we did not regress out the global signal. The regressed

rs-fMRI images were parcellated into 114 ROIs [1] in the cortical


regions, 57 per hemisphere, which are derived from the 17 networks


using the functional atlas in Thomas Yeo et al. (2011). Subsequently,


the mean rs-fMRI time series at each ROI was computed and band

pass filtered from 0.01 to 0.1 Hz to exploit the characteristics of low


frequency fluctuations, thus resulting in a 114-dimensional vector for


each sample. Subjects with excessive head motion during scan acqui
sition [2] were excluded from further analysis.


3 | METHODS


In this section, we describe our experimental approaches for dis

tinguishing drug-naïve MDD patients from HCs based on rs-fMRI time


series. As shown in the overall procedure (Figure 1), we first estimate


EC by a group sparse representation along with SEM in an


unsupervised manner. This allows to impose similar connectional pat

terns among subjects but maintain individual differences in their net

work characteristics. We transform the estimated connectivity map


into a vectorial feature space and further reduce its dimension based


on statistically significant features while eliminating the redundant


and less informative features in a univariate manner. The selected


imaging feature vector and the phenotypic information (e.g., age, gen

der, etc.) of the subjects are incorporated into a population graph that



FIGURE 1 Overall framework of the proposed method for MDD identification. Test samples were marked with gray boxes to indicate that
the test sample labels are never used during training. GCNs, graph convolutional networks; GSL, group-constrained Sparse LASSO; MDD, major

depressive disorder; SEM, structural equation model


JUN ET AL . 5001



forms the basis for our GCNs. A vertex represents each subject's


acquisition, and an edge weight encodes the pairwise similarities of


phenotypic information. By operating the spectral graph convolutions


through the layers, the GCNs perform a binary classification between


the MDD patients and HCs. In addition to MDD identification, we fur

ther introduce an SA method for our trained GCNs to detect discrimi

native EC measures.


3.1 | Sparse estimation of EC


To estimate the fMRI-derived features in the ML pipeline of MDD


diagnosis, FC coefficients have been typically used (Bhaumik


et al., 2017; Sundermann et al., 2017; Wang et al., 2017; Yoshida


et al., 2017; Zhong et al., 2017). However, to validate the potential of


the EC as a biomarker, we estimate the EC coefficients by leveraging


the concept of SEM (Suk et al., 2015; Wee et al., 2014). Assume that


a sequence of T-length mean time series of rs-fMRI from R ROIs is

provided for subject n, that is, X n = x� [1] n [,] [���][,] [x] [r] n [,] [���][,] [x] [R] n � R [T][ ×][ R], where


         - T
x [r] n [=][ x] h [r] n,1 [,] [���][,] [x] n [r],t [,][���][,] [x] [r] n,T i R . In this study, we hypothesize that the

response of an ROI can be represented by a linear combination of


those of other ROIs. That is, given the time course of the other ROIs

excluding a target rth ROI, X [n] n [r] [R] [T][ ×][ R] ð [−][1] Þ, we can formulate the time

course of the target ROI as x [r] n [=][ X] n [n][r] [w] n [n][r] [+][ e][, where][ w] [n] n [r] [R] [R][−] [1] [ is a]


regression coefficient vector, and e is a zero-mean Gaussian distrib

uted error vector. It should be noted that these learnable regression

coefficients of N subjects, W [n] 1 [r] :N [=][ w] h [n] 1 [r] [,] [���][,] [w] n [n][r] [,] [���][,] [w] [n] N [r] i R ð [R] [−][1] Þ × N, indi
cate the causal relations between a target ROI and the other ROIs.


Further, motivated by a recent study (Supekar, Menon, Rubin,


Musen, & Greicius, 2008) that validated the effect of sparsity con

straints for detecting robust connections from noisy connectivities, we


apply a group-constrained sparse least absolute shrinkage and selection


(LASSO) (Wee, Yap, Zhang, Wang, & Shen, 2012) into our estimation of


the EC. This sparse representation through ℓ 1 -norm penalization can


provide a biologically plausible interpretation, following the fact that a


brain region typically forms relatively few numbers of connections.

Hence, the objective function, ℒ(W [\][r] ), is defined as follows:



resulting in f n R [m], where m is a reduced dimension. Thus, a feature

matrix for all N subjects, F = [f 1, ���, f n, ���, f N ] [>] R [N][ ×][ m], is fed into our


classifier as the input.


3.2 | Population graph construction


For classification, we use the GCNs (Parisot et al., 2018) based on a


population graph. The population graph is represented as a


weighted undirected graph G = Vf, ℰ, Wg, where V and ℰ are finite

sets of j V j = N vertices and edges respectively, and W R [N][ ×][ N] denotes


an weighted adjacency matrix. Specifically, each vertex corresponds


to a subject and the edges encode the phenotypic similarities between


every pair of subjects. To construct the aforementioned graph, the


following two factors need to be determined: (a) the vertex feature


vector assigned for each vertex and (b) the weighted adjacency matrix.


In this study, we define f n described in Section 3.1 as our feature vec

tor for each vertex. Regarding the adjacency matrix, we consider the


similarities of both imaging and nonimaging phenotypic features


(e.g., age, gender) between subjects (Parisot et al., 2018). Given a set


H
of H phenotypic measures p n = p� [h] n � h = 1 [for subject][ n][, each weight][ W] [ij]

between subject i and j is defined as follows:



W ij = exp − [k][f] [i] 2 [ −] σ [f] [2][j] [k] [2]



H

δ� p [h] i [,] [p] [h] j � ð2Þ

!X h = 1



where σ is a predefined kernel width of a Gaussian similarity function.


With respect to δ(�), it depends on the type of phenotypic measure.


For example, δ(�) is defined as the Kronecker delta function for cate

gorical measures (e.g., subject's gender) or the unistep function for

quantitative measures (e.g., subject's age) satisfying 1 iff j p [h] i [−] [p] [h] j [j][ <][ γ][;]

0 otherwise, where γ is a threshold to be determined. Therefore,


according to Equation (2), the edge weights increase when two sub

jects have a high similarity of vertex feature vectors and/or pheno

typic measures. It is noteworthy that this population graph


incorporates not only nonimaging features, but also imaging features,


compared with many existing studies that use only imaging features


for brain disease prediction.


3.3 | Graph convolutional networks for MDD

identification


After constructing the population graph represented in Section 3.2,


we learn the GCNs to predict the target labels of MDD/HC. To this


end, we introduce a spectral graph convolution as the main building


block in GCNs, which generalizes the conventional convolution opera

tion in the Euclidean domain to irregular graphs. It requires the eigen

decomposition of the graph Laplacian (Chung & Graham, 1997) to be


computed, followed by a graph Fourier transform (GFT) (Shuman,


Narang, Frossard, Ortega, & Vandergheynst, 2013).


First, our population graph is represented by its Laplacian matrix

ℒ, formulated as ℒ = D −W, where D = diagð d 0, …, d N −1 Þ R [N][ ×][ N] is the



ℒ�W [n] 1 [r] :N � = [1] 2



N
X kx [r] n [−][X] n [n][r] [w] [n] n [r] [k] [2] 2 [+][ α][k][W] [n] 1 [r] :N [k] 2,1 ð1Þ

n = 1



where α > 0 is a regularization parameter that indicates the magnitude of


sparsity and k �k 2,1 denotes an ℓ 2,1 -norm. The ℓ 2,1 -norm is derived from

the summation of ℓ 2 -norms of kw [n] n [r] [k] 1 [that is an individually imposed][ ℓ] 1 [-]


norm for each subject. This group-constrained sparsity not only cap

tures the consistent characteristics among subjects, but also retains


intersubject variability. It is noteworthy that self-to-self connections


are ignored by filling the rth element with zeros for each ROI, where

we newly define W [^] n1r:N [R] [R][ ×][ N] [. The resulting unsupervised representa-]



^ nr R
tion, W 1:N
n o r



r = 1 [, is regarded as the EC coefficients for all subjects.]



Finally, we concatenate the estimated connectivities of all ROIs

for a subject n such that hw^ [n] n [1] [,] [���][,][ ^][w] [n] n [r] [,] [���][,][ ^][w] [n] n [R] i R [R] [2] . Then, we conduct

LASSO feature selection method to select informative features, thus


5002 JUN ET AL .



diagonal degree matrix and d i = [P] j [W] [ij] is the degree of vertex i.


Because ℒ is a symmetric semidefinite matrix, it can be eigen

decomposed such that ℒ = UΛU [>], into a complete set of orthonormal

eigenvectors U = [u 0, …, u N − 1 ] R [N][ ×][ N] and the diagonal matrix of non
negative eigenvalues Λ = diag([λ 0, …, λ N − 1 ]) R [N][ ×][ N] (0 ≤ λ 0 ≤ ��� ≤ λ N

− 1 [). Particularly, it can be normalized as][ ℒ] [=][ I] N [−] [D] [−] [1][=][2] [WD] [−] [1][=][2] [, where]

I N R [N][ ×][ N] is an identity matrix, and the eigenvalues belong to the


range of [−1, 1]. Accordingly, ℒ contains information about the con

nections between subjects and their similarities.


Following the property of the GFT, given vertex features F and a


filter g θ that is a diagonal matrix parameterized with Fourier coeffi
cients θ R [N], the spectral convolutions are operated in the Fourier


domain as g θ - F = g θ (ℒ)F = g θ (UΛU [>] )F = U g θ (Λ)U [>] F. Specifically, in


this study, we apply filter approximation by representing g θ (Λ) as a


Kth order Chebyshev polynomial function of the eigenvalues


(Defferrard, Bresson, & Vandergheynst, 2016; Hammond,

Vandergheynst, & Gribonval, 2011), g θ Λð Þ = [P] [K] k = 0 [θ] [k] [Λ] [k] [, where][ θ] f [k] g [K] k = 0

is a set of polynomial coefficients. This provides the benefits of


K-localization and cost-effective computation of convolution. Thus,


the convolution can be rewritten as follows:



parameters are updated by backpropagating the following two


gradients:



1



: ð5Þ
A



∂J F out


=
X
∂ℋ [ð Þ] i [l] j = 1



∂J K

0 ~~@~~ ∂ℋ ð [l][ + 1] Þ X k = 0 θ i,j k ℒ [k] !



~~@~~



∂J



∂J
∂θ i,j k = ℒ [k] ℋ [ð Þ] i [l]



∂J ∂J

∂ℋ ðj [l][ + 1] Þ, ∂ℋ i [l]



∂J



j = 1



K
X



∂ℋ ð [l][ + 1] Þ
j



K
Xθ k Λ [k]

k = 0 !



K
g θ � F = U X



K
U [>] F = X



K K
Xθ k U� Λ [k] U [>] �F = X

k = 0 k = 0



Xθ k ℒ [k] F: ð3Þ

k = 0



On the basis of the spectral graph convolution, the overall model


comprises multiple convolutional layers and a fully connected layer


for the final prediction. In terms of the convolutional layer, layer-wise


activations are propagated, thus resulting in the representation of the


jth output graph for the (l + 1)th layer activation from the lth layer


activation, as follows:



F in
ℋ ð [l][ + 1] Þ = σ
j X

i = 1



K
Xθ i,j k ℒ [k] ℋ [ð Þ] i [l]

k = 0



!



+ b [ð Þ][l]
j



!



ð4Þ



After training the GCNs, during the test, test samples are


predicted with labels that maximize the probabilities of the softmax


output.


3.4 | Sensitivity analysis for interpretation of GCNbased prediction


Many previous works have developed the methods to explain the pre

dictions of deep learning models such as SA (Baehrens et al., 2010;


Simonyan, Vedaldi, & Zisserman, 2013) and layer-wise relevance

propagation (Bach et al., 2015), and so forth. Recently, SA has been


used in various applications such as medical diagnosis (Khan


et al., 2001) and ecological modeling (Gevrey, Dimopoulos, &


Lek, 2003), and so forth. However, to the best of our knowledge,


interpretation techniques for GCNs have not been investigated yet.


Thus, we devise a novel SA method for analyzing our trained GCN


model. That is, in addition to the diagnosis, it provides an interpreta

tion of what enables the GCNs to reach their individual predictions,


thus allowing the identification of significantly altered EC measures in


MDD patients.


SA is a gradient-based model interpretation method. As shown in


the Figure 2, it computes the norm k �k q over partial derivatives for a


differentiable prediction function with respect to the input (i.e., a sen

sitivity of the prediction based on the changes in the input). Given our


prediction function g and the vertex feature input f n for subject n, rel

evance scores in SA are defined as follows:



where σ(�) is a nonlinear activation function such as a rectified linear


unit (ReLU) and θ i,j k is the (F in × F out ) vector of polynomial coefficients

to be learned, and b [ð Þ] j [l] denotes the (1 × F out ) bias vector in the lth layer.

Here, we assume that by the GCN training, the vertices connected


with high edge weights become more similar as they pass through


multiple layers .


Finally, the final prediction layer comprises the fully connected


layer followed by a softmax activation function. That is, the GCNs


output a prediction label ^y n that describes the brain state (e.g., MDD


or HC) of a subject n. The loss function Jð ^y, yÞ is defined by the differ

ence between ^y and the actual label y among test vertices, where a


cross-entropy loss function is used in our implementation. Basically,


training the GCNs follows a transductive learning scheme. In other


word, during the training, we use the whole data including labeled


training and unlabeled test samples to construct the whole population


graph. In addition, the features of test samples are exploited to per

form the convolutions of training samples. The GCNs are trained to


minimize the loss evaluated on the labeled training samples, and the



where k �k q is the norm of the partial derivative. To represent the


magnitude to which variations of the input contribute to the output,


the ℓ 1 or ℓ 2 -norm can be used (Kardynska & Smieja, 2016). A high rel

evance score implies that changes in the EC value influence the diag

nosis of MDD significantly.


4 | EXPERIMENTAL SETTINGS AND

RESULTS


In this section, we validate the effectiveness of the proposed method


for MDD identification by considering the following scenarios:


(a) using FC or EC as features, (b) applying the feature selection or


not, and (c) using GCNs or other ML method as a classifier.



∂g
R n =

����∂f n



��� R [m] : ð6Þ
~~�~~ q


JUN ET AL . 5003


FIGURE 2 A schematic diagram of sensitivity analysis (SA) for our trained graph convolutional networks (GCNs). Gray lined arrows represent
forward computation for major depressive disorder (MDD)/healthy control (HC) prediction, and purple dashed arrows denote gradient
backpropagation of prediction with respect to input, resulting in the relevance scores



Furthermore, we identify the discriminant connectivities from the


magnitude of resulting relevance scores in our SA method. All the


[codes are available at “https://github.com/ejju92/EC_GCN.”](https://github.com/ejju92/EC_GCN)


4.1 | Experimental settings


For performance evaluation, we took a 10-fold stratified cross

validation technique (Bishop, 2006). Specifically, we partitioned the


samples of each class (i.e., drug-naïve MDD patients and HCs) into


10 folds and used samples of 1 fold for testing and those of the


remaining folds for training. Since we only have a total of 73 samples,


including 29 drug-naïve MDD patients and 44 HCs, that is, about


67 samples for the training set, we used the whole data including


labeled training and unlabeled test set to construct population graph,


as described in Section 3.3. However, the features of test set were


used for the convolutions of training samples during training, and the


loss is calculated only on a subset of training set. Note that the test


sample labels were never used during training. As such, we repeated


the above process 10 times by setting another different samples of


1 fold as the test set and rest as training set. The average of the


results is reported in Section 4.2.


For constructing the population graph, we set σ = 1, γ = 2, and


considered the ages and genders of the subjects as the phenotypic


measures for adjacency matrix representation. We trained our GCNs


with a single hidden layer that approximates the convolutions with


third-order Chebyshev polynomials, with parameters optimized by a


grid search. For regularization, we applied dropout among the input,



hidden, and prediction layers during training. The training hyper

parameters are chosen as follows: a dropout rate of 0.3, a learning

rate of 0.05, and an ℓ 2 regularization of 5 × 10 [−][4] with 200 epochs.


In this study, we considered comparable scenarios in terms of the


feature type, feature selection, and classifier. For the extracted fea

tures, we compared FC and EC features. Many existing works (Azari


et al., 1992; Van Dijk et al., 2009; Wang et al., 2007) have used the


FC as a common measure of representative features from rs-fMRI


time-series, demonstrating competitive performances in brain disease


prediction tasks. Specifically, we estimated the FC by calculating


pairwise Pearson correlation coefficients (Ye et al., 2015) between


ROIs. Finally, we used its vectorized upper triangular part, thereby


resulting in an R(R − 1)/2-dimensional feature vector for each

subject. [3]


In addition, we validated the effect of feature selection. Our fea

ture vector is high dimensional with possibilities of including noisy


features that may lead to performance degradation. Hence, we


attempted to retain the features with the highest discrimination pow

ers while eliminating redundant and less informative features using


LASSO feature selection method.


To evaluate our proposed method, we compared it with other


ML/deep learning methods. Regarding to the ML method, a linear


SVM is exploited, which is a widely used classifier for brain disease


diagnosis (Chen et al., 2016; Craddock et al., 2009; Fan et al., 2011).


The SVM estimates an optimal hyperplane that best separates the


two classes. We selected the model parameter C that balances

between a regularization term in the set of {10 [−][5], 10 [−][4], …, 10 [4] } by


nested cross-validation.


5004 JUN ET AL .



For the deep learning method, we evaluated BrainNetCNN


(Kawahara et al., 2017) and discriminative/generative long short-term


memory (LSTM-DG) (Dvornek, Li, Zhuang, & Duncan, 2019). The


BrainNetCNN (Kawahara et al., 2017) is based on a CNN framework


to capture the topological locality of structural brain networks. By


taking the connectivity matrix as input, it uses novel edge-to-edge,


edge-to-node, and node-to-graph convolutional filters for neuro

development prediction. With respect to the LSTM-DG (Dvornek


et al., 2019), i.e., joint LSTM-DG network, it performs a multi-task


learning of brain disorder identification and rs-fMRI time-series data


generation, given the rs-fMRI ROI time-series as input.


When calculating the relevance scores in the SA, we used the


ℓ 1 -norm that is the absolute of the partial derivative.


4.2 | Performance results and analysis


For a quantitative evaluation of the comparable scenarios illustrated


in Section 4.1, we considered the following metrics:




- ACCuracy (ACC) = (TP + TN)/(TP + TN + FP + FN).


- SENsitivity (SEN) = TP/(TP + FN).


- SPECificity (SPE) = TN/(TN + FP).


- Area under the curve (AUC).


where TP, TN, FP, and FN denote true positive, true negative,


false positive, and false negative, respectively. Specifically, higher


values of the sensitivity and specificity represent the lower chances of


misdiagnosing each clinical label. We summarized the experimental


results under various conditions in Table 2.


As presented in Table 2, our method of GCNs w/LASSO demon

strated the best performance with respect to all the metrics, com

pared to other competitive methods including SVM, BrainNetCNN


(Kawahara et al., 2017), and LSTM-DG (Dvornek et al., 2019). From


the experimental results, the following findings can be inferred: fea

ture selection helps improve the performance in all scenarios. In par

ticular, the effect of feature selection resulted in significant

performance gains for high dimensional (R [2] ) EC feature vector, which


is approximately twice higher than that of FC (R × (R − 1)/2) given



Method Metric Effective connectivity Functional connectivity


SVM ACC 0.626 ± 0.144 [a] 0.553 ± 0.252*


SEN 0.266 ± 0.199 [a] 0.350 ± 0.262*


SPE 0.870 ± 0.188 [a] 0.690 ± 0.287*


AUC 0.568 ± 0.156 [a] 0.520 ± 0.249*



TABLE 2 Classification performance

of various scenarios. The mean and SD

over 10-fold cross-validation are

represented. For each imaging feature,
the highest performance is bolded in

terms of each evaluation metric



BrainNetCNN (Kawahara ACC 0.557 ± 0.103* 0.587 ± 0.153 [a]
et al., 2017) SEN 0.200 ± 0.233* 0.433 ± 0.386 [a]


SPE 0.785 ± 0.248* 0.710 ± 0.245 [a]


AUC 0.492 ± 0.086* 0.571 ± 0.172 [a]





GCNs ACC 0.591 ± 0.095* 0.539 ± 0.139*


SEN 0.283 ± 0.258* 0.066 ± 0.133*


SPE 0.820 ± 0.244* 0.850 ± 0.204*


AUC 0.563 ± 0.211* 0.428 ± 0.168*


Note: *: p < .05.

Abbreviations: ACC: ACCuracy; AUC, area under the curve; GCNs, graph convolutional networks; SEN,

SENsitivity; SPE, SPECificity; SVM, support vector machine.

a No statistical difference from the McNemar's test.
b The reference method for the statistical tests with other methods.


JUN ET AL . 5005


**[f]** [i]



R ROIs. More specifically, the quantitative improvements for FC/EC in


accuracy were 5.0/7.2% in SVM and 2.5/15% in GCNs, respectively.


In addition, the proposed method (GCNs w/LASSO) achieved the


highest AUC in both EC and FC scenarios, implying that their predic

tions were not biased toward the majority class. It is noteworthy that


in our dataset, because the number of samples available for each class


was not balanced, that is, MDD patients (29) versus HC (44), the per

formance results could have been likely inflated. Nevertheless, our


method achieved the AUC of 0.791 in EC and 0.665 in FC, respec

tively, demonstrating the power of our method to still identify the


minority class well.


To demonstrate the statistical power of our method, we con

ducted a power (1-probability of Type II error) analysis with R package


(Kohl, 2019) that is based on a previous research (Flahault, Cadilhac, &


Thomas, 2005). As shown in Table 2, the mean sensitivity (SD) of our


classifier generated from 10-fold cross-validation is 0.566 ± 0.300. As

the formula of a confidence interval is mean � Z ~~p~~ [SD] ~~[f]~~ n **[f]** [i], the mean sensitiv

–
ity (95% CI) and marginal error is 0.566 (0.380 0.752) and 0.186,


respectively. With α (probability of Type I error) = 0.05, sensitiv

ity = 0.566, marginal error = 0.186, Z = 1.96, number of cases = 29,


and number of controls = 44, the power of our classifier is estimated


to 63.6%. When considering that most researchers set the statistical


power to the range between 60 and 80% (OECD, 2014), the value of


our statistical power is adequate.


In addition, in order to validate whether any observed difference


between the proposed method and others is statistically significant,


we conducted the McNemar' statistical test. We observed that the


proposed method outperformed statistically (p − value < .05); the


competing methods of BrainNetCNN (Kawahara et al., 2017) and


GCNs for EC feature, SVM, GCNs, GCNs w/LASSO for FC feature,


and LSTM-DG (Dvornek et al., 2019).


We compared the computational time [4] of the proposed method


with that of our comparative methods in terms of training and test


time (second) per epoch, as presented in Table 3. We measured the


time on a NVIDIA GTX 1070 GPU. It is noteworthy that as our GCNs


are tuning network parameters in a transductive manner, basically the


learning process occurs in a testing phase only. Thus, the training and


test time is identical.


Furthermore, we conducted a comparative experiment to estimate


EC through GC analysis (GCA) for comparison with that of our pro

posed method. By using the estimated EC as feature, we performed


MDD identification using GCNs, SVM, and BrainNetCNN (Kawahara


et al., 2017) as classifier. The results are summarized in Table 4. It is


noteworthy that with the GCA features, our proposed method was still


superior to the competing methods in ACC, SEN, and AUC.



4.3 | SA-based interpretation


As described in Section 3.4, we conducted the SA for our GCNs to


identify significantly altered EC measures in MDD patients compared


to HCs. From the SA, we obtained the relevance scores estimated for

N subjects, R = Rf n g [N] n = 1 [. Here, after averaging them over all subjects,]

the mean relevance scores R [^] were considered for analysis. Specifi

cally, to investigate the discriminative EC measures, we selected the


connectivities whose relevance scores were higher than (μ + 1.5 * σ),


where μ and σ denote the mean and SD of the mean relevance scores,


respectively. The selected connections are presented in Table 5 and


Figure 3. The larger the relevance score values, the greater the impor

tance of corresponding EC measures for the diagnosis of MDD.


Basically, we inputed the EC (EC) feature vector selected by our


feature selection method, that is, LASSO, into the GCNs, and then


applied SA to the learned GCN to investigate the discriminant connec

**[f]** [i] tivities for MDD identification from input feature vector. Through the


LASSO feature selection, a total of 107 connectivities are selected


from the 114 × 113/2 = 6,441 connectivities when considering the


union of connectivities selected from all folds in cross-validation, as


shown in Table A2.


We examined the resulting LASSO coefficients for 13 connectivi

ties chosen in the SA, as presented in Table 5. Considering that the


mean coefficient for 107 connectivities is −0.00024, it is noteworthy


that the coefficients for 13 connectivities have significantly high


values and thus we believe that our GCNs well captured the informa

tive features and their relations.


5 | DISCUSSIONS


In this study, we successfully distinguished drug-naïve MDD patients


from HCs using GCNs. Hitherto, ML algorithms have been widely


used for diagnosing MDD (Gao, Calhoun, & Sui, 2018). The accuracies


of the performances ranged from good to excellent. For example, Lord


et al. (Lord et al., 2012) and Wang et al. (Wang et al., 2017) reported


99.0 and 95.0% accuracy, respectively. Therefore, from the sheer


number of reported accuracies, the difference in performance


between ours and previous studies appears slight.


However, two distinguished features ensure the intrinsic reliabil

ity of our results. One is that we conducted a diagnostic evaluation of


participants in the drug-naïve state. Measuring neuroimaging mate

rials in the drug-naïve state is substantially important because drugs


such as antidepressants have substantial effects on the structural


(Dusi, Barlati, Vita, & Brambilla, 2015) and functional (Wessa &




**[f]** [i]


TABLE 3 Comparison of the

computational time between the

proposed method and the competitive

methods in terms of training and test

time pear epoch




**[f]** [i]


Measure GCNs SVM BrainNetCNN LSTM-DG

(Kawahara et al., 2017) (Dvornek et al., 2019)


Training time (s) 0.00375 0.00116 0.31375 0.21620


Test time (s) 0.00375 0.00015 2.18694 0.07804


Abbreviations: GCNs, graph convolutional networks; LSTM-DG, discriminative/generative long short
term memory; SVM, support vector machine.


5006 JUN ET AL .



Lois, 2015) aspects of the brain. Another important methodological


factor is that we ensured diagnostic stability for 6 months. Owing to


the operational diagnostic criteria of the DSM series, diagnostic


TABLE 4 Performance comparison between the case of using the

GCA-EC and ours. The mean and SD over 10-fold cross-validation are

represented. For each method, the highest performance is bolded in

terms of each evaluation measure


Method Measure GCA-EC Ours


SVM ACC 0.576 ± 0.102 0.626 ± 0.144


SEN 0.066 ± 0.133 0.266 ± 0.199


SPE 0.915 ± 0.187 0.870 ± 0.188


AUC 0.490 ± 0.077 0.568 ± 0.156



BrainNetCNN (Kawahara ACC 0.519 ± 0.129 0.557 ± 0.103
et al., 2017) SEN 0.266 ± 0.409 0.200 ± 0.233


SPE 0.720 ± 0.423 0.785 ± 0.248


AUC 0.493 ± 0.078 0.492 ± 0.086



GCNs w/LASSO ACC 0.658 ± 0.187 0.741 ± 0.130


SEN 0.633 ± 0.233 0.566 ± 0.300


SPE 0.684 ± 0.233 0.869 ± 0.166


AUC 0.738 ± 0.220 0.791 ± 0.153


Abbreviations: ACC: ACCuracy; AUC, area under the curve; GCA-EC,

effective connectivity estimated by Granger causality analysis; GCNs,

graph convolutional networks; SEN, SENsitivity; SPE, SPECificity; SVM,

support vector machine.



changes are not rare from a longitudinal perspective. For example, in


the Korean population (Kim, Woo, Chae, & Bahk, 2011), the diagnostic


consistency of MDD by DSM-IV was only 84.8% in the first year. No


matter how excellent the discriminating algorithms are, they are


meaningless if the index diagnosis of MDD is changed to other


indexes. To avoid the potential pitfall of cross-sectional design, it is


necessary to ensure longitudinal diagnostic stability. However, if the


participation in the study is postponed until 1 or 2 years after the ini

tial diagnosis, the confounding effects of the antidepressants can


become problematic. Therefore, as suggested in a recent review


(Kim & Na, 2018), we partially solved this issue using the MRI of par

ticipants whose diagnostic stability were confirmed for at least


6 months. Many previous ML studies did not provide reliable informa

tion of these critical methodological issues. Both the aforementioned


studies that reported better discriminating performances than our


results (Lord et al., 2012; Wang et al., 2017) did not mention the


selection procedure of participants in terms of longitudinal diagnostic


instability. Regarding antidepressants medication, one study reported


that all the participants were taking antidepressants (Lord


et al., 2012), and another study did not provide medication-related


information. We believe that the well-defined selection process of the


participants rendered our results more reliable than those of previ

ously conducted studies.


5.1 | Discriminative features analyses


Through the SA of our GCNs, we demonstrated that the dorsal pre

frontal cortex received decreased connectivity from the precentral


ventral, striate cortex, parietal medial, inferior parietal lobule, para

hippocampal cortex. The dorsal prefrontal has long been known as


a key region of depression, wherein cognitive reappraisal occurs in


a top-down manner (Alexander & Brown, 2011; Ochsner, Silvers, &



TABLE 5 Discriminant effective connectivities from the SA of our GCNs. For each connection, we presented the index and name of the ROI,

RS, and corresponding LASSO coefficient. The relevance scores are sorted in the descending order


Index Source ROI Index Destination ROI RS value LASSO coefficient


19 Precentral ventral, left 24 Dorsal prefrontal cortex, left 0.99684 −1.19076


108 Anterior temporal, right 27 Orbital frontal cortex, left 0.97552 −0.01477


112 Retrosplenial, right 56 Parahippocampal cortex, left 0.92768 −0.75947


3 Striate cortex, left 24 Dorsal prefrontal cortex, left 0.82408 0.03766


79 Parietal medial, right 24 Dorsal prefrontal cortex, left 0.81150 −0.40647


38 Inferior parietal lobule, left 39 Dorsal prefrontal cortex, left 0.73504 0.68513


23 Inferior parietal lobule, left 44 Cingulate posterior, left 0.63800 −0.06180


59 Extrastriate cortex, right 58 Striate cortex, right 0.62140 0.72370


20 Insula, left 19 Precentral ventral, left 0.61601 0.31163


109 Dorsal prefrontal cortex, right 82 Dorsal prefrontal cortex, right 0.61421 0.37992


29 Temporal pole, left 94 Cingulate anterior, right 0.59003 −0.00683


113 Parahippocampal cortex, right 24 Dorsal prefrontal cortex, left 0.54320 0.19291


93 Lateral prefrontal cortex, right 35 Lateral ventral prefrontal cortex, left 0.51404 −0.41892


Abbreviations: GCNs, graph convolutional networks; ROI, region of interest; RS, relevance score; SA, sensitivity analysis.


JUN ET AL . 5007


FIGURE 3 Discriminative effective connectivities from the sensitivity analysis (SA) of our graph convolutional networks (GCNs). Each color
denotes the following brain networks: (1) central visual network, (2) peripheral visual network, (3) somatomotor network, (4) dorsal attention
network, (5) salience/ventral attention network, (6) limbic network, (7) control network, (8) default network, and (9) temporal parietal network. All
the above networks follow 17 brain networks defined in the study of Thomas Yeo et al. (2011)



Buhle, 2012). Disturbed connectivity with this region may result in


biased selective attention to negative events and the related emo

tions such as depressive feeling, sadness, and shamefulness, which


may contribute to the pathophysiology of MDD. However, the


directions among the connectivities that contributed to the onset


of depression have not been elucidated. By measuring the EC, we


identified the directionality in the aberrant connectivity with this


region.


Another interesting finding from the results of the SA is the


abnormal connectivity from right retrosplenial cortices to the left


parahippocampal cortices. The retrosplenial cortex is located in the


posterior corpus callosum, the Brodmann areas 29 and 30. Mean

while, the retrosplenial and parahippocampal cortices are jointly


involved in visuospatial memory (Epstein, 2008; Mitchell, Czajkowski,


Zhang, Jeffery, & Nelson, 2018); they are crucial in emotion regulation


(Bubb, Kinnavane, & Aggleton, 2017; Maddock, 1999). Animal studies


revealed that the retrosplenial cortex receives inputs primarily from


the parahippocampal and prefrontal cortex (Sugar, Witter, van


Strien, & Cappaert, 2011; Suzuki & Amaral, 1994). Indeed, the retro

splenial cortex is activated more than other regions in response to


negative emotional words (Maddock & Buonocore, 1997). A possible


mechanism by which the disturbed connectivity between the



retrosplenial and parahippocampal cortices contribute to the MDD is


through associative functions. Both the retrosplenial and para

hippocampal cortices play a key role in the processing of contextual


associations in MDD (Harel, Tennyson, Fava, & Bar, 2016). Broad


scope and lively association exhibit a reciprocal relationship with posi

tive mood and increased activity; narrow scope and ruminative pat

tern of thoughts tend to be associated with depressed mood,


pessimistic thoughts of the future, and decreased energy (Bar, 2009;


Harel et al., 2016; Nolen-Hoeksema, 2000). We speculate that the


decoupling of the retrosplenial and parahippocampal can result in


inappropriate associative processing that, in turn, contributes to the


negative view of future.


5.2 | Limitations


This study has a few limitations that must be noted. First, the sam

ple size (29 MDD patients and 44 HCs) may not be sufficiently


large. Indeed, a recent study reported the characteristics of EC from


the rs-fMRI of MDD patients (n = 336) as compared to HC


(n = 350) (Rolls et al., 2018). However, a fundamental difference


exists between the previous study and our study. Whereas the


5008 JUN ET AL .



previous study primarily examined the characteristics of EC in MDD


via group-level analysis, we aimed to discriminate MDD patients


from HCs using the individual-level approach. To the best of our


knowledge, a GCN-based deep learning model for distinguishing


MDD patients from the HCs has not been developed. Second,


detailed sociodemographic variables (e.g., marital status, cohabita

tion, and socioeconomic status) and clinical variables (e.g., current


and past suicide attempt, family history of psychiatric disorder,


and/or suicide death) were not fully obtained in the MDD group.


Third, we discussed abnormal EC (e.g., disturbed bidirectional con

nectivity between parahippocampal and retrosplenial cortices) in


relation with the characteristic symptoms of MDD (e.g., negative


scope and rumination). However, we could not directly confirm such


connections between EC and symptomatology in the case of MDD.


Future studies require a larger sample size and relevant instruments


for the investigation of symptoms.


6 | CONCLUSION


In this study, we successfully estimated EC from rs-fMRI and devel

oped the GCN model for discriminating drug-naïve MDD patients


from HCs. We empirically exhibited the superiority of our method in


various MDD classification scenarios, in terms of extracted features,


feature selection, and classifiers. Because the performance ability did


not provide any insight into the discriminant connectivity for the diag

nosis of MDD, we devised a novel interpretation approach of our


trained GCNs. Specifically, we applied the SA for the GCNs and


selected the connectivities with high relevance scores. From the


results of the SA, we could successfully identify regions that were pre

viously identified as those associated with the MDD symptoms in the


psychiatry literature. Thus, our results showed that EC may be promis

ing for building deep learning-based models in the field of neuroimag

ing. Further studies with a larger sample size are required to validate


our findings.


ACKNOWLEDGMENTS


This research was supported by Research Program To Solve Social


Issues of the National Research Foundation of Korea (NRF) funded by


the Ministry of Science and ICT (NRF-2017R1A2B4002090) and par

tially by Institute of Information and communications Technology


Planning and Evaluation (IITP) grant funded by the Korea government


(MSIT) (No. 2019-0-00079, Artificial Intelligence Graduate School


Program [Korea University]).


CONFLICT OF INTEREST


The authors declare no conflict of interest.


DATA AVAILABILITY STATEMENT


The data that support the findings of this study are available on


request from the corresponding author (B. J. H.). The data are not


publicly available due to restrictions, for example, their containing


information that could compromise the privacy of research


participants.



ETHICS STATEMENT


This research obtained ethics approval from Korea University Anam


Hospital, Seoul. All the participants agreed to join the research and


gave informed consent before taking part.


INFORMED CONSENT


In accordance with the Declaration of Helsinki, all the 73 participants


signed a written informed consent prior to participating in the study.


ORCID


Eunji Jun [https://orcid.org/0000-0002-3121-7734](https://orcid.org/0000-0002-3121-7734)


Kyoung-Sae Na [https://orcid.org/0000-0002-0148-9827](https://orcid.org/0000-0002-0148-9827)


Wooyoung Kang [https://orcid.org/0000-0003-4733-027X](https://orcid.org/0000-0003-4733-027X)


Jiyeon Lee [https://orcid.org/0000-0002-8400-2729](https://orcid.org/0000-0002-8400-2729)


Heung-Il Suk [https://orcid.org/0000-0001-7019-8962](https://orcid.org/0000-0001-7019-8962)


Byung-Joo Ham [https://orcid.org/0000-0002-0108-2058](https://orcid.org/0000-0002-0108-2058)


ENDNOTES


1 For the names of all the regions, refer to Table A1.


2 We excluded patients with a displacement of greater than 2.5 mm

and/or an angular rotation of greater than 2 [∘] in any direction.


3 In this paper, 114 × 113/2 = 6,441 dimensional vector.


4 The Python time module was used.


REFERENCES


Alexander, W. H., & Brown, J. W. (2011). Medial prefrontal cortex as an
action-outcome predictor. Nature Neuroscience, 14(10), 1338–1344.
American Psychiatric Association. (2013). Diagnostic and statistical manual

of mental disorders. BMC Medicine, 17, 133–137.

Anand, A., Li, Y., Wang, Y., Wu, J., Gao, S., Bukhari, L., … Lowe, M. J.
(2005). Activity and connectivity of brain mood regulating circuit in

depression: A functional magnetic resonance study. Biological Psychiatry, 57(10), 1079–1088.

Azari, N., Rapoport, S., Grady, C., Schapiro, M., Salerno, J., GonzalesAviles, A., & Horwitz, B. (1992). Patterns of interregional correlations

of cerebral glucose metabolic rates in patients with dementia of the
Alzheimer type. Neurodegeneration, 1(1), 101–111.

Bach, S., Binder, A., Montavon, G., Klauschen, F., Müller, K.-R., & Samek, W.

(2015). On pixel-wise explanations for non-linear classifier decisions by
layer-wise relevance propagation. PLoS One, 10(7), e0130140.

Baehrens, D., Schroeter, T., Harmeling, S., Kawanabe, M., Hansen, K., &
MÃžller, K.-R. (2010). How to explain individual classification decisions. Journal of Machine Learning Research, 11(Jun), 1803–1831.
Bar, M. (2009). A cognitive neuroscience hypothesis of mood and depression. Trends in Cognitive Sciences, 13(11), 456–463.

Bhaumik, R., Jenkins, L. M., Gowins, J. R., Jacobs, R. H., Barba, A.,

Bhaumik, D. K., & Langenecker, S. A. (2017). Multivariate pattern analy
sis strategies in detection of remitted major depressive disorder using

resting state functional connectivity. NeuroImage: Clinical, 16, 390–398.
Bishop, C. M. (2006). Pattern recognition and machine learning, New York,

NY: Springer.
Brosch, T., Tam, R., & Alzheimer's Disease Neuroimaging Initiative. (2013).

Manifold learning of brain MRIs by deep learning. In International Con
ference on Medical Image Computing and Computer-Assisted Intervention, (pp. 633–640). Springer.
Bubb, E. J., Kinnavane, L., & Aggleton, J. P. (2017). Hippocampal–dience
–
phalic cingulate networks for memory and emotion: An anatomical

guide. Brain and Neuroscience Advances, 1, 2398212817723443.
Büchel, C., & Friston, K. J. (1997). Modulation of connectivity in visual

pathways by attention: Cortical interactions evaluated with structural


JUN ET AL . 5009



equation modelling and fMRI. Cerebral Cortex (New York, NY: 1991), 7
(8), 768–778.

Cao, L., Guo, S., Xue, Z., Hu, Y., Liu, H., Mwansisya, T. E., … Liu, Z. (2014).

Aberrant functional connectivity for diagnosis of major depressive dis
order: A discriminant analysis. Psychiatry and Clinical Neurosciences, 68
(2), 110–119.
Chen, H., Duan, X., Liu, F., Lu, F., Ma, X., Zhang, Y., … Chen, H. (2016). Mul
tivariate classification of autism spectrum disorder using frequency
specific resting-state functional connectivity: A multi-center study.

Progress in Neuro-Psychopharmacology and Biological Psychiatry,

64, 1–9.

Chung, F. R., & Graham, F. C. (1997). Spectral graph theory (Vol. 92), Provi
dence, RI: American Mathematical Society.
Craddock, R. C., Holtzheimer, P. E., Hu, X. P., & Mayberg, H. S. (2009). Dis
ease state prediction from resting state functional connectivity. Magnetic Resonance in Medicine, 62(6), 1619–1628.
Defferrard, M., Bresson, X., & Vandergheynst, P. (2016). Convolutional

neural networks on graphs with fast localized spectral filtering.

–
Advances in neural information processing systems (pp. 3844 3852).

Paper presented at NIPS, Barcelona, Spain.

Drysdale, A. T., Grosenick, L., Downar, J., Dunlop, K., Mansouri, F.,
Meng, Y., … Liston, C. (2017). Resting-state connectivity biomarkers

define neurophysiological subtypes of depression. Nature Medicine, 23
(1), 28–38.

Dusi, N., Barlati, S., Vita, A., & Brambilla, P. (2015). Brain structural effects

of antidepressant treatment in major depression. Current Neuropharmacology, 13(4), 458–465.
Dvornek, N. C., Li, X., Zhuang, J., & Duncan, J. S. (2019). Jointly discrimina
tive and generative recurrent neural networks for learning from fMRI.

In International Workshop on Machine Learning in Medical Imaging,

–
(pp. 382 390). Springer.
Epstein, R. A. (2008). Parahippocampal and retrosplenial contributions to
human spatial navigation. Trends in Cognitive Sciences, 12(10),

388–396.

Fan, Y., Liu, Y., Wu, H., Hao, Y., Liu, H., Liu, Z., & Jiang, T. (2011). Discrimi
nant analysis of functional connectivity patterns on Grassmann manifold. NeuroImage, 56(4), 2058–2067.
Flahault, A., Cadilhac, M., & Thomas, G. (2005). Sample size calculation

should be performed for design accuracy in diagnostic test studies.
Journal of Clinical Epidemiology, 58(8), 859–862.
Friston, K., Ungerleider, L., Jezzard, P., & Turner, R. (1994). Characterizing mod
ulatory interactions between areas v1 and v2 in human cortex: A new
treatment of functional MRI data. Human Brain Mapping, 2(4), 211–224.
Gao, S., Calhoun, V. D., & Sui, J. (2018). Machine learning in major depres
sion: From classification to treatment outcome prediction. CNS Neuroscience & Therapeutics, 24(11), 1037–1052.
Geng, X., Xu, J., Liu, B., & Shi, Y. (2018). Multivariate classification of major

depressive disorder using the effective connectivity and functional

connectivity. Frontiers in Neuroscience, 12, 38.
Gevrey, M., Dimopoulos, I., & Lek, S. (2003). Review and comparison of

methods to study the contribution of variables in artificial neural network models. Ecological Modelling, 160(3), 249–264.
Granger, C. (1969). Investigating causal relations by econometric models
and cross-spectral methods. Econometrica, 37(3), 424–438.

Greicius, M. D., Flores, B. H., Menon, V., Glover, G. H., Solvason, H. B.,

Kenna, H., … Schatzberg, A. F. (2007). Resting-state functional connectivity

in major depression: Abnormally increased contributions from subgenual
cingulate cortex and thalamus. Biological Psychiatry, 62(5), 429–437.
Guo, H., Cheng, C., Cao, X., Xiang, J., Chen, J., & Zhang, K. (2014). Resting
state functional connectivity abnormalities in first-onset unmedicated
depression. Neural Regeneration Research, 9(2), 153–163.

Hamilton, J. P., Chen, G., Thomason, M. E., Schwartz, M. E., & Gotlib, I. H.

(2011). Investigating neural primacy in major depressive disorder: Mul
tivariate granger causality analysis of resting-state fMRI time-series
data. Molecular Psychiatry, 16(7), 763–772.



Hamilton, M. (1960). A rating scale for depression. Journal of Neurology,
Neurosurgery, and Psychiatry, 23(1), 56–62.
Hammond, D. K., Vandergheynst, P., & Gribonval, R. (2011). Wavelets on

graphs via spectral graph theory. Applied and Computational Harmonic
Analysis, 30(2), 129–150.
Harel, E. V., Tennyson, R. L., Fava, M., & Bar, M. (2016). Linking major

depression and the neural substrates of associative processing. Cognitive, Affective, & Behavioral Neuroscience, 16(6), 1017–1026.

Hu, S., Cao, Y., Zhang, J., Kong, W., Yang, K., Zhang, Y., & Li, X. (2012).

Granger causality's shortcomings and new causality measure. Cognitive

Neurodynamics, 6, 33–42.
Jing, B., Long, Z., Liu, H., Yan, H., Dong, J., Mo, X., … Li, H. (2017). Identify
ing current and remitted major depressive disorder with the Hurst

exponent: A comparative study on two automated anatomical labeling
atlases. Oncotarget, 8(52), 90452–90464.
Kardynska, M., & Smieja, J. (2016). L1 and L2 norms in sensitivity analysis

of signaling pathway models. In 2016 21st International Conference on
Methods and Models in Automation and Robotics (MMAR),

–
(pp. 589 594). IEEE.

Kawahara, J., Brown, C. J., Miller, S. P., Booth, B. G., Chau, V.,

Grunau, R. E., … Hamarneh, G. (2017). BrainNetCNN: Convolutional

neural networks for brain networks; towards predicting neuro
development. NeuroImage, 146, 1038–1049.

Kempton, M. J., Salvador, Z., Munafò, M. R., Geddes, J. R., Simmons, A.,
Frangou, S., & Williams, S. C. (2011). Structural neuroimaging studies

in major depressive disorder: Meta-analysis and comparison with bipolar disorder. Archives of General Psychiatry, 68(7), 675–690.

Khan, J., Wei, J. S., Ringner, M., Saal, L. H., Ladanyi, M., Westermann, F., …
Meltzer, P. S. (2001). Classification and diagnostic prediction of can
cers using gene expression profiling and artificial neural networks.
Nature Medicine, 7(6), 673–679.
Kim, W., Woo, Y. S., Chae, J.-H., & Bahk, W.-M. (2011). The diagnostic sta
bility of DSM-IV diagnoses: An examination of major depressive disor
der, bipolar I disorder, and schizophrenia in korean patients. Clinical
Psychopharmacology and Neuroscience, 9(3), 117–121.
Kim, Y.-K., & Na, K.-S. (2018). Application of machine learning classifica
tion for structural brain MRI in mood disorders: Critical review from a

clinical perspective. Progress in Neuro-Psychopharmacology and Biologi
cal Psychiatry, 80, 71–80.
Kohl, M. (2019). MKmisc: Miscellaneous functions from M. Kohl. R pack
age version 1.6.
Li, M., Das, T., Deng, W., Wang, Q., Li, Y., Zhao, L., … Li, T. (2017). Clinical

utility of a short resting-state MRI scan in differentiating bipolar from
unipolar depression. Acta Psychiatrica Scandinavica, 136(3), 288–299.
Liao, W., Ding, J., Marinazzo, D., Xu, Q., Wang, Z., Yuan, C., … Chen, H. (2011).

Small-world directed networks in the human brain: Multivariate granger
causality analysis of resting-state fMRI. NeuroImage, 54(4), 2683–2694.

Lorant, V., Deliège, D., Eaton, W., Robert, A., Philippot, P., & Ansseau, M.
(2003). Socioeconomic inequalities in depression: A meta-analysis.
American Journal of Epidemiology, 157(2), 98–112.
Lord, A., Horn, D., Breakspear, M., & Walter, M. (2012). Changes in com
munity structure of resting state functional connectivity in unipolar
depression. PLoS One, 7(8), e41282.
Ma, Z., Li, R., Yu, J., He, Y., & Li, J. (2013). Alterations in regional homoge
neity of spontaneous brain activity in late-life subthreshold depression. PLoS One, 8(1), e53148.

Maddock, R. J. (1999). The retrosplenial cortex and emotion: New insights

from functional neuroimaging of the human brain. Trends in Neurosciences, 22(7), 310–316.

Maddock, R. J., & Buonocore, M. H. (1997). Activation of left posterior cin
gulate gyrus by the auditory presentation of threat-related words: An
fMRI study. Psychiatry Research: Neuroimaging, 75(1), 1–14.
McIntosh, A. R., Rajah, M. N., & Lobaugh, N. J. (1999). Interactions of pre
frontal cortex in relation to awareness in sensory learning. Science,
284(5419), 1531–1533.


5010 JUN ET AL .



Mitchell, A. S., Czajkowski, R., Zhang, N., Jeffery, K., & Nelson, A. J. (2018).

Retrosplenial cortex and its role in spatial cognition. Brain and Neuro
science Advances, 2, 2398212818757098.

Nolen-Hoeksema, S. (2000). The role of rumination in depressive disorders

and mixed anxiety/depressive symptoms. Journal of Abnormal Psychology, 109(3), 504–511.
Ochsner, K. N., Silvers, J. A., & Buhle, J. T. (2012). Functional imaging stud
ies of emotion regulation: A synthetic review and evolving model of

the cognitive control of emotion. Annals of the New York Academy of

Sciences, 1251, E1–E24.

OECD. (2014). Detailed review paper (DRP) on Molluscs life-cycle toxicity

testing.

Parisot, S., Darlix, A., Baumann, C., Zouaoui, S., Yordanova, Y., Blonski, M.,

… Paragios, N. (2016). A probabilistic atlas of diffuse WHO grade II glioma locations in the brain. PLoS One, 11(1), e0144200.

Parisot, S., Ktena, S. I., Ferrante, E., Lee, M., Guerrero, R., Glocker, B., &

Rueckert, D. (2018). Disease prediction using graph convolutional net
works: Application to autism spectrum disorder and Alzheimer's dis
ease. Medical Image Analysis, 48, 117–130.
Park, H.-J., & Friston, K. (2013). Structural and functional brain networks:
From connections to cognition. Science, 342(6158), 1238411.
Penny, W. D., Stephan, K. E., Mechelli, A., & Friston, K. J. (2004). Modelling

functional integration: A comparison of structural equation and

dynamic causal models. NeuroImage, 23, S264–S274.

Ramasubbu, R., Brown, M. R., Cortese, F., Gaxiola, I., Goodyear, B.,
Greenshaw, A. J., … Greiner, R. (2016). Accuracy of automated classifi
cation of major depressive disorder as a function of symptom severity.

NeuroImage: Clinical, 12, 320–331.

Rolls, E. T., Cheng, W., Gilson, M., Qiu, J., Hu, Z., Ruan, H., … Feng, J.
(2018). Effective connectivity in depression. Biological Psychiatry: Cognitive Neuroscience and Neuroimaging, 3(2), 187–197.

Schlösser, R. G., Wagner, G., Koch, K., Dahnke, R., Reichenbach, J. R., &
Sauer, H. (2008). Fronto-cingulate effective connectivity in major

depression: A study with fMRI and dynamic causal modeling.
NeuroImage, 43(3), 645–655.

Shuman, D. I., Narang, S. K., Frossard, P., Ortega, A., & Vandergheynst, P.
(2013). The emerging field of signal processing on graphs: Extending

high-dimensional data analysis to networks and other irregular
domains. IEEE Signal Processing Magazine, 30(3), 83–98.
Simonyan, K., Vedaldi, A., & Zisserman, A. (2013). Deep inside con
volutional networks: Visualising image classification models and

saliency maps. ArXiv Preprint ArXiv, 1312, 6034.
Steele, J. D., & Paulus, M. P. (2019). Pragmatic neuroscience for clinical
psychiatry. The British Journal of Psychiatry, 215(1), 404–408.
Sugar, J., Witter, M. P., van Strien, N., & Cappaert, N. (2011). The retrosplenial cortex: Intrinsic connectivity and connections with the (Para)

hippocampal region in the rat. An interactive connectome. Frontiers in

Neuroinformatics, 5, 7.

Suk, H.-I., Wee, C.-Y., Lee, S.-W., & Shen, D. (2015). Supervised discrimina
tive group sparse representation for mild cognitive impairment diagnosis. Neuroinformatics, 13(3), 277–295.

Sundermann, B., Feder, S., Wersching, H., Teuber, A., Schwindt, W., Kugel, H.,
… Pfleiderer, B. (2017). Diagnostic classification of unipolar depression

based on resting-state functional connectivity MRI: Effects of generalization to a diverse sample. Journal of Neural Transmission, 124(5), 589–605.
Supekar, K., Menon, V., Rubin, D., Musen, M., & Greicius, M. D. (2008).

Network analysis of intrinsic functional brain connectivity in
Alzheimer's disease. PLoS Computational Biology, 4(6), e1000100.
Suzuki, W. L., & Amaral, D. G. (1994). Perirhinal and parahippocampal cor
tices of the macaque monkey: Cortical afferents. Journal of Comparative Neurology, 350(4), 497–533.

Tong, T., Gray, K., Gao, Q., Chen, L., Rueckert, D., & Alzheimer's Disease
Neuroimaging Initiative. (2017). Multi-modal classification of

Alzheimer's disease using nonlinear graph fusion. Pattern Recognition,

63, 171–181.



van Dijk, K. R., Hedden, T., Venkataraman, A., Evans, K. C., Lazar, S. W., &
Buckner, R. L. (2009). Intrinsic functional connectivity as a tool for

human connectomics: Theory, properties, and optimization. Journal of
Neurophysiology, 103(1), 297–321.
Wang, K., Liang, M., Wang, L., Tian, L., Zhang, X., Li, K., & Jiang, T. (2007).

Altered functional connectivity in early Alzheimer's disease: A restingstate fMRI study. Human Brain Mapping, 28(10), 967–978.
Wang, X., Ren, Y., & Zhang, W. (2017). Depression disorder classification

of fMRI data using sparse low-rank functional brain network and

graph-based features. Computational and Mathematical Methods in

Medicine, 2017, 1–11.

Wee, C.-Y., Yap, P.-T., Zhang, D., Wang, L., & Shen, D. (2012). Constrained

sparse functional connectivity networks for MCI classification. In Inter
national Conference on Medical Image Computing and Computer-Assisted

–
Intervention (MICCAI), (pp. 212 219). Springer.
Wee, C.-Y., Yap, P.-T., Zhang, D., Wang, L., & Shen, D. (2014). Group
constrained sparse fMRI connectivity modeling for mild cognitive impairment identification. Brain Structure and Function, 219(2), 641–656.

Wei, M., Qin, J., Yan, R., Li, H., Yao, Z., & Lu, Q. (2013). Identifying major

depressive disorder using Hurst exponent of resting-state brain networks. Psychiatry Research: Neuroimaging, 214(3), 306–312.
Wessa, M., & Lois, G. (2015). Brain functional effects of psychopharmaco
logical treatment in major depression: A focus on neural circuitry of
affective processing. Current Neuropharmacology, 13(4), 466–479.

Wolz, R., Aljabar, P., Hajnal, J. V., Lötjönen, J., Rueckert, D., & Alzheimer's
Disease Neuroimaging Initiative. (2012). Nonlinear dimensionality

reduction combining MR imaging with non-imaging information. Medical Image Analysis, 16(4), 819–830.
World Health Organization. (2017). Depression and other common mental

disorders: Global health estimates. Technical Report, World Health

Organization.
Wu, G.-R., & Marinazzo, D. (2015). Point-process deconvolution of fMRI

BOLD signal reveals effective connectivity alterations in chronic pain
patients. Brain Topography, 28(4), 541–547.
Ye, M., Yang, T., Qing, P., Lei, X., Qiu, J., & Liu, G. (2015). Changes of func
tional brain networks in major depressive disorder: A graph theoretical
analysis of resting-state fMRI. PLoS One, 10(9), e0133775.

Yeo, B. T., Krienen, F. M., Sepulcre, J., Sabuncu, M. R., Lashkari, D.,
Hollinshead, M., … Buckner, L. (2011). The organization of the human

cerebral cortex estimated by intrinsic functional connectivity. Journal
of Neurophysiology, 106(3), 1125–1165.

Yoshida, K., Shimizu, Y., Yoshimoto, J., Takamura, M., Okada, G., Okamoto, Y.,

… Doya, K. (2017). Prediction of clinical depression scores and detection

of changes in whole-brain using resting-state functional MRI data with
partial least squares regression. PLoS One, 12(7), e0179638.
Zeng, L.-L., Shen, H., Liu, L., & Hu, D. (2014). Unsupervised classification of

major depression using functional connectivity MRI. Human Brain
Mapping, 35(4), 1630–1641.

Zhong, X., Shi, H., Ming, Q., Dong, D., Zhang, X., Zeng, L.-L., & Yao, S.
(2017). Whole-brain resting-state functional connectivity identified

major depressive disorder: A multivariate pattern analysis in two inde
pendent samples. Journal of Affective Disorders, 218, 346–352.
Zhuang, J., Peltier, S., He, S., LaConte, S., & Hu, X. (2008). Mapping the

connectivity with structural equation modeling in an fMRI study of
shape-from-motion task. NeuroImage, 42(2), 799–806.


JUN ET AL . 5011


APPENDIX


Name of the ROIs in the Yeo template


TABLE A1 The index and name of the ROIs in the Yeo template (Thomas Yeo et al., 2011). The indices 1–57 and the indices 58–114 refer,

respectively, to the left- and right-hemispheric regions


Index ROI label Index ROI label


1 Striate cortex (Striate) 58 Striate cortex (Striate)


2 Extrastriate cortex (ExStr) 59 Extrastriate cortex (ExStr)


3 Striate cortex (Striate) 60 Striate cortex (Striate)


4 Extrastriate inferior (ExStrInf) 61 Extrastriate inferior (ExStrInf)


5 Extrastriate superior (ExStrSup) 62 Extrastriate superior (ExStrSup)


6 Somatomotor A (SomMotA) 63 Somatomotor A (SomMotA)


7 Central (cent) 64 Central (cent)


8 S2 (S2) 65 S2 (S2)


9 Insula (Ins) 66 Insula (Ins)


10 Auditory (Aud) 67 Auditory (Aud)


11 Temporal occipital (TempOcc) 68 Temporal occipital (TempOcc)


12 Parietal occipital (ParOcc) 69 Parietal occipital (ParOcc)


13 Superior parietal lobule (SPL) 70 Superior parietal lobule (SPL)


14 Temporal occipital (TempOcc) 71 Temporal occipital (TempOcc)


15 Postcentral (PostC) 72 Postcentral (PostC)


16 Frontal eye fields (FEF) 73 Frontal eye fields (FEF)


17 Precentral ventral (PrCv) 74 Precentral ventral (PrCv)


18 Parietal operculum (ParOper) 75 Parietal operculum (ParOper)


19 Precentral ventral (PrCv) 76 Precentral (PrC)


20 Insula (Ins) 77 Precentral ventral (PrCv)


21 Parietal medial (ParMed) 78 Insula (Ins)


22 Frontal medial (FrMed) 79 Parietal medial (ParMed)


23 Inferior parietal lobule (IPL) 80 Frontal medial (FrMed)


24 Dorsal prefrontal cortex (PFCd) 81 Inferior parietal lobule (IPL)


25 Lateral prefrontal cortex (PFCl) 82 Dorsal prefrontal cortex (PFCd)


26 Ventral prefrontal cortex (PFCv) 83 Lateral prefrontal cortex (PFCl)


27 Orbital frontal cortex (OFC) 84 Lateral ventral prefrontal cortex (PFClv)


28 Medial posterior prefrontal cortex (PFCmp) 85 Ventral prefrontal cortex (PFCv)


29 Temporal pole (TempPole) 86 Medial posterior prefrontal cortex (PFCmp)


30 Orbital frontal cortex (OFC) 87 Cingulate anterior (Cinga)


31 Temporal (Temp) 88 Temporal pole (TempPole)


32 Intraparietal sulcus (IPS) 89 Orbital frontal cortex (OFC)


33 Dorsal prefrontal cortex (PFCd) 90 Temporal (Temp)


34 Lateral prefrontal cortex (PFCl) 91 Intraparietal sulcus (IPS)


35 Lateral ventral prefrontal cortex (PFClv) 92 Dorsal prefrontal cortex (PFCd)


36 Cingulate anterior (Cinga) 93 Lateral prefrontal cortex (PFCl)


37 Temporal (Temp) 94 Cingulate anterior (Cinga)


38 Inferior parietal lobule (IPL) 95 Temporal (Temp)


39 Dorsal prefrontal cortex (PFCd) 96 Inferior parietal lobule (IPL)


40 Lateral prefrontal cortex (PFCl) 97 Lateral dorsal prefrontal cortex (PFCld)


41 Lateral ventral prefrontal cortex (PFClv) 98 Lateral ventral prefrontal cortex (PFClv)


(Continues)


5012 JUN ET AL .


TABLE A1 (Continued)


43 Precuneus (pCun) 100 Precuneus (pCun)


44 Cingulate posterior (Cingp) 101 Cingulate posterior (Cingp)


45 Inferior parietal lobule (IPL) 102 Temporal (Temp)


46 Dorsal prefrontal cortex (PFCd) 103 Inferior parietal lobule (IPL)


47 Posterior cingulate cortex (PCC) 104 Dorsal prefrontal cortex (PFCd)


48 Medial prefrontal cortex (PFCm) 105 Posterior cingulate cortex (PCC)


49 Temporal (Temp) 106 Medial prefrontal cortex (PFCm)


50 Inferior parietal lobule (IPL) 107 Temporal (Temp)


51 Dorsal prefrontal cortex (PFCd) 108 Anterior temporal (AntTemp)


52 Lateral prefrontal cortex (PFCl) 109 Dorsal prefrontal cortex (PFCd)


53 Ventral prefrontal cortex (PFCv) 110 Ventral prefrontal cortex (PFCv)


54 Inferior parietal lobule (IPL) 111 Inferior parietal lobule (IPL)


55 Retrosplenial (Rsp) 112 Retrosplenial (Rsp)


56 Parahippocampal cortex (PHC) 113 Parahippocampal cortex (PHC)


57 Temporal parietal (TempPar) 114 Temporal parietal (TempPar)


Note: Central visual network = (1–12, 58–59); peripheral visual network = (3–5, 60–63); somatomotor network = (6–10, 63–67); dorsal attention
network = (11–17, 68–74); salience/ventral attention network = (18–28, 75–87); limbic = (29–30, 88–89); control network = (31–44, 90–101); default

network = (45–56, 102–113); temporal parietal = (57, 114).


TABLE A2 Discriminant effective connectivities selected by LASSO feature selection method from all folds in cross-validation. We

highlighted the connectivities selected from sensitivity analysis. For corresponding connections, the index and name of the ROI are presented


Index Source ROI Index Destination ROI


62 Extrastriate superior, right 5 Extrastriate superior, left


111 Inferior parietal lobule, right 54 Inferior parietal lobule, left


46 Dorsal prefrontal cortex, left 104 Dorsal prefrontal cortex, right


38 Inferior parietal lobule, left 39 Dorsal prefrontal cortex, left


41 Lateral ventral prefrontal cortex, left 39 Dorsal prefrontal cortex, left


20 Insula, left 19 Precentral ventral, left


35 Lateral ventral prefrontal cortex, left 84 Lateral ventral prefrontal cortex, right


54 Inferior parietal lobule, left 12 Parietal occipital, left


69 Parietal occipital, right 12 Parietal occipital, left


32 Intraparietal sulcus, left 91 Intraparietal sulcus, right


67 Auditory, right 66 Insula, right


79 Parietal medial, right 21 Parietal medial, left


88 Temporal pole, right 113 Parahippocampal cortex, right


103 Inferior parietal lobule, right 50 Inferior parietal lobule, left


48 Medial prefrontal cortex, left 46 Dorsal prefrontal cortex, left


39 Dorsal prefrontal cortex, left 82 Dorsal prefrontal cortex, right


23 Inferior parietal lobule, left 44 Cingulate posterior, left


82 Dorsal prefrontal cortex, right 39 Dorsal prefrontal cortex, left


97 Lateral dorsal prefrontal cortex, right 104 Dorsal prefrontal cortex, right


3 Striate cortex, left 24 Dorsal prefrontal cortex, left


6 Somatomotor A, left 15 Postcentral, left


75 Parietal operculum, right 84 Lateral ventral prefrontal cortex, right


JUN ET AL . 5013


TABLE A2 (Continued)


Index Source ROI Index Destination ROI


34 Lateral prefrontal cortex, left 17 Precentral ventral, left


89 Orbital frontal cortex, right 30 Orbital frontal cortex, left


81 Inferior parietal lobule, right 75 Parietal operculum, right


15 Postcentral, left 6 Somatomotor A, left


16 Frontal eye fields, left 24 Dorsal prefrontal cortex, left


16 Frontal eye fields, left 33 Dorsal prefrontal cortex, left


19 Precentral ventral, left 24 Dorsal prefrontal cortex, left


77 Precentral ventral, right 19 Precentral ventral, left


22 Frontal medial, left 24 Dorsal prefrontal cortex, left


25 Lateral prefrontal cortex, left 24 Dorsal prefrontal cortex, left


3 Striate cortex, left 4 Extrastriate inferior, left


41 Lateral ventral prefrontal cortex, left 98 Lateral ventral prefrontal cortex, right


72 Postcentral, right 91 Intraparietal sulcus, right


20 Insula, left 78 Insula, right


80 Frontal medial, right 82 Dorsal prefrontal cortex, right


1 Striate cortex, left 58 Striate cortex, right


16 Frontal eye fields, left 22 Frontal medial, left


46 Dorsal prefrontal cortex, left 24 Dorsal prefrontal cortex, left


10 Auditory, left 67 Auditory, right


85 Ventral prefrontal cortex, right 26 Ventral prefrontal cortex, left


32 Intraparietal sulcus, left 13 Superior parietal lobule, left


112 Retrosplenial, right 55 Retrosplenial, left


18 Parietal operculum, left 94 Cingulate anterior, right


24 Dorsal prefrontal cortex, left 76 Precentral, right


47 Posterior cingulate cortex, left 105 Posterior cingulate cortex, right


109 Dorsal prefrontal cortex, right 82 Dorsal prefrontal cortex, right


93 Lateral prefrontal cortex, right 35 Lateral ventral prefrontal cortex, left


26 Ventral prefrontal cortex, left 85 Ventral prefrontal cortex, right


29 Temporal pole, left 76 Precentral, right


26 Ventral prefrontal cortex, left 94 Cingulate anterior, right


29 Temporal pole, left 94 Cingulate anterior, right


8 S2, left 65 S2, right


76 Precentral, right 24 Dorsal prefrontal cortex, left


68 Temporal occipital, right 69 Parietal occipital, right


79 Parietal medial, right 24 Dorsal prefrontal cortex, left


39 Dorsal prefrontal cortex, left 76 Precentral, right


82 Dorsal prefrontal cortex, right 24 Dorsal prefrontal cortex, left


98 Lateral ventral prefrontal cortex, right 89 Orbital frontal cortex, right


17 Precentral ventral, left 74 Precentral ventral, right


61 Extrastriate inferior, right 4 Extrastriate inferior, left


92 Dorsal prefrontal cortex, right 24 Dorsal prefrontal cortex, left


10 Auditory, left 9 Insula, left


92 Dorsal prefrontal cortex, right 33 Dorsal prefrontal cortex, left


59 Extrastriate cortex, right 58 Striate cortex, right


70 Superior parietal lobule, right 13 Superior parietal lobule, left


6 Somatomotor A, left 63 Somatomotor A, right


(Continues)


5014 JUN ET AL .


TABLE A2 (Continued)


Index Source ROI Index Destination ROI


102 Temporal, right 24 Dorsal prefrontal cortex, left


99 Medial posterior prefrontal cortex, right 42 Medial posterior prefrontal cortex, left


55 Retrosplenial, left 112 Retrosplenial, right


17 Precentral ventral, left 27 Orbital frontal cortex, left


66 Insula, right 76 Precentral, right


91 Intraparietal sulcus, right 96 Inferior parietal lobule, right


24 Dorsal prefrontal cortex, left 27 Orbital frontal cortex, left


113 Parahippocampal cortex, right 24 Dorsal prefrontal cortex, left


77 Precentral ventral, right 76 Precentral, right


44 Cingulate posterior, left 101 Cingulate posterior, right


80 Frontal medial, right 76 Precentral, right


111 Inferior parietal lobule, right 69 Parietal occipital, right


80 Frontal medial, right 94 Cingulate anterior, right


4 Extrastriate inferior, left 61 Extrastriate inferior, right


41 Lateral ventral prefrontal cortex, left 27 Orbital frontal cortex, left


97 Lateral dorsal prefrontal cortex, right 40 Lateral prefrontal cortex, left


22 Frontal medial, left 16 Frontal eye fields, left


78 Insula, right 20 Insula, left


92 Dorsal prefrontal cortex, right 94 Cingulate anterior, right


13 Superior parietal lobule, left 70 Superior parietal lobule, right


98 Lateral ventral prefrontal cortex, right 76 Precentral, right


3 Striate cortex, left 5 Extrastriate superior, left


11 Temporal occipital, left 14 Temporal occipital, left


12 Parietal occipital, left 14 Temporal occipital, left


96 Inferior parietal lobule, right 38 Inferior parietal lobule, left


111 Inferior parietal lobule, right 103 Inferior parietal lobule, right


74 Precentral ventral, right 27 Orbital frontal cortex, left


11 Temporal occipital, left 68 Temporal occipital, right


82 Dorsal prefrontal cortex, right 27 Orbital frontal cortex, left


84 Lateral ventral prefrontal cortex, right 27 Orbital frontal cortex, left


23 Inferior parietal lobule, left 50 Inferior parietal lobule, left


75 Parietal operculum, right 81 Inferior parietal lobule, right


112 Retrosplenial, right 56 Parahippocampal cortex, left


60 Striate cortex, right 61 Extrastriate inferior, right


99 Medial posterior prefrontal cortex, right 27 Orbital frontal cortex, left


10 Auditory, left 57 Temporal parietal, left


14 Temporal occipital, left 57 Temporal parietal, left


45 Inferior parietal lobule, left 50 Inferior parietal lobule, left


108 Anterior temporal, right 27 Orbital frontal cortex, left


