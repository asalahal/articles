[View Article Online](https://doi.org/10.1039/d0cp03596c)


[View Journal](https://pubs.rsc.org/en/journals/journal/CP)

# PCCP


Physical Chemistry Chemical Physics

##### Accepted Manuscript


This article can be cited before page numbers have been issued, to do this please use: P. Gao, J. Zhang,





This is an Accepted Manuscript, which has been through the
Royal Society of Chemistry peer review process and has been
accepted for publication.


Accepted Manuscripts are published online shortly after acceptance,
before technical editing, formatting and proof reading. Using this free
service, authors can make their results available to the community, in
citable form, before we publish the edited article. We will replace this
Accepted Manuscript with the edited and formatted Advance Article as
soon as it is available.


You can find more information about Accepted Manuscripts in the
[Information for Authors.](http://www.rsc.org/Publishing/Journals/guidelines/AuthorGuidelines/JournalPolicy/accepted_manuscripts.asp)


Please note that technical editing may introduce minor changes to the
text and/or graphics, which may alter content. The journal’s standard
[Terms & Conditions and the](http://www.rsc.org/help/termsconditions.asp) [Ethical guidelines still apply. In no event](http://www.rsc.org/publishing/journals/guidelines/)
shall the Royal Society of Chemistry be held responsible for any errors
or omissions in this Accepted Manuscript or any consequences arising
from the use of any information it contains.

#### rsc.li/pccp






Page 1 of 7 Physical Chemistry Chemical Physics



[View Article Online](https://doi.org/10.1039/d0cp03596c)
DOI: 10.1039/D0CP03596C


###### Accurate Predictions of Aqueous Solubility of Drug Molecules via Multilevel Graph Convolutional Network (MGCN) and SchNet Ar- chitectures

Peng Gao, _[b]_ Jie Zhang, _[a][∗][,][c]_,Yuzhu Sun, _[c]_ and Jianguo Yu, _[c]_


Deep learning based methods have been widely applied to predict various kinds of molecular properties
in pharmaceutical industry with increasingly more success. In this study, we propose two novel
models for aqueous solubility predictions, based on Multilevel Graph Convolutional Network (MGCN)
and SchNet architectures, respectively. The advantage of MGCN lies in the fact that it could
extract the graph features of the target molecules directly from the (3D) structural information;
therefore, it doesn’t need to rely on lots of intra-molecular descriptors to learn the features, which
are of significance for accurate predictions of the molecular properties. The SchNet performs well
at modelling the interatomic interactions inside molecule, and such a deep learning architecture is
also capable of extracting structural information and further predicting the related properties. The
actual accuracy of these two novel approaches were systematically benchmarked with four different
independent datasets. We found that both the MGCN and SchNet models performed well for aqueous
solubility predictions. In the future, we believe such promising predictive models are applicable to
enhance the efficiency of the screenings, crystallization and delivery of drug molecules, essentially as
a useful tool to promote the development of molecular pharmaceutics.



1 Introduction


In the field of molecular pharmaceutics, aqueous solubility is an
important index, which needs to be considered carefully during
the discovery of ideal drug molecules. Without a sufficient solubility, a designed drug molecule is difficult to be absorbed or
delivered efficiently inside human body. [1–7] Thus, despite the increasing interest in improving the efficiency of drug discovery, a
reliable approach for direct yet accurate aqueous solubility prediction still remains to be a challenging goal for computational
pharmaceutics. Currently, standard alchemical free energy methods were routinely applied by researchers to predict the solubility of drug molecules, via simulating the thermodynamic integration and free energy perturbation. [8] And, moreover, one recently
developed approach conductor-like screening model for real solvents (COSMO-RS), combing quantum chemical surface calculations with fluid phase thermodynamics, was also proved to be accurate for complex solubility prediction. [9] However, considering


_a_ _Centre of Chemistry and Chemical Biology, Guangzhou Regenerative Medicine and_
_Health-Guangdong Laboratory, Guangzhou 53000, China;_ _[b]_ _School of Chemistry and_
_Molecular Bioscience, University of Wollongong, NSW 2500, Australia;_ _c_ _School of_
_Chemical Engineering, East China University of Science and Technology, Shanghai_
_200237, China E-mail: j.chang@mail.ecust.edu.cn;_

_†_ Electronic Supplementary Information (ESI) available at https://github.com/jeahz/Aqueous-solubility-prediction-with-GCN; See DOI: 10.5281/zenodo.4057261.



the high computational cost of running these advanced methods,
as well as their low transferability among different class of drug
molecules, their applications in real practice may be limited.


And, in recent years, increasingly more data sets of aqueous solubility of drug molecules were collected and organised
by experimental researchers, indicating that a statistical or a
more advanced data-driven approach may be available for this
kind of property predictions without running complicate computations. [10] However, to abstract a reliable rule of solubility prediction from the large amount of existing experimental data, a decent tool is needed. [11–14] Currently, deep learning is emerging as
a powerful data-driven approach in many fields of chemistry, [15–33]

for instance, in accelerating conformer optimisation, [34–36] developing molecular dynamics (MD) force fields, [37–40] assisting materials design, [41–44] and even predicting atomic and molecular
properties. [45–50] To the best of our knowledge, some deep learning based approaches have been proposed to predict the aqueous solubility of drug molecules and display considerable accuracy, like Undirected Graph Recursive Neural Network (UG-RNN),
which extracts information from the acyclic graphs of the molecular structures. [51]


The dissolution of drug is a dynamic process, indicating that
such a physical phenomenon actually occurs via two simultaneous
yet reverse processes: dissolution and precipitation. [52] An equi

1–7 | 1


Physical Chemistry Chemical Physics Page 2 of 7


[View Article Online](https://doi.org/10.1039/d0cp03596c)
DOI: 10.1039/D0CP03596C


_̸_



librium can be reached only if these two processes can proceed
with the same rate. The aqueous solubility of a drug molecule
largely depends on the solubility equilibrium, which can balance
the interactions between the solved molecule and water. And,
inter-molecular force between a drug molecule and water is determined by its conformation features. Therefore, to develop the
data-driven tool for aqueous solubility prediction, the key factor

lies in the overall and accurate extraction of the solved molecule’s

structural information. Currently, two promising deep learning
architectures, Multilevel Graph Convolutional Network (MGCN)
and SchNet, have drawn considerable attention, due to their high
applicability in molecular property predictions. [53] The former is
able to obtain accurate information of the target molecules directly via resolving its 3D structure at different atomic levels;
while the later can accurately model the interatomic interactions _̸_
inside the molecule, and therefore, can also extract valuable information for molecular property predictions. [54]


In this study, two models based on MGCN and SchNet architectures were developed, respectively, for molecular solubility prediction. The performance of each approach were demonstrated
in details; and a systematic benchmark of these two approaches
was also conducted. We hope the developed data-driven tools in
this study could enhance the efficiency of drug discovery.


2 Computational Details


**2.1** **Multilevel Graph Convolutional Network (MGCN)**


Molecules can be expressed in the form of graph with inclusion of
the chemical information, as atoms can be represented by nodes,
and chemical bonds by edges. Generally, such a undirected graph
network is capable of providing valuable structure information
that is correlated with various molecular properties. However,
due to the fact that the regular graph neural network doesn’t
apply decent symmetry functions to extract the atomic environment information accurately, especially the spacial information,
their prediction accuracy may be limited. To efficiently resolve
the intra-molecular environment and improve the prediction accuracy of molecular property, a novel approach, multilevel graph
convolutional network (MGCN) was recently proposed, such an
architecture treats the interatomic interactions more decently. [53]

It can model the intra-molecular interactions between two, three
and even more atoms, and such a level by level mechanism (multilevel) enables the accurate predictions of molecular properties
within the framework of graph neural network.


A general architecture of MGCN was illustrated in Figure 1, and
the key components include atom embedding, radial basis function (RBF), interaction and readout layers. The embedding layer
can initialise the atomic/interatomic information by generating a
node and edge containing matrix, and the RBF layer can convert
this generated matrix into a distance tensor. Then, several interaction layers, which are the most crucial part of this novel neural
network, can play their roles. That said, the decent interactions

between atoms inside molecules can be modelled at different in
teratomic levels. Actually, the level by level interaction layer was
designed via a hierarchical architecture by Liu and co-workers. [53]

With such a continuous architecture, MGCN could effectively ex

2 | 1–7



tract the structural information of the target molecule; and the
successful preservation of this kind of information is crucial for
the accurate predictions of various properties. Following the interaction layers, the atom representations obtained at different
levels can be recorded at the readout layer. At this stage, the final
prediction of a specific property based on these recorded information can be conducted. And, numerically speaking, such a property actually satisfy the summation over all the interatomic contributions. As shown in formula (1), interaction layer of MGCN
involves multilevel interactions between nodes and their neighbours, such as pair-wise interaction, triple-wise interaction and

etc..


_N_
_a_ _[l]_ _i_ [+][1] = ∑ _h_ _v_ ( _a_ _[l]_ _j_ _[,]_ _[e]_ _[l]_ _i j_ _[,]_ _[d]_ _[i j]_ [)] (1)
_j_ =1 _,_ _j_ = _̸_ _i_


where, _a_ _[l]_ _i_ [is atom-wise representation of molecular graph,]
which involves ( _l_ + 1) _−_ _wise_ interactions between ith atom and
its neighbours of layer l; Edge embedding ( _e_ _i j_ ) provides the extra
bond information between _**i**_ th atom and _**j**_ th atom; Distance tensor ( _d_ _i j_ ) controls the magnitude of impact in each pair of atoms;
_h_ _v_ is the function that collects the message from the neighbours
of the _**i**_ th atom to generate node embedding _a_ _[l]_ _i_ [+][1] . In this study,
the default hyper parameters of MGCN were applied. [53]


**2.2** **SchNet**


SchNet is taken as a variant of the Deep Tensor Neural Networks (DTNNs), [55] which is capable of resolving atomic environments inside molecules via performing high-accuracy interatomic
modelling. The SchNet architecture owns many similar building
blocks, which are also fundamental to DTNNs; however, there ex
ists a essential difference between these two architectures. Within

the framework of DTNN, the interactions between atoms can be
modelled via applying tensor layers, the interatomic distances
and atom representations are read from a fixed parameterized
tensor. Usually, a low-rank factor is applied to improve the computational efficiency. The SchNet architecture applies continuousfilter convolutions with filter-generating networks to deal with
the interactions between atoms more accurately. Such a novel
approach was recently applied by K. T. Schütt and co-workers in
modelling potential-energy surface, the obtained accuracy can be
comparable with quantum calculations. [54] Among the blocks of
SchNet, the most important ones are atom embedding, interaction refinement and atom-wise contributions layers. Atom embedding indicates the initialisation of atoms inside the molecule,
and is only dependent on the its chemical environment. For the
interaction refinement, the SchNet applies a filter generator to
keep the filters continuous, to handle the situation that the atoms
do not lie on a regular grid and thus tend to lack the implicit
neighbourhood relationship.


_N_
_a_ _[l]_ _i_ [+][1] = ∑ _a_ _[l]_ _j_ _[◦]_ _[ω]_ _[l]_ [(] _[d]_ _[i j]_ [)] (2)
_j_ =0


where, _ω_ _[l]_ is a filter-generating network to map the atom positions to the corresponding values of the filter bank; _◦_ represents


Page 3 of 7 Physical Chemistry Chemical Physics


Fig. 1 Illustrations of the solubility prediction model with SchNet and MGCN architectures



[View Article Online](https://doi.org/10.1039/d0cp03596c)
DOI: 10.1039/D0CP03596C



the element-wise multiplication. Formula (2) indicates that interaction layer of SchNet adopts a continuous filter generator to cope
with the non-euclidean geometry of molecular structure. Compared to MGCN that adopts a multilevel interaction layer, SchNet
can stack several interaction layers on each other. And in this
study, the default hyper parameters of SchNet were applied. [54,55]


In this study, both of our models were trained for 30,000 epochs
with mini-batch stochastic gradient descent using the ADAM optimizer, the learning rate was set to 0.0001.


**2.3** **Data sets for model development**


In this study, two independent data sets were applied for predictive model training and validation comparison. The Delaney
data set provides experimental solubility of drug molecules (log
mol/L at 25 _[◦]_ C). [56] And, the Huuskonen data set, which was
built by Jarmo Huuskonen from the AQUASOL database and the
PHYSPROP database, is mainly composed of experimental data
of organic molecules. [57] The solubility of these molecules were
also expressed in log mol/L, and the measurement temperature
is ranged from 20 to 25 _[◦]_ C. The solubility challenge data set and
intrinsic solubility data set were also applied as test sets to assess the performance of the predictive models. [58,59] Moreover, to
note, there exist considerable similarities between Delaney and
Huuskonen in molecular properties distribution; and therefore,

the cross-validations of the models between these two data sets

are also important.


The two original data sets (Delaney and Huuskonen) were randomly divided with a ratio of 9:1; and 90% was applied as the



Table 1 The summary of the performances of MGCN and SchNet models
in aqueous solubility predictions on Delaney and Huuskonen data sets.


Data set Model **R** [2] _a_ ) **MAE** _b_ ) **RMSE**

Delaney MGCN 0.9979 0.0904 0.1250
SchNet 0.9981 0.0573 0.0983
Ref [51] UG-RNN 0.9200 0.4300 0.5800
Ref [51] UG-RNN + logP 0.9100 0.4600 0.6100
Huuskonen MGCN 0.9996 0.0349 0.0529

SchNet 0.9996 0.0357 0.0493
Ref [51] UG-RNN 0.9100 0.4600 0.6000
Ref [51] UG-RNN + logP 0.9100 0.4700 0.6100


_a_ ) The mean absolute error of predictions. _b_ ) The root man
square error of predictions.


training set, and the remaining 10% as the test set. To reasonably
assess the performance of the predictive models, some statistics

results were summarised in Table 1


3 Results and Discussions


**3.1** **Delaney data set**

The two predictive models, based on MGCN and SchNet architectures, respectively, were first trained with Delaney data set,
the performance of these two models are shown in Figure 2 for
comparison. We can see that both the two novel models perform better than the ones based on undirected graph recursive
neural networks (UG-RNN) [51] in aqueous solubility prediction, as


1–7 | 3


Physical Chemistry Chemical Physics Page 4 of 7


[View Article Online](https://doi.org/10.1039/d0cp03596c)
DOI: 10.1039/D0CP03596C



Fig. 2 The comparison between predicted and experimental aqueous solubility, the predictive models were trained on Delaney data set. (a).The
predictions were made using MGCN architecture; (b).The predictions
were made using SchNet architecture.


Fig. 3 The comparison between predicted and experimental aqueous solubility, the predictive models were trained on Huuskonen data set.(a).The
predictions were made using MGCN architecture; (b).The predictions
were made using SchNet architecture.


the prediction errors were largely reduced compared to previous
study. And moreover, from Figure 2, we can also see that the
MAE (mean absolute error) value of the MGCN predictive model
is larger than that of the SchNet predictive model, indicating that
the SchNet architecture is more suitable for solubility prediction

of this data set.


**3.2** **Huuskonen data set**

To further testify the performance of these two predictive models,
we also try another data set, Huuskonen, and the corresponding performance of these two models are shown in Figure 3. On
this data set, MGCN and SchNet approaches also perform better
than undirected graph recursive neural networks (UG-RNN) [51] in
aqueous solubility prediction. From Figure 3, it is the same as
Figure 2 that both the MAE value of the MGCN predictive model
is larger than that of the SchNet predictive model, indicating that
for aqueous solubility prediction of Huuskonen data set, SchNet
model performs better.


**3.3** **The performance of the predictive models on the se-**
**lected drug molecules**

To further testify the performances of the two predictive models,
based on MGCN and SchNet architectures, respectively, we also
loaded the developed models (trained on Delaney and Huuskonen data sets, respectively) for aqueous solubility predictions of
a separate set of typical drug molecules. The performance of the
developed predictive models were shown in Table 2. We can see


4 | 1–7



Fig. 4 The cross-validation test of MGCN predictive model between
Delaney and Huuskonen data sets. (a).The comparison between the predicted and experimental solubility of molecules in Delaney data sets using
the predictive model trained on Husskonen data set; (b).The comparison
between the predicted and experimental solubility of molecules in Huuskonen data sets using the predictive model trained on Delaney data

set.


Fig. 5 The cross-validation test of SchNet predictive model between
delaney and Huuskonen data sets. (a).The comparison between the predicted and experimental solubility of molecules in delaney data sets using
the predictive model trained on husskonen data set; (b).The comparison
between the predicted and experimental solubility of molecules in Huuskonen data sets using the predictive model trained on delaney data

set.


that the prediction results match well with experimentally measured values, indicating the high applicability of our models.


**3.4** **Cross-validation test**

To further assess the transferrability of the developed models, a
cross-validation study was also conducted among the two large
data sets, Delaney and Huuskonen. The models trained on each
data set were applied for predictions of the other, the crossvalidation test results were shown in Figure 4 and 5. In one
aspect, the low errors of the validation tests indicate the high
transferability of the predictive models; and in another aspect,
we also realise that the original data sets of experimental aqueous solubility remain to be organised and extended for further
prediction accuracy improvement.
To systematically find out the reason behind the error increase
of the cross-validation tests, we conducted a Tanimoto similarity
analysis using RDKit [60] among Delaney and Huuskonen data sets,
the details were shown in Figure 6 and 7, respectively. It is clear
that the averaged prediction error is negatively correlated with
the similarity values between these two data sets. And, we also
realise that both the diversity of the molecules contained in the
original data set, and the similarity of the application data set


Page 5 of 7 Physical Chemistry Chemical Physics


Table 2 The predicted aqueous solubility of some selected drug molecules



[View Article Online](https://doi.org/10.1039/d0cp03596c)
DOI: 10.1039/D0CP03596C



Compound name CAS no. _a_ ) Exptl.logS _b_ ) Pred.logS _c_ ) Pred.logS _d_ ) Pred.logS _e_ ) Pred.logS
antipyrine 60-80-0 -0.56 0.8564 -1.3842 0.6378 -1.7001
asprin 50-78-2 -1.72 -1.0584 -1.5845 -2.0521 -1.6428
atrazine 1912-24-9 -3.85 -3.9660 -4.6352 -4.0082 -4.3301

benzocaine 94-09-7 -2.32 -2.1275 -1.9652 -2.3767 -1.9817

chlordane 57-74-9 -6.86 -6.7745 -7.1772 -6.9353 -6.7126
chlorpyrifos 2921-88-2 -5.49 -4.9529 -5.1989 -5.1410 -4.7998
diazepam 439-14-5 -3.76 -3.6784 -4.3361 -3.6819 -3.6483
diazinon 33-41-5 -3.64 -3.6248 -3.5678 -3.5631 -3.1145

diuron 330-54-1 -3.80 -3.6668 -3.7366 -3.7868 -3.3698

lindane 58-89-9 -4.64 -4.5902 -4.6260 -4.6892 -4.6308

malathion 121-75-5 -3.37 -3.2140 -3.6760 -3.4222 -2.1675

nitrofurantoin 67-20-9 -3.47 -2.8553 -1.7759 -1.7766 -1.7375
parathion 56-38-2 -4.66 -4.6128 -3.8443 -4.6224 -3.8991
2,2 _′_,4,5,5 _′_ -PCB 37680-73-2 -7.89 -7.2174 -7.7384 -7.5750 -7.5650
phenobarbital 50-06-6 -2.34 -2.0179 -2.1703 -2.2519 -2.2446
phenolphthalein 77-09-8 -2.90 -2.7303 -3.5711 -2.8850 -3.7299
phenytoin 57-41-0 -3.99 -3.9674 -2.3859 -3.9698 -2.5555
prostaglandin E2 363-24-6 -2.47 -3.7244 -3.6960 -4.1595 -3.7884
testosterone 58-22-0 -4.09 -3.9088 -4.1144 -4.0576 -4.1095
theophylline 58-55-9 -1.39 -1.4580 -0.8613 -1.4331 -1.2754
_f_ ) **MAE** – – **0.3398** **0.5312** **0.3186** **0.5961**


_a_ ) Experimental data were taken from Ref 56 . _b_ ) The predictive model was trained on Delaney data set using MGCN architecture. _c_ ) The
predictive model was trained on Huuskonen data set using MGCN architecture. _[d]_ [)] The predictive model was trained on Delaney data
set using SchNet architecture. _[e]_ [)] The predictive model was trained on Huuskonen data set using SchNet architecture. _[f]_ [)] The mean
absolute error in predictions.



Fig. 6 The correlation between the Tanimoto similarity and average MAE
values of the cross-validation test between Delaney and Huuskonen data
sets, using MGCN predictive model. (a).The plot of the averaged MAE
values between the predicted and experimental solubility of molecules
in Huuskonen data set using the predictive model trained on Delaney
data set, with the tanimoto similarity of molecules in Huuskonen data
set with respect to molecules in Delaney data set; (b).The plot of the
averaged MAE values between the predicted and experimental solubility
of molecules in Delaney data set using the predictive model trained on
Huuskonen data set, with the tanimoto similarity of molecules in Delaney
data set with respect to molecules in Huuskonen data set.


with respect to the original one, can impact the model’s prediction
performance.

Considering the higher prediction errors of the cross-validation
tests shown above, we combined all the four data sets of experimental aqueous solubility, including Delaney, Huuskonen, solubility challenge, [58] and intrinsic solubility [59] data sets; and used
this combined data set to re-train the new predictive models. In



Fig. 7 The correlation between the Tanimoto similarity and average
MAE values of the cross-validation between Delaney and Huuskonen data
sets, using SchNet predictive model. (a).The plot of the averaged MAE
values between the predicted and experimental solubility of molecules
in Huuskonen data set using the predictive model trained on Delaney
data set, with the tanimoto similarity of molecules in Huuskonen data
set with respect to molecules in Delaney data set; (b).The plot of the
averaged MAE values between the predicted and experimental solubility
of molecules in Delaney data set using the predictive model trained on
Huuskonen data set, with the tanimoto similarity of molecules in Delaney
data set with respect to molecules in Huuskonen data set.


1–7 | 5


Physical Chemistry Chemical Physics Page 6 of 7


[View Article Online](https://doi.org/10.1039/d0cp03596c)
DOI: 10.1039/D0CP03596C


Conflicts of interest


There is no conflict of interests.


Acknowledgements


We thank the Australian Government for supporting Peng’s PhD
study via proding him an Australian International Postgraduate
Award scholarship. We also thank the NCI system, which is supported by the Australian Government (Project id: v15) to offer
computational resource to complete this project.



Fig. 8 The comparison between predicted and experimental aqueous
solubility, the predictive models were trained on the combined data set,
which includes Delaney, Huuskonen, solubility-challenging, and solubilityintrinsic data sets. (a).The predictions were made using MGCN architecture; (b).The predictions were made using SchNet architecture. The
orange points indicate the validation test of delaney data set; the blue
points indicate the validation test of Huuskonen data set; the wine points
indicate the validation test of solubility challenge data set; the dark cyan
points indicate the validation test of intrinsic solubility data set.


Figure 8, the performance of the new models were presented. We
can see that the overall MAE values were 0.2766 and 0.2898, for
MGCN and SchNet approaches, respectively. This has verified our
idea that the extension of data set can strengthen the predictive
models. However, considering the fact that the experimental measurement conditions may vary and even be inconsistent, thus the
performance of the predictive models may be limited. Therefore,
further organisation of the collected experimental data remains
to be of great importance for future study.


4 Conclusions


In this study, two novel deep learning architectures, MGCN and
SchNet, were applied for aqueous solubility predictions. These

two architectures were all focused on accurate solution and ex
traction of molecular structure information, therefore, suitable
for chemical properties predictions, and sometimes, their capabilities even outperform quantum methods. The predicted results of
aqueous solubility of the two predictive models, on Delaney and
Huuskonen data sets, match well with experimental values. And
moreover, a cross-validation study of the developed predictive
models was also conducted among these two independent data
sets, and the prediction errors were observed to be larger. One
possible reason for this lies in the fact that the sizes of the selected
data sets were relatively small, not covering enough molecular information. The Tanimoto similarity analysis was also carried out
to verify the correlation between the molecular similarity values
and the magnitude of the predicted errors. We also realise that
the current versions of the data set of experimental aqueous solubility can be reorganised and extended for more accurate models
development. Among the two proposed architectures, MGCN and
SchNet, we tend to recommend SchNet for potential researchers
to conduct aqueous solubility predictions. The most important
contribution of this study lies in the fact that it brings forth a new
graph neural network based deep learning methodology, which is
computationally affordable yet accurate, for chemical properties
prediction; and we believe in the future it can be applied in other
molecular representations prediction beyond aqueous solubility.


6 | 1–7



Notes and references


1 A. Alhalaweh, L. Roy, N. Rodríguez-Hornedo and S. P. Velaga,
_Molecular Pharmaceutics_, 2012, **9**, 2605–2612.

2 J. H. Fagerberg, Y. Al-Tikriti, G. Ragnarsson and C. A.
Bergström, _Molecular Pharmaceutics_, 2012, **9**, 1942–1952.

3 M. P. Lipert and N. Rodríguez-Hornedo, _Molecular Pharma-_
_ceutics_, 2015, **12**, 3535–3546.

4 J. Brinkmann, F. Rest, C. Luebbert and G. Sadowski, _Molecular_

_Pharmaceutics_, 2020, **17**, 2499–2507.

5 M. M. Knopp, L. Tajber, Y. Tian, N. E. Olesen, D. S. Jones,
A. Kozyra, K. Löbmann, K. Paluch, C. M. Brennan, R. Holm,
A. M. Healy, G. P. Andrews and T. Rades, _Molecular Pharma-_

_ceutics_, 2015, **12**, 3408–3419.

6 W. Zhang, A. Haser, H. H. Hou and K. Nagapudi, _Molecular_
_Pharmaceutics_, 2018, **15**, 1714–1723.

7 D. S. Palmer and J. B. O. Mitchell, _Molecular Pharmaceutics_,

2014, **11**, 2962–2972.

8 H. Liu, S. Dai and D.-e. Jiang, _The Journal of Physical Chem-_
_istry B_, 2014, **118**, 2719–2725.

9 J. Alsenz and M. Kuentz, _Molecular Pharmaceutics_, 2019, **16**,

4661–4669.

10 S. Zheng, X. Yan, Y. Yang and J. Xu, _Journal of Chemical Infor-_
_mation and Modeling_, 2019, **59**, 914–923.

11 S. Jaeger, S. Fulle and S. Turk, _Journal of Chemical Information_
_and Modeling_, 2018, **58**, 27–35.

12 P. Pogány, N. Arad, S. Genway and S. D. Pickett, _Journal of_
_Chemical Information and Modeling_, 2019, **59**, 1136–1146.

13 W. L. Chen, _Journal of Chemical Information and Modeling_,
2006, **46**, 2230–2255.

14 M. Fernandez, F. Ban, G. Woo, O. Isaev, C. Perez, V. Fokin,
A. Tropsha and A. Cherkasov, _Journal of Chemical Information_
_and Modeling_, 2019, **59**, 1306–1313.

15 A. C. Mater and M. L. Coote, _Journal of Chemical Information_
_and Modeling_, 2019, **59**, 2545–2559.

16 Y. Xu, J. Pei and L. Lai, _Journal of Chemical Information and_
_Modeling_, 2017, **57**, 2672–2685.

17 G. Klambauer, S. Hochreiter and M. Rarey, _Journal of Chemical_
_Information and Modeling_, 2019, **59**, 945–946.

18 Y. Zhou, S. Cahya, S. A. Combs, C. A. Nicolaou, J. Wang, P. V.
Desai and J. Shen, _Journal of Chemical Information and Mod-_
_eling_, 2019, **59**, 1005–1016.

19 F. Imrie, A. R. Bradley, M. van der Schaar and C. M. Deane,
_Journal of Chemical Information and Modeling_, 2018, **58**,

2319–2330.


Page 7 of 7 Physical Chemistry Chemical Physics



20 J. L. Baylon, N. A. Cilfone, J. R. Gulcher and T. W. Chittenden,
_Journal of Chemical Information and Modeling_, 2019, **59**, 673–

688.

21 N. Ståhl, G. Falkman, A. Karlsson, G. Mathiason and
J. Boström, _Journal of Chemical Information and Modeling_,
2019, **59**, 3166–3176.

22 N. Sturm, J. Sun, Y. Vandriessche, A. Mayr, G. Klambauer,
L. Carlsson, O. Engkvist and H. Chen, _Journal of Chemical In-_
_formation and Modeling_, 2019, **59**, 962–972.

23 J. A. Morrone, J. K. Weber, T. Huynh, H. Luo and W. D. Cornell, _Journal of Chemical Information and Modeling_, 0, **0**, null.

24 G. Scalia, C. A. Grambow, B. Pernici, Y.-P. Li and W. H.
Green, _Journal of Chemical Information and Modeling_, 2020,
**60**, 2697–2717.

25 M. Fernandez, F. Ban, G. Woo, M. Hsing, T. Yamazaki,
E. LeBlanc, P. S. Rennie, W. J. Welch and A. Cherkasov, _Jour-_
_nal of Chemical Information and Modeling_, 2018, **58**, 1533–

1543.

26 P. Morris, R. St. Clair, W. E. Hahn and E. Barenholtz, _Journal_
_of Chemical Information and Modeling_, 0, **0**, null.

27 J. G. Meyer, S. Liu, I. J. Miller, J. J. Coon and A. Gitter, _Journal_
_of Chemical Information and Modeling_, 2019, **59**, 4438–4449.

28 K. Yang, K. Swanson, W. Jin, C. Coley, P. Eiden, H. Gao,
A. Guzman-Perez, T. Hopper, B. Kelley, M. Mathea, A. Palmer,
V. Settels, T. Jaakkola, K. Jensen and R. Barzilay, _Journal of_
_Chemical Information and Modeling_, 2019, **59**, 3370–3388.

29 J.-H. Yuan, S. B. Han, S. Richter, R. C. Wade and D. B.
Kokh, _Journal of Chemical Information and Modeling_, 2020,
**60**, 1685–1699.

30 A. P. A. Janssen, S. H. Grimm, R. H. M. Wijdeven, E. B.
Lenselink, J. Neefjes, C. A. A. van Boeckel, G. J. P. van Westen
and M. van der Stelt, _Journal of Chemical Information and_
_Modeling_, 2019, **59**, 1221–1229.

31 M. Ragoza, J. Hochuli, E. Idrobo, J. Sunseri and D. R. Koes,
_Journal of Chemical Information and Modeling_, 2017, **57**, 942–

957.

32 S. Korkmaz, _Journal of Chemical Information and Modeling_,
2020, **Article ASAP**,.

33 P. Gao, J. Zhang, Q. Peng, J. Zhang and V.-A. Glezakou, _Jour-_
_nal of Chemical Information and Modeling_, 2020, **60**, 3746–

3754.

34 S. A. Meldgaard, E. L. Kolsbjerg and B. Hammer, _The Journal_
_of Chemical Physics_, 2018, **149**, 134104.

35 R. Ouyang, Y. Xie and D.-e. Jiang, _Nanoscale_, 2015, **7**, 14817–

14821.

36 K. H. Sørensen, M. S. Jørgensen, A. Bruix and B. Hammer, _The_
_Journal of Chemical Physics_, 2018, **148**, 241734.

37 J. Behler, _The Journal of Chemical Physics_, 2016, **145**, 170901.

38 J. Wang, S. Olsson, C. Wehmeyer, A. Pérez, N. E. Charron,
G. de Fabritiis, F. Noé and C. Clementi, _ACS Central Science_,

2019, **5**, 755–767.

39 V. Botu, R. Batra, J. Chapman and R. Ramprasad, _The Journal_



[View Article Online](https://doi.org/10.1039/d0cp03596c)
DOI: 10.1039/D0CP03596C


_of Physical Chemistry C_, 2017, **121**, 511–522.

40 J. Behler, _Angewandte Chemie International Edition_, 2017, **56**,
12828–12840.

41 R. B. Wexler, J. M. P. Martirez and A. M. Rappe, _Journal of the_
_American Chemical Society_, 2018, **140**, 4678–4683.

42 A. Mansouri Tehrani, A. O. Oliynyk, M. Parry, Z. Rizvi,
S. Couper, F. Lin, L. Miyagi, T. D. Sparks and J. Brgoch, _Jour-_
_nal of the American Chemical Society_, 2018, **140**, 9844–9853.

43 G. Panapitiya, G. Avendaño-Franco, P. Ren, X. Wen, Y. Li and
J. P. Lewis, _Journal of the American Chemical Society_, 2018,

**140**, 17508–17514.

44 Y. Bai, L. Wilbraham, B. J. Slater, M. A. Zwijnenburg, R. S.
Sprick and A. I. Cooper, _Journal of the American Chemical So-_
_ciety_, 2019, **141**, 9063–9071.

45 S. H. Martínez, M. A. Fernandez-Herrera, V. Uc-Cetina and
G. Merino, _Journal of Chemical Information and Modeling_, 0,
**0**, null.

46 X. Li, X. Yan, Q. Gu, H. Zhou, D. Wu and J. Xu, _Journal of_
_Chemical Information and Modeling_, 2019, **59**, 1044–1049.

47 C. W. Coley, R. Barzilay, W. H. Green, T. S. Jaakkola and K. F.
Jensen, _Journal of Chemical Information and Modeling_, 2017,

**57**, 1757–1772.

48 X. Wang, Z. Li, M. Jiang, S. Wang, S. Zhang and Z. Wei, _Jour-_
_nal of Chemical Information and Modeling_, 2019, **59**, 3817–

3828.

49 M. Rupp, R. Ramakrishnan and O. A. von Lilienfeld, _The Jour-_
_nal of Physical Chemistry Letters_, 2015, **6**, 3309–3313.

50 J. Cuny, Y. Xie, C. J. Pickard and A. A. Hassanali, _Journal of_
_Chemical Theory and Computation_, 2016, **12**, 765–773.

51 A. Lusci, G. Pollastri and P. Baldi, _Journal of Chemical Infor-_
_mation and Modeling_, 2013, **53**, 1563–1575.

52 P. Sanphui, V. K. Devi, D. Clara, N. Malviya, S. Ganguly and
G. R. Desiraju, _Molecular Pharmaceutics_, 2015, **12**, 1615–

1622.

53 C. Lu, Q. Liu, C. Wang, Z. Huang, P. Lin and L. He, _Molecular_
_Property Prediction: A Multilevel Quantum Interactions Model-_
_ing Perspective_, 2019.

54 K. T. Schütt, H. E. Sauceda, P.-J. Kindermans, A. Tkatchenko
and K.-R. Müller, _The Journal of Chemical Physics_, 2018, **148**,

241722.

55 K. T. Schütt, F. Arbabzadah, S. Chmiela, K.-R. Müller and

A. Tkatchenko, _Nat Commun_, 2017, **8**, 13890.

56 J. S. Delaney, _Journal of Chemical Information and Computer_

_Sciences_, 2004, **44**, 1000–1005.

57 J. Huuskonen, _Journal of Chemical Information and Computer_

_Sciences_, 2000, **40**, 773–777.

58 A. Llinàs, R. C. Glen and J. M. Goodman, _Journal of Chemical_
_Information and Modeling_, 2008, **48**, 1289–1303.

59 B. Louis, V. K. Agrawal and P. V. Khadikar, _European journal_
_of medicinal chemistry_, 2010, **45**, 4018—4025.

60 G. A. Landrum, _http://www.rdkit.org_, 2018.


1–7 | 7


